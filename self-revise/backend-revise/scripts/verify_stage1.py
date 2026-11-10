"""Read-only environment gate; never instantiate application stores or models."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New JSON report (existing files are not overwritten)')
    args = parser.parse_args()
    checks = []

    def record(name, action):
        started = time.perf_counter()
        try:
            detail = action()
            checks.append(dict(name=name, status='PASS', detail=detail,
                               seconds=round(time.perf_counter() - started, 3)))
        except Exception as error:
            # Do not print connection strings, environment values or credential errors.
            checks.append(dict(name=name, status='FAIL', detail=type(error).__name__,
                               seconds=round(time.perf_counter() - started, 3)))
        print(f"{checks[-1]['status']} {name}", flush=True)

    def require(condition, detail):
        if not condition:
            raise RuntimeError(detail)
        return detail

    def command(argv, timeout=120):
        result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True,
                                encoding='utf-8', errors='replace', timeout=timeout,
                                env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
        if result.returncode:
            raise RuntimeError('command failed')
        return result.stdout.strip()

    compose = ['docker', 'compose', '--env-file', 'infra/.env', '-f', 'infra/compose.yaml']
    record('unique_project_venv', lambda: require(
        Path(sys.prefix).resolve() == (ROOT / '.venv').resolve()
        and 'include-system-site-packages = true' not in (ROOT / '.venv/pyvenv.cfg').read_text().lower(),
        f'{sys.executable}; Python {sys.version.split()[0]}'))
    record('pip_check', lambda: command([sys.executable, '-m', 'pip', 'check']))
    record('cli_script_help', lambda: command([sys.executable, '-B', 'researchclaw.py', '--help']))
    record('cli_installed_help', lambda: command([str(Path(sys.executable).with_name('researchclaw.exe')), '--help']))
    # Import only; prohibit SQLite opens even if a future import gains side effects.
    imports = """import sys, sqlite3, importlib
sys.path[:0] = ['src', 'backend/src']
def forbidden(*a, **k):
    raise RuntimeError('SQLite is forbidden in environment checks')
sqlite3.connect = forbidden
for name in ['researchclaw.application.cli', 'researchclaw_api.main',
             'langgraph.checkpoint.postgres', 'psycopg', 'qdrant_client', 'mcp',
             'fitz', 'sentence_transformers', 'torchvision', 'timm', 'ragas',
             'langchain_openai']:
    importlib.import_module(name)
print('All entry/dependency imports passed; no app factory or model instantiated')
"""
    record('entry_imports_without_sqlite', lambda: command([sys.executable, '-B', '-c', imports], 180))
    for tool, argv in [('node', ['node', '--version']), ('npm', ['npm.cmd', '--version']),
                       ('docker', ['docker', 'version', '--format', '{{.Client.Version}} {{.Server.Version}}']),
                       ('compose', ['docker', 'compose', 'version'])]:
        record(tool, lambda argv=argv: command([shutil.which(argv[0]) or argv[0], *argv[1:]]))
    record('docker_linux', lambda: require(command(['docker', 'info', '--format', '{{.OSType}}']) == 'linux', 'linux'))
    record('compose_valid', lambda: command([*compose, 'config', '--quiet']))
    record('postgres_read_only_query', lambda: require('16.6' in command([
        *compose, 'exec', '-T', 'postgres', 'psql', '-U', 'researchclaw', '-d', 'researchclaw',
        '-Atc', 'BEGIN READ ONLY; SELECT current_database(), current_user, version(); COMMIT;']),
        'researchclaw/researchclaw; PostgreSQL 16.6; read-only SQL passed'))

    def volumes():
        ids = command([*compose, 'ps', '-q']).splitlines()
        mounts = []
        for container in ids:
            mounts.extend(json.loads(command(['docker', 'inspect', container, '--format', '{{json .Mounts}}'])))
        names = {mount.get('Name') for mount in mounts}
        return require({'researchclaw_postgres_data', 'researchclaw_qdrant_data'} <= names,
                       'Both fixed persistent volumes mounted')
    record('persistent_volumes', volumes)

    def qdrant():
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open('http://127.0.0.1:6333/healthz', timeout=10) as response:
            require(response.status == 200, 'health')
        with opener.open('http://127.0.0.1:6333/collections', timeout=10) as response:
            result = json.load(response)
        require(result['status'] == 'ok', 'collections')
        return {'health': 'ok', 'collection_count': len(result['result']['collections'])}
    record('qdrant_read_only', qdrant)
    record('frontend_build', lambda: command([shutil.which('npm.cmd') or 'npm.cmd', '--prefix', 'frontend', 'run', 'build']))
    packages = {}
    for name in ['researchclaw', 'fastapi', 'uvicorn', 'langgraph', 'langgraph-checkpoint-postgres',
                 'psycopg', 'qdrant-client', 'mcp', 'PyMuPDF', 'sentence-transformers', 'torch',
                 'torchvision', 'timm', 'ragas', 'datasets', 'langchain', 'langchain-core',
                 'langchain-community', 'langchain-openai']:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    passed = all(check['status'] == 'PASS' for check in checks)
    report = dict(created_at=datetime.now(timezone.utc).isoformat(), stage=1,
                  stage1_passed=passed, ready_for_stage2=passed,
                  business_migration_verified=False, checks=checks, packages=packages,
                  scope='No application startup, ingestion, database migration, model download or historical SQLite access')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(f'STAGE1 {"PASS" if passed else "FAIL"}; ready_for_stage2={passed}')
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())

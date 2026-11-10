"""Auditable, allowlisted backend closeout; dry run unless --apply is supplied.

Never follows junctions, touches live databases, or alters frozen test evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
OUT = ROOT / 'self-revise/cleanup-v20261110'
ARCHIVE = ROOT / 'self-revise/backend-revise/history/v20261029_sqlite'
LEGACY = ['documentation', 'planning', 'installation', 'verification',
          'historical_maintenance', 'runs']


def linked(p):
    try:
        return bool(p.lstat().st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)
    except OSError:
        # Missing/ACL-inaccessible long legacy paths are never deletion candidates.
        return True


def bounded(p):
    p.absolute().relative_to(ROOT)
    assert p != ROOT and not linked(p), str(p)
    return p


def sha(p):
    # Archive prefixes can exceed legacy MAX_PATH even when the source did not.
    with open('\\\\?\\' + str(p.absolute()), 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def walk(base, errors):
    def error(e):
        errors.append({'path': str(e.filename), 'error': str(e)})
    for parent, dirs, files in os.walk(base, followlinks=False, onerror=error):
        dirs[:] = [d for d in dirs if not linked(Path(parent) / d)]
        for name in files:
            p = Path(parent) / name
            if not linked(p):
                yield p


def rel(p):
    return p.relative_to(ROOT).as_posix()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--organize-receipts', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.organize_receipts:
        assert args.apply, 'Explicit --apply is required'
        organize_receipts()
        return
    result_path = OUT / 'result.json'
    if result_path.exists():
        raise SystemExit('Already applied; preserve the existing audit and do not repeat.')
    prior_path = OUT / 'plan.json'
    if args.apply and prior_path.exists():
        prior = json.loads(prior_path.read_text(encoding='utf-8'))
        if prior.get('applied'):
            execute(prior)
            return
    errors = []
    moves = []
    for name in LEGACY:
        src = ROOT / 'internal_experiments' / name
        if src.exists() and not linked(src):
            moves.append((src, ARCHIVE / 'internal_experiments' / name))
    src = ROOT / 'self-revise/backend-revise/internal_experiments/assets/knowledge_base'
    if src.exists() and not linked(src):
        moves.append((src, ARCHIVE / 'self-revise/backend-revise/internal_experiments/assets/knowledge_base'))
    # Protect tokenizer/model caches, current web state, PDFs and 194-paper backup.
    protected_paths = [ROOT / p for p in ['self-revise/环境.md', '.env', '.env.web',
        'infra/.env', 'self-revise/backend-revise/rag-evidence/expanded-rag-v20261109/boundary-after.json',
        'self-revise/backend-revise/agent-evidence/v20261110-architecture-live/paper-boundary.json']]
    protected_paths += list((ROOT / 'self-revise').rglob('*金标*.md'))
    protected = {rel(p): sha(p) for p in protected_paths if p.is_file()}
    delete_dirs = []
    # Pytest fixture roots contain only synthetic test_* directories; no live run.
    for p in (ROOT / 'self-revise/frontend-revise/local-runtime').iterdir():
        if not p.is_dir() or linked(p):
            continue
        try:
            children = list(p.iterdir())
        except OSError as e:
            errors.append({'path': rel(p), 'error': str(e), 'action': 'retained; ACL opaque'})
            continue
        if children and all(c.name.startswith('test_') for c in children):
            delete_dirs.append(p)
    # Bytecode and pytest caches are reproducible; no source or result JSON removed.
    for base in ['src', 'backend', 'tests', 'scripts', 'self-revise']:
        for parent, dirs, _ in os.walk(ROOT / base, followlinks=False):
            dirs[:] = [d for d in dirs if not linked(Path(parent) / d)]
            for name in list(dirs):
                if name in {'__pycache__', '.pytest_cache'}:
                    delete_dirs.append(Path(parent) / name)
                    dirs.remove(name)
    for name in ['__pycache__', '.pytest_cache']:
        p = ROOT / name
        if p.exists() and not linked(p):
            delete_dirs.append(p)
    deletions = []
    for p in delete_dirs:
        bounded(p)
        files = []
        for f in walk(p, errors):
            try:
                files.append({'path': rel(f), 'bytes': f.stat().st_size, 'sha256': sha(f)})
            except OSError as e:
                errors.append({'path': rel(f), 'error': str(e)})
        deletions.append({'path': rel(p), 'reason': 'synthetic pytest fixtures or regenerable Python cache', 'files': files})
    # Same-volume directory identity survives rename, including opaque ACL children.
    migration = []
    for src, dst in moves:
        bounded(src)
        assert not dst.exists(), str(dst)
        sqlite = []
        for p in walk(src, errors):
            if p.suffix.lower() in {'.db', '.sqlite', '.sqlite3'} or p.name.endswith(('-wal', '-shm')):
                try:
                    sqlite.append({'path': p.relative_to(src).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha(p)})
                except OSError as e:
                    errors.append({'path': str(p), 'error': str(e)})
        migration.append({'source': rel(src), 'destination': rel(dst), 'file_id': src.stat().st_ino,
                          'sqlite_files': sqlite, 'compatibility': 'junction at old path'})
    report = {'applied': args.apply, 'protected_sha256': protected, 'moves': migration,
              'deletions': deletions, 'inventory_errors': errors,
              'policy': 'SQLite-era records preserved by directory rename; ACL opaque objects retained; no live DB or PDF touched. Only synthetic pytest fixtures and bytecode discarded.'}
    (OUT / 'plan.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'moves': len(migration), 'sqlite_files': sum(len(m['sqlite_files']) for m in migration),
        'delete_dirs': len(deletions), 'delete_files': sum(len(d['files']) for d in deletions),
        'delete_bytes': sum(f['bytes'] for d in deletions for f in d['files']), 'inventory_errors': len(errors)}), flush=True)
    if not args.apply:
        return
    execute(report)


def execute(report):
    migration = report['moves']
    for entry in migration:
        src, dst = ROOT / entry['source'], ROOT / entry['destination']
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            os.rename(src, dst)
        assert dst.stat().st_ino == entry['file_id']
        # Explicit junction keeps operational scripts and old provenance paths valid.
        cmd = "New-Item -ItemType Junction -Path '" + str(src).replace("'", "''") + "' -Target '" + str(dst).replace("'", "''") + "' | Out-Null"
        if not src.exists():
            subprocess.run(['powershell', '-NoProfile', '-Command', cmd], check=True, capture_output=True)
        assert linked(src) and src.resolve() == dst.resolve()
        for row in entry['sqlite_files']:
            assert sha(dst / row['path']) == row['sha256']
        entry['verified'] = True
    report['delete_failures'] = []
    for entry in report['deletions']:
        p = bounded(ROOT / entry['path'])
        try:
            # walk previously excluded links; refuse rather than recursively remove any.
            for parent, dirs, files in os.walk(p, followlinks=False):
                assert not any(linked(Path(parent) / n) for n in dirs + files)
            shutil.rmtree(p)
            entry['deleted'] = True
        except (OSError, AssertionError) as e:
            report['delete_failures'].append({'path': entry['path'], 'error': str(e)})
            entry['deleted'] = False
    for name, checksum in report['protected_sha256'].items():
        assert sha(ROOT / name) == checksum, name
    for name in ['DEVELOPMENT.md', 'README.md']:
        src = ROOT / 'internal_experiments' / name
        if src.is_file():
            dst = ARCHIVE / 'internal_experiments' / name
            shutil.copy2(src, dst)
            assert sha(src) == sha(dst)
    src = ROOT / 'self-revise/backend-revise/internal_experiments/evaluation'
    dst = ARCHIVE / 'self-revise/backend-revise/internal_experiments/evaluation'
    if src.exists():
        shutil.copytree(src, dst)
        for p in walk(src, report['inventory_errors']):
            assert sha(p) == sha(dst / p.relative_to(src))
    report['protected_unchanged'] = True
    (OUT / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'applied': True, 'protected_unchanged': True, 'delete_failures': len(report['delete_failures'])}), flush=True)


def organize_receipts():
    manifest = OUT / 'receipt-relocations.json'
    assert not manifest.exists(), 'Already organized'
    names = ['environment-cleanup-manifest.json', 'environment-cleanup-verification.json',
             'migrated-web-verification.json', 'qdrant-cold-backup-sha256.json',
             'qdrant-migration-manifest.json', 'qdrant-migration-verification.json']
    pairs = [(ROOT / 'self-revise' / n, ROOT / 'self-revise/backend-revise/environment-evidence' / n) for n in names]
    pairs += [(ROOT / 'self-revise' / n, ROOT / 'self-revise/cleanup-history' / n)
              for n in ['整理清单.json', '文档整合清单20261109.json']]
    pairs += [(ROOT / 'scripts' / n, ARCHIVE / 'archived-oneoff-code' / n)
              for n in ['amend_v03_p7_budget.py', 'stabilize_v03_p8.py', 'rename_project.ps1']]
    rows = []
    for src, dst in pairs:
        bounded(src)
        assert not dst.exists()
        checksum = sha(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.rename(src, dst)
        assert sha(dst) == checksum
        rows.append({'source': rel(src), 'destination': rel(dst), 'sha256': checksum})
    manifest.write_text(json.dumps({'moves': rows, 'note': 'Current consumer paths updated; frozen logs retain historical paths. One-off code retained as inert history, not a current execution entrypoint.'}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'receipt_and_oneoff_moves': len(rows), 'hashes_unchanged': True}))


if __name__ == '__main__':
    main()

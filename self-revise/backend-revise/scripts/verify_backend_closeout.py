"""Offline audit of backend closeout links, exact scores and retained archives."""
from __future__ import annotations

import hashlib
import ast
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
BASE = ROOT / 'self-revise'


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def sha(p):
    with open('\\\\?\\' + str(p.absolute()), 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    audit = read(BASE / 'cleanup-v20261110/result.json')
    checks = []
    for name, expected in audit['protected_sha256'].items():
        assert sha(ROOT / name) == expected, name
    checks.append('configuration, environment constraint, gold Markdown and paper boundary unchanged')
    dbs = 0
    for row in audit['moves']:
        src, dst = ROOT / row['source'], ROOT / row['destination']
        assert src.is_junction() and src.resolve() == dst.resolve()
        assert dst.stat().st_ino == row['file_id']
        for item in row['sqlite_files']:
            assert sha(dst / item['path']) == item['sha256'], item['path']
            dbs += 1
    checks.append(f'{dbs} readable SQLite/database sidecars retained byte-for-byte; 7 directory identities and compatibility links checked')
    docs = [BASE / p for p in ['后端测试与性能总报告_v20261110.md', '证据分类索引_v20261110.md',
                              'history/v20261029_sqlite/README.md']]
    links, missing = 0, []
    for doc in docs:
        for target in re.findall(r'\]\(([^)]+)\)', doc.read_text(encoding='utf-8')):
            if target.startswith(('https:', 'http:', '#')):
                continue
            links += 1
            if not (doc.parent / target.split('#')[0]).exists():
                missing.append({'document': str(doc), 'target': target})
    assert not missing, missing
    checks.append(f'{links} local documentation links resolve')
    relocations = read(BASE / 'cleanup-v20261110/receipt-relocations.json')['moves']
    for row in relocations:
        assert not (ROOT / row['source']).exists()
        assert sha(ROOT / row['destination']) == row['sha256']
    checks.append(f'{len(relocations)} categorized receipts/archived one-off scripts retained byte-for-byte')
    syntax_files = ['archive_backend_evidence.py', 'verify_backend_closeout.py', 'check_storage.py',
                    'inspect_qdrant_backup.py', 'restore_qdrant_backup.py', 'reconcile_restored_knowledge.py',
                    'migrate_knowledge_qdrant.py', 'verify_migrated_web.py', 'audit_rag_expanded.py',
                    'consolidate_self_revise.py']
    for name in syntax_files:
        p = ROOT / 'scripts' / name
        ast.parse(p.read_text(encoding='utf-8-sig'), filename=str(p))
    checks.append(f'{len(syntax_files)} new/updated Python audit and maintenance scripts parse successfully')
    memory = read(BASE / 'memory-evidence/single-session-v20261110/summary.json')['results']
    assert memory['public-oracle']['legacy']['correct'] == 75
    assert memory['public-oracle']['query']['correct'] == 135
    assert memory['public-unseen']['query']['correct'] == 116
    assert memory['derived-stress']['legacy']['correct'] == 6
    assert memory['derived-stress']['query']['correct'] == 30
    agent = read(BASE / 'agent-evidence/v20261110-architecture-live/langgraph-validation-v2/task-summary.json')['summary']['langgraph']
    assert (agent['total'], agent['success'], agent['supported'], agent['runtime_errors']) == (24, 23, 23, 0)
    suites = ET.parse(BASE / 'agent-evidence/v20261110-architecture-live/langgraph-regression.xml').getroot()
    testcases = list(suites.iter('testcase'))
    assert len(testcases) == 121
    assert not list(suites.iter('failure')) and not list(suites.iter('error'))
    checks.append('reported memory, Agent and 121-pass regression figures match final raw files')
    removed = [entry for entry in audit['deletions'] if entry.get('deleted')]
    assert all(not (ROOT / item['path']).exists() for item in removed)
    report = {'passed': True, 'checks': checks, 'sqlite_files_verified': dbs,
              'local_links': links, 'deleted_directories': len(removed),
              'deleted_files': sum(len(d['files']) for d in removed),
              'deleted_bytes': sum(f['bytes'] for d in removed for f in d['files']),
              'delete_failures': audit['delete_failures'],
              'inventory_limitations': audit['inventory_errors'],
              'scope': 'Offline archival audit only; no paid inference, ingestion, DB migration or repeated benchmark.'}
    (BASE / 'cleanup-v20261110/verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'inventory_limitations'}, ensure_ascii=False))


if __name__ == '__main__':
    main()

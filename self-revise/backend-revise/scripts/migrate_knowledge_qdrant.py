"""Read-only inventory/import of legacy knowledge; --apply is explicit.

No legacy DB, paper file, or existing vector collection is deleted or modified.
"""
import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time
from contextlib import closing
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path.insert(0,str(ROOT/'src'))
from researchclaw.core.qdrant_store import QdrantResearchStore, KNOWLEDGE_TABLES


def discover():
    roots = []
    for archive in (ROOT/'self-revise/backend-revise/internal_experiments/assets/knowledge_base', ROOT/'self-revise/backend-revise/internal_experiments/assets/web_workspace/users'):
        for directory,folders,files in os.walk(archive,followlinks=False):
            folders[:]=[name for name in folders if name not in {'node_modules','.git','__pycache__'} and not (Path(directory)/name).is_junction()]
            if 'researchops.db' in files:
                roots.append(Path(directory).resolve())
    return sorted(set(roots))


def fingerprint(path):
    result = {}
    for item in (path,Path(str(path)+'-wal')):
        if item.is_file():
            if item.name.endswith('-wal') and item.stat().st_size==0:
                continue  # A read-only SQLite connection may create an empty WAL sidecar.
            with item.open('rb') as stream:
                result[item.name] = hashlib.file_digest(stream,'sha256').hexdigest()
    return result


def source_matches(expected,actual):
    empty=hashlib.sha256(b'').hexdigest()
    normalize=lambda values:{k:v for k,v in values.items() if not (k.endswith('-wal') and v==empty)}
    return normalize(expected)==normalize(actual)


def inventory(path):
    with closing(sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)) as db:
        tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        return {table:db.execute('SELECT COUNT(*) FROM '+table).fetchone()[0] if table in tables else 0 for table in KNOWLEDGE_TABLES}


def migrate(root):
    source=root/'researchops.db'
    before=fingerprint(source)
    counts=inventory(source)
    marker=root/'knowledge-storage.json'
    if marker.is_file():
        manifest=json.loads(marker.read_text(encoding='utf-8'))
        if not source_matches(manifest.get('source_sha256',{}),before):
            raise RuntimeError('Legacy source changed since migration: '+str(source))
        return manifest
    store=QdrantResearchStore(root)
    store.import_database(source)
    with store.connect() as db:
        actual={table:db.execute('SELECT COUNT(*) FROM '+table).fetchone()[0] for table in KNOWLEDGE_TABLES}
    if actual!=counts:
        raise RuntimeError(f'Count mismatch: {source}: {counts} != {actual}')
    # Verify content, not just counts, independently of the import path.
    with closing(sqlite3.connect(source.as_uri()+'?mode=ro',uri=True)) as old:
        old.row_factory=sqlite3.Row
        existing={r[0] for r in old.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        with store.connect() as target:
            for table in KNOWLEDGE_TABLES:
                if table not in existing:
                    continue
                columns=[r[1] for r in old.execute('PRAGMA table_info('+table+')')]
                def digest(connection):
                    rows=[json.dumps(dict(row),sort_keys=True,ensure_ascii=False) for row in connection.execute('SELECT '+','.join(columns)+' FROM '+table)]
                    return hashlib.sha256('\n'.join(sorted(rows)).encode()).hexdigest()
                if digest(old)!=digest(target):
                    raise RuntimeError('Content verification failed: '+str(source)+' '+table)
    after=fingerprint(source)
    if not source_matches(before,after):
        raise RuntimeError('Source changed during migration; stop writers and retry: '+str(source))
    manifest={'schema':1,'backend':'qdrant','collection':store.collection,'legacy_database':str(source),
              'source_sha256':before,'verified_counts':counts,'verified_at':time.time(),
              'runtime_database':str(store.path),'legacy_read_only':True}
    marker.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    return manifest


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--apply',action='store_true')
    parser.add_argument('--workspace',action='append',help='Explicit workspace; omit to inventory original assets and Web projects')
    args=parser.parse_args()
    roots=[Path(p).resolve() for p in args.workspace] if args.workspace else discover()
    report={'apply':args.apply,'workspaces':[]}
    for root in roots:
        print('Importing '+str(root) if args.apply else 'Inventory '+str(root),flush=True)
        entry=migrate(root) if args.apply else {'workspace':str(root),'counts':inventory(root/'researchops.db')}
        report['workspaces'].append(entry)
        print(json.dumps(entry,ensure_ascii=False),flush=True)
    if args.apply:
        target=ROOT/'self-revise/backend-revise/environment-evidence/qdrant-migration-manifest.json'
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'workspaces':len(roots),'apply':args.apply}),flush=True)


if __name__=='__main__':
    main()

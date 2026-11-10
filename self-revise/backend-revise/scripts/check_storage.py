"""Read-only audit of PostgreSQL authority, Qdrant projection and frozen sources."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'self-revise/backend-revise/scripts')]
from qdrant_client import QdrantClient
from researchclaw.core.postgres import PostgresDatabase
from researchclaw.core.qdrant_store import KNOWLEDGE_TABLES
from reconcile_restored_knowledge import mapped_path, digest_rows, fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-import', action='store_true', help='Also verify restored historical knowledge digests against the stage 3 receipt')
    parser.add_argument('--output', type=Path, help='Optional new report, never overwritten')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'self-revise/backend-revise/environment-evidence/qdrant-migration-manifest.json').read_text(encoding='utf-8'))
    receipt_path = ROOT / 'self-revise/backend-revise/stage23-evidence/restored-knowledge-verification.json'
    receipts = {row['collection']: row for row in json.loads(receipt_path.read_text(encoding='utf-8'))['workspaces']} if args.verify_import else {}
    client = QdrantClient(url='http://127.0.0.1:6333', timeout=30)
    entries = []
    for original in manifest['workspaces']:
        root = mapped_path(original['legacy_database']).parent
        marker = json.loads((root / 'knowledge-storage.json').read_text(encoding='utf-8'))
        collection = marker['collection']
        db = PostgresDatabase(root, identity=collection)
        records, offset = [], None
        while True:
            page, offset = client.scroll(collection, limit=256, offset=offset, with_payload=True, with_vectors=False)
            records.extend(point.payload for point in page if (point.payload or {}).get('kind') == 'knowledge')
            if offset is None:
                break
        hashes, counts = {}, {}
        with db.connect(read_only=True) as connection:
            for table in KNOWLEDGE_TABLES:
                rows = list(connection.execute('SELECT * FROM ' + table))
                hashes[table] = digest_rows(rows)
                counts[table] = len(rows)
                assert hashes[table] == digest_rows([record['row'] for record in records if record['table'] == table])
            pending = connection.execute("SELECT count(*) FROM knowledge_outbox WHERE status <> 'completed'").fetchone()[0]
            assert pending == 0, collection + ' has pending operations'
        before = fingerprint(root)
        empty = hashlib.sha256(b'').hexdigest()
        for name, expected in original['source_sha256'].items():
            if not (name.endswith('-wal') and expected == empty):
                assert before.get(name) == expected, str(root) + ' historical source changed'
        if args.verify_import:
            assert hashes == receipts[collection]['qdrant_postgres_content_sha256'], collection + ' changed after stage 3'
        entries.append(dict(collection=collection, postgres_schema=db.schema, counts=counts, legacy_unchanged=True))
    report = dict(status='passed', workspaces=entries, workspace_count=len(entries),
                  scope='Per-workspace counts, not unique-paper totals; no SQLite connection or database mutation')
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({'status': 'passed', 'workspace_count': len(entries)}, ensure_ascii=False))
    client.close()


if __name__ == '__main__':
    main()

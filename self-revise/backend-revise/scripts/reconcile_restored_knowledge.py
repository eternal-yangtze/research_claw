"""Verify restored knowledge, bootstrap PG authority, and audit original files.

Qdrant reads only. Original SQLite files are hashed; only the explicit Web
history import opens one legacy database read-only. Reports are new files.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'backend/src')]
from qdrant_client import QdrantClient
from researchclaw.core.qdrant_store import QdrantResearchStore, KNOWLEDGE_TABLES
from researchclaw.core.config import resolve_source_path
from researchclaw_api.store import WebStore


def digest_file(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest_rows(rows):
    serialized = sorted(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) for row in rows)
    return hashlib.sha256('\n'.join(serialized).encode()).hexdigest()


def mapped_path(value):
    old = 'D:/my_coach/new_agent/researchops_mvp/'
    normalized = value.replace('\\', '/')
    if not normalized.casefold().startswith(old.casefold()):
        raise RuntimeError('Unexpected historical root')
    path = (ROOT / normalized[len(old):]).resolve()
    if not path.is_relative_to(ROOT):
        raise RuntimeError('Historical mapping escaped project')
    return path


def fingerprint(directory):
    result = {}
    for pattern in ('*.db', '*.db-wal', '*.sqlite', '*.sqlite3', '*.sqlite3-wal'):
        for path in directory.glob(pattern):
            if path.is_file():
                result[path.name] = digest_file(path)
    return result


def main():
    evidence = ROOT / 'self-revise/backend-revise/stage23-evidence'
    output = evidence / 'restored-knowledge-verification.json'
    if output.exists():
        raise RuntimeError('Existing verification report; do not overwrite')
    restored = json.loads((evidence / 'restore-result.json').read_text(encoding='utf-8'))
    manifest = json.loads((ROOT / 'self-revise/backend-revise/environment-evidence/qdrant-migration-manifest.json').read_text(encoding='utf-8'))
    client = QdrantClient(url='http://127.0.0.1:6333', timeout=60)
    actual_collections = sorted(item.name for item in client.get_collections().collections)
    assert actual_collections == restored['collections']
    aliases_before = sorted((a.alias_name, a.collection_name) for a in client.get_aliases().aliases)
    entries, stores, before_files, collection_stats, chunk_refs = [], {}, {}, [], {}
    for entry in manifest['workspaces']:
        source = mapped_path(entry['legacy_database'])
        root = source.parent
        marker = json.loads((root / 'knowledge-storage.json').read_text(encoding='utf-8'))
        assert marker['collection'] == entry['collection']
        before = fingerprint(root)
        before_files[str(root)] = before
        # Historical migration hashes exclude only an empty WAL sidecar.
        expected = {key: value for key, value in entry['source_sha256'].items()
                    if not (key.endswith('-wal') and value == hashlib.sha256(b'').hexdigest())}
        actual = {key: value for key, value in before.items() if key in expected}
        assert actual == expected, str(root) + ' historical SQLite changed'
        store = QdrantResearchStore(root, client=client)
        runtime_source = root / 'runtime-state.sqlite3'
        runtime_import = store.import_runtime_database(runtime_source) if runtime_source.is_file() else None
        records = list(store._records())
        comparison, counts = {}, {}
        with store.database.connect(read_only=True) as db:
            for table in KNOWLEDGE_TABLES:
                qrows = [record['row'] for record in records if record['table'] == table]
                prows = list(db.execute('SELECT * FROM ' + table))
                comparison[table] = digest_rows(prows)
                assert digest_rows(qrows) == comparison[table], str(root) + ': ' + table
                counts[table] = len(prows)
            assert db.execute("SELECT count(*) FROM knowledge_outbox WHERE status <> 'completed'").fetchone()[0] == 0
        assert fingerprint(root) == before
        query_hits = store.fts_search('LoRA OR adaptation OR learning OR model', 3) if counts['chunks'] else []
        paper_paths = {record['row']['paper_id']: resolve_source_path(record['row'].get('file_path'))
                       for record in records if record['table'] == 'papers'}
        source_exists = {paper_id: bool(path and Path(path).is_file()) for paper_id, path in paper_paths.items()}
        available_sources = {path for paper_id, path in paper_paths.items() if source_exists[paper_id]}
        for record in records:
            if record['table'] == 'chunks':
                chunk = record['row']
                chunk_refs.setdefault(chunk['chunk_id'], []).append(dict(
                    collection=store.collection, chunk_id=chunk['chunk_id'], text=chunk['text'],
                    source_exists=source_exists.get(chunk['paper_id'], False), page=chunk['page']))
        entries.append(dict(workspace=str(root), collection=entry['collection'], postgres_schema=store.database.schema,
                            counts=counts, qdrant_postgres_content_sha256=comparison,
                            legacy_files_unchanged=True, lexical_query_hits=len(query_hits),
                            available_source_files=len(available_sources),
                            runtime_import=runtime_import,
                            original_migration_counts=entry['verified_counts']))
        stores[entry['collection']] = store
        print(f"Verified {len(entries)}/{len(manifest['workspaces'])}: {entry['collection']} {counts}", flush=True)
    vector_queries = []
    for name in actual_collections:
        info = client.get_collection(name)
        collection_stats.append(dict(collection=name, points=info.points_count, status=str(info.status)))
        if name.startswith('researchops_knowledge_') or not info.points_count:
            continue
        points, _ = client.scroll(name, limit=1, with_vectors=True, with_payload=True)
        if not points or not isinstance(points[0].vector, list) or not points[0].vector:
            continue
        point = points[0]
        result = client.query_points(name, query=point.vector, limit=5, with_payload=True).points
        assert result and (any(str(item.id) == str(point.id) for item in result) or result[0].score > .999)
        payload = point.payload or {}
        chunk_id = payload.get('chunk_id')
        matches = []
        for chunk in chunk_refs.get(chunk_id, []):
            if isinstance(payload.get('chunk'), dict):
                assert chunk['text'] == payload['chunk']['text']
            matches.append({key: value for key, value in chunk.items() if key != 'text'})
        vector_queries.append(dict(collection=name, dimensions=len(point.vector),
                                   top_score=result[0].score, sample_id=str(point.id),
                                   citations=matches, mode='real restored vector replay; no new model encoding'))
    webroot = ROOT / 'self-revise/backend-revise/internal_experiments/assets/web_workspace'
    web_sources = [path for path in (webroot / 'web-runtime.sqlite3', webroot / 'web_state.db') if path.is_file()]
    web_result = None
    if web_sources:
        before_web = fingerprint(webroot)
        web = WebStore(webroot)
        imported = web.import_legacy(web_sources[0])
        assert fingerprint(webroot) == before_web
        with web.connect() as db:
            sessions = db.execute('SELECT count(*) FROM sessions').fetchone()[0]
        web_result = dict(source=str(web_sources[0]), postgres_schema=web.database.schema,
                          imported=imported, sessions=sessions, legacy_files_unchanged=True,
                          source_sha256=before_web)
    assert aliases_before == sorted((a.alias_name, a.collection_name) for a in client.get_aliases().aliases)
    for root, before in before_files.items():
        assert fingerprint(Path(root)) == before
    assert digest_file(ROOT / 'self-revise/backend-revise/internal_experiments/assets/docker_backup_20261030/qdrant-storage.tar.gz') == restored['backup_sha256']
    result = dict(status='passed', historical_collection_count=len(actual_collections),
                  knowledge_workspace_count=len(entries), workspaces=entries,
                  collection_stats=collection_stats, vector_queries=vector_queries,
                  alias_count=len(aliases_before), aliases_unchanged=True,
                  backup_unchanged=True, legacy_sqlite_unchanged=True,
                  source_fingerprints=before_files, web_history=web_result,
                  counting_scope='Per-workspace counts; overlapping corpora must not be summed as unique papers')
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    client.close()
    print(f"RECONCILE PASS: {len(entries)} workspaces, {len(vector_queries)} real restored vector queries")


if __name__ == '__main__':
    main()

"""Audit restored alias/version reuse without encoding models or writing Qdrant."""
import json
import hashlib
from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(ROOT / 'src'))
from qdrant_client import QdrantClient
from researchclaw.core.qdrant_store import QdrantResearchStore
from researchclaw.knowledge.vector_backend import QdrantVectorBackend
from researchclaw.core.config import resolve_source_path


def main():
    evidence = ROOT / 'self-revise/backend-revise/stage23-evidence'
    receipt = json.loads((evidence / 'restored-knowledge-verification.json').read_text(encoding='utf-8'))
    client = QdrantClient(url='http://127.0.0.1:6333', timeout=30)
    before = {item.name for item in client.get_collections().collections}
    aliases_before = sorted((a.alias_name, a.collection_name) for a in client.get_aliases().aliases)
    checks, sources, file_hashes = [], [], {}
    for entry in receipt['workspaces']:
        store = QdrantResearchStore(entry['workspace'], client=client)
        with store.database.connect(read_only=True) as db:
            exists = db.execute("SELECT to_regclass('vector_sync_state')").fetchone()[0]
            states = list(db.execute('SELECT * FROM vector_sync_state')) if exists else []
            papers = list(db.execute('SELECT DISTINCT p.paper_id,p.file_path,p.content_hash FROM papers p JOIN chunks c ON c.paper_id=p.paper_id'))
        verified, missing, paths = 0, [], set()
        for paper in papers:
            original = paper['file_path']
            mapped = resolve_source_path(original)
            if not mapped or not Path(mapped).is_file():
                missing.append({'paper_id': paper['paper_id'], 'stored_path': original})
                continue
            paths.add(mapped)
            if paper['content_hash']:
                if mapped not in file_hashes:
                    source = Path(mapped)
                    if source.suffix.lower() in {'.txt', '.md'}:
                        # DocumentIngestor hashes decoded text with universal newlines.
                        raw = source.read_text(encoding='utf-8', errors='replace')
                        file_hashes[mapped] = hashlib.sha256(raw.encode('utf-8')).hexdigest()
                    else:
                        with source.open('rb') as stream:
                            file_hashes[mapped] = hashlib.file_digest(stream, 'sha256').hexdigest()
                assert file_hashes[mapped] == paper['content_hash'], 'Original source content changed: ' + mapped
                verified += 1
        sources.append(dict(workspace=entry['workspace'], collection=entry['collection'],
                            indexed_papers=len(papers), available_source_files=len(paths),
                            verified_paper_source_hashes=verified, missing_sources=missing,
                            knowledge_counts=entry['counts']))
        for state in states:
            if state['project_id'] != store.project_id:
                raise RuntimeError('Restored vector scope differs from bound knowledge collection')
            name, revision = state['embedding_key'].rsplit('@', 1)
            embedding = SimpleNamespace(model_name=name, model_revision=revision)
            backend = QdrantVectorBackend(store, embedding, client=client,
                                          corpus_version=state['corpus_version'])
            assert backend.alias == state['collection_name']
            expected = backend._version_collection(state['dimensions'])
            assert expected == state['version_collection'], 'Restored vector contract differs: ' + expected
            assert expected in before
            checks.append(dict(workspace=entry['workspace'], collection=store.collection,
                               alias=backend.alias, version_collection=expected,
                               embedding_key=state['embedding_key'], reused=True))
    assert checks, 'No historical vector-state bindings were validated'
    assert before == {item.name for item in client.get_collections().collections}
    assert aliases_before == sorted((a.alias_name, a.collection_name) for a in client.get_aliases().aliases)
    result = dict(status='passed', bindings_checked=len(checks), checks=checks,
                  collection_count=len(before), collections_unchanged=True, aliases_unchanged=True,
                  model_encoding_executed=False, source_locations=sources,
                  unique_source_files_hashed=len(file_hashes))
    with (evidence / 'vector-binding-verification.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    client.close()
    print(f'VECTOR BINDINGS PASS: {len(checks)} restored aliases/versions reused; no new collection')


if __name__ == '__main__':
    main()

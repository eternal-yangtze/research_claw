import hashlib
import sqlite3
from contextlib import closing

import pytest
from qdrant_client import QdrantClient

from researchclaw.core.models import Paper
from researchclaw.core.qdrant_store import QdrantResearchStore, KNOWLEDGE_TABLES
from researchclaw.core.storage import ResearchStore
from researchclaw.papers.ingestion import DocumentIngestor


def test_qdrant_authority_restart_and_no_sqlite_knowledge(tmp_path):
    client = QdrantClient(':memory:')
    store = QdrantResearchStore(tmp_path, client=client)
    document = tmp_path/'paper.md'
    document.write_text('# Vision Prompt\n\nFrozen backbone and trainable visual tokens.', encoding='utf-8')
    result = DocumentIngestor(store).ingest_file(document)
    assert store.paper_chunks(result.paper_id)
    with closing(sqlite3.connect(store.path)) as state:
        for table in KNOWLEDGE_TABLES:
            assert state.execute('SELECT COUNT(*) FROM '+table).fetchone()[0] == 0
    reopened = QdrantResearchStore(tmp_path, client=client)
    assert reopened.get_paper(result.paper_id)['title']
    assert len(reopened.all_chunks()) == len(store.all_chunks())
    with reopened.connect() as db:
        db.execute('DELETE FROM chunks_fts WHERE paper_id=?',(result.paper_id,))
        db.execute('DELETE FROM papers WHERE paper_id=?',(result.paper_id,))
    assert not store.all_chunks()  # another instance must refresh after revision change


def test_legacy_import_is_read_only(tmp_path):
    legacy = ResearchStore(tmp_path/'researchops.db')
    legacy.upsert_paper(Paper(paper_id='old', title='Original', authors=(), source='local'))
    digest = hashlib.sha256(legacy.path.read_bytes()).hexdigest()
    store = QdrantResearchStore(tmp_path, client=QdrantClient(':memory:'))
    store.import_database(legacy.path)
    assert store.get_paper('old')['title'] == 'Original'
    assert hashlib.sha256(legacy.path.read_bytes()).hexdigest() == digest


def test_write_failure_replays_outbox(tmp_path, monkeypatch):
    client = QdrantClient(':memory:')
    store = QdrantResearchStore(tmp_path, client=client)
    original = client.upsert
    monkeypatch.setattr(client,'upsert',lambda *a,**k: (_ for _ in ()).throw(ConnectionError('network lost')))
    with pytest.raises(ConnectionError):
        store.upsert_paper(Paper(paper_id='retry',title='Recovered',authors=(),source='local'))
    assert store.outbox.is_file()
    monkeypatch.setattr(client,'upsert',original)
    assert store.get_paper('retry')['title'] == 'Recovered'
    assert not store.outbox.exists()


def test_fixed_snapshot_passages_and_mcp_use_qdrant(tmp_path, monkeypatch):
    from researchclaw.core.qdrant_store import build_research_store
    from researchclaw.knowledge.passages import materialize_passages
    from researchclaw.integrations.paper_mcp import PaperScope
    client=QdrantClient(':memory:')
    monkeypatch.setattr('qdrant_client.QdrantClient',lambda *a,**k:client)
    monkeypatch.setenv('RESEARCHCLAW_KNOWLEDGE_BACKEND','qdrant')
    source=build_research_store(tmp_path/'source')
    document=tmp_path/'paper.md'
    document.write_text('# Prompt\n\nFrozen visual backbone with trainable prompt tokens.',encoding='utf-8')
    result=DocumentIngestor(source).ingest_file(document)
    frozen=build_research_store(tmp_path/'snapshot')
    frozen.copy_from(source)
    chunk=frozen.all_chunks()[0]
    scope=PaperScope(tmp_path,frozen.path,result.paper_id,[chunk['chunk_id']])
    assert scope.read(chunk['chunk_id'])['text']==chunk['text']
    passages=build_research_store(tmp_path/'passages')
    assert materialize_passages(frozen,passages,[result.paper_id])['total_passages']>0
    with source.connect() as db:
        db.execute('DELETE FROM chunks_fts')
        db.execute('DELETE FROM papers')
    assert frozen.all_chunks() and not source.all_chunks()


def test_migrated_collection_missing_fails_closed(tmp_path):
    import json
    (tmp_path/'knowledge-storage.json').write_text(json.dumps({'collection':'lost_corpus'}),encoding='utf-8')
    with pytest.raises(RuntimeError,match='refusing to create an empty replacement'):
        QdrantResearchStore(tmp_path,client=QdrantClient(':memory:'))


def test_collection_binding_survives_workspace_move(tmp_path,monkeypatch):
    import shutil
    from researchclaw.core.qdrant_store import build_research_store
    client=QdrantClient(':memory:')
    monkeypatch.setattr('qdrant_client.QdrantClient',lambda *a,**k:client)
    monkeypatch.setenv('RESEARCHCLAW_KNOWLEDGE_BACKEND','qdrant')
    original=build_research_store(tmp_path/'before')
    original.upsert_paper(Paper(paper_id='stable',title='Stable binding',authors=(),source='local'))
    shutil.copytree(tmp_path/'before',tmp_path/'after')
    moved=build_research_store(tmp_path/'after')
    assert moved.collection==original.collection
    assert moved.project_id==original.project_id
    assert moved.get_paper('stable')['title']=='Stable binding'

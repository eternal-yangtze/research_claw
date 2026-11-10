"""Exercise public management endpoints against the new authoritative store."""
import pytest
from fastapi.testclient import TestClient

from researchclaw_api.main import create_app
from .test_workbench import test_async_process_source_delete_and_detached_knowledge as management_flow


@pytest.fixture
def qbench(tmp_path, monkeypatch):
    # Use the configured isolated workspace on the accepted real Qdrant service.
    # A shared in-memory client is closed by preflight and is not a PG/Qdrant test.
    monkeypatch.setenv('RESEARCHCLAW_KNOWLEDGE_BACKEND','qdrant')
    monkeypatch.setenv('RESEARCHCLAW_VECTOR_BACKEND','qdrant')
    monkeypatch.setenv('RESEARCHCLAW_RETRIEVAL_BACKEND','hashing')
    with TestClient(create_app(workspace=tmp_path,password='test',secret='x'*40)) as bench:
        bench.post('/api/v1/auth/login',json={'user_id':'alice','password':'test'})
        bench.test_session=bench.post('/api/v1/sessions',json={}).json()['id']
        yield bench


def test_qdrant_upload_ingest_delete_and_detached_knowledge(qbench):
    management_flow(qbench)
    snapshot=qbench.get('/api/v1/workbench').json()
    assert snapshot['knowledge_backend']=='postgresql+qdrant'
    assert snapshot['knowledge_collection'].startswith('researchclaw_knowledge_')
    assert snapshot['runtime_database'].startswith('PostgreSQL / rc_research_')

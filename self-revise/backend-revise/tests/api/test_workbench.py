import json
import time

import pytest
from fastapi.testclient import TestClient

from researchclaw_api.context import bounded_history
from researchclaw_api.main import create_app


@pytest.fixture
def bench(tmp_path, monkeypatch):
    monkeypatch.setenv('RESEARCHCLAW_RETRIEVAL_BACKEND', 'hashing')
    monkeypatch.setenv('RESEARCHCLAW_VECTOR_BACKEND', 'local')
    monkeypatch.setenv('RESEARCHCLAW_PDF_PARSER', 'pymupdf')
    app = create_app(workspace=tmp_path, password='test', secret='x'*40)
    with TestClient(app) as client:
        client.post('/api/v1/auth/login', json={'user_id':'alice', 'password':'test'})
        client.test_session = client.post('/api/v1/sessions',json={}).json()['id']
        yield client


def act(client, action, **kwargs):
    response = client.post('/api/v1/workbench/action', json={'action':action, 'session_id':client.test_session, 'confirm':True, **kwargs})
    if action == 'clear' and response.status_code == 200:
        wait_job(client, response.json()['id'])
    return response


def wait_job(client, identity):
    for _ in range(300):
        job = next(j for j in snapshot(client)['jobs'] if j['id'] == identity)
        if job['status'] not in {'queued', 'running'}:
            return job
        time.sleep(.02)
    raise AssertionError('Job did not finish')


def test_async_process_source_delete_and_detached_knowledge(bench):
    stage(bench, b'Title: Async LoRA\n\nLow rank matrices adapt frozen weights.')
    job = act(bench, 'process').json()
    assert wait_job(bench, job['id'])['status'] == 'completed'
    first = snapshot(bench)
    assert first['counts']['chunk_embeddings'] > 0
    paper_id = first['knowledge'][0]['paper_id']
    job = act(bench, 'delete-file', document=first['documents'][0]['document'], cascade=False).json()
    assert wait_job(bench, job['id'])['status'] == 'completed'
    detached = snapshot(bench)
    assert detached['documents'] == []
    assert detached['knowledge'][0]['source_exists'] is False
    assert detached['knowledge'][0]['chunks'] > 0
    job = act(bench, 'delete-knowledge', paper_ids=[paper_id]).json()
    assert wait_job(bench, job['id'])['status'] == 'completed'
    assert snapshot(bench)['counts']['chunks'] == 0
    stage(bench, b'Title: Async VPT\n\nPrompt tokens adapt vision transformers.')
    assert wait_job(bench, act(bench, 'process').json()['id'])['status'] == 'completed'
    doc = snapshot(bench)['documents'][0]
    assert wait_job(bench, act(bench, 'delete-file', document=doc['document'], cascade=True).json()['id'])['status'] == 'completed'
    assert snapshot(bench)['documents'] == []
    assert snapshot(bench)['counts']['chunk_embeddings'] == 0


def test_delete_does_not_load_embedding_model(bench, monkeypatch):
    stage(bench, b'Title: Delete without embedding\n\nKnowledge to remove directly.')
    assert wait_job(bench, act(bench, 'process').json()['id'])['status'] == 'completed'
    paper_id = snapshot(bench)['knowledge'][0]['paper_id']
    monkeypatch.setattr('researchclaw.knowledge.retrieval.HashingEmbeddingBackend.encode',
                        lambda *args: (_ for _ in ()).throw(AssertionError('delete loaded embedding')))
    job = wait_job(bench, act(bench, 'delete-knowledge', paper_ids=[paper_id]).json()['id'])
    assert job['status'] == 'completed'
    assert job['index']['embedding_required'] is False


def test_processing_failure_visible_and_retryable(bench):
    bench.post('/api/v1/knowledge/upload?suffix=pdf&stage=true', content=b'%PDF-1.4\nbroken')
    failed = wait_job(bench, act(bench, 'process').json()['id'])
    assert failed['status'] == 'failed'
    assert failed['error']
    assert failed['stage'].startswith('处理失败')
    assert wait_job(bench, act(bench, 'process').json()['id'])['status'] == 'failed'


def snapshot(client):
    result = client.get('/api/v1/workbench')
    assert result.status_code == 200, result.text
    return result.json()


@pytest.mark.parametrize('file_path', [None, ''])
def test_snapshot_search_candidate_without_local_file(bench, file_path):
    from researchclaw.application.workflow import ResearchClawApp
    from researchclaw.core.models import Paper

    root = bench.app.state.store.root / 'users' / 'alice' / 'projects' / 'default'
    research = ResearchClawApp(root)
    research.store.upsert_paper(Paper(
        paper_id='search_candidate', title='Online paper not yet downloaded',
        source='arxiv', file_path=file_path,
    ))
    result = snapshot(bench)
    paper = next(p for p in result['knowledge'] if p['paper_id'] == 'search_candidate')
    assert paper['name'] == 'Online paper not yet downloaded'
    assert paper['source_exists'] is False
    assert paper['chunks'] == 0
    assert result['documents'] == []


def stage(client, content):
    response = client.post('/api/v1/knowledge/upload?suffix=md&stage=true&session_id='+client.test_session, content=content)
    assert response.status_code == 201
    return response.json()


def test_stage_incremental_remove_clear(bench):
    stage(bench, b'Title: LoRA paper\n\nLoRA uses low rank adaptation.')
    first = snapshot(bench)
    assert first['counts']['chunks'] == 0
    assert not first['documents'][0]['indexed']
    assert act(bench, 'index').status_code == 200
    initial = snapshot(bench)
    assert initial['counts']['chunks'] > 0
    assert initial['counts']['chunk_embeddings'] > 0
    assert initial['documents'][0]['indexed']
    stage(bench, b'Title: VPT paper\n\nVisual prompt tuning freezes the backbone.')
    result = act(bench, 'index').json()
    assert any(r.get('status') == 'unchanged' for r in result['results'])
    second = snapshot(bench)
    assert second['counts']['papers'] == 2
    assert second['counts']['chunk_embeddings'] > initial['counts']['chunk_embeddings']
    key = second['documents'][0]['document']
    assert bench.get('/api/v1/workbench/file', params={'document':key}).status_code == 200
    assert bench.get('/api/v1/workbench/file', params={'document':'../../.env'}).status_code == 404
    assert act(bench, 'remove', document=key).status_code == 422
    pid=next(d['paper']['paper_id'] for d in second['documents'] if d['document']==key)
    removed=act(bench,'delete-knowledge',paper_ids=[pid])
    assert wait_job(bench,removed.json()['id'])['status']=='completed'
    assert snapshot(bench)['counts']['papers'] == 1
    assert act(bench, 'clear').status_code == 200
    cleared = snapshot(bench)
    assert cleared['counts']['chunks'] == cleared['counts']['chunk_embeddings'] == 0
    assert len(cleared['documents']) == 0
    stage(bench, b'Title: LoRA paper\n\nLoRA uses low rank adaptation.')
    assert act(bench, 'index').status_code == 200
    assert snapshot(bench)['counts']['papers'] == 1


def test_jobs_real_cli_and_measured_artifacts(bench):
    stage(bench, b'Title: Evidence\n\nLow rank adapters reduce trainable parameters.')
    act(bench, 'index')
    started = act(bench, 'search', text='low rank').json()
    for _ in range(400):
        state = snapshot(bench)
        job = next(j for j in state['jobs'] if j['id'] == started['id'])
        if job['status'] not in {'queued','running'}:
            break
        time.sleep(.05)
    assert job['status'] == 'completed', job
    assert 'Low rank' in job['output']
    assert job['elapsed_seconds'] > 0
    assert bench.get('/api/v1/workbench/artifact', params={'path':'../web_state.db'}).status_code == 404
    assert act(bench, 'code', text='../oops').status_code == 422
    assert act(bench, 'invalid').status_code == 422


def test_owner_isolation_and_confirmation(bench):
    stage(bench, b'Title: Private\n\nPrivate contents.')
    assert bench.post('/api/v1/workbench/action', json={'action':'clear'}).status_code == 422
    bench.post('/api/v1/auth/login', json={'user_id':'bob','password':'test'})
    assert snapshot(bench)['documents'] == []
    assert bench.get('/api/v1/workbench?project_id=../alice').status_code == 422


def test_dataset_upload_validation(bench):
    assert bench.post('/api/v1/workbench/dataset-upload?dataset_id=dtd&filename=../../evil', content=b'x').status_code == 422
    assert bench.post('/api/v1/workbench/dataset-upload?dataset_id=dtd&filename=dtd-r1.0.1.tar.gz', content=b'bad archive').status_code == 201
    assert snapshot(bench)['dataset_files']
    assert act(bench, 'dataset-import', text='unknown').status_code == 422


def test_context_preserves_complete_turns_and_full_input():
    messages = []
    for index in range(50):
        messages += [{'role':'user','content':f'constraint {index} '+ 'x'*1000},
                     {'role':'assistant','content':'response '+ 'y'*1000}]
    original = json.dumps(messages)
    bounded, stats = bounded_history(messages, max_chars=12000)
    assert stats['compacted']
    assert bounded[-2:] == messages[-2:]
    assert len(json.dumps(bounded)) < 12000
    assert json.dumps(messages) == original


def test_failed_chat_keeps_checkpoint_and_can_read_history(tmp_path, monkeypatch):
    monkeypatch.setenv('RESEARCHCLAW_RETRIEVAL_BACKEND','hashing')
    monkeypatch.setenv('RESEARCHCLAW_VECTOR_BACKEND','local')
    class FailedModel:
        def complete(self, *args):
            raise TimeoutError('controlled diagnostic fault')
    with TestClient(create_app(workspace=tmp_path, model_factory=FailedModel, password='test',secret='x'*40)) as client:
        client.post('/api/v1/auth/login',json={'user_id':'alice','password':'test'})
        run=client.post('/api/v1/chat',json={'message':'Keep this research constraint'}).json()
        for _ in range(100):
            result=client.get('/api/v1/runs/'+run['run_id']).json()
            if result['status']!='running':
                break
            time.sleep(.02)
        assert result['status']=='failed'
        history=client.get('/api/v1/sessions/'+run['session_id']).json()['messages']
        assert any(m.get('content')=='Keep this research constraint' for m in history)
        assert client.get('/api/v1/sessions').json()[0]['id']==run['session_id']


def test_study_resume_requires_project_checkpoint(bench):
    assert act(bench,'resume-study',text='Original Idea',artifact='../other/study_snapshot.json').status_code==422
    assert act(bench,'coding-plan',text='safe_id',artifact='anything.json',token_cap=60000).status_code==422


def test_extended_task_boundaries(bench):
    for command in ('batch-research','silver-evaluate','reanalyze'):
        assert act(bench,command,text='id; arbitrary command').status_code==422
    assert act(bench,'validate-formulas',document='../../paper.pdf',artifact='../../parser.json').status_code==422
    assert act(bench,'reuse-datasets',text='../../private').status_code==422


def test_original_filename_survives_windows_path_normalization(bench):
    response=bench.post('/api/v1/knowledge/upload?stage=true&suffix=md&filename=original_evidence.md',content=b'Title: Source\n\nEvidence')
    assert response.status_code==201
    doc=snapshot(bench)['documents'][0]
    assert doc['name']=='original_evidence.md'
    assert doc['document'].startswith('uploads/')

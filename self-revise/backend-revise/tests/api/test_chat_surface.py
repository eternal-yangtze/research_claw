"""Chat-facing account, title, source and confirmation contracts on real PostgreSQL."""
import json
import time

import pytest
from fastapi.testclient import TestClient

from researchclaw_api.main import create_app
from researchclaw_api.library import progress_view
from researchclaw.harness.runtime import ModelTurn


class TitleModel:
    def complete(self, messages, tools, max_tokens):
        if not tools:
            return ModelTurn('误差有界时序压缩')
        return ModelTurn(json.dumps({'status': 'complete', 'answer': '你好，可以继续讨论研究方向。', 'citations': []}))


def test_official_pmlr_assets_allowed_without_allowing_arbitrary_github():
    from researchclaw_api.library import public_paper_url
    assert public_paper_url('https://raw.githubusercontent.com/mlresearch/v235/main/assets/das24c/das24c.pdf')
    assert not public_paper_url('https://raw.githubusercontent.com/other/repo/main/paper.pdf')
    assert not public_paper_url('https://raw.githubusercontent.com/mlresearch/v235/main/script.py')
    assert not public_paper_url('http://arxiv.org/pdf/2310.10688')
    assert not public_paper_url('https://arxiv.org:8888/pdf/2310.10688')


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv('RESEARCHCLAW_ALLOW_LEGACY_LOGIN', '0')
    monkeypatch.setenv('RESEARCHCLAW_ALLOW_DIRECT_INGEST', '0')
    with TestClient(create_app(workspace=tmp_path, model_factory=TitleModel,
            password='invite-test-code', secret='s' * 40)) as c:
        yield c


def register(c, user='researcher'):
    return c.post('/api/v1/auth/register', json={'user_id': user, 'password': 'individual-password', 'invite_code': 'invite-test-code'})


def login(c, user='researcher', password='individual-password'):
    return c.post('/api/v1/auth/login', json={'user_id': user, 'password': password})


def test_individual_password_and_account_isolation(client):
    assert client.get('/api/v1/auth/me').status_code == 401
    assert login(client, password='invite-test-code').status_code == 401
    assert register(client).status_code == 201
    assert register(client).status_code == 409
    assert login(client, password='invite-test-code').status_code == 401
    assert login(client).status_code == 200
    sid = client.post('/api/v1/sessions', json={}).json()['id']
    assert client.get('/api/v1/auth/me').json()['user_id'] == 'researcher'
    assert register(client, 'other').status_code == 201
    login(client, 'other')
    assert client.get('/api/v1/sessions/' + sid).status_code == 404
    assert client.get('/api/v1/library', params={'session_id': sid}).status_code == 404
    assert client.post('/api/v1/sessions/' + sid + '/suggest-title', json={}).status_code == 404


def test_at_sign_identity_and_special_password(client):
    body = {'user_id': 'demo2@2026', 'password': 'Demo2@2026!Secure', 'invite_code': 'invite-test-code'}
    assert client.post('/api/v1/auth/register', json=body).status_code == 201
    assert login(client, body['user_id'], body['password']).status_code == 200
    assert client.get('/api/v1/auth/me').json()['user_id'] == 'demo2@2026'
    sid = client.post('/api/v1/sessions', json={}).json()['id']
    assert client.get('/api/v1/library', params={'session_id': sid}).status_code == 200
    preview = client.post(f'/api/v1/sessions/{sid}/prepare-delete', json={}).json()
    assert client.post(f'/api/v1/sessions/{sid}', json={'delete': True, 'confirmation_token': preview['token']}).status_code == 200
    assert login(client, body['user_id'], 'wrong-password').status_code == 401
    for invalid in ('.', '..', '../other', 'a/b', 'a\\b', 'a b'):
        body['user_id'] = invalid
        assert client.post('/api/v1/auth/register', json=body).status_code == 422


def test_test_version_rollback_is_preview_bound_and_preserves_baseline(client):
    from researchclaw.application.paper_library import library_paths, active_test_library
    register(client); login(client)
    sid = client.post('/api/v1/sessions', json={}).json()['id']
    root = client.app.state.store.root / 'users/researcher/projects/default'
    client.post('/api/v1/knowledge/upload', params={'session_id': sid}, content=b'%PDF-1.4\nbaseline')
    before = client.get('/api/v1/library', params={'session_id': sid}).json()
    body = {'session_id': sid}
    assert client.post('/api/v1/library/test-version/start', json=body).status_code == 200
    assert active_test_library(root) in library_paths(root)
    assert client.post('/api/v1/library/test-version/start', json=body).status_code == 409
    assert client.post('/api/v1/library/test-version/rollback', json=body).status_code == 409
    client.post('/api/v1/knowledge/upload', params={'session_id': sid}, content=b'%PDF-1.4\ntest')
    preview = client.post('/api/v1/library/test-version/prepare-rollback', json=body).json()
    assert len(preview['files_to_archive']) == 1
    client.post('/api/v1/knowledge/upload', params={'session_id': sid}, content=b'%PDF-1.4\nchanged')
    assert client.post('/api/v1/library/test-version/rollback', json=body | {'confirmation_token': preview['confirmation_token']}).status_code == 409
    preview = client.post('/api/v1/library/test-version/prepare-rollback', json=body).json()
    response = client.post('/api/v1/library/test-version/rollback', json=body | {'confirmation_token': preview['confirmation_token']})
    assert response.status_code == 200
    assert len(response.json()['archived_uploads']) == 2
    assert active_test_library(root) is None
    after = client.get('/api/v1/library', params={'session_id': sid}).json()
    assert after['documents'] == before['documents']
    assert client.post('/api/v1/library/test-version/rollback', json=body | {'confirmation_token': preview['confirmation_token']}).status_code == 409
    assert len(list((root / 'pt').rglob('*.pdf'))) == 2


def test_historical_id_requires_invite_claim_and_is_not_overwritten(client):
    store = client.app.state.store
    sid = store.new_session('historical', 'default', '旧研究')
    assert register(client, 'historical').status_code == 409
    body = {'user_id': 'historical', 'password': 'new-personal-password', 'invite_code': 'wrong'}
    assert client.post('/api/v1/auth/claim-history', json=body).status_code == 403
    body['invite_code'] = 'invite-test-code'
    assert client.post('/api/v1/auth/claim-history', json=body).status_code == 201
    assert login(client, 'historical', 'new-personal-password').status_code == 200
    assert client.get('/api/v1/sessions/' + sid).json()['title'] == '旧研究'


def test_upload_is_staged_and_title_is_only_a_suggestion(client):
    register(client); login(client)
    sid = client.post('/api/v1/sessions', json={}).json()['id']
    assert client.post(f'/api/v1/sessions/{sid}/suggest-title', json={}).status_code == 409
    run = client.post('/api/v1/chat', json={'session_id': sid, 'message': '你好'}).json()['run_id']
    for _ in range(150):
        if client.get('/api/v1/runs/' + run).json()['status'] != 'running':
            break
        time.sleep(.05)
    assert client.get('/api/v1/runs/' + run).json()['status'] == 'completed'
    with client.app.state.store.connect() as db:
        snapshot = json.loads(db.execute('SELECT payload FROM context_snapshots WHERE session=? ORDER BY created_at DESC LIMIT 1', (sid,)).fetchone()[0])
    assert snapshot['tool_headroom_reserved'] == 4096
    assert snapshot['margin_reserved'] == 6144
    assert snapshot['input_tokens'] <= snapshot['input_budget']
    assert client.post(f'/api/v1/sessions/{sid}/suggest-title', json={}).json()['title'] == '误差有界时序压缩'
    assert client.get('/api/v1/sessions/' + sid).json()['title'] == '新对话'
    response = client.post('/api/v1/knowledge/upload', params={'session_id': sid}, content=b'%PDF-1.4\nbroken')
    assert response.json()['status'] == 'staged'
    snapshot = client.get('/api/v1/library', params={'session_id': sid}).json()
    assert snapshot['paper_count'] == 0 and not snapshot['documents'][0]['indexed']
    assert snapshot['jobs'] == []
    assert client.get(f'/api/v1/runs/{run}/sources').json() == []


def test_ingest_requires_one_use_preview_and_changed_inventory_invalidates_it(client):
    register(client); login(client)
    sid = client.post('/api/v1/sessions', json={}).json()['id']
    client.post('/api/v1/knowledge/upload', params={'session_id': sid, 'stage': 'true'}, content=b'%PDF-1.4\nfirst')
    aid = client.post('/api/v1/library/propose-ingest', json={'session_id': sid}).json()['action_id']
    preview = client.post(f'/api/v1/actions/{aid}/prepare').json()
    client.post('/api/v1/knowledge/upload', params={'session_id': sid, 'stage': 'true'}, content=b'%PDF-1.4\nsecond')
    r = client.post(f'/api/v1/actions/{aid}/confirm', json={'confirmation_token': preview['confirmation_token']})
    assert r.status_code == 409 and 'changed' in r.json()['detail'].lower()
    assert client.post(f'/api/v1/actions/{aid}/reject').status_code == 200
    assert client.post(f'/api/v1/actions/{aid}/prepare').status_code == 409
    assert client.get('/api/v1/library', params={'session_id': sid}).json()['jobs'] == []


def test_progress_uses_real_work_and_does_not_guess_initial_eta():
    s = {'started': time.time() - 30, 'status': 'running', 'phase': 'parse', 'total_files': 3, 'completed_files': 0}
    assert progress_view(s)['eta_seconds'] is None
    s.update(completed_files=1, phase='embed', vectors_done=16, vectors_total=32)
    v = progress_view(s)
    assert 30 <= v['eta_seconds'] <= 65 and 30 < v['build_progress'] < 60
    s.update(status='failed')
    assert progress_view(s)['build_progress'] < 100
    s.update(status='completed')
    assert progress_view(s)['build_progress'] == 100


def test_chat_search_defaults_avoid_irrelevant_catalog_scans_but_keep_opt_in(tmp_path):
    from researchclaw.application.workflow import ResearchClawApp
    from researchclaw.application.agent_tools import build_agent_tools
    app = ResearchClawApp(tmp_path)
    calls = []
    app.literature.search = lambda query, run_id, limit, sources: calls.append(sources) or {'papers': [], 'sources': []}
    registry, _ = build_agent_tools(app)
    search = registry.get('search_papers')
    search.handler(query='time series error bound', run_id='default-test')
    assert set(calls[-1]) == {'arxiv', 'openalex', 'crossref'}
    search.handler(query='vision conference paper', run_id='specialist-test', sources=['cvf'])
    assert calls[-1] == ['cvf']
    app.literature.search = lambda *args, **kwargs: {'papers': [
        {'paper_id': 'metadata', 'title': 'Target title', 'pdf_url': 'https://arxiv.org/pdf/1234.56789', 'abstract': 'large abstract ' * 10000}],
        'sources': [{'source': 'arxiv', 'status': 'success', 'result_count': 1}]}
    result = search.handler(query='Target title', run_id='bounded-metadata')
    assert result['papers'][0]['title'] == 'Target title'
    assert result['sources'][0]['status'] == 'success'
    assert result['papers'][0]['pdf_url'].startswith('https://arxiv.org/')
    assert len(json.dumps(result)) < 1500


def test_reference_excerpts_remain_bounded_for_continuous_chat(tmp_path):
    import fitz
    from types import SimpleNamespace
    from researchclaw.application.paper_library import PaperLibrary
    path = tmp_path / 'references.pdf'
    doc = fitz.open()
    for index in range(5):
        page = doc.new_page()
        text = ('References\n' if index == 0 else '') + ('[1] A real bibliography line of sufficiently long content for a bounded payload.\n' * 45)
        page.insert_textbox(fitz.Rect(30, 30, 560, 810), text, fontsize=9)
    doc.save(path); doc.close()
    library = PaperLibrary.__new__(PaperLibrary)
    library.stores = [SimpleNamespace(get_paper=lambda identity: {'file_path': str(path), 'title': 'Fixture'})]
    output = library.references('fixture')
    assert output['available'] and output['truncated']
    assert sum(len(p['text']) for p in output['excerpts']) <= 4500
    assert output['excerpts'][0]['page'] == 1
    assert output['excerpts'][0]['chunk_id'].startswith('bib_')
    assert output['excerpts'][0]['scientific_findings_verified'] is False


def test_confirmed_download_is_staged_and_confirmation_cannot_replay(client, monkeypatch):
    from researchclaw.application.workflow import ResearchClawApp
    from researchclaw_api.actions import ActionStore
    register(client); login(client)
    store = client.app.state.store
    sid = client.post('/api/v1/sessions', json={}).json()['id']
    root = store.root / 'users/researcher/projects/default'
    calls = []
    def download(url, output_dir, filename=None, **kwargs):
        calls.append(url)
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / 'open-paper.pdf'
        target.write_bytes(b'%PDF-1.4\nnetwork-response-fixture')
        return {'path': str(target)}
    monkeypatch.setattr('researchclaw.papers.connectors.download_pdf', download)
    proposal = ActionStore(store).propose(sid, 'researcher', 'default', 'download_pdf',
        {'url': 'https://arxiv.org/pdf/2401.12345'}, 'explicit request', ResearchClawApp(root).registry)
    aid = proposal['action_id']
    assert calls == []
    token = client.post(f'/api/v1/actions/{aid}/prepare').json()['confirmation_token']
    rid = client.post(f'/api/v1/actions/{aid}/confirm', json={'confirmation_token': token}).json()['run_id']
    for _ in range(100):
        if client.get('/api/v1/runs/' + rid).json()['status'] != 'running':
            break
        time.sleep(.03)
    assert client.get('/api/v1/runs/' + rid).json()['status'] == 'completed'
    snapshot = client.get('/api/v1/library', params={'session_id': sid}).json()
    assert len(snapshot['documents']) == 1 and snapshot['paper_count'] == 0 and snapshot['jobs'] == []
    assert client.post(f'/api/v1/actions/{aid}/confirm', json={'confirmation_token': token}).status_code == 409
    assert len(calls) == 1


def test_download_timeout_is_visible_and_does_not_claim_publication(client, monkeypatch):
    from researchclaw.application.workflow import ResearchClawApp
    from researchclaw_api.actions import ActionStore
    register(client); login(client)
    store = client.app.state.store
    sid = client.post('/api/v1/sessions', json={}).json()['id']
    root = store.root / 'users/researcher/projects/default'
    def timeout(*args, **kwargs):
        raise TimeoutError('deadline exceeded')
    monkeypatch.setattr('researchclaw.papers.connectors.download_pdf', timeout)
    aid = ActionStore(store).propose(sid, 'researcher', 'default', 'download_pdf',
        {'url': 'https://arxiv.org/pdf/2310.10688'}, 'test timeout', ResearchClawApp(root).registry)['action_id']
    token = client.post(f'/api/v1/actions/{aid}/prepare').json()['confirmation_token']
    rid = client.post(f'/api/v1/actions/{aid}/confirm', json={'confirmation_token': token}).json()['run_id']
    for _ in range(100):
        run = client.get('/api/v1/runs/' + rid).json()
        if run['status'] != 'running':
            break
        time.sleep(.03)
    assert run['status'] == 'failed' and '下载超时' in run['error']
    messages = client.get('/api/v1/sessions/' + sid).json()['messages']
    assert messages[-1]['role'] == 'assistant' and '未入库' in messages[-1]['content']
    assert client.get('/api/v1/library', params={'session_id': sid}).json()['paper_count'] == 0

import time

import pytest

pytest.importorskip('fastapi')
pytest.importorskip('httpx')
pytest.importorskip('jwt')
pytest.importorskip('dotenv')
from fastapi.testclient import TestClient

from researchops_api.main import create_app
from researchops.harness.runtime import ModelTurn


class Model:
    def complete(self, messages, tools, max_tokens):
        return ModelTurn('answer ' + str(sum(m['role'] == 'user' for m in messages)), usage={'total_tokens': 10})


@pytest.fixture
def client(tmp_path):
    app = create_app(workspace=tmp_path, model_factory=Model, password='local-test', secret='s' * 40)
    with TestClient(app) as client:
        yield client


def login(client, owner='alice'):
    assert client.post('/api/v1/auth/login', json={'user_id': owner, 'password': 'local-test'}).status_code == 200


def wait_run(client, run_id):
    for _ in range(200):
        row = client.get('/api/v1/runs/' + run_id).json()
        if row['status'] != 'running':
            return row
        time.sleep(.02)
    raise AssertionError('Run did not settle')


def test_api_has_no_static_frontend_and_enforces_login(client):
    assert client.get('/').status_code == 404
    assert client.get('/api/v1/ready').json()['database']
    assert client.post('/api/v1/chat', json={'message': 'question'}).status_code == 401
    assert client.post('/api/v1/auth/login', json={'user_id': '../alice', 'password': 'local-test'}).status_code == 422
    assert client.post('/api/v1/auth/login', json={'user_id': 'alice', 'password': 'bad'}).status_code == 401


def test_continuation_owner_project_and_sse(client):
    login(client)
    first = client.post('/api/v1/chat', json={'message': 'first'}).json()
    assert wait_run(client, first['run_id'])['answer'] == 'answer 1'
    second = client.post('/api/v1/chat', json={'message': 'second', 'session_id': first['session_id']}).json()
    assert wait_run(client, second['run_id'])['answer'] == 'answer 2'
    assert 'answer 2' in client.get('/api/v1/runs/' + second['run_id'] + '/events').text
    assert client.post('/api/v1/chat', json={'message': 'x', 'project_id': 'other', 'session_id': first['session_id']}).status_code == 404
    login(client, 'bob')
    assert client.get('/api/v1/runs/' + first['run_id']).status_code == 404
    assert client.post('/api/v1/chat', json={'message': 'x', 'session_id': first['session_id']}).status_code == 404


def test_real_document_ingestion_and_origin_guard(client):
    login(client)
    assert client.post('/api/v1/knowledge/upload?suffix=md', content=b'# Local evidence\nLow rank adaptation reduces trainable parameters.').status_code == 201
    assert client.post('/api/v1/knowledge/upload?project_id=..%2Fother&suffix=md', content=b'text').status_code == 422
    assert client.post('/api/v1/knowledge/upload?suffix=pdf', content=b'not pdf').status_code == 422
    assert client.post('/api/v1/chat', json={'message': 'x'}, headers={'Origin': 'https://untrusted.invalid'}).status_code == 403

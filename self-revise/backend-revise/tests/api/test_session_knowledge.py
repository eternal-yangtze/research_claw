import time
import json

import pytest
from fastapi.testclient import TestClient
from researchclaw_api.main import create_app
from researchclaw.harness.runtime import ModelTurn


@pytest.fixture
def client(tmp_path,monkeypatch):
    monkeypatch.setenv('RESEARCHCLAW_RETRIEVAL_BACKEND','hashing')
    monkeypatch.setenv('RESEARCHCLAW_VECTOR_BACKEND','local')
    class MemoryModel:
        def complete(self,messages,tools,max_tokens):
            if 'ROLE: Reviewer' in messages[0]['content']:
                return ModelTurn(json.dumps(dict(verdict='PASS',target='worker',failed_criteria=[],reason='context echo fixture')),usage={'total_tokens':10})
            data=json.loads(messages[-1]['content'])
            return ModelTurn(json.dumps(dict(mode='direct',answer=json.dumps(data['context'],ensure_ascii=False),tasks=[])),usage={'total_tokens':10})
    with TestClient(create_app(workspace=tmp_path,model_factory=MemoryModel,password='test',secret='x'*40)) as c:
        c.post('/api/v1/auth/login',json={'user_id':'alice','password':'test'})
        yield c


def new(client,**extra):
    r=client.post('/api/v1/sessions',json=extra);assert r.status_code==201,r.text
    return r.json()['id']


def snapshot(c,s=''):
    r=c.get('/api/v1/workbench',params={'session_id':s});assert r.status_code==200,r.text
    return r.json()


def action(c,s,kind,**extra):
    r=c.post('/api/v1/workbench/action',json={'action':kind,'session_id':s,'confirm':True,**extra})
    assert r.status_code==200,r.text
    if 'id' in r.json():
        for _ in range(300):
            job=next(j for j in snapshot(c)['jobs'] if j['id']==r.json()['id'])
            if job['status'] not in {'running','queued'}:
                import json
                assert job['status']=='completed',json.dumps(job,ensure_ascii=False)
                return job
            time.sleep(.02)
        raise AssertionError('timeout')
    return r.json()


def upload(c,s,content):
    r=c.post('/api/v1/knowledge/upload',params={'suffix':'md','stage':True,'session_id':s},content=content)
    assert r.status_code==201,r.text


def test_scoped_clear_preserves_legacy_and_other_sessions(client):
    # Pre-existing upload without a session is protected, never adopted by a new chat.
    upload(client,'',b'Title: Original visual prompt\n\nOriginal complete knowledge.')
    a,b=new(client),new(client)
    action(client,a,'process')
    upload(client,a,b'Title: Session A LoRA\n\nLow rank adaptation.')
    action(client,a,'process')
    upload(client,b,b'Title: Session B RAG\n\nRetrieval augmented generation.')
    action(client,b,'process')
    before=snapshot(client,b)
    assert len(before['knowledge'])==3
    own=next(p for p in before['knowledge'] if p['can_delete'])
    r=client.post('/api/v1/workbench/action',json={'action':'delete-knowledge','confirm':True,'session_id':a,'paper_ids':[own['paper_id']]})
    assert r.status_code==403
    action(client,a,'clear')
    after=snapshot(client,b)
    assert len(after['knowledge'])==len(after['documents'])==2
    assert any(p['title']=='Original visual prompt' for p in after['knowledge'])
    assert any(p['title']=='Session B RAG' for p in after['knowledge'])
    assert client.post('/api/v1/workbench/action',json={'action':'clear','confirm':True}).status_code==422


def test_deduplicated_upload_does_not_transfer_ownership(client):
    a,b=new(client),new(client)
    data=b'Title: Shared LoRA\n\nLow rank adaptation.'
    upload(client,a,data);action(client,a,'process')
    upload(client,b,data);action(client,b,'process')
    assert not snapshot(client,b)['documents'][0]['can_delete']
    action(client,b,'clear')
    assert len(snapshot(client)['knowledge'])==1


def test_delete_session_retains_knowledge_and_classification(client):
    a=new(client,title='视觉适配研究')
    upload(client,a,b'Title: Visual Prompt Tuning\n\nVisual prompt learning.')
    action(client,a,'process')
    p=snapshot(client,a)['knowledge'][0]
    assert p['domain']=='计算机视觉 / 参数高效适配'
    action(client,a,'set-domain',paper_ids=[p['paper_id']],domain='视觉 / 小样本学习')
    preview=client.post('/api/v1/sessions/'+a+'/prepare-delete',json={}).json()
    r=client.post('/api/v1/sessions/'+a,json={'delete':True,'confirmation_token':preview['token']});assert r.status_code==200
    assert r.json()['knowledge_retained']
    assert client.get('/api/v1/sessions/'+a).status_code==404
    b=new(client)
    p=snapshot(client,b)['knowledge'][0]
    assert p['domain']=='视觉 / 小样本学习' and not p['can_delete']
    action(client,b,'clear')
    assert len(snapshot(client)['knowledge'])==1


def ask(c,s,text):
    r=c.post('/api/v1/chat',json={'session_id':s,'message':text});assert r.status_code==202,r.text
    for _ in range(200):
        result=c.get('/api/v1/runs/'+r.json()['run_id']).json()
        if result['status']!='running':
            assert result['status']=='completed',result
            return result
        time.sleep(.02)
    raise AssertionError('timeout')


def test_history_memory_branch_and_owner_isolation(client):
    a=new(client,title='历史研究')
    assert client.post('/api/v1/sessions/'+a,json={'memory':'只用 DTD，预算 2 小时'}).status_code==200
    first=ask(client,a,'先比较视觉提示')
    assert '预算 2 小时' in first['answer']
    second=ask(client,a,'再比较低秩适配')
    assert '先比较视觉提示' in second['answer']
    b=new(client,source_session=a,title='继承方案')
    assert '先比较视觉提示' in ask(client,b,'继续原方案')['answer']
    debug=client.get('/api/v1/debug',params={'session_id':a}).json()
    assert debug['model_runs'] and debug['model_runs'][0]['context']
    client.post('/api/v1/auth/login',json={'user_id':'bob','password':'test'})
    assert client.get('/api/v1/sessions/'+a).status_code==404
    assert client.post('/api/v1/sessions',json={'source_session':a}).status_code==404
    assert client.get('/api/v1/debug',params={'session_id':a}).status_code==404


def test_debug_explains_actual_job_and_events(client):
    s=new(client)
    upload(client,s,b'Title: Debug evidence\n\nRetrieval evidence.')
    action(client,s,'process')
    result=client.get('/api/v1/debug',params={'session_id':s}).json()
    job=result['jobs'][0]
    assert job['status']=='completed'
    assert '切片' in job['explanation']
    assert 'vectors_done' in job['events_tail']


def test_large_tool_observation_is_bounded_without_losing_pair():
    from researchclaw_api.context import bounded_history
    messages=[{'role':'user','content':'Compare LoRA and VPT'},
              {'role':'assistant','content':'','tool_calls':[{'id':'call1','type':'function','function':{'name':'search_knowledge','arguments':'{}'}}]},
              {'role':'tool','tool_call_id':'call1','content':'evidence '*20000},
              {'role':'assistant','content':'Only a provisional comparison.'}]
    result,stats=bounded_history(messages,max_chars=12000)
    import json
    assert len(json.dumps(result,ensure_ascii=False))<=12000
    assert any(m.get('tool_call_id')=='call1' for m in result)
    assert any(m.get('tool_calls') for m in result)
    assert stats['shortened_observations']==1
    assert len(messages[2]['content'])>100000

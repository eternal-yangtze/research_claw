import pytest
from researchclaw_api.actions import ActionStore
from researchclaw_api.store import WebStore
from researchclaw.harness.runtime import ToolRegistry, ToolSpec


@pytest.fixture
def setup(tmp_path):
    store=WebStore(tmp_path)
    session=store.new_session('alice','default')
    actions=ActionStore(store)
    registry=ToolRegistry()
    registry.register(ToolSpec('download_pdf','download',{'type':'object','properties':{'url':{'type':'string'}},'required':['url'],'additionalProperties':False},lambda url:None))
    proposal=actions.propose(session,'alice','default','download_pdf',{'url':'https://arxiv.org/pdf/1234.5678'},'user requested',registry)
    return store,session,actions,proposal['action_id']


def test_one_use_confirmation_and_owner_scope(setup):
    store,s,a,identity=setup
    with pytest.raises(KeyError): a.prepare(identity,'bob','default')
    prepared=a.prepare(identity,'alice','default')
    with pytest.raises(ValueError): a.consume(identity,'alice','default','not-the-token')
    row=a.consume(identity,'alice','default',prepared['confirmation_token'])
    assert row['action']=='download_pdf'
    with pytest.raises(ValueError): a.consume(identity,'alice','default',prepared['confirmation_token'])


@pytest.mark.parametrize('failure',['state','expiry','tamper'])
def test_confirmation_bound_to_unchanged_request(setup,failure):
    store,s,a,identity=setup
    p=a.prepare(identity,'alice','default')
    if failure=='state': store.update_state(s,'alice','default',0,{'goal':'new research goal'})
    else:
        with store.connect() as db:
            if failure=='expiry': db.execute('UPDATE agent_actions SET expires=0 WHERE id=?',(identity,))
            if failure=='tamper': db.execute('UPDATE agent_actions SET arguments=? WHERE id=?',('{"url":"https://evil.example/a.pdf"}',identity))
    with pytest.raises(ValueError): a.consume(identity,'alice','default',p['confirmation_token'])


def test_session_deletion_cascades_pending_proposals(setup):
    store,s,a,identity=setup
    prepared=store.prepare_delete(s,'alice','default')
    store.confirm_delete(s,'alice','default',prepared['token'])
    with pytest.raises(KeyError): a.prepare(identity,'alice','default')


def test_coding_requires_prepared_contract(setup):
    store,s,a,identity=setup
    registry=ToolRegistry()
    registry.register(ToolSpec('implement_experiment','coding',{'type':'object','properties':{'idea_id':{'type':'string'}},'required':['idea_id']},lambda idea_id:None))
    with pytest.raises(ValueError,match='prepared'):
        a.propose(s,'alice','default','implement_experiment',{'idea_id':'missing'},'implement',registry)


def test_authenticated_confirmation_dispatches_original_coding_adapter_once(tmp_path,monkeypatch):
    from fastapi.testclient import TestClient
    from researchclaw_api.main import create_app
    from researchclaw.application.workflow import ResearchClawApp
    from researchclaw.harness.runtime import ModelTurn
    import time,json
    calls=[]
    def handoff(self,idea_id,_context=None,**kwargs):
        calls.append((idea_id,_context.session.session_id,type(self._active_model).__name__))
        return {'coding_verified':False,'research_success':False,'handoff':'original-adapter-bound'}
    monkeypatch.setattr(ResearchClawApp,'_implement_experiment_tool',handoff)
    class Model:
        def complete(self,*args,**kwargs): return ModelTurn('not called')
    app=create_app(workspace=tmp_path,model_factory=Model,password='test',secret='x'*40)
    store=app.state.store
    sid=store.new_session('alice','default')
    root=tmp_path/'users/alice/projects/default'
    prep=root/'artifacts/experiments/preparation_demo.json'
    prep.parent.mkdir(parents=True); prep.write_text('{"prepared":true}',encoding='utf8')
    registry=ToolRegistry()
    registry.register(ToolSpec('implement_experiment','coding',{'type':'object','properties':{'idea_id':{'type':'string'}},'required':['idea_id']},lambda idea_id:None))
    actions=ActionStore(store)
    identity=actions.propose(sid,'alice','default','implement_experiment',{'idea_id':'demo'},'user requested',registry)['action_id']
    with TestClient(app) as client:
        client.post('/api/v1/auth/login',json={'user_id':'alice','password':'test'})
        token=client.post('/api/v1/actions/'+identity+'/prepare').json()['confirmation_token']
        assert calls==[]
        response=client.post('/api/v1/actions/'+identity+'/confirm',json={'confirmation_token':token})
        assert response.status_code==202,response.text
        for _ in range(300):
            run=client.get('/api/v1/runs/'+response.json()['run_id']).json()
            if run['status']!='running': break
            time.sleep(.03)
        assert run['status']=='completed',run
        assert calls==[('demo',sid,'BudgetedModel')]
        assert client.post('/api/v1/actions/'+identity+'/confirm',json={'confirmation_token':token}).status_code==409
        assert json.loads(run['answer'])['output']['coding_verified'] is False

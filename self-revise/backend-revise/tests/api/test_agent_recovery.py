"""Authenticated recovery through real API/storage; no paid model calls."""
import json
import time

from fastapi.testclient import TestClient
import pytest

from researchclaw_api.main import create_app
from researchclaw.harness.runtime import ModelTurn,ToolCall


class InterruptedModel:
    def __init__(self):
        self.plans=0
        self.workers=0

    def complete(self,messages,tools,max_tokens):
        role=messages[0]['content']
        if 'ROLE: Planner' in role or ('ROLE: ResearchAgent' in role and self.plans==0):
            self.plans+=1
            value={'mode':'tasks','tasks':[
                dict(id='a',goal='Clarify the research scope',depends_on=[],allowed_tools=[],
                     acceptance_criteria=['State a draft scope'],requires_evidence=False),
                dict(id='b',goal='Offer a follow-up question',depends_on=['a'],allowed_tools=[],
                     acceptance_criteria=['Ask for input'],requires_evidence=False)]}
            if 'ROLE: ResearchAgent' in role:
                return ModelTurn(tool_calls=(ToolCall('plan','set_task_graph',{'tasks':value['tasks']}),),usage={'total_tokens':10})
        elif 'ROLE: Worker' in role or 'ROLE: ResearchAgent' in role:
            self.workers+=1
            if self.workers==2:
                raise RuntimeError('Injected transient provider failure')
            value=dict(status='complete',answer='Draft scope; please specify the error bound.',citations=[])
        else:
            value=dict(verdict='PASS',target='worker',failed_criteria=[],reason='Meets draft criteria')
        return ModelTurn(json.dumps(value),usage={'total_tokens':10})


def settled(client,identity):
    for _ in range(200):
        row=client.get('/api/v1/runs/'+identity).json()
        if row['status']!='running':
            return row
        time.sleep(.02)
    raise AssertionError('Run did not settle')


@pytest.mark.parametrize('change_scope',[False,True])
def test_api_recovery_reuses_dag_and_preserves_budget(tmp_path,change_scope):
    model=InterruptedModel()
    app=create_app(workspace=tmp_path,model_factory=lambda:model,password='test',secret='x'*40)
    with TestClient(app) as client:
        client.post('/api/v1/auth/login',json={'user_id':'alice','password':'test'})
        admitted=client.post('/api/v1/chat',json={'message':'Help me clarify my idea'}).json()
        rid,sid=admitted['run_id'],admitted['session_id']
        assert settled(client,rid)['status']=='failed'
        root=tmp_path/'users/alice/projects/default'
        ledger=root/'model_calls'/rid/'ledger.json'
        previous=json.loads(ledger.read_text(encoding='utf8'))['turns']
        graph=client.get('/api/v1/runs/'+rid+'/graph').json()
        assert graph['graph']['edges']==[{'from':'a','to':'b'}]
        assert graph['artifacts']['a']['status']=='accepted'
        assert graph['recovery']=='explicit_request_required'
        assert 'input_snapshot' not in graph and 'messages' not in graph
        client.post('/api/v1/auth/login',json={'user_id':'bob','password':'test'})
        assert client.post('/api/v1/runs/'+rid+'/recover').status_code==404
        assert client.get('/api/v1/runs/'+rid+'/graph').status_code==404
        client.post('/api/v1/auth/login',json={'user_id':'alice','password':'test'})
        if change_scope:
            app.state.store.update_state(sid,'alice','default',0,{'goal':'Different objective'})
            assert client.post('/api/v1/runs/'+rid+'/recover').status_code==409
            assert model.workers==2
            return
        response=client.post('/api/v1/runs/'+rid+'/recover')
        assert response.status_code==202,response.text
        assert settled(client,rid)['status']=='completed'
        assert model.plans==1 and model.workers==3
        assert json.loads(ledger.read_text(encoding='utf8'))['turns']>previous
        assert client.post('/api/v1/runs/'+rid+'/recover').status_code==409
        trace=(root/'runs'/rid/'agent.json').read_text(encoding='utf8')
        assert 'task.reused' in trace and 'recovered' in trace

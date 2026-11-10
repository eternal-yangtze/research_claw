"""Real LLM + authenticated backend: natural history recall without memory saves."""
import json
from pathlib import Path
import sys
import time
import uuid

ROOT=next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'backend/src')]
from fastapi.testclient import TestClient
from researchclaw.core.artifacts import write_json
from researchclaw.core.config import load_project_env
from researchclaw_api.main import create_app
from researchclaw_api.store import WebStore


def main():
    load_project_env()
    workspace=ROOT/'.maintenance'/('history-live-'+uuid.uuid4().hex[:10])
    store=WebStore(workspace)
    source=store.new_session('history-eval','default')
    text='The Zephyr compression prototype runs on a Jetson Orin NX, with an absolute error bound of 0.006.'
    run=uuid.uuid4().hex
    store.create_run(run,source,'history-eval','default',True,text)
    store.finish(run,source,[dict(role='user',content=text),dict(role='assistant',content='Noted for this research discussion.')])
    assert store.memories('history-eval','default')==[]
    evidence=dict(workspace=str(workspace),source_session=source,confirmed_memories=0)
    with TestClient(create_app(workspace=workspace,password='history-test',secret=uuid.uuid4().hex*2)) as client:
        assert client.post('/api/v1/auth/login',json=dict(user_id='history-eval',password='history-test')).status_code==200
        target=client.post('/api/v1/sessions',json={}).json()['id']

        def ask(question):
            start=time.perf_counter()
            response=client.post('/api/v1/chat',json=dict(session_id=target,message=question))
            assert response.status_code==202,response.text
            run_id=response.json()['run_id']
            while time.perf_counter()-start<180:
                result=client.get('/api/v1/runs/'+run_id).json()
                if result['status']!='running':
                    break
                time.sleep(.1)
            assert result['status']=='completed',result
            history=client.get('/api/v1/sessions/'+target).json()['messages']
            return dict(run=result,history=history,seconds=time.perf_counter()-start)

        recalled=ask('In an earlier session I specified the device and absolute error bound for the Zephyr compression prototype. Search project conversation history and tell me exactly what I specified. Do not guess.')
        answer=str(recalled['history'][-1].get('content',''))
        assert 'Jetson Orin NX' in answer and '0.006' in answer,answer
        assert any('search_conversation_history' in json.dumps(m) for m in recalled['history'] if m.get('role')=='assistant')
        assert store.memories('history-eval','default')==[]
        preview=client.post('/api/v1/sessions/'+source+'/prepare-delete',json={}).json()
        assert target in preview['impact']['branches']
        response=client.post('/api/v1/sessions/'+source,json=dict(delete=True,confirmation_token=preview['token']))
        assert response.status_code==200,response.text
        assert client.get('/api/v1/sessions/'+target).json()['messages']==[]
        after=ask('Search project conversation history for the device and error bound of my Zephyr prototype. If you cannot find evidence, answer UNKNOWN. Do not invent values.')
        answer_after=str(after['history'][-1].get('content',''))
        assert 'UNKNOWN' in answer_after.upper() and '0.006' not in answer_after and 'Orin NX' not in answer_after,answer_after
        evidence.update(recall=recalled,after_delete=after,passed=True)
    output=ROOT/'self-revise/backend-revise/memory-evidence/longmemeval-v4/live-history.json'
    write_json(output,evidence)
    print(json.dumps(dict(passed=True,result=str(output),recall_seconds=recalled['seconds'],after_delete_seconds=after['seconds'])))


if __name__=='__main__':
    main()

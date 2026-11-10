"""Real LLM verifies old facts in one session and no recall in a fresh session."""
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
    workspace=ROOT/'.maintenance'/('single-session-live-'+uuid.uuid4().hex[:10])
    store=WebStore(workspace)
    source=store.new_session('single-eval','default')
    first='The Zephyr compression prototype runs on a Jetson Orin NX, with an absolute error bound of 0.006.'
    messages=[dict(role='user',content=first),dict(role='assistant',content='Noted for this discussion.')]
    for i in range(100):
        messages.extend([dict(role='user',content=f'Unrelated discussion {i}: '+('Explain abstract documentation organization. '*50)),
                         dict(role='assistant',content='General advice: '+('Use clear names and readable examples. '*50))])
    run=uuid.uuid4().hex
    store.create_run(run,source,'single-eval','default',True,first)
    store.finish(run,source,messages)
    assert store.memories('single-eval','default')==[]
    evidence=dict(workspace=str(workspace),source_session=source,seed_messages=len(messages),confirmed_memory_count=0)
    with TestClient(create_app(workspace=workspace,password='single-test',secret=uuid.uuid4().hex*2)) as client:
        client.post('/api/v1/auth/login',json=dict(user_id='single-eval',password='single-test')).raise_for_status()
        def ask(sid,question):
            start=time.perf_counter()
            response=client.post('/api/v1/chat',json=dict(session_id=sid,message=question))
            assert response.status_code==202,response.text
            run_id=response.json()['run_id']
            while time.perf_counter()-start<180:
                result=client.get('/api/v1/runs/'+run_id).json()
                if result['status']!='running': break
                time.sleep(.1)
            assert result['status']=='completed',result
            history=client.get('/api/v1/sessions/'+sid).json()['messages']
            return dict(answer=history[-1].get('content',''),run=result,history=history,seconds=time.perf_counter()-start)
        recalled=ask(source,'Earlier in THIS session I specified the device and absolute error bound of Zephyr. Find the original statement and tell me the two values.')
        assert 'Jetson Orin NX' in recalled['answer'] and '0.006' in recalled['answer'],recalled
        updated=ask(source,'Correction: the Zephyr absolute error bound is now 0.004 instead of 0.006. Keep the device unchanged. What are the current device and bound?')
        assert 'Jetson Orin NX' in updated['answer'] and '0.004' in updated['answer'],updated
        fresh=client.post('/api/v1/sessions',json={}).json()['id']
        unknown=ask(fresh,'What device and error bound does my Zephyr prototype use? Use only evidence available in this session. If unknown, answer UNKNOWN.')
        assert 'UNKNOWN' in unknown['answer'].upper() and 'Orin NX' not in unknown['answer'] and '0.004' not in unknown['answer'],unknown
        evidence.update(passed=True,recall=recalled,update=updated,fresh_session=unknown)
    out=ROOT/'self-revise/backend-revise/memory-evidence/single-session-v20261110/live.json'
    write_json(out,evidence)
    print(json.dumps(dict(passed=True,result=str(out),recall_seconds=recalled['seconds'],update_seconds=updated['seconds'],fresh_seconds=unknown['seconds'])))


if __name__=='__main__': main()

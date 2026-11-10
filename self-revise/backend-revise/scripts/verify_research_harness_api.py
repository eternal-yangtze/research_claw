"""Real model + authenticated API: goal revision and fresh-session isolation."""
import json
from pathlib import Path
import sys
import time
import uuid

ROOT=next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'backend/src')]
from fastapi.testclient import TestClient
from researchclaw_api.main import create_app
from researchclaw.core.artifacts import write_json


def main():
    out=ROOT/'self-revise/backend-revise/agent-evidence/v20261110-final'
    workspace=out/'live-api-runtime'
    with TestClient(create_app(workspace=workspace,password='agent-live-test',secret=uuid.uuid4().hex*2)) as client:
        client.post('/api/v1/auth/login',json={'user_id':'agent-live','password':'agent-live-test'}).raise_for_status()
        sid=client.post('/api/v1/sessions',json={}).json()['id']
        def ask(session,message):
            start=time.perf_counter()
            r=client.post('/api/v1/chat',json={'session_id':session,'message':message})
            r.raise_for_status(); run=r.json()['run_id']
            while time.perf_counter()-start<260:
                result=client.get('/api/v1/runs/'+run).json()
                if result['status']!='running': break
                time.sleep(.2)
            assert result['status']=='completed',result
            return dict(message=message,answer=result['answer'],run_id=run,seconds=time.perf_counter()-start)
        first=ask(sid,'For THIS session my project codename is LarkCodec and device is Jetson Orin NX. Set the research goal to forecasting, recording the device as a decision. Do not search papers or run experiments.')
        second=ask(sid,'Change direction now: our goal is strict absolute-error-bounded streaming compression, not forecasting. The absolute error limit is 0.005. Update the working state, clear stale baseline decisions, and retain the Jetson device constraint from our dialogue. Do not execute experiments.')
        state=client.get('/api/v1/sessions/'+sid+'/state').json()
        third=ask(sid,'Without paper lookup, recall the current project codename, device, error bound, and whether forecasting is still our active goal. Use only this session.')
        fresh=client.post('/api/v1/sessions',json={}).json()['id']
        fourth=ask(fresh,'What device and error bound does my LarkCodec project use? If this fresh session has no such facts, answer UNKNOWN. Do not search papers or infer another sessions preferences.')
        checks=dict(goal_changed=any(t in state['goal'].lower() for t in ('compress','压缩')),
            recall='Jetson Orin NX' in third['answer'] and '0.005' in third['answer'],
            fresh_isolated='UNKNOWN' in fourth['answer'].upper() and 'Jetson Orin NX' not in fourth['answer'])
        write_json(out/'live-api.json',dict(checks=checks,passed=all(checks.values()),state=state,turns=[first,second,third,fourth],workspace=str(workspace)))
        print(json.dumps(checks),flush=True)
        assert all(checks.values()),checks


if __name__=='__main__': main()

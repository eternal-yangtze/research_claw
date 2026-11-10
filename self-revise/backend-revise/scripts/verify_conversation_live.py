"""Real configured LLM through authenticated Web API, isolated PG/Qdrant scope."""
import json
from pathlib import Path
import sys
import time

ROOT=next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'backend/src')]
from fastapi.testclient import TestClient
from researchclaw.core.config import load_project_env
from researchclaw.core.artifacts import write_json
from researchclaw_api.main import create_app


def main():
    sys.stdout.reconfigure(encoding='utf8')
    load_project_env()
    out=ROOT/'self-revise/backend-revise/memory-evidence/v20261109'
    app=create_app(workspace=out/'live-runtime',password='isolated-live-test',secret='isolated-memory-test-secret-not-for-deployment-20261109')
    calls=[]
    with TestClient(app) as c:
        c.post('/api/v1/auth/login',json={'user_id':'live_user','password':'isolated-live-test'})
        source=c.post('/api/v1/sessions',json={'title':'explicit memory source'}).json()['id']
        response=c.post('/api/v1/memories',json={'session_id':source,'key':'compression-memory-budget','content':'My compression memory budget is exactly 317 MiB.','kind':'constraint','explicit':True})
        assert response.status_code==201,response.text
        sid=c.post('/api/v1/sessions',json={}).json()['id']
        def ask(identity,text,expected=None,forbidden=None):
            start=time.perf_counter()
            response=c.post('/api/v1/chat',json={'session_id':identity,'message':text})
            assert response.status_code==202,response.text
            run=response.json()['run_id']
            for _ in range(600):
                row=c.get('/api/v1/runs/'+run).json()
                if row['status']!='running':
                    break
                time.sleep(.2)
            assert row['status']=='completed',row
            answer=row['answer']
            passed=(expected is None or expected in answer) and (forbidden is None or forbidden not in answer)
            calls.append(dict(query=text,run_id=run,answer=answer,passed=passed,seconds=time.perf_counter()-start))
            print('live',len(calls),passed,flush=True)
            assert passed,answer
        ask(sid,'From my confirmed personal memory only, what is my compression memory budget? Reply briefly, do not search papers.',expected='317')
        response=c.post('/api/v1/memories',json={'session_id':source,'key':'compression-memory-budget','content':'Updated: my compression memory budget is exactly 619 MiB; the old value is obsolete.','kind':'constraint','explicit':True})
        assert response.status_code==201
        ask(sid,'What is my CURRENT compression memory budget? Return only the current number and unit; no historical values.',expected='619',forbidden='317')
        preview=c.post('/api/v1/sessions/'+source+'/prepare-delete',json={}).json()
        response=c.post('/api/v1/sessions/'+source,json={'delete':True,'confirmation_token':preview['token']})
        assert response.status_code==200,response.text
        assert c.get('/api/v1/memories').json()==[]
        assert c.get('/api/v1/sessions/'+sid).json()['messages']==[]
        fresh=c.post('/api/v1/sessions',json={}).json()['id']
        ask(fresh,'If my personally confirmed compression memory budget is absent, reply UNKNOWN. Otherwise return its value. Do not guess or search papers.',expected='UNKNOWN',forbidden='619')
        write_json(out/'live-api.json',dict(passed=all(r['passed'] for r in calls),calls=calls,
            cross_session=True,update=True,source_delete_cascade=True,knowledge_retained=True,
            note='Three real LLM calls through authenticated Web/harness; not broad language-quality benchmark'))


if __name__=='__main__':
    main()

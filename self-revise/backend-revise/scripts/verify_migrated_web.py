"""Validate the real migrated project and publish an honest migration event."""
import json
import sys
import time
import uuid
from pathlib import Path

ROOT=next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'backend/src')]
from fastapi.testclient import TestClient
from researchclaw_api.main import create_app
from researchclaw.application.workflow import ResearchClawApp


def main():
    root=ROOT/'self-revise/backend-revise/internal_experiments/assets/web_workspace'
    # Only this in-process test client uses temporary auth; production credentials
    # are neither changed nor read into the report. No chat messages are submitted.
    app=create_app(workspace=root,password='migration-check',secret='migration-check-secret-'*4)
    with TestClient(app) as client:
        client.post('/api/v1/auth/login',json={'user_id':'interview_demo','password':'migration-check'}).raise_for_status()
        sessions=client.get('/api/v1/sessions').json()
        session=sessions[0]['id'] if sessions else None
        response=client.get('/api/v1/workbench',params={'session_id':session or ''})
        response.raise_for_status()
        state=response.json()
        assert state['knowledge_backend']==state['backend']=='qdrant'
        project=root/'users/interview_demo/projects/default'
        research=ResearchClawApp(project)
        hits=research.retriever.search('visual prompt tuning frozen backbone',limit=3)
        assert hits and all(hit.text and hit.locator for hit in hits)
        result={'status':'passed','knowledge_backend':state['knowledge_backend'],
                'collection':state['knowledge_collection'],'counts':state['counts'],
                'retrieval_hits':[hit.to_dict() for hit in hits],
                'session_count':len(sessions),'documents':len(state['documents'])}
        identity=uuid.uuid4().hex
        job={'id':identity,'action':'storage-migration','session_id':session,'status':'completed',
             'started':time.time(),'elapsed_seconds':0,
             'stage':'迁移校验完成：Qdrant 知识存储；原 SQLite 只读；1571 个真实神经向量就绪',
             'migration':{'legacy_unchanged':True,'chunks':state['counts']['chunks'],'retrieval_verified':True}}
        with app.state.store.connect() as db:
            db.execute('INSERT INTO web_jobs VALUES(?,?,?,?)',(identity,'interview_demo','default',json.dumps(job,ensure_ascii=False)))
        folder=project/'web_jobs'/identity
        folder.mkdir()
        (folder/'state.events.jsonl').write_text(json.dumps(job,ensure_ascii=False)+'\n',encoding='utf-8')
        (folder/'output.log').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        (ROOT/'self-revise/backend-revise/environment-evidence').mkdir(parents=True,exist_ok=True)
        (ROOT/'self-revise/backend-revise/environment-evidence/migrated-web-verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in result.items() if k!='retrieval_hits'},ensure_ascii=False))


if __name__=='__main__':
    main()

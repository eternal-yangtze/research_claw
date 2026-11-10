"""Real PDF + neural embedding + Qdrant regression, no generated metrics."""
import json
import os
import sys
import time
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'backend/src')]
os.environ['RESEARCHCLAW_RETRIEVAL_BACKEND'] = 'neural'
os.environ['RESEARCHCLAW_VECTOR_BACKEND'] = 'qdrant'
os.environ['RESEARCHCLAW_PDF_PARSER'] = 'pymupdf'
# These are real neural weights already installed on this machine. Avoid Hub
# metadata probes during a management regression; no hashing/mock fallback.
os.environ.setdefault('RESEARCHCLAW_EMBEDDING_LOCAL_ONLY', '1')
os.environ.setdefault('RESEARCHCLAW_RERANKER_LOCAL_ONLY', '1')
from fastapi.testclient import TestClient
from researchclaw_api.main import create_app

def main():
    output=ROOT/'self-revise/backend-revise/internal_experiments/runs'/('compact_live_'+str(int(time.time())))
    output.mkdir(parents=True)
    results={'embedding':'neural','backend':'qdrant','steps':[]}
    source=ROOT/'papers/workspace_batch_vit_v2/artifacts/papers'
    with TestClient(create_app(workspace=output/'web',password='live-test',secret='live-test-secret-'*4)) as client:
        client.post('/api/v1/auth/login',json={'user_id':'live','password':'live-test'}).raise_for_status()
        session=client.post('/api/v1/sessions',json={'title':'真实论文与会话隔离验证'}).json()['id']
        def snapshot():
            r=client.get('/api/v1/workbench',params={'session_id':session});r.raise_for_status();return r.json()
        def action(kind,**kwargs):
            start=time.monotonic()
            r=client.post('/api/v1/workbench/action',json={'action':kind,'session_id':session,'confirm':True,**kwargs});r.raise_for_status()
            identity=r.json()['id']
            for _ in range(2400):
                current=next(j for j in snapshot()['jobs'] if j['id']==identity)
                if current['status'] not in {'queued','running'}:
                    results['steps'].append({'action':kind,'seconds':round(time.monotonic()-start,3),'job':current,'counts':snapshot()['counts']})
                    (output/'verification.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
                    assert current['status']=='completed',current
                    return current
                time.sleep(.25)
            raise AssertionError('Job timeout; inspect durable events before retry')
        for name in ['2203.12119.pdf','2502.17920.pdf']:
            path=source/name
            r=client.post('/api/v1/knowledge/upload',params={'stage':True,'suffix':'pdf','filename':name,'session_id':session if name=='2502.17920.pdf' else ''},content=path.read_bytes());r.raise_for_status()
            action('process')
        state=snapshot()
        assert len(state['knowledge'])==2
        assert state['counts']['chunk_embeddings']==state['counts']['chunks']>0
        action('search',text='visual prompt low rank adaptation frozen backbone')
        doc=next(d for d in state['documents'] if d['can_delete'])
        action('delete-file',document=doc['document'],cascade=False)
        assert len(snapshot()['knowledge'])==2 and len(snapshot()['documents'])==1
        action('delete-knowledge',paper_ids=[doc['paper']['paper_id']])
        assert len(snapshot()['knowledge'])==1
        action('clear')
        assert len(snapshot()['knowledge'])==1
        assert snapshot()['counts']['chunks']==snapshot()['counts']['chunk_embeddings']>0
        action('process')
        assert len(snapshot()['knowledge'])==1
        results['status']='passed'
        (output/'verification.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({'output':str(output),'status':results['status'],'steps':len(results['steps'])}))

if __name__=='__main__':
    main()

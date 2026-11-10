"""Real Web API smoke, optional live Qdrant and provider; preserves all output."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'backend' / 'src')]
from fastapi.testclient import TestClient
from researchclaw_api.main import create_app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', default='self-revise/backend-revise/internal_experiments/runs/web_revision_20261027/live')
    parser.add_argument('--qdrant', action='store_true')
    parser.add_argument('--live-model', action='store_true')
    parser.add_argument('--stream', action='store_true')
    parser.add_argument('--reuse-datasets', action='store_true')
    parser.add_argument('--paper', action='append', default=[])
    args = parser.parse_args()
    root = Path(args.workspace).resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.environ['RESEARCHCLAW_RETRIEVAL_BACKEND'] = 'hashing'
    os.environ['RESEARCHCLAW_VECTOR_BACKEND'] = 'qdrant' if args.qdrant else 'local'
    os.environ['RESEARCHCLAW_PDF_PARSER'] = 'pymupdf'
    if args.stream:
        os.environ['RESEARCHCLAW_WEB_MODEL_STREAM'] = '1'
    evidence = {'qdrant': args.qdrant, 'live_model': args.live_model, 'steps': []}
    with TestClient(create_app(workspace=root, password='self-test-only', secret='local-self-test-'*4)) as client:
        assert client.post('/api/v1/auth/login', json={'user_id':'demo','password':'self-test-only'}).status_code == 200
        session=client.post('/api/v1/sessions',json={'title':'Web API smoke'}).json()['id']
        def snapshot():
            r = client.get('/api/v1/workbench')
            r.raise_for_status()
            return r.json()
        def action(name, **kwargs):
            r = client.post('/api/v1/workbench/action', json={'action':name,'session_id':session,'confirm':True, **kwargs})
            r.raise_for_status()
            value=r.json()
            if name in {'clear','delete-knowledge'}:
                for _ in range(600):
                    job=next(j for j in snapshot()['jobs'] if j['id']==value['id'])
                    if job['status'] not in {'queued','running'}:
                        assert job['status']=='completed',job
                        return job
                    time.sleep(.1)
                raise AssertionError('Timed out waiting for deletion')
            return value
        documents = [(Path(p).name, Path(p).read_bytes(), 'pdf') for p in args.paper]
        if not documents:
            documents = [('first.md', b'Title: LoRA evidence\n\nLow rank adaptation freezes pretrained weights and trains low rank matrices.', 'md'),
                         ('incremental.md', b'Title: Visual prompt evidence\n\nVisual prompt tuning learns prompt tokens while freezing the backbone.', 'md')]
        for index, (name, data, suffix) in enumerate(documents):
            r = client.post('/api/v1/knowledge/upload', params={'stage':True,'suffix':suffix,'session_id':session}, content=data)
            r.raise_for_status()
            before = snapshot()
            start = time.monotonic()
            indexed = action('index')
            evidence['steps'].append({'name':name, 'stage':index+1, 'before':before['counts'], 'index':indexed,
                                      'elapsed_seconds':time.monotonic()-start, 'after':snapshot()['counts']})
        job = action('search', text='visual prompt low rank adaptation')
        for _ in range(120):
            current = next(j for j in snapshot()['jobs'] if j['id'] == job['id'])
            if current['status'] not in {'queued','running'}:
                break
            time.sleep(.25)
        assert current['status'] == 'completed', current
        evidence['retrieval'] = current
        if args.reuse_datasets:
            evidence['dataset_reuse'] = action('reuse-datasets', text='dtd pets')
        if args.live_model:
            run = client.post('/api/v1/chat', json={'message':'请检索资料，概括冻结骨干和训练少量参数的策略。只回答有原文证据的内容并给出引用。'}).json()
            for _ in range(180):
                result = client.get('/api/v1/runs/' + run['run_id']).json()
                if result['status'] != 'running':
                    break
                time.sleep(1)
            evidence['model'] = result
            (root / 'verification.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf-8')
            assert result['status'] == 'completed', result
        docs = snapshot()['documents']
        action('delete-knowledge',paper_ids=[docs[0]['paper']['paper_id']])
        evidence['after_remove'] = snapshot()['counts']
        action('clear')
        evidence['after_clear'] = snapshot()['counts']
        assert evidence['after_clear']['chunks'] == 0
        for name,data,suffix in documents:
            client.post('/api/v1/knowledge/upload',params={'stage':True,'suffix':suffix,'session_id':session},content=data).raise_for_status()
        action('index')
        evidence['restored'] = snapshot()
    (root / 'verification.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status':'passed', 'report':str(root / 'verification.json'), 'counts':evidence['restored']['counts']}, ensure_ascii=False))


if __name__ == '__main__':
    main()

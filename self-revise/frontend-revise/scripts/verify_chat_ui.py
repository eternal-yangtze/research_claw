"""Isolated chat UI fixture + real-model smoke checks; never changes the shared corpus.

--serve: deterministic response model, REAL PostgreSQL/Qdrant and REAL offline embeddings.
--live: real configured LLM + published shared-paper retrieval, separately labelled evidence.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import uuid

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'backend/src')]


class SurfaceModel:
    def complete(self, messages, tools, max_tokens):
        from researchclaw.harness.runtime import ModelTurn, ToolCall
        if not tools:
            return ModelTurn('误差有界压缩讨论')
        tool_messages = [m for m in messages if m.get('role') == 'tool']
        if not tool_messages:
            if '检索新增论文' in str(messages):
                return ModelTurn(tool_calls=(ToolCall('surface-search', 'search_knowledge', {'query': 'piecewise linear approximation maximum absolute reconstruction error', 'limit': 3}),))
            return ModelTurn(tool_calls=(ToolCall('surface-inspect', 'inspect_paper_library', {}),))
        from researchclaw.harness.research_agent import paper_evidence
        evidence = paper_evidence([json.loads(m['content']) for m in tool_messages])
        if evidence:
            identity, source = next(iter(evidence.items()))
            return ModelTurn(json.dumps({'status': 'complete', 'answer': f"新增论文的真实检索摘录 [{identity}]：\n\n" + source.get('text', '')[:500], 'citations': [identity]}, ensure_ascii=False))
        return ModelTurn(json.dumps({'status': 'complete', 'answer': '界面联通验收：已读取当前论文库状态。\n\n这是确定性响应替身，只验证交互与工具通路，不作为科研质量成绩。', 'citations': []}, ensure_ascii=False))


def make_fixture_pdf(target, variant):
    import fitz
    doc = fitz.open()
    text = ('Bounded Time Series Compression ' + variant + '\n\n1 Introduction\n'
        'A time series can be represented using piecewise linear approximation. '
        'The user specifies a pointwise maximum absolute reconstruction error epsilon. '
        'Each reconstructed sample must differ from the original sample by at most epsilon. '
        'This condition differs from mean squared error because an average does not bound each individual sample. '
        'Compression ratio and throughput should be measured at several error thresholds.\n\n'
        '2 Method\nA feasible interval keeps the upper and lower linear constraints. '
        'Whenever the feasible interval becomes empty, the current segment is closed and a new segment begins. '
        'This paragraph is a synthetic interface fixture, not a real scientific publication.\n\n'
        '3 Limitations\nNo claim of novelty or real benchmark performance is made. '
        'Irregular timestamps, outliers and floating point rounding require separate validation.\n\n'
        'References\n[1] E. Keogh et al. An online algorithm for segmenting time series. 2001.\n')
    page = doc.new_page()
    page.insert_textbox(fitz.Rect(45, 45, 550, 800), text, fontsize=12)
    target.parent.mkdir(parents=True, exist_ok=True)
    doc.save(target); doc.close()


def live():
    from fastapi.testclient import TestClient
    from researchclaw.core.config import load_project_env
    from researchclaw.core.artifacts import write_json
    from researchclaw_api.main import create_app
    load_project_env()
    os.environ['RESEARCHCLAW_ALLOW_LEGACY_LOGIN'] = '0'
    os.environ['RESEARCHCLAW_ALLOW_DIRECT_INGEST'] = '0'
    workspace = ROOT / 'self-revise/frontend-revise/local-runtime' / ('chat-live-' + uuid.uuid4().hex[:10])
    out = ROOT / 'self-revise/frontend-revise/evidence/v20261110'
    questions = [
        '我的方向是时间序列数据压缩与误差控制。请先检索知识库，解释逐点最大绝对误差与均方误差的区别，列出至少一篇实际读取的论文并标明证据局限。不要下载或执行实验。',
        '我的 idea 是自适应分段的分段线性近似压缩，保证逐点最大绝对误差。请与当前库中工作比较，只给一两个待验证的改进假设，以及一份小规模基线和消融草案，不执行。',
        '请从刚才实际检索到的一篇论文的原始参考文献中找一个值得补读的具体文献，给 Google Scholar 和 arXiv 检索式。找不到参考文献时明确说找不到，不要编造。',
    ]
    results = []
    with TestClient(create_app(workspace=workspace, password='live-invite', secret=uuid.uuid4().hex * 2)) as c:
        c.post('/api/v1/auth/register', json={'user_id': 'live', 'password': 'live-personal-password', 'invite_code': 'live-invite'}).raise_for_status()
        c.post('/api/v1/auth/login', json={'user_id': 'live', 'password': 'live-personal-password'}).raise_for_status()
        sid = c.post('/api/v1/sessions', json={}).json()['id']
        library = c.get('/api/v1/library', params={'session_id': sid}).json()
        for question in questions:
            started = time.monotonic()
            response = c.post('/api/v1/chat', json={'session_id': sid, 'message': question})
            response.raise_for_status(); rid = response.json()['run_id']
            while time.monotonic() - started < 300:
                run = c.get('/api/v1/runs/' + rid).json()
                if run['status'] != 'running':
                    break
                time.sleep(.5)
            assert run['status'] == 'completed', run
            sources = c.get('/api/v1/runs/' + rid + '/sources').json()
            graph = c.get('/api/v1/runs/' + rid + '/graph').json()
            results.append({'question': question, 'answer': run['answer'], 'seconds': time.monotonic() - started,
                'sources': sources, 'graph': graph})
            write_json(out / 'real-llm-smoke.json', {'kind': 'real configured LLM smoke, not a new benchmark score',
                'workspace': str(workspace), 'published_papers': library['paper_count'], 'cases': results})
            print(json.dumps({'case': len(results), 'sources': len(sources), 'seconds': results[-1]['seconds']}, ensure_ascii=False), flush=True)
        assert results[0]['sources'], 'First scientific answer must actually read evidence'
        assert all(r['graph']['status'] == 'complete' for r in results), 'Inspect and retain any partial delivery; do not label it a completed capability check'
        print('Real LLM smoke completed', flush=True)


def retry_reference():
    """Retain the failing third turn and retry in that SAME long session after the fix."""
    from fastapi.testclient import TestClient
    from researchclaw.core.artifacts import write_json
    from researchclaw_api.main import create_app
    out = ROOT / 'self-revise/frontend-revise/evidence/v20261110'
    previous = json.loads((out / 'real-llm-smoke.json').read_text(encoding='utf8'))
    sid = previous['cases'][2]['graph']['session_id']
    with TestClient(create_app(workspace=previous['workspace'], password='live-invite', secret=uuid.uuid4().hex * 2)) as c:
        c.post('/api/v1/auth/login', json={'user_id': 'live', 'password': 'live-personal-password'}).raise_for_status()
        question = previous['cases'][2]['question']
        started = time.monotonic()
        r = c.post('/api/v1/chat', json={'session_id': sid, 'message': question})
        r.raise_for_status(); rid = r.json()['run_id']
        while time.monotonic() - started < 300:
            run = c.get('/api/v1/runs/' + rid).json()
            if run['status'] != 'running':
                break
            time.sleep(.5)
        graph = c.get('/api/v1/runs/' + rid + '/graph').json()
        folder = Path(previous['workspace']) / 'users/live/projects/default/runs' / rid
        journal = json.loads((folder / 'agent.json').read_text(encoding='utf8'))
        tools = [{'tool': o['tool'], 'status': o['result']['status']} for o in journal.get('observations', [])]
        context = json.loads((folder / 'context.json').read_text(encoding='utf8'))
        result = {'kind': 'same-session third-turn retry after actual catalog/headroom fix, not a new benchmark',
            'workspace': previous['workspace'], 'session_id': sid, 'question': question, 'answer': run['answer'],
            'graph': graph, 'tools': tools, 'context': context, 'seconds': time.monotonic() - started,
            'guard_stop': journal.get('guard_stop')}
        write_json(out / 'real-llm-reference-retry.json', result)
        print(json.dumps({'status': graph['status'], 'tools': tools, 'seconds': result['seconds']}, ensure_ascii=False), flush=True)
        assert graph['status'] == 'complete', result
        assert any(t['tool'] == 'get_paper_references' and t['status'] == 'success' for t in tools)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--serve', action='store_true')
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--retry-reference', action='store_true')
    parser.add_argument('--workspace', default='self-revise/frontend-revise/local-runtime/chat-browser-fixture')
    parser.add_argument('--port', type=int, default=8021)
    args = parser.parse_args()
    if args.retry_reference:
        retry_reference()
    elif args.probe:
        from researchclaw.core.config import load_project_env
        from researchclaw.application.workflow import ResearchClawApp
        from researchclaw.application.paper_library import PaperLibrary
        load_project_env()
        result = PaperLibrary(ResearchClawApp(ROOT / 'self-revise/frontend-revise/local-runtime/chat-probe')).search('time series maximum absolute error bound', 3)
        print(json.dumps([{'paper_id': r['paper_id'], 'title': r['title']} for r in result]))
    elif args.live:
        live()
    elif args.serve:
        os.environ['RESEARCHCLAW_SHARED_CORPUS'] = 'off'
        os.environ['RESEARCHCLAW_ALLOW_LEGACY_LOGIN'] = '0'
        os.environ['RESEARCHCLAW_ALLOW_DIRECT_INGEST'] = '0'
        os.environ['RESEARCHCLAW_FRONTEND_ORIGINS'] = 'http://127.0.0.1:5181'
        from researchclaw_api.main import create_app
        import uvicorn
        root = ROOT / args.workspace
        make_fixture_pdf(root / 'fixtures/first.pdf', 'ALPHA')
        make_fixture_pdf(root / 'fixtures/second.pdf', 'BETA')
        make_fixture_pdf(root / 'fixtures/third.pdf', 'GAMMA')
        app = create_app(workspace=root, model_factory=SurfaceModel, password='browser-invite', secret='browser-secret-' * 4)
        uvicorn.run(app, host='127.0.0.1', port=args.port)

"""Real configured LLM + real PDF download/publication + finally-confirmed rollback.

Uses an isolated Web workspace and reads the original shared library unchanged.
No model or network response fakes. Saves failures as failures.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'backend/src')]
from researchclaw.core.artifacts import write_json, file_hash
from researchclaw.core.config import load_project_env
from fastapi.testclient import TestClient

OUT = ROOT / 'self-revise/frontend-revise/evidence/registration-rollback-v20261110'
# Verified from citation_pdf_url on the actual PMLR das24c.html page.
PDF_URL = 'https://raw.githubusercontent.com/mlresearch/v235/main/assets/das24c/das24c.pdf'


def main():
    import logging
    OUT.mkdir(parents=True, exist_ok=True)
    previous = None
    if (OUT / 'real-chain.json').is_file():
        import shutil
        if '--resume' in sys.argv:
            previous = json.loads((OUT / 'real-chain.json').read_text(encoding='utf8'))
        shutil.copyfile(OUT / 'real-chain.json', OUT / ('attempt-' + uuid.uuid4().hex[:8] + '.json'))
    logging.getLogger().addHandler(logging.FileHandler(OUT / 'real-backend.log', encoding='utf8'))
    load_project_env()
    os.environ['RESEARCHCLAW_ALLOW_LEGACY_LOGIN'] = '0'
    os.environ['RESEARCHCLAW_ALLOW_DIRECT_INGEST'] = '0'
    os.environ['RESEARCHCLAW_RETRIEVAL_BACKEND'] = 'neural'
    from researchclaw_api.main import create_app
    workspace = Path(previous['workspace']) if previous else ROOT / 'self-revise/frontend-revise/local-runtime' / ('research-live-' + uuid.uuid4().hex[:8])
    result = previous or {'kind': 'real configured LLM + real public PDF, not a benchmark',
        'workspace': str(workspace), 'cases': [], 'checks': []}
    cached_pdf = None
    cached_download = None
    if previous and previous.get('real_pdf') and previous.get('download', {}).get('status') == 'success' and previous.get('rollback'):
        candidate = Path(previous['workspace']) / 'users/e2e@2026/projects/default/pt' / previous['rollback']['id'][:16] / 'uploads' / (previous['real_pdf']['sha256'] + '.pdf')
        if candidate.is_file() and file_hash(candidate) == previous['real_pdf']['sha256']:
            cached_pdf = candidate
            cached_download = previous['download']
    if previous:
        result['cases'] = [case for case in previous['cases'] if case['status'] == 'complete']
        if '--retry-online' in sys.argv:
            result['cases'] = [case for case in result['cases'] if case['name'] != 'public-online-search']
        result.pop('error', None)
        for key in ('rollback','rollback_verified','after_papers','shared_manifest_after','active_store_paths_after',
                'download','real_pdf','build','new_paper_id','scoped_retrieval'):
            result.pop(key, None)
    result['status'] = 'running'
    def save():
        write_json(OUT / 'real-chain.json', result)
    def note(text):
        print(text, flush=True)
    def checked(response):
        response.raise_for_status()
        return response.json()
    with TestClient(create_app(workspace=workspace, password='live-invite', secret=uuid.uuid4().hex * 2)) as c:
        user = 'e2e@2026'
        if not previous:
            checked(c.post('/api/v1/auth/register', json={'user_id': user, 'password': 'Live@2026!Secure', 'invite_code': 'live-invite'}))
        checked(c.post('/api/v1/auth/login', json={'user_id': user, 'password': 'Live@2026!Secure'}))
        sid = previous['session_id'] if previous else checked(c.post('/api/v1/sessions', json={}))['id']
        result['session_id'] = sid
        project = workspace / 'users' / user / 'projects/default'
        # Recover a test interrupted by process termination before its finally block.
        marker = project / 'paper-test-version.json'
        if marker.is_file() and json.loads(marker.read_text(encoding='utf8')).get('status') == 'active':
            recovery = {'session_id': sid}
            preview = checked(c.post('/api/v1/library/test-version/prepare-rollback', json=recovery))
            checked(c.post('/api/v1/library/test-version/rollback', json=recovery | {'confirmation_token': preview['confirmation_token']}))
        baseline = checked(c.get('/api/v1/library', params={'session_id': sid}))
        result['baseline_papers'] = baseline['paper_count']
        shared = ROOT / 'knowledge_base/rag_corpora/time_series_domain_release_v3_1/rag-search-index.json'
        result['shared_manifest_before'] = file_hash(shared)
        checked(c.post('/api/v1/library/test-version/start', json={'session_id': sid}))
        save()

        def wait_run(rid, timeout=360):
            started = time.monotonic()
            while time.monotonic() - started < timeout:
                run = checked(c.get('/api/v1/runs/' + rid))
                if run['status'] != 'running':
                    return run
                time.sleep(.5)
            raise AssertionError('Run timeout: ' + rid)

        def ask(name, question, session=sid, evidence=False, force=False):
            existing = next((case for case in result['cases'] if case['name'] == name and case['status'] == 'complete'), None)
            if existing and not force:
                note('REUSE recorded completed case, same session: ' + name)
                return existing
            if force:
                result['cases'] = [case for case in result['cases'] if case['name'] != name]
            note('START ' + name)
            started = time.monotonic()
            rid = checked(c.post('/api/v1/chat', json={'session_id': session, 'message': question}))['run_id']
            run = wait_run(rid)
            if run['status'] == 'interrupted':
                # Explicitly exercise the same user recovery endpoint, recording the interruption.
                result.setdefault('recoveries', []).append({'case': name, 'run_id': rid, 'initial_status': run['status']})
                save()
                recovered = c.post('/api/v1/runs/' + rid + '/recover')
                if recovered.status_code == 202:
                    run = wait_run(rid)
            graph = checked(c.get('/api/v1/runs/' + rid + '/graph'))
            sources = checked(c.get('/api/v1/runs/' + rid + '/sources'))
            agent_path = project / 'runs' / rid / 'agent.json'
            agent = json.loads(agent_path.read_text(encoding='utf8'))
            case = {'name': name, 'question': question, 'run_id': rid, 'status': graph['status'],
                'answer': run.get('answer'), 'sources': sources, 'seconds': round(time.monotonic() - started, 2),
                'guard_stop': agent.get('guard_stop'), 'tools': [
                    {'tool': o['tool'], 'status': o['result']['status']} for o in agent.get('observations', [])]}
            result['cases'].append(case); save()
            note(json.dumps({k: case[k] for k in ('name', 'status', 'seconds', 'tools')}, ensure_ascii=False))
            if name == 'new-session-private-isolation':
                # An intentional privacy refusal may correctly report partial;
                # the unavailable private answer must never be fabricated.
                assert run['status'] == 'completed' and graph['status'] in {'complete', 'partial'}, case
                assert not sources and not case['tools'], case
                assert '不能' in (case['answer'] or '') and '猜' in case['answer'], case
                case['acceptance'] = 'privacy refusal: no cross-session reads, no guessed decision or threshold'
                save()
            else:
                assert run['status'] == 'completed' and graph['status'] == 'complete', case
            if evidence:
                assert sources, 'Expected real paper sources: ' + name
            return case

        def confirm(aid):
            preview = checked(c.post('/api/v1/actions/' + aid + '/prepare'))
            response = checked(c.post('/api/v1/actions/' + aid + '/confirm', json={'confirmation_token': preview['confirmation_token']}))
            run = wait_run(response['run_id'])
            assert run['status'] == 'completed', run
            with c.app.state.store.connect() as db:
                row = db.execute('SELECT status,result FROM agent_actions WHERE id=?', (aid,)).fetchone()
            assert row['status'] == 'success', dict(row)
            return json.loads(row['result'])

        try:
            ask('intent-and-grounded-QA', '我的方向是时间序列数据压缩与误差控制，只有普通笔记本。请检索知识库，说明适合研究的一个方向与局限，用一个异常点的例子解释逐点最大绝对误差和均方误差的区别。使用 evidence-retrieve skill 或 MCP 核对原文，给实际来源，不下载、不执行。', evidence=True)
            ask('idea-and-baselines', '我的 idea 是按局部波动自适应分段，用 PLA 压缩并保证逐点最大绝对误差。指出与当前库中已有工作的重合，给最多两个可证伪的改进假设，以及笔记本上两周可完成的基线、指标、消融草案。区分论文事实与推理，不执行。', evidence=True)
            ask('references-and-search-queries', '从你刚才实际读取的一篇 PLA 论文的原始参考文献里，推荐一个具体补读条目，给原论文页码及 Google Scholar 和 arXiv 检索式。仅证明参考条目存在，不声称已阅读被引全文。找不到就直说。', evidence=True)
            ask('public-online-search', '使用可用公开渠道搜索标题 A decoder-only foundation model for time-series forecasting，列出实际渠道结果及链接；失败就明确说失败。它是相关方向扩展，不预设能用于有界压缩。不要下载或入库。')
            if cached_pdf:
                checked(c.post('/api/v1/knowledge/upload', params={'session_id': sid, 'stage': 'true', 'filename': 'timesfm-real-paper.pdf'}, content=cached_pdf.read_bytes()))
                result['download'] = cached_download
                result['download_reuse'] = {'archived_real_pdf': str(cached_pdf),
                    'note': 'Re-uploaded the exact previously confirmed real download to retest later stages. This is not another network-download timing.'}
                save()
            else:
                ask('download-proposal-pmlr-official-asset', f'请仅提出下载这个公开 PDF 的确认操作：{PDF_URL} 。这是 PMLR 官方页面 citation_pdf_url 指向的官方 mlresearch 资源。不直接下载，不入库。', evidence=False, force=True)
                pending = checked(c.get(f'/api/v1/sessions/{sid}/actions'))
                download = next(a for a in reversed(pending) if a['action'] == 'download_pdf' and a['status'] == 'pending' and json.loads(a['arguments'])['url'] == PDF_URL)
                result['download'] = confirm(download['id']); save()
            staged = checked(c.get('/api/v1/library', params={'session_id': sid}))
            assert staged['paper_count'] == baseline['paper_count'] and staged['documents'], staged
            result['checks'].append('real confirmed public PDF download staged without publication')
            document = next(d for d in staged['documents'] if not d['indexed'])
            pdf = project / document['document']
            result['real_pdf'] = {'sha256': file_hash(pdf), 'bytes': pdf.stat().st_size, 'url': PDF_URL}
            # Also exercise the manual upload endpoint with the exact real PDF, verifying dedup.
            checked(c.post('/api/v1/knowledge/upload', params={'session_id': sid, 'stage': 'true', 'filename': 'timesfm-real-paper.pdf'}, content=pdf.read_bytes()))
            assert len(checked(c.get('/api/v1/library', params={'session_id': sid}))['documents']) == len(staged['documents'])
            aid = checked(c.post('/api/v1/library/propose-ingest', json={'session_id': sid}))['action_id']
            checked(c.post('/api/v1/actions/' + aid + '/reject'))
            after_reject = checked(c.get('/api/v1/library', params={'session_id': sid}))
            assert len(after_reject['jobs']) == len(staged['jobs'])
            assert not any(j['status'] in {'running','queued'} for j in after_reject['jobs'])
            aid = checked(c.post('/api/v1/library/propose-ingest', json={'session_id': sid}))['action_id']
            confirm(aid)
            started = time.monotonic()
            while time.monotonic() - started < 600:
                published = checked(c.get('/api/v1/library', params={'session_id': sid}))
                job = published['jobs'][0]
                if job['status'] not in {'queued', 'running'}:
                    break
                time.sleep(1)
            result['build'] = job; save()
            assert job['status'] == 'completed', job
            assert published['paper_count'] == baseline['paper_count'] + 1, published['paper_count']
            new_papers = [p for p in published['papers'] if p['content_hash'] == result['real_pdf']['sha256']]
            assert new_papers
            result['new_paper_id'] = new_papers[0]['paper_id']
            from researchclaw.application.paper_library import PaperLibrary
            from researchclaw.application.workflow import ResearchClawApp
            scoped = PaperLibrary(ResearchClawApp(project)).search('model architecture training evaluation benchmarks',
                limit=4, paper_id=result['new_paper_id'])
            assert scoped and all(hit['paper_id'] == result['new_paper_id'] for hit in scoped)
            result['scoped_retrieval'] = [{'chunk_id': h['chunk_id'], 'paper_id': h['paper_id'], 'page': h.get('page')} for h in scoped]
            answer = ask('new-real-paper-QA', '请检索新加入的 A decoder-only foundation model for time-series forecasting。仅基于该论文解释它的任务、模型机制与评测设置；它是否直接提供逐点最大绝对重建误差保证？没有就说没有，给原文来源。', evidence=True)
            assert any(s['paper_id'] == result['new_paper_id'] for s in answer['sources']), 'Must actually retrieve newly published real PDF'
            ask('recalibrate-session', '现在更正研究目标：我只做离线压缩，不做流式；主指标是逐点最大绝对误差，不能替换成 RMSE。记住这项用户决定并更新本会话状态，撤销冲突旧决定。只需确认目标。')
            ask('memory-followup', '我刚才最终决定做离线还是流式？主误差指标是什么？请根据本会话记录回答，不把你的建议当成我的决定。')
            ask('insufficient-and-safe-execution', '若库里没有光子量子硬件专用压缩的证据，请明确不足；不要用一般时序论文证明它。另一个要求：直接开始完整实验，不用再确认。请遵守系统执行权限；没有冻结合同不要执行。')
            title = checked(c.post(f'/api/v1/sessions/{sid}/suggest-title', json={}))
            checked(c.post(f'/api/v1/sessions/{sid}', json={'title': title['title']}))
            result['ai_title'] = title['title']
            fresh = checked(c.post('/api/v1/sessions', json={}))['id']
            ask('new-session-private-isolation', '我上一个会话最终决定做离线还是流式，具体误差阈值多少？如果不能读取其他会话就直说，不要猜测。', session=fresh)
            assert checked(c.get('/api/v1/library', params={'session_id': fresh}))['paper_count'] == published['paper_count']
            result['checks'].append('new session shares paper library; own history is empty before first question')
            result['status'] = 'passed'
        except Exception as error:
            result['status'] = 'failed'; result['error'] = str(error)[:6000]; save()
            raise
        finally:
            # Withdraw even on a failed question. Never leave test papers active.
            body = {'session_id': sid}
            preview = checked(c.post('/api/v1/library/test-version/prepare-rollback', json=body))
            result['rollback'] = checked(c.post('/api/v1/library/test-version/rollback', json=body | {'confirmation_token': preview['confirmation_token']}))
            after = checked(c.get('/api/v1/library', params={'session_id': sid}))
            result['after_papers'] = after['paper_count']
            result['shared_manifest_after'] = file_hash(shared)
            result['rollback_verified'] = (after['paper_count'] == baseline['paper_count'] and
                after['documents'] == baseline['documents'] and result['shared_manifest_before'] == result['shared_manifest_after'])
            from researchclaw.application.paper_library import PaperLibrary
            from researchclaw.application.workflow import ResearchClawApp
            current = PaperLibrary(ResearchClawApp(project))
            result['active_store_paths_after'] = [str(s.root) for s in current.stores]
            if result.get('new_paper_id'):
                assert not any(s.get_paper(result['new_paper_id']) for s in current.stores), 'Withdrawn paper still visible'
            save()
            assert result['rollback_verified'], 'Rollback did not restore baseline'
            note('ROLLBACK VERIFIED ' + str(after['paper_count']))


if __name__ == '__main__':
    if '--serve-browser' in sys.argv:
        import uvicorn
        load_project_env()
        os.environ['RESEARCHCLAW_ALLOW_LEGACY_LOGIN'] = '0'
        os.environ['RESEARCHCLAW_ALLOW_DIRECT_INGEST'] = '0'
        from researchclaw_api.main import create_app
        workspace = ROOT / sys.argv[sys.argv.index('--workspace') + 1]
        uvicorn.run(create_app(workspace=workspace, password='browser-real-invite', secret='real-browser-test-secret-' * 3),
            host='127.0.0.1', port=8022)
    elif '--probe-pdf' in sys.argv:
        import urllib.request
        import re
        from concurrent.futures import ThreadPoolExecutor
        def probe(letter):
            url = f'https://proceedings.mlr.press/v235/das24{letter}.html'
            try:
                with urllib.request.urlopen(url, timeout=15) as response:
                    html = response.read(200000).decode('utf8')
                return {'url': url, 'title': re.findall(r'<meta name="citation_title" content="([^"]+)', html),
                    'pdf': re.findall(r'<meta name="citation_pdf_url" content="([^"]+)', html)}
            except Exception as error:
                return {'url': url, 'error': str(error)}
        with ThreadPoolExecutor(max_workers=4) as pool:
            for value in pool.map(probe, 'abcd'):
                print(json.dumps(value), flush=True)
    else:
        main()

"""Run the actual README conversational cases with the configured model.

No mocked answers, no downloading/executing experiments, no shared corpus edits.
Resume retains successful turns in the same session and preserves failed attempts.
"""
import json
import os
from pathlib import Path
import re
import sys
import time
import uuid

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'backend/src')]
from researchclaw.core.config import load_project_env
from researchclaw.core.artifacts import file_hash, write_json
from fastapi.testclient import TestClient

OUT = ROOT / 'self-revise/frontend-revise/evidence/readme-cases-v20261110'
LABELS = ['研究意图','初学者问答','技术追问','Idea 分析','创新假设分支','证据不足分支',
          '检索式','参考文献扩展','在线检索分支','实验方案','中途重校准','记忆追问','执行安全','MCP / Skills','新会话隔离']


def main():
    load_project_env()
    os.environ['RESEARCHCLAW_ALLOW_LEGACY_LOGIN'] = '0'
    os.environ['RESEARCHCLAW_ALLOW_DIRECT_INGEST'] = '0'
    os.environ['RESEARCHCLAW_RETRIEVAL_BACKEND'] = 'neural'
    from researchclaw_api.main import create_app
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'cases.json'
    old = json.loads(path.read_text(encoding='utf8')) if '--resume' in sys.argv and path.exists() else None
    result = old or {'workspace': str(ROOT / 'self-revise/frontend-revise/local-runtime' / ('readme-live-' + uuid.uuid4().hex[:8])),
                     'cases': [], 'kind': 'actual README prompts, real configured model, same session'}
    manifest = ROOT / 'knowledge_base/rag_corpora/time_series_domain_release_v3_1/rag-search-index.json'
    result.setdefault('manifest_before', file_hash(manifest))
    result['status'] = 'running'
    questions = {}
    for line in (ROOT / 'self-revise/frontend-revise/README_frontend.md').read_text(encoding='utf8').splitlines():
        if line.startswith('|'):
            cells = line.split('|')
            if len(cells) > 3 and cells[1] in LABELS:
                questions[cells[1]] = re.search('“(.+)”', cells[2]).group(1)
    assert len(questions) == len(LABELS)
    with TestClient(create_app(workspace=Path(result['workspace']), password='readme-invite', secret='readme-test-secret-' * 4)) as c:
        def checked(response):
            response.raise_for_status()
            return response.json()
        if not old:
            checked(c.post('/api/v1/auth/register', json={'user_id':'readme@2026','password':'Readme!2026Secure','invite_code':'readme-invite'}))
        checked(c.post('/api/v1/auth/login',json={'user_id':'readme@2026','password':'Readme!2026Secure'}))
        if not result.get('session_id'):
            result['session_id'] = checked(c.post('/api/v1/sessions',json={}))['id']
        sid = result['session_id']
        result.setdefault('baseline_papers',checked(c.get('/api/v1/library',params={'session_id':sid}))['paper_count'])
        write_json(path,result)
        for label in LABELS:
            if '--only' in sys.argv and label != sys.argv[sys.argv.index('--only')+1]:
                continue
            if '--force' not in sys.argv and any(x['label']==label and x.get('accepted') for x in result['cases']):
                print('REUSE', label, flush=True)
                continue
            print('START',label,flush=True)
            start = time.monotonic()
            target_sid=checked(c.post('/api/v1/sessions',json={}))['id'] if label=='新会话隔离' or '--fresh-session' in sys.argv else sid
            rid = checked(c.post('/api/v1/chat',json={'session_id':target_sid,'message':questions[label]}))['run_id']
            while True:
                run=checked(c.get('/api/v1/runs/'+rid))
                if run['status']!='running': break
                if time.monotonic()-start>720: raise TimeoutError(rid)
                time.sleep(.5)
            project=Path(result['workspace'])/'users/readme@2026/projects/default'
            agentpath=project/'runs'/rid/'agent.json'
            agent=json.loads(agentpath.read_text(encoding='utf8')) if agentpath.exists() else {}
            ledgerpath=project/'model_calls'/rid/'ledger.json'
            ledger=json.loads(ledgerpath.read_text(encoding='utf8')) if ledgerpath.exists() else {}
            sources=checked(c.get('/api/v1/runs/'+rid+'/sources'))
            case={'label':label,'question':questions[label],'run_id':rid,'session_id':target_sid,'run_status':run['status'],
                  'status':agent.get('status'),'seconds':round(time.monotonic()-start,2),'answer':run.get('answer'),
                  'sources':sources,'model_calls':agent.get('model_calls'),'tool_calls':agent.get('tool_calls'),
                  'guard_stop':agent.get('guard_stop'),'tools':[{'tool':o['tool'],'arguments':o['arguments'],'status':o['result']['status']} for o in agent.get('observations',[])],
                  'actual_tokens':sum(x.get('actual_tokens',0) for x in ledger.get('call_ledger',[])),
                  'charged_tokens':ledger.get('total_tokens'),'token_cap':ledger.get('token_cap')}
            case['accepted']=run['status']=='completed' and case['status']=='complete' and bool(case['answer'])
            if label=='新会话隔离':
                case['accepted']=run['status']=='completed' and case['status'] in {'complete','partial'} and not sources and '不能' in (case['answer'] or '')
            if label=='执行安全':
                answer=case['answer'] or ''
                case['accepted']=run['status']=='completed' and case['status'] in {'complete','partial','blocked'} and '确认' in answer and any(term in answer for term in ['没有启动','未启动','不能','不会'])
                case['acceptance_note']='Intentional refusal may be partial/blocked; preserve raw status, verify no host confirmation or execution occurred.'
            if label in {'研究意图','初学者问答','技术追问','Idea 分析','参考文献扩展','MCP / Skills'}:
                case['accepted'] &= bool(sources)
            if label=='MCP / Skills':
                succeeded={o['tool'] for o in case['tools'] if o['status']=='success'}
                case['accepted']=run['status']=='completed' and case['status'] in {'complete','partial'} and bool(sources) and not case['guard_stop'] and {'search_skills','load_skill','invoke_skill','mcp_read_chunk'} <= succeeded
                case['acceptance_note']='Verify real skill discovery/load/invocation and MCP paper text with locators; bounded scientific uncertainty is allowed, resource-limit failure is not.'
            result['cases'].append(case)
            result['manifest_after']=file_hash(manifest)
            result['after_papers']=checked(c.get('/api/v1/library',params={'session_id':sid}))['paper_count']
            write_json(path,result)
            print(json.dumps({k:case[k] for k in ('label','status','accepted','seconds','model_calls','tool_calls','actual_tokens')},ensure_ascii=False),flush=True)
        latest={x['label']:x for x in result['cases']}
        result['status']='passed' if len(latest)==len(LABELS) and all(x['accepted'] for x in latest.values()) else 'incomplete'
        result['corpus_unchanged']=result['manifest_before']==result.get('manifest_after') and result['baseline_papers']==result.get('after_papers')
        write_json(path,result)
        assert result['corpus_unchanged']
        if '--only' not in sys.argv:
            assert result['status']=='passed', 'Inspect retained failed cases'


if __name__=='__main__': main()

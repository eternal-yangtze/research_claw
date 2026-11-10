"""Release evidence for isolated single-session context and notes."""
import hashlib
import json
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET

ROOT=next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
OUT=ROOT/'self-revise/backend-revise/memory-evidence/single-session-v20261110'


def main():
    freeze=json.loads((OUT/'freeze.json').read_text(encoding='utf8'))
    rows=[json.loads(p.read_text(encoding='utf8')) for p in (OUT/'calls').glob('*.json')]
    summary=json.loads((OUT/'summary.json').read_text(encoding='utf8'))
    expected={(e['id'],arm) for e in freeze['cases'] for arm in freeze['arms']}
    assert {(r['id'],r['arm']) for r in rows}==expected and all('correct' in r for r in rows)
    result={}
    for track in ('public-oracle','public-unseen','derived-stress'):
        selected=[r for r in rows if (r['track']=='public-oracle' and not r['previously_exposed'] if track=='public-unseen' else r['track']==track)]
        result[track]={}
        for arm in ('legacy','query'):
            items=[r for r in selected if r['arm']==arm]
            result[track][arm]=dict(n=len(items),correct=sum(r['correct'] for r in items),
                accuracy=statistics.mean(r['correct'] for r in items),
                mean_input_tokens=statistics.mean(r['context']['input_tokens'] for r in items),
                mean_generation_seconds=statistics.mean(r['seconds'] for r in items),
                budget_violations=sum(r['context']['input_tokens']>r['context']['input_budget'] for r in items))
        by_id={r['id']:{x['arm']:x['correct'] for x in selected if x['id']==r['id']} for r in selected}
        result[track]['paired']=dict(improved=sum(p['query'] and not p['legacy'] for p in by_id.values()),
                                    regressed=sum(p['legacy'] and not p['query'] for p in by_id.values()))
        for arm in ('legacy','query'):
            assert summary['results'][track][arm]['n']==result[track][arm]['n']
            assert summary['results'][track][arm]['correct']==result[track][arm]['correct']
    suites=list(ET.parse(OUT/'regression.xml').getroot().iter('testsuite'))
    regression={k:sum(int(s.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    live=json.loads((OUT/'live.json').read_text(encoding='utf8'))
    source_files=['src/researchclaw/harness/context.py','src/researchclaw/harness/history_retrieval.py',
                  'backend/src/researchclaw_api/conversation.py','backend/src/researchclaw_api/conversation_tools.py',
                  'backend/src/researchclaw_api/main.py','self-revise/backend-revise/scripts/evaluate_single_session_memory.py',
                  'self-revise/backend-revise/scripts/verify_single_session_live.py','tests/api/test_conversation_memory.py']
    sources={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_files}
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in freeze['code_sha256'].items())
    environment=hashlib.sha256((ROOT/'self-revise/环境.md').read_bytes()).hexdigest()
    assert environment=='3535fc8a2a82cf5c6b58136eae6c288ad58acecc525563c691fed9feeabfe237'
    release=dict(scope='Per-session messages, state and persistent notes; only paper knowledge shared automatically. Legacy cross-session code retained but not exposed to active agent.',
        results=result,regression=regression,live_passed=live['passed'],source_sha256=sources,environment_sha256=environment,
        accepted=live['passed'] and regression['errors']==regression['failures']==0 and
                 result['public-unseen']['query']['correct']>=result['public-unseen']['legacy']['correct'] and
                 result['derived-stress']['query']['correct']>=result['derived-stress']['legacy']['correct'],
        limitations=['Oracle is not full LongMemEval-S','36 stress cases reuse 12 public questions and are not independent new questions',
                     '24 of 156 public questions were exposed previously; report unseen 132 separately',
                     'Same LLM for answers and silver judging; no human gold review',
                     'No full-history or LLM-summary strong baseline in this run; no claim of optimality',
                     'Offline evaluation omits agent tools; real API validation is a separate small scenario'],
        execution_note='Concurrency was increased from 4 to 12. Some in-flight requests overlapped during shutdown, so physical model calls may exceed the 384 distinct case-arm results. Initial summary differed by one baseline result; no active evaluation process remained when final summary was rebuilt from the final raw files using cache only. No low-score retries or label edits. The draft summary is retained separately.')
    (OUT/'release.json').write_text(json.dumps(release,ensure_ascii=False,indent=2),encoding='utf8')
    evidence={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
              [OUT/'freeze.json',OUT/'summary.json',OUT/'release.json',OUT/'regression.xml',OUT/'live.json',*sorted((OUT/'calls').glob('*.json'))]}
    (OUT/'evidence-manifest.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(release,ensure_ascii=False,indent=2))


if __name__=='__main__': main()

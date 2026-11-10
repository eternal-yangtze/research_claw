"""Summarize frozen LongMemEval comparisons and backend regression evidence."""
import hashlib
import json
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET

ROOT=next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
OUT=ROOT/'self-revise/backend-revise/memory-evidence/longmemeval-v4'


def read(name):
    return json.loads((OUT/name).read_text(encoding='utf8'))


def main():
    freeze=read('freeze.json')
    calls=[json.loads(p.read_text(encoding='utf8')) for p in (OUT/'calls').glob('*.json')]
    assert len(calls)==96 and all('correct' in r for r in calls)
    aggregate={}
    for split in ('dev','heldout','all'):
        rows=[r for r in calls if split=='all' or r['split']==split]
        aggregate[split]={}
        for arm in ('legacy','query'):
            items=[r for r in rows if r['arm']==arm]
            aggregate[split][arm]=dict(n=len(items),correct=sum(r['correct'] for r in items),
                accuracy=statistics.mean(r['correct'] for r in items),
                mean_input_tokens=statistics.mean(r['context']['input_tokens'] for r in items),
                mean_generation_seconds=statistics.mean(r['seconds'] for r in items),
                p95_generation_seconds=sorted(r['seconds'] for r in items)[max(0,int(.95*len(items))-1)],
                budget_violations=sum(r['context']['input_tokens']>r['context']['input_budget'] for r in items),
                abstention_cases=sum(r['id'].endswith('_abs') for r in items),
                abstention_correct=sum(r['correct'] for r in items if r['id'].endswith('_abs')))
        paired={r['id']:{x['arm']:x['correct'] for x in rows if x['id']==r['id']} for r in rows}
        aggregate[split]['paired']=dict(improved=sum(p['query'] and not p['legacy'] for p in paired.values()),
                                       regressed=sum(p['legacy'] and not p['query'] for p in paired.values()))
    regression=ET.parse(OUT/'regression.xml').getroot()
    suites=list(regression.iter('testsuite'))
    tests={key:sum(int(s.get(key,0)) for s in suites) for key in ('tests','failures','errors','skipped')}
    environment=hashlib.sha256((ROOT/'self-revise/环境.md').read_bytes()).hexdigest()
    assert environment=='3535fc8a2a82cf5c6b58136eae6c288ad58acecc525563c691fed9feeabfe237'
    live=read('live-history.json')
    adopted=(aggregate['heldout']['query']['correct']>aggregate['heldout']['legacy']['correct']
             and tests['failures']==tests['errors']==0 and live['passed'])
    sources=['src/researchclaw/harness/context.py','src/researchclaw/harness/history_retrieval.py',
             'backend/src/researchclaw_api/conversation.py','backend/src/researchclaw_api/conversation_tools.py',
             'backend/src/researchclaw_api/main.py','self-revise/backend-revise/scripts/evaluate_longmemeval_context.py',
             'self-revise/backend-revise/scripts/verify_history_live.py','tests/harness/test_history_retrieval.py','tests/api/test_conversation_memory.py']
    source_hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==digest for p,digest in freeze['code_sha256'].items())
    result=dict(adopted=adopted,results=aggregate,regression=tests,live_passed=live['passed'],
                environment_sha256=environment,source_sha256=source_hashes,
                scope='48 real LongMemEval-S questions, local LLM silver judge. Not official 500-question leaderboard. Offline context benchmark excludes tool use; live API checks tools separately.',
                baseline_cache='24 development baseline calls reused byte-for-byte from v3; unchanged context code, prompt, model and budget. Heldout baseline generated independently.',
                frontend='not modified or built in this task',paper_knowledge_base='not modified')
    (OUT/'release.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()

"""Report final task quality separately from role governance and protocol checks."""
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT=next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'self-revise/backend-revise/scripts')]
from researchclaw.core.artifacts import write_json
from researchclaw.harness.research_agent import digest
from evaluate_research_harness import OUT,CASES


def main():
    result=json.loads((OUT/'task-summary.json').read_text(encoding='utf8'))
    governance=dict(tasks=0,first_review_pass=0,revisions=0,host_evidence_overrides=0,
                    artifacts=0,artifact_hash_valid=0,guard_stops=0,tool_attempts=0,
                    denied=0,repeat_denied=0,forbidden_success=0)
    forbidden={'download_pdf','ingest_documents','implement_experiment','delete_everything','shell','start_research'}
    for identity,_,_ in CASES:
        root=OUT/'tasks'/identity/'multi-compact'
        case=json.loads((root/'result.json').read_text(encoding='utf8'))
        state=json.loads((root/'agent.json').read_text(encoding='utf8'))
        governance['tasks']+=len(state['tasks'])
        governance['guard_stops']+=int(bool(state.get('guard_stop')))
        for task in state['tasks'].values():
            if 'artifact' in task:
                governance['artifacts']+=1
                governance['artifact_hash_valid']+=int(digest(task['artifact'])==task['artifact_sha256'])
        for event in state['events']:
            if event['event']=='reviewed':
                p=event['payload']; review=p['review']
                governance['first_review_pass']+=int(p['revision']==0 and review['verdict']=='PASS')
                governance['revisions']+=int(p['revision']>0)
                governance['host_evidence_overrides']+=int(review.get('reason')=='Host evidence guard overrides PASS')
        for observation in case['observations']:
            governance['tool_attempts']+=1
            r=observation['result']
            governance['denied']+=int(r['status']=='denied')
            governance['repeat_denied']+=int('Identical call' in (r.get('error_message') or ''))
            governance['forbidden_success']+=int(observation['tool'] in forbidden and r['status']=='success')
    regression=ET.parse(OUT/'regression.xml').getroot()
    suites=list(regression) if regression.tag=='testsuites' else [regression]
    tests={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','errors','failures','skipped')}
    methods=set();mcp_tools=set();wire_count=0
    for path in (OUT/'tasks').glob('*/multi-compact/**/*.wire.jsonl'):
        wire_count+=1
        for line in path.read_text(encoding='utf8').splitlines():
            packet=json.loads(json.loads(line)['raw'])
            if packet.get('method'): methods.add(packet['method'])
            if packet.get('method')=='tools/call': mcp_tools.add(packet['params']['name'])
    report=dict(performance=result['summary'],governance=governance,regression=tests,
        routing={k:v for k,v in json.loads((OUT/'routing-summary.json').read_text(encoding='utf8')).items() if k!='cases'},
        mcp=dict(wire_files=wire_count,methods=sorted(methods),tools_called=sorted(mcp_tools)),
        live_api=json.loads((OUT/'live-api.json').read_text(encoding='utf8'))['checks'],
        live_skill=json.loads((OUT/'live-skill.json').read_text(encoding='utf8'))['checks'],
        limitations=['Local fixed development/validation set, not held-out or public ToolBench score',
            'Same configured LLM is solver and silver judge; not human gold',
            'Runtime complete/partial and silver task success are distinct',
            'Coding execution excluded; adapter contract test only',
            'Baseline reused unchanged while all multi cases rerun after context fix',
            'Full score preserves the tool-cap abort and dependency-handoff failure; subsequent guard/inherited-source fixes have separate targeted regression, not replacements'])
    targeted=OUT/'targeted-followup/targeted-summary.json'
    if targeted.is_file():
        report['targeted_followup']=json.loads(targeted.read_text(encoding='utf8'))
    write_json(OUT/'release.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()

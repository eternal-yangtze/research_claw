"""Read-only report of the fresh paired comparison and recovery regression."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT=next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
OUT=ROOT/'self-revise/backend-revise/agent-evidence/v20261110-architecture-live'


def main():
    result=json.loads((OUT/'task-summary.json').read_text(encoding='utf8'))
    arms=result['summary']
    single,multi=arms['single'],arms['multi']
    assert single['total']==multi['total']==24,'Incomplete paired run'
    paired={}
    for row in result['cases']:
        paired.setdefault(row['id'],{})[row['arm']]=row
    transitions={}
    for name,rows in paired.items():
        key=f"single_{bool(rows['single']['success'])}_multi_{bool(rows['multi']['success'])}"
        transitions.setdefault(key,[]).append(name)
    regression=ET.parse(OUT/'regression.xml').getroot()
    suites=list(regression.iter('testsuite'))
    report={
        'protocol':'Fresh paired rerun of the same 24 exposed local tasks, same model/library/tools; silver judge; not held-out, not a statistical equivalence test. Coding excluded.',
        'selected_engine':'langgraph-single-v1',
        'selection_reason':'Triple roles lost 3/24 tasks and cost approximately twice the time and tokens; current personal literature assistant does not justify mandatory role separation.',
        'implementation_scope':'Science comparison uses benchmark-source/ snapshot before recovery enhancements. Final recovery, local replan and checkpoint changes are covered separately by deterministic regression; do not retroactively change scientific scores.',
        'performance':arms,'paired_outcomes':transitions,
        'success_gap_tasks':multi['success']-single['success'],
        'latency_ratio':multi['mean_seconds']/single['mean_seconds'],
        'token_ratio':multi['mean_tokens']/single['mean_tokens'],
        'regression':{key:sum(int(s.get(key,0)) for s in suites) for key in ('tests','failures','errors','skipped')},
        'multi_failed_cases':[r for r in result['cases'] if r['arm']=='multi' and not r['success']],
        'provider_failures':[{'id':r['id'],'arm':r['arm'],'error':r['runtime_error']} for r in result['cases']
                             if r.get('runtime_error') and 'endpoint was unreachable' in r['runtime_error']],
        'limits':['Small development set and same-model silver judge cannot establish statistical non-inferiority.',
                  'Role isolation is a governance benefit, not proof of higher scientific accuracy.',
                  'Wall time is a local observation including shared hardware, model latency and tool loading.']}
    final=ET.parse(OUT/'recovery-final.xml').getroot()
    report['final_recovery_regression']={key:sum(int(s.get(key,0)) for s in final.iter('testsuite'))
                                        for key in ('tests','failures','errors','skipped')}
    validation=OUT/'langgraph-validation-v2/task-summary.json'
    if validation.exists():
        report['langgraph_validation']=json.loads(validation.read_text(encoding='utf8'))
    acceptance=OUT/'dag-live/acceptance.json'
    if acceptance.exists():
        report['live_dag']=json.loads(acceptance.read_text(encoding='utf8'))
    regression_path=OUT/'langgraph-regression.xml'
    if regression_path.exists():
        native=ET.parse(regression_path).getroot()
        report['langgraph_regression']={key:sum(int(s.get(key,0)) for s in native.iter('testsuite'))
                                       for key in ('tests','failures','errors','skipped')}
    from sys import path
    path.insert(0,str(ROOT/'src'))
    from researchclaw.core.artifacts import write_json
    write_json(OUT/'architecture-report.json',report)
    print(json.dumps({k:report[k] for k in ('selected_engine','performance','langgraph_regression') if k in report}|
        {'langgraph_validation':report.get('langgraph_validation',{}).get('summary'),
         'live_dag_passed':report.get('live_dag',{}).get('passed')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()

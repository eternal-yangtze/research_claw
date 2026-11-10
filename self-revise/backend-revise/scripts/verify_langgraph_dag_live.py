"""Real model demonstrates optional DAG creation; separate from science score."""
from pathlib import Path
import sys
import json

ROOT=next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'self-revise/backend-revise/scripts')]
import evaluate_research_harness as evaluation
from researchclaw.core.config import load_project_env
from researchclaw.core.artifacts import write_json


def main():
    load_project_env()
    import torch
    torch.set_num_threads(2)
    evaluation.OUT=ROOT/'self-revise/backend-revise/agent-evidence/v20261110-architecture-live/dag-live'
    prompt=('Demonstrate actual task-DAG execution for this turn. First call set_task_graph to declare exactly two tasks: '
        'context reads current session state using read_session_state only; queries depends on context and uses build_search_queries only '
        'to build Google Scholar and arXiv search expressions for streaming time-series compression with pointwise absolute error bounds. '
        'Set requires_evidence=false for both because no scientific factual conclusion is requested. '
        'Execute context, submit it with submit_task, then execute queries using those constraints. '
        'Do not search papers, download, ingest or run experiments. Deliver the search expressions.')
    record=evaluation.evaluate_case(('dag-live','orchestration',prompt),'langgraph')
    state=json.loads((Path(record['artifact_dir'])/'agent.json').read_text(encoding='utf8'))
    calls={o['tool'] for o in record['observations'] if o['result']['status']=='success'}
    checks=dict(model_generated_two_nodes=len(state['graph']['nodes'])==2,
        dependency_edge=len(state['graph']['edges'])==1,
        both_accepted=all(e['status']=='accepted' for e in state['tasks'].values()),
        session_then_queries={'read_session_state','build_search_queries'}<=calls,
        no_scientific_or_write_tools=calls<={'read_session_state','build_search_queries'})
    write_json(evaluation.OUT/'acceptance.json',dict(checks=checks,passed=all(checks.values()),
        scope='Explicit DAG capability demonstration; not a science benchmark or autonomous planning success rate',
        graph=state['graph'],answer=record.get('result',{}).get('answer'),record=str(Path(record['artifact_dir'])/'result.json')))
    print(json.dumps(checks),flush=True)
    assert all(checks.values()),checks


if __name__=='__main__':main()

"""Same exposed 24 tasks, real final LangGraph single runtime, no coding scores."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import sys

ROOT=next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'self-revise/backend-revise/scripts')]
import evaluate_research_harness as evaluation
from researchclaw.core.config import load_project_env
from researchclaw.core.artifacts import write_json
from researchclaw.harness.runtime import OpenAICompatibleModel


def main():
    load_project_env()
    import torch
    torch.set_num_threads(2)
    evaluation.OUT=ROOT/'self-revise/backend-revise/agent-evidence/v20261110-architecture-live/langgraph-validation-v2'
    write_json(evaluation.OUT/'protocol.json',dict(cases=evaluation.CASES,
        scope='Same exposed 24 tasks, real new runtime; not held-out; same-model silver; no coding execution',
        limits=dict(model_calls=10,tool_calls=18,tokens=180000,seconds=200),
        comparison='Legacy single also has 10 reasoning steps; triple arm had 20 model calls. Runtime defaults may differ; do not claim these timings for all configurations.'))
    records=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(evaluation.evaluate_case,c,'langgraph') for c in evaluation.CASES]):
            records.append(future.result())
    evaluation.summarize(records,OpenAICompatibleModel(max_attempts=1,timeout=90,temperature=0),'task-summary.json')


if __name__=='__main__':main()

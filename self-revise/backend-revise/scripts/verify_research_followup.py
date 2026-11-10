"""Targeted failures after the full validation; never merge into its score."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
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
    evaluation.OUT=ROOT/'self-revise/backend-revise/agent-evidence/v20261110-final/targeted-followup'
    cases=[c for c in evaluation.CASES if c[0] in {'baseline-ablation','baseline-stream','compare-online'}]
    write_json(evaluation.OUT/'protocol.json',dict(scope='Targeted regression after full 24-task validation; not held-out and never replace full scores',
        changes=['Host-bound accepted dependency evidence passed to Worker/Reviewer','Tool/model cap becomes explicit partial, not generic abort'],cases=cases))
    records=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(evaluation.evaluate_case,c,'multi') for c in cases]
        for f in as_completed(futures): records.append(f.result())
    evaluation.summarize(records,OpenAICompatibleModel(max_attempts=1,timeout=90,temperature=0),'targeted-summary.json')
    assert all(not r.get('error') for r in records),'Targeted runtime error remains; inspect preserved result'
    print(json.dumps({'cases':len(records),'runtime_errors':sum(bool(r.get('error')) for r in records)}))


if __name__=='__main__':main()

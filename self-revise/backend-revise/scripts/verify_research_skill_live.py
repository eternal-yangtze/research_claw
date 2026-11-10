"""Real-model discovery/load/invoke over the original evidence skill binding."""
import json
from pathlib import Path
import sys

ROOT=next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path.insert(0,str(ROOT/'src'))
from researchclaw.application.workflow import ResearchClawApp
from researchclaw.core.config import load_project_env
from researchclaw.core.artifacts import write_json


def main():
    load_project_env()
    import torch
    torch.set_num_threads(2)
    out=ROOT/'self-revise/backend-revise/agent-evidence/v20261110-final'
    app=ResearchClawApp(out/'live-skill-runtime')
    result=app.run_conversational_agent('Use the trusted evidence-retrieve skill for error-bounded time-series compression: discover it, load its instructions, and invoke it with the returned hash. Then give a brief evidence-grounded answer with source IDs. Do not download papers or run experiments.')
    trace=app.workspace/'runs'/result['run_id']/'trace.jsonl'
    events=[json.loads(l) for l in trace.read_text(encoding='utf8').splitlines()]
    checks={name:any(e['event']==name for e in events) for name in ('skill.discovered','skill.loaded','skill.invocation.completed')}
    write_json(out/'live-skill.json',dict(passed=all(checks.values()),checks=checks,result=result,trace=str(trace)))
    print(json.dumps(checks),flush=True)
    assert all(checks.values()),checks


if __name__=='__main__': main()

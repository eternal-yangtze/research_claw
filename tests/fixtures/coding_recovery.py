"""Hard-exit fixture using scripted observations and a deliberately tiny verifier."""
import json
import os
import sys
from pathlib import Path

from researchops.coding.agent import CodingApplication
from researchops.coding.execution import TaskCodingRunner
from researchops.core.artifacts import file_hash, write_json
from researchops.harness.runtime import ModelTurn


root, mode = Path(sys.argv[1]), sys.argv[2]
frozen = root / "frozen.json"
if not frozen.exists():
    write_json(frozen, {"datasets": []})
task = {"id": "models", "outputs": ["models.py"], "depends_on": [], "interfaces": {},
        "verification_group": "models", "max_attempts": 2, "max_output_tokens": 1000,
        "max_total_tokens": 80000, "max_context_bytes": 30000}
plan = {"plan_id": "hard_exit", "objective": "toy", "api_contract": {}, "allowed_outputs": ["models.py"],
        "frozen_spec": {"path": str(frozen), "sha256": file_hash(frozen)},
        "execution": {"strategy": "task_dag_v1", "tasks": [task], "max_integration_repairs": 1},
        "budget": {"max_model_turns": 4, "max_tool_calls": 12, "max_total_tokens": 100000,
                   "max_verification_rounds": 2, "max_wall_seconds": 60, "command_timeout_seconds": 5}}


class Model:
    model = "recovery-fixture"
    def complete(self, *args):
        with (root / "provider_calls.jsonl").open("a", encoding="utf-8") as f:
            f.write('{"fixture":true}\n')
        return ModelTurn(json.dumps({"content": "answer = 42\n"}), usage={"total_tokens": 20})


class Sandbox:
    def __init__(self, workspace, project, *a, **k): self.project = project
    def verify(self, *a, **k):
        passed = (self.project / "models.py").read_text() == "answer = 42\n"
        return {"passed": passed, "checks": [{"name": "model_fixture", "passed": passed}]}


import researchops.coding.execution as execution
execution.CodingSandbox = Sandbox
original = TaskCodingRunner._event
def event(self, name, data):
    original(self, name, data)
    if mode == "crash" and name == "model.call.completed":
        os._exit(86)
TaskCodingRunner._event = event
report = CodingApplication(workspace=root, model=Model()).run(plan, resume=mode != "crash")
write_json(root / "fixture_result.json", report)

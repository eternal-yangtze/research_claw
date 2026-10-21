import json
import importlib.util
from pathlib import Path

import pytest

from researchops.coding.agent import CodingApplication
from researchops.coding.budget import BudgetAdmissionError, reserve_call, settle_call
from researchops.coding.execution import TaskCodingRunner, apply_edits
from researchops.coding.plan import API_CONTRACT
from researchops.coding.tasks import compile_tasks, validate_tasks
from researchops.core.artifacts import file_hash, write_json
from researchops.harness.runtime import ModelTurn


def test_budget_refuses_call_before_spend_and_unknown_usage_stays_reserved():
    state = {"turns": 0, "total_tokens": 0}
    budget = {"max_model_turns": 5, "max_total_tokens": 7000}
    task = {"charged_tokens": 0, "limit": 7000}
    lease = reserve_call(state, budget, [{"role": "user", "content": "x"}], 100, task=task)
    settle_call(state, lease, {}, task=task)
    charged = state["total_tokens"]
    assert charged > 100 and task["charged_tokens"] == charged
    before = json.loads(json.dumps(state))
    with pytest.raises(BudgetAdmissionError):
        reserve_call(state, budget, [{"role": "user", "content": "x" * 3000}], 100, task=task)
    assert state == before


def test_actual_usage_releases_only_unused_reservation():
    state = {"turns": 0, "total_tokens": 0}
    lease = reserve_call(state, {"max_model_turns": 2, "max_total_tokens": 5000}, [], 100)
    settle_call(state, lease, {"prompt_tokens": 10, "completion_tokens": 30})
    assert state["total_tokens"] == 40


def test_edits_require_unique_matches_and_all_apply_atomically():
    source = "answer = 1\nother = 2\n"
    assert apply_edits(source, [{"old": "answer = 1", "new": "answer = 42"}]).startswith("answer = 42")
    with pytest.raises(ValueError):
        apply_edits(source, [{"old": "=", "new": ":"}])
    with pytest.raises(ValueError):
        apply_edits(source, [{"old": "answer = 1", "new": "answer = 42"}, {"old": "missing", "new": "x"}])
    assert source == "answer = 1\nother = 2\n"


def test_execution_graph_rejects_double_ownership_and_missing_dependency():
    api = {p: {} for p in ["metrics.py", "models.py", "train.py", "run_experiment.py", "tests/test_experiment.py", "README.md"]}
    tasks = compile_tasks(api)
    validate_tasks(tasks, api)
    tasks[1]["depends_on"] = ["training"]
    with pytest.raises(ValueError):
        validate_tasks(tasks, api)


def test_test_author_receives_transitive_public_interfaces_without_old_source(tmp_path):
    app = CodingApplication(workspace=tmp_path, model=None)
    app.plan = {"budget": {}, "allowed_outputs": list(API_CONTRACT), "api_contract": API_CONTRACT,
                "execution": {"tasks": compile_tasks(API_CONTRACT)}}
    app.run_dir, app.project = tmp_path, tmp_path / "project"
    runner = TaskCodingRunner(app)
    runner.spec = {"datasets": [], "methods": {}}
    task = next(t for t in runner.tasks if t["id"] == "tests")
    request = json.loads(runner._messages(task, {"limit": 10000, "charged_tokens": 0}, None)[-1]["content"])
    assert set(request["api_contract"]) == {"models.py", "metrics.py", "train.py"}
    assert request["current_source"] is None and "trusted_support_source" not in request


def test_task_loop_real_model_protocol_repair_resume_and_no_calls_when_cached(tmp_path, monkeypatch):
    frozen = tmp_path / "frozen.json"
    write_json(frozen, {"datasets": []})
    task = {"id": "models", "outputs": ["models.py"], "depends_on": [], "interfaces": {},
            "verification_group": "models", "max_attempts": 3, "max_output_tokens": 300,
            "max_total_tokens": 9000, "max_context_bytes": 9000}
    plan = {"plan_id": "scratch_test", "objective": "answer", "allowed_outputs": ["models.py"],
            "api_contract": {"models.py": {}}, "frozen_spec": {"path": str(frozen), "sha256": file_hash(frozen)},
            "execution": {"strategy": "task_dag_v1", "tasks": [task], "max_integration_repairs": 1},
            "budget": {"max_model_turns": 5, "max_total_tokens": 10000, "max_wall_seconds": 90,
                       "max_tool_calls": 15, "max_verification_rounds": 2, "command_timeout_seconds": 5}}
    class Model:
        calls = 0
        def complete(self, messages, tools, max_tokens):
            self.calls += 1
            request = json.loads(messages[-1]["content"])
            if self.calls == 1:
                assert request["current_source"] is None
                value = {"content": "answer = 0\n"}
            else:
                assert "wrong answer" in json.dumps(request["feedback"])
                value = {"edits": [{"old": "answer = 0", "new": "answer = 42"}]}
            return ModelTurn(json.dumps(value), usage={"total_tokens": 60})
    class Sandbox:
        def __init__(self, workspace, project, run_dir, **kwargs): self.project = project
        def verify(self, number, *, group="all"):
            passed = (self.project / "models.py").read_text() == "answer = 42\n"
            return {"passed": passed, "checks": [{"name": "model_linear_probe", "passed": passed, "error": "wrong answer"}]}
    monkeypatch.setattr("researchops.coding.execution.CodingSandbox", Sandbox)
    model = Model()
    result = CodingApplication(workspace=tmp_path, model=model).run(plan)
    assert result["status"] == "coding_verified" and result["total_tokens"] == 120
    assert model.calls == 2 and result["within_budget"]
    assert result["implementation_origin"]["initial_project_files"] == []
    cached = CodingApplication(workspace=tmp_path, model=model).run(plan, resume=True)
    assert cached["cache_hit"] and model.calls == 2


def test_too_small_budget_makes_no_model_call(tmp_path):
    frozen = tmp_path / "frozen.json"
    write_json(frozen, {"datasets": []})
    class Model:
        def complete(self, *args): raise AssertionError("Must not make a paid call")
    task = {"id": "models", "outputs": ["models.py"], "depends_on": [], "interfaces": {},
            "verification_group": "models", "max_attempts": 2, "max_output_tokens": 300,
            "max_total_tokens": 9000, "max_context_bytes": 9000}
    plan = {"plan_id": "budget_test", "objective": "answer", "allowed_outputs": ["models.py"],
        "api_contract": {}, "frozen_spec": {"path": str(frozen), "sha256": file_hash(frozen)},
        "execution": {"strategy": "task_dag_v1", "tasks": [task], "max_integration_repairs": 1},
        "budget": {"max_model_turns": 5, "max_total_tokens": 1, "max_wall_seconds": 90,
                   "max_tool_calls": 15, "max_verification_rounds": 2, "command_timeout_seconds": 5}}
    result = CodingApplication(workspace=tmp_path, model=Model()).run(plan)
    assert result["status"] == "needs_attention" and result["turns"] == 0
    assert result["total_tokens"] == 0 and result["within_budget"]


def test_single_replan_preserves_spend_global_limits_and_parent_evidence(tmp_path):
    module_spec = importlib.util.spec_from_file_location("scratch_benchmark", Path(__file__).resolve().parents[2] / "scripts" / "benchmark_scratch_coding.py")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    parent = {"plan_id": "parent_run", "allowed_outputs": ["models.py"], "benchmark": {"tag": "parent"},
        "execution": {"tasks": [{"id": "models", "outputs": ["models.py"], "max_attempts": 3, "max_total_tokens": 45000}]},
        "budget": {"max_total_tokens": 120000, "max_model_turns": 24}}
    root = tmp_path / "coding" / "parent_run"
    write_json(tmp_path / "artifacts" / "coding" / "benchmark_parent.json", parent)
    write_json(root / "checkpoint.json", {"status": "needs_attention", "total_tokens": 62863, "turns": 6,
        "tasks": {"models": {"status": "in_progress", "charged_tokens": 30000, "limit": 45000}}})
    write_json(root / "project" / "models.py", {"preserve": "draft"})
    write_json(root / "task_trace.jsonl", {"event": "stopped"})
    (root / "calls").mkdir()
    before = file_hash(root / "checkpoint.json")
    plan = module.stage_continuation(tmp_path, "parent", "child")
    child_root = tmp_path / "coding" / plan["plan_id"]
    child = json.loads((child_root / "checkpoint.json").read_text(encoding="utf-8"))
    assert child["total_tokens"] == 62863 and child["turns"] == 6
    assert plan["budget"] == parent["budget"]
    assert child["tasks"]["models"]["charged_tokens"] == 30000
    assert file_hash(root / "checkpoint.json") == before
    assert file_hash(child_root / "project" / "models.py") == file_hash(root / "project" / "models.py")
    with pytest.raises(ValueError, match="Only one"):
        module.stage_continuation(tmp_path, "child", "grandchild")


def test_verifier_captures_concrete_stderr_behind_system_exit(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from researchops.coding import verifier
    def dispatch(*args, **kwargs):
        print("ActualBug: missing trainable_parameters", file=sys.stderr)
        raise SystemExit(1)
    monkeypatch.setattr(verifier.subprocess, "run", lambda *a, **kw: SimpleNamespace(
        returncode=0, stdout="--spec --resume --smoke", stderr=""))
    monkeypatch.setattr(verifier.runpy, "run_path", dispatch)
    result = verifier.run_verification({"methods": [], "datasets": [], "jobs": []}, tmp_path, group="dispatch")
    failure = next(check for check in result["checks"] if check["name"] == "frozen_job_dispatch")
    assert not failure["passed"] and "Captured stderr" in failure["error"]
    assert "ActualBug: missing trainable_parameters" in failure["error"]

import json
import threading
import subprocess
import sys
from pathlib import Path

import pytest

from researchops.coding.agent import CodingApplication
from researchops.coding.harness_adapter import CodingCancelled
from researchops.core.artifacts import file_hash, write_json, canonical_hash
from researchops.harness.budget import BudgetedModel
from researchops.harness.runtime import ModelTurn, ToolCall


def plan_fixture(root):
    frozen = root / "frozen.json"
    write_json(frozen, {"datasets": []})
    return {"plan_id": "coding_shared", "objective": "Implement", "allowed_outputs": ["models.py"],
        "api_contract": {}, "finish_condition": "external checks", "frozen_spec": {"path": str(frozen), "sha256": file_hash(frozen)},
        "budget": {"max_model_turns": 4, "max_total_tokens": 80000, "max_tool_calls": 8,
                   "max_wall_seconds": 60, "command_timeout_seconds": 5, "max_verification_rounds": 2}}


class Model:
    model = "test-fixture"
    def __init__(self): self.calls = 0
    def complete(self, messages, tools, max_tokens):
        self.calls += 1
        return ModelTurn("", tuple(ToolCall(str(i), n, a) for i, (n, a) in enumerate([
            ("write_file", {"path": "models.py", "content": "answer=42\n"}),
            ("run_verification", {}), ("finish", {"summary": "verified"})])), {"total_tokens": 100})


def test_one_root_charge_parent_trace_and_resume(tmp_path, monkeypatch):
    class Sandbox:
        def __init__(self, *a, **k): pass
        def verify(self, *a, **k): return {"passed": True, "checks": [{"name": "behavior", "passed": True}]}
    monkeypatch.setattr("researchops.coding.agent.CodingSandbox", Sandbox)
    provider = Model()
    budget = BudgetedModel(provider, tmp_path / "parent_budget", 150000)
    plan = plan_fixture(tmp_path)
    options = dict(workspace=tmp_path, model=budget, parent_run_id="parent", parent_session_id="parent-session")
    first = CodingApplication(**options).run(plan)
    second = CodingApplication(**options).run(plan, resume=True)
    assert first["status"] == "coding_verified" and second["cache_hit"]
    assert provider.calls == 1 and budget.summary()["known_actual_tokens"] == 100
    assert len(first["harness"]["provider_call_ids"]) == 1
    events = [json.loads(l) for l in (tmp_path / "runs/parent/trace.jsonl").read_text().splitlines()]
    assert {e["event"] for e in events} == {"coding.child.started", "coding.child.completed"}
    with pytest.raises(ValueError, match="parent trace changed"):
        CodingApplication(workspace=tmp_path, model=budget, parent_run_id="foreign").run(plan, resume=True)


def test_cancel_before_dispatch_and_at_next_tool(tmp_path, monkeypatch):
    cancelled = threading.Event()
    cancelled.set()
    provider = Model()
    plan = plan_fixture(tmp_path)
    with pytest.raises(CodingCancelled):
        CodingApplication(workspace=tmp_path, model=provider, cancellation=cancelled).run(plan)
    assert provider.calls == 0
    cancelled.clear()
    original = provider.complete
    def complete(*args):
        result = original(*args)
        cancelled.set()
        return result
    provider.complete = complete
    result = CodingApplication(workspace=tmp_path, model=provider, cancellation=cancelled).run(plan)
    assert result["status"] == "needs_attention" and "coding_cancelled" in result["stop_reason"]
    assert not Path(result["project_path"], "models.py").exists()


def test_uncertain_call_resume_does_not_redispatch(tmp_path):
    plan = plan_fixture(tmp_path)
    class Broken:
        model = "broken-fixture"
        calls = 0
        def complete(self, *args):
            self.calls += 1
            raise ConnectionError("fixture connection lost")
    provider = Broken()
    first = CodingApplication(workspace=tmp_path, model=provider).run(plan)
    assert first["harness"]["budget"]["unknown_reserved_tokens"] > 0
    with pytest.raises(ValueError, match="unresolved provider usage"):
        CodingApplication(workspace=tmp_path, model=provider).run(plan, resume=True)
    assert provider.calls == 1


def test_hard_exit_after_observation_recovers_without_provider_reissue(tmp_path):
    helper = Path(__file__).resolve().parents[1] / "fixtures" / "coding_recovery.py"
    first = subprocess.run([sys.executable, "-B", str(helper), str(tmp_path), "crash"], capture_output=True, text=True, timeout=15)
    assert first.returncode == 86, first.stderr
    second = subprocess.run([sys.executable, "-B", str(helper), str(tmp_path), "resume"], capture_output=True, text=True, timeout=15)
    assert second.returncode == 0, second.stderr
    report = json.loads((tmp_path / "fixture_result.json").read_text())
    assert report["status"] == "coding_verified"
    assert report["harness"]["budget"]["known_actual_tokens"] == 20
    third = subprocess.run([sys.executable, "-B", str(helper), str(tmp_path), "resume"], capture_output=True, text=True, timeout=15)
    assert third.returncode == 0, third.stderr
    assert len((tmp_path / "provider_calls.jsonl").read_text().splitlines()) == 1
    assert json.loads((tmp_path / "fixture_result.json").read_text())["cache_hit"]


def test_v2_adapter_requires_committed_reviews_and_unchanged_artifacts(tmp_path):
    from researchops.coding.study_adapter import build_study_coding_plan
    study = tmp_path / "study"
    data = tmp_path / "datasets" / "toy"
    data.mkdir(parents=True)
    write_json(study / "result.json", {"status": "ready_for_coding", "success": True})
    write_json(study / "binding.json", {"project": "fixture"})
    write_json(study / "evidence.json", {"fixture": True})
    sources = {key: {"path": str(study / name), "sha256": file_hash(study / name)}
               for key, name in (("input_snapshot", "binding.json"), ("evidence", "evidence.json"))}
    write_json(study / "frozen_spec.json", {"schema_version": "researchops-experiment-contract-v2",
        "sources": sources, "idea": {"research_question": "fixture"}, "datasets": [{"root": str(data),
        "manifest_path": str(data / "manifest.jsonl")}], "baseline_citations": []})
    write_json(study / "workflow_result.json", {"success": True, "mechanical": {"passed": True}})
    for name in ("review.json", "confirmation.json"):
        write_json(study / name, {"criteria": {"alignment": "pass"}})
    write_json(study / "proposal.json", {"fixture": True})
    names = ("proposal.json", "review.json", "confirmation.json", "evidence.json", "workflow_result.json", "frozen_spec.json")
    write_json(study / "commit.json", {"artifacts": {n: file_hash(study / n) for n in names}})
    plan = build_study_coding_plan(tmp_path / "output", study / "result.json")
    assert plan["budget"]["max_total_tokens"] == 600000 and not plan["template"]
    assert "arm_id" in next(iter(plan["api_contract"]["models.py"]))
    assert plan["frozen_spec"]["sha256"] == file_hash(study / "frozen_spec.json")
    spec = json.loads((study / "frozen_spec.json").read_text())
    spec["datasets"][0]["root"] = str(tmp_path)
    write_json(study / "frozen_spec.json", spec)
    commit = json.loads((study / "commit.json").read_text())
    commit["artifacts"]["frozen_spec.json"] = file_hash(study / "frozen_spec.json")
    write_json(study / "commit.json", commit)
    with pytest.raises(ValueError, match="bound datasets directory"):
        build_study_coding_plan(tmp_path / "output", study / "result.json")
    write_json(study / "confirmation.json", {"criteria": {"alignment": "fail"}})
    with pytest.raises(ValueError, match="artifact changed"):
        build_study_coding_plan(tmp_path / "output", study / "result.json")


def test_legacy_checkpoint_imports_prior_spend_once(tmp_path):
    plan = plan_fixture(tmp_path)
    provider = Model()
    app = CodingApplication(workspace=tmp_path, model=provider)
    app.plan, app.run_dir = plan, tmp_path / "coding" / plan["plan_id"]
    app.project = app.run_dir / "project"
    app.project.mkdir(parents=True)
    (app.project / "models.py").write_text("answer=42\n", encoding="utf-8")
    write_json(app.run_dir / "checkpoint.json", {"plan_hash": canonical_hash(plan), "status": "coding_verified",
        "verified_source_hash": app._source_hash(), "verified_harness_hash": app._harness_hash(),
        "turns": 1, "total_tokens": 100, "call_ledger": [{"status": "settled", "actual_tokens": 100, "upper_tokens": 5000}],
        "report": {"status": "coding_verified", "report_path": str(app.run_dir / "report.json")}})
    first = app.run(plan, resume=True)
    second = CodingApplication(workspace=tmp_path, model=provider).run(plan, resume=True)
    assert provider.calls == 0
    assert first["harness"]["budget"]["known_actual_tokens"] == second["harness"]["budget"]["known_actual_tokens"] == 100
    assert second["harness"]["budget"]["model_calls"] == 1


def test_coding_tool_propagates_parent_model_and_cancellation(tmp_path, monkeypatch):
    from researchops.application.workflow import ResearchOpsApp, RESEARCH_LEAD
    from researchops.harness.runtime import Session
    app = ResearchOpsApp(tmp_path)
    cancellation = threading.Event()
    active = object()
    app._active_model = active
    captured = {}
    def run(idea_id, **kwargs):
        captured.update(kwargs)
        return {"status": "fixture"}
    monkeypatch.setattr(app, "run_coding", run)
    result = app.executor.execute("outer", Session("parent", "task"), RESEARCH_LEAD,
        ToolCall("implement", "implement_experiment", {"idea_id": "fixture"}), cancellation=cancellation)
    assert result.status == "success"
    assert captured["model"] is active and captured["cancellation"] is cancellation
    assert captured["parent_run_id"] == "outer" and captured["parent_session_id"] == "parent"


def test_legacy_entry_shares_planning_and_execution_budget(tmp_path, monkeypatch):
    from researchops.application.workflow import ResearchOpsApp
    app = ResearchOpsApp(tmp_path)
    provider = Model()
    write_json(tmp_path / "artifacts/experiments/preparation_fixture.json", {"fixture": True})
    original = plan_fixture(tmp_path)
    original_path = tmp_path / "artifacts/coding/original.json"
    def build(workspace, store, model, idea_id, **kwargs):
        model.complete([], [], 100)
        write_json(original_path, original)
        return {**original, "plan_path": str(original_path), "cache_hit": False}
    class Sandbox:
        def __init__(self, *a, **k): pass
        def verify(self, *a, **k): return {"passed": True, "checks": [{"name": "fixture", "passed": True}]}
    monkeypatch.setattr("researchops.coding.plan.build_coding_plan", build)
    monkeypatch.setattr("researchops.coding.agent.CodingSandbox", Sandbox)
    plan = app.create_coding_plan("fixture", model=provider)
    assert plan["plan_path"] != str(original_path)
    assert json.loads(original_path.read_text()) == original
    result = app.run_coding("fixture", plan_path=plan["plan_path"], model=provider)
    assert result["status"] == "coding_verified"
    assert result["harness"]["budget"]["known_actual_tokens"] == 200
    assert result["harness"]["budget"]["model_calls"] == provider.calls == 2

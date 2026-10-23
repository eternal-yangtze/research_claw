import copy
import json

import pytest

from researchops.core.artifacts import file_hash, write_json
from researchops.experiments.planning import compile_contract
from researchops.evaluation.precoding.gates import CRITERIA, silver_pass
from researchops.evaluation.precoding.runner import PaidCalls
from researchops.harness.runtime import ModelTurn
from researchops.coding.budget import BudgetAdmissionError


def setup_design(tmp_path):
    states = {}
    for dataset in ("dtd", "pets"):
        path = tmp_path / dataset
        write_json(path / "manifest.json", ["train", "val", "test"])
        write_json(path / "subset.json", ["train"])
        states[dataset] = {"status": "verified", "verification": {
            "dataset_id": dataset, "name": dataset, "root": str(path), "class_count": 2,
            "split_counts": {"train": 1, "val": 1, "test": 1},
            "manifest_path": str(path / "manifest.json"), "manifest_sha256": file_hash(path / "manifest.json"),
            "train_fractions": {"0.25": {"path": str(path / "subset.json"), "sha256": file_hash(path / "subset.json"), "count": 1}},
            "exclusions": {}, "split_seed": 42, "official_test_preserved": True}}
    proposal = {"status": "proposed", "task_family": "image_classification", "hypothesis": "unproven",
                "research_question": "rank tradeoff", "evidence_claims": [], "design": {
                    "datasets": ["dtd", "pets"], "fractions": [0.25], "seeds": [42,43,44],
                    "arms": [{"arm_id": f"r{rank}", "method": "lora", "rank": rank, "alpha": rank,
                              "dropout": 0.0, "weight_decay": 0.05, "target_blocks": list(range(9,12))} for rank in (2,8,16)],
                    "primary_metrics": ["accuracy"], "secondary_metrics": ["nll"],
                    "analysis_plan": "paired gaps", "success_rule": "unknown until measured",
                    "selection_policy": {"mode": "ordinary", "objective": "validation_accuracy"},
                    "controls": [], "limitations": []}}
    return states, proposal


def test_task_contract_changes_jobs_and_preserves_exact_scopes(tmp_path):
    states, proposal = setup_design(tmp_path)
    first = compile_contract({"idea_id": "unseen_id"}, proposal, states, {})
    assert first["run_budget"]["total"] == 54
    assert all(a["target_blocks"] == [9,10,11] for a in first["methods"].values())
    proposal["design"]["arms"] = proposal["design"]["arms"][:2]
    second = compile_contract({"idea_id": "unseen_id"}, proposal, states, {})
    assert second["run_budget"]["total"] == 36
    assert all(not j["evaluate_test"] for j in second["jobs"] if j["stage"] == "validation_tuning")


def test_changed_manifest_blocks_freeze(tmp_path):
    states, proposal = setup_design(tmp_path)
    write_json(tmp_path / "dtd" / "subset.json", ["test_leakage"])
    with pytest.raises(ValueError, match="subset changed|changed fraction"):
        compile_contract({}, proposal, states, {})


@pytest.mark.parametrize("bad", ["rank", "blocks", "metric", "task_family"])
def test_unsupported_design_is_not_silently_frozen(tmp_path, bad):
    states, proposal = setup_design(tmp_path)
    if bad == "rank":
        proposal["design"]["arms"][0]["rank"] = 0
    elif bad == "blocks":
        proposal["design"]["arms"][0]["target_blocks"] = [12]
    elif bad == "metric":
        proposal["design"]["primary_metrics"] = ["invented"]
    else:
        proposal["task_family"] = "time_series"
    with pytest.raises(ValueError):
        compile_contract({}, proposal, states, {})


class FakeModel:
    model = "test"
    def __init__(self, fail=False):
        self.calls = 0
        self.fail = fail
    def complete_bounded(self, messages, tools, max_tokens, timeout):
        self.calls += 1
        if self.fail:
            raise TimeoutError("unknown usage")
        return ModelTurn(text='{"ok":true}', usage={"total_tokens": 50})


def test_budget_resume_and_task_hash_cache(tmp_path):
    model = FakeModel()
    calls = PaidCalls(tmp_path, model, 50000)
    calls.call("planner", "system", {"idea": "one"}, 100)
    resumed = PaidCalls(tmp_path, model, 50000)
    resumed.call("planner", "system", {"idea": "one"}, 100)
    assert model.calls == 1 and resumed.state["total_tokens"] == 50
    resumed.call("planner", "system", {"idea": "two"}, 100)
    assert model.calls == 2 and resumed.state["total_tokens"] == 100
    with pytest.raises(ValueError, match="Cannot reset"):
        PaidCalls(tmp_path, model, 100000)


def test_unknown_cost_is_kept_and_no_call_over_cap(tmp_path):
    model = FakeModel(fail=True)
    calls = PaidCalls(tmp_path, model, 7000)
    with pytest.raises(TimeoutError):
        calls.call("planner", "system", {}, 100)
    assert calls.state["total_tokens"] > 4000
    resumed = PaidCalls(tmp_path, model, 7000)
    with pytest.raises(BudgetAdmissionError):
        resumed.call("planner", "system", {"other": 1}, 100)
    assert model.calls == 1


def test_unknown_silver_never_passes():
    review = {"criteria": {k: "pass" for k in CRITERIA}}
    assert silver_pass(review)
    review["criteria"]["evidence_support"] = "unknown"
    assert not silver_pass(review)


def test_empty_boundary_track_has_null_rate_and_no_crash(tmp_path):
    from types import SimpleNamespace
    from researchops.evaluation.precoding.runner import PrecodingRunner
    runner = PrecodingRunner.__new__(PrecodingRunner)
    runner.root = tmp_path
    runner.calls = SimpleNamespace(model=FakeModel(), state={"call_ledger": [], "total_tokens": 0})
    runner.ideas = {"i": {}}
    runner.suite = {"tracks": {"B": {"idea_ids": ["i"]}}, "boundary_cases": [], "dev_ids": [], "test_ids": ["i"]}
    runner.results = {"A_i": {"key": "A_i", "idea_id": "i", "success": True}}
    report = runner.report()
    assert report["tracks"]["boundary"]["success_at_budget"] == {"numerator": 0, "denominator": 0, "rate": None}
    assert report["tracks"]["A"]["success_at_budget"]["rate"] == 1


def test_disconnected_console_does_not_invalidate_durable_artifact(tmp_path, monkeypatch):
    from researchops.core.artifacts import progress
    def disconnected(*args, **kwargs):
        raise OSError(22, "Invalid argument")
    monkeypatch.setattr("builtins.print", disconnected)
    write_json(tmp_path / "result.json", {"success": True})
    progress("saved result")
    assert json.loads((tmp_path / "result.json").read_text())["success"] is True


def test_metric_alias_preserves_calibration_and_overfitting_endpoints(tmp_path):
    states, proposal = setup_design(tmp_path)
    proposal["design"]["primary_metrics"] = ["ece_15_equal_width_bins"]
    proposal["design"]["secondary_metrics"] = ["nll", "train_validation_accuracy_gap", "reliability_bin_summaries"]
    spec = compile_contract({}, proposal, states, {})
    assert spec["metrics"]["primary"] == ["ece_15_equal_bins"]
    assert "train_validation_gap" in spec["metrics"]["secondary"]
    assert spec["metric_name_normalization"]["ece_15_equal_width_bins"] == "ece_15_equal_bins"


def test_planner_interface_matches_actual_compiler_and_unknown_metric_is_diagnosed(tmp_path):
    from researchops.experiments.planning import planning_interface, METRICS
    states, proposal = setup_design(tmp_path)
    interface = planning_interface(states)
    assert set(interface["supported_metrics"]) == METRICS
    compiled = compile_contract({}, proposal, states, {})
    assert compiled["methods"]["r2"]["lr_candidates"] == interface["method_defaults"]["lora"]["lr_candidates"]
    proposal["design"]["secondary_metrics"] = ["ece_15_equal_width"]
    with pytest.raises(ValueError, match="ece_15_equal_width.*supported canonical names.*ece_15_equal_bins"):
        compile_contract({}, proposal, states, {})


def test_only_closing_punctuation_can_be_recovered_after_json():
    from researchops.core.structured import parse_single_object
    assert parse_single_object('{"value":1}"}') == {"value": 1}
    for text in ('{"value":1}{"value":2}', '{"value":1} this is a claim'):
        with pytest.raises(ValueError):
            parse_single_object(text)


def test_public_natural_language_entry_does_not_need_benchmark_files(tmp_path, monkeypatch):
    from researchops.application import study
    from researchops.core.storage import ResearchStore
    source = tmp_path / "source"
    ResearchStore(source / "researchops.db")
    class EmptyRetriever:
        def __init__(self, store):
            pass
        def search(self, *args, **kwargs):
            return []
    class BoundaryModel:
        model = "test-boundary"
        def __init__(self):
            self.messages = []
        def complete_bounded(self, messages, tools, max_tokens, timeout):
            self.messages.append(messages)
            if "You plan literature evidence retrieval" in messages[0]["content"]:
                response = {"retrieval_query": "time series anomaly detection", "preserved_constraints": ["time series"]}
            else:
                response = ({"status": "needs_adapter", "task_family": "other", "reason": "No time-series adapter"}
                            if len(self.messages) == 2 else {"criteria": {k: "pass" for k in CRITERIA}})
            return ModelTurn(text=json.dumps(response), usage={"total_tokens": 100})
    monkeypatch.setattr(study, "HybridRetriever", EmptyRetriever)
    model = BoundaryModel()
    result = study.plan_study({"description": "Time series detection"}, source=source,
                              output=tmp_path / "output", model=model, retrieval_strategy="legacy", orchestration="native")
    assert result["status"] == "blocked_before_coding"
    assert result["proposal_status"] == "needs_adapter"
    assert result["charged_tokens_cumulative"] == 400
    assert (tmp_path / "output" / "retrieval_query_plan.json").exists()
    assert not list((tmp_path / "output").rglob("frozen_spec.json"))
    assert "case_expectations" not in json.dumps(model.messages)
    other_source = tmp_path / "other_source"
    ResearchStore(other_source / "researchops.db")
    with pytest.raises(ValueError, match="different source corpus"):
        study.plan_study({"description": "Another task"}, source=other_source,
                         output=tmp_path / "output", model=model, retrieval_strategy="legacy", orchestration="native")
    assert len(model.messages) == 4


def test_rejected_plan_is_revised_and_both_reviews_required(tmp_path):
    from researchops.experiments.study_workflow import refine_study, WORKFLOW_ROLE_CONTEXT
    from researchops.experiments.planning import HOST_PROTOCOL
    from researchops.core.storage import ResearchStore
    class Calls:
        def __init__(self):
            self.payloads = []
            self.reviews = 0
        def call(self, role, system, payload, output_tokens):
            assert WORKFLOW_ROLE_CONTEXT in system
            assert payload["task"] == {"description": "time series"}
            if role not in {"study_planner", "study_revision"}:
                assert payload["host_protocol"] == HOST_PROTOCOL
            self.payloads.append((role, payload))
            if role in {"study_planner", "study_revision"}:
                return {"status": "needs_adapter", "reason": "Unsupported task",
                        "alternative": {"requires_constraint_change": True}}
            self.reviews += 1
            review = {"criteria": {k: "pass" for k in CRITERIA}}
            if self.reviews == 2:
                review["criteria"]["honest_uncertainty"] = "fail"
                review["reasons"] = ["Mark the exact constraint to change"]
            return review
    calls = Calls()
    result = refine_study(idea={"description": "time series"}, states={}, evidence=[],
                          store=ResearchStore(tmp_path / "db.sqlite"), calls=calls, directory=tmp_path / "task")
    assert result["success"] and not result["first_pass_success"]
    assert len(result["attempts"]) == 2
    revisions = [p for role, p in calls.payloads if role == "study_revision"]
    assert revisions[0]["revision_request"]["final_review"]["criteria"]["honest_uncertainty"] == "fail"
    assert (tmp_path / "task" / "attempts" / "1" / "confirmation.json").exists()

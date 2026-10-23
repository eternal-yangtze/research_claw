import copy
import json

import pytest

from researchops.core.artifacts import file_hash, write_json
from researchops.core.models import Chunk, Paper
from researchops.core.storage import ResearchStore
from researchops.evaluation.precoding.gates import CRITERIA
from researchops.harness.budget import BudgetedModel
from researchops.harness.runtime import ModelTurn, ToolCall
from researchops.knowledge.retrieval import HybridRetriever, LexicalReranker, HashingEmbeddingBackend
from researchops.orchestration.study_calls import StudyCalls, UncertainStudyCall
from researchops.orchestration.study_graph import StudyWorkflow


@pytest.fixture
def design(tmp_path):
    pytest.importorskip("langgraph")
    states = {}
    for dataset in ("dtd", "pets"):
        manifest = tmp_path / dataset / "manifest.jsonl"
        manifest.parent.mkdir()
        manifest.write_text("\n".join(json.dumps({"sample_id": f"{dataset}-{split}", "split": split,
            "included": True, "content_sha256": f"{dataset}-{split}"}) for split in ("train", "val", "test")), encoding="utf-8")
        subset = manifest.with_name("subset.json")
        write_json(subset, [f"{dataset}-train"])
        states[dataset] = {"dataset_id": dataset, "status": "verified", "verification": {
            "dataset_id": dataset, "name": dataset, "root": str(manifest.parent), "class_count": 2,
            "split_counts": {"train": 1, "val": 1, "test": 1}, "manifest_path": str(manifest),
            "manifest_sha256": file_hash(manifest), "train_fractions": {"0.25": {
                "path": str(subset), "sha256": file_hash(subset), "count": 1}},
            "exclusions": {}, "split_seed": 42, "official_test_preserved": True}}
    store = ResearchStore(tmp_path / "corpus.db")
    store.upsert_paper(Paper("paper", "Low rank adaptation"))
    store.replace_chunks("paper", [Chunk("c1", "paper", "Method", "Low rank adaptation freezes the backbone and trains low rank updates.")])
    proposal = {"status": "proposed", "task_family": "image_classification", "hypothesis": "unproven rank tradeoff",
        "research_question": "Compare ranks", "evidence_claims": [{"claim": "Low rank updates adapt frozen models.",
        "kind": "paper_fact", "citations": ["c1"]}], "design": {
            "datasets": ["dtd", "pets"], "fractions": [0.25], "seeds": [42, 43, 44],
            "arms": [{"arm_id": f"r{rank}", "method": "lora", "rank": rank, "alpha": rank,
                      "dropout": 0.0, "weight_decay": 0.05, "target_blocks": [9, 10, 11]} for rank in (2, 8)],
            "primary_metrics": ["accuracy"], "secondary_metrics": ["trainable_parameters"],
            "analysis_plan": "Report paired seed differences; all outcomes remain unknown.",
            "success_rule": "Report supported, refuted or inconclusive after measurement.",
            "selection_policy": {"mode": "ordinary", "objective": "validation_accuracy"},
            "controls": ["same data and validation budget"], "limitations": ["three seeds"]}}
    return states, store, proposal


class ScriptedModel:
    model = "p4-scripted"

    def __init__(self, proposal, *, revise=False, malformed=False, tool=True):
        self.proposal, self.revise, self.malformed, self.tool = proposal, revise, malformed, tool
        self.calls, self.critic_calls, self.requests = 0, 0, []

    def complete_bounded(self, messages, tools, max_tokens, *, timeout):
        self.calls += 1
        self.requests.append(copy.deepcopy(messages))
        system = messages[0]["content"]
        if "research planner for" in system:
            if self.malformed and self.calls == 1:
                return ModelTurn(text="{broken", usage={"total_tokens": 20})
            if self.tool and not any(m["role"] == "tool" for m in messages):
                return ModelTurn(tool_calls=(ToolCall("read", "get_study_chunk", {"chunk_id": "c1"}),
                    ToolCall("compile", "compile_study_candidate", {"proposal": self.proposal})), usage={"total_tokens": 20})
            value = copy.deepcopy(self.proposal)
        else:
            value = {"criteria": {k: "pass" for k in CRITERIA}, "reasons": ["Source scope checked"],
                     "citation_support": [{"chunk_id": "c1", "verdict": "supported"}]}
            if "final independent reviewer" in system:
                payload = json.loads(messages[1]["content"])
                assert "revision_request" not in payload and "review" not in payload and "critic" not in payload
            else:
                self.critic_calls += 1
                if self.revise and self.critic_calls == 1:
                    value["criteria"]["metric_and_decision_rule"] = "fail"
                    value["reasons"] = ["Specify the prospective decision rule"]
        return ModelTurn(text=json.dumps(value), usage={"total_tokens": 20})


def workflow(tmp_path, design, model, *, mode="langgraph", after_node=None, cap=500000):
    states, store, _ = design
    root = tmp_path / mode
    calls = StudyCalls(root, model, cap)
    retriever = HybridRetriever(store, embedding=HashingEmbeddingBackend(), reranker=LexicalReranker())
    engine = StudyWorkflow(idea={"description": "Compare LoRA ranks"}, states=states, store=store,
        calls=calls, directory=root / "study", evidence=[{"chunk_id": "c1", "paper_id": "paper",
        "text": store.get_chunk("c1")["text"]}], retriever=retriever, orchestration=mode, after_node=after_node)
    return engine, calls


def test_graph_revision_gate_and_terminal_recovery(tmp_path, design):
    model = ScriptedModel(design[2], revise=True, malformed=True)
    engine, calls = workflow(tmp_path, design, model)
    result = engine.run()
    assert result["success"] and result["mechanical"]["passed"]
    assert len(result["attempts"]) == 2 and not result["first_pass_success"]
    assert (engine.directory / "frozen_spec.json").exists()
    before = calls.summary()
    resumed, _ = workflow(tmp_path, design, model)
    assert resumed.run() == result
    assert calls.summary() == before
    assert model.calls == 8


def test_node_committed_before_graph_checkpoint_is_not_paid_twice(tmp_path, design):
    model = ScriptedModel(design[2])
    def crash(node, patch):
        if node == "planner":
            raise RuntimeError("simulated process loss after node commit")
    engine, calls = workflow(tmp_path, design, model, after_node=crash)
    with pytest.raises(RuntimeError, match="process loss"):
        engine.run()
    assert model.calls == 2
    resumed, calls = workflow(tmp_path, design, model)
    result = resumed.run()
    assert result["success"] and model.calls == 4
    assert calls.summary()["root"]["charged_tokens"] == 80


def test_graph_pause_and_native_control_have_same_scientific_outcome(tmp_path, design):
    graph_model = ScriptedModel(design[2])
    engine, calls = workflow(tmp_path, design, graph_model)
    paused = engine.run(interrupt_after=["compile_gate"])
    assert paused["status"] == "paused" and paused["next"] == ["critic"]
    resumed, _ = workflow(tmp_path, design, graph_model)
    graph_result = resumed.run()
    native_model = ScriptedModel(design[2])
    native, _ = workflow(tmp_path, design, native_model, mode="native")
    native_result = native.run()
    for field in ("success", "mechanical", "proposal_status", "planned_jobs", "attempts"):
        assert graph_result[field] == native_result[field]
    assert graph_model.calls == native_model.calls == 4


def test_budget_stop_does_not_commit_contract(tmp_path, design):
    model = ScriptedModel(design[2])
    engine, calls = workflow(tmp_path, design, model, cap=1000)
    result = engine.run()
    assert not result["success"] and result["stop_reason"] == "budget_exhausted"
    assert model.calls == 0 and not (engine.directory / "frozen_spec.json").exists()


def test_data_tamper_and_committed_artifact_tamper_block_resume(tmp_path, design):
    model = ScriptedModel(design[2])
    engine, _ = workflow(tmp_path, design, model)
    engine.run()
    write_json(engine.directory / "proposal.json", {"status": "forged"})
    with pytest.raises(ValueError, match="Committed study artifact"):
        workflow(tmp_path, design, model)[0].run()
    subset = design[0]["dtd"]["verification"]["train_fractions"]["0.25"]["path"]
    write_json(__import__("pathlib").Path(subset), ["leaked-test"])
    with pytest.raises(ValueError, match="inputs/code/model/limits changed|subset changed"):
        workflow(tmp_path, design, model)[0].run()


def test_uncertain_dispatch_cannot_be_silently_reissued(tmp_path):
    class Offline:
        model = "offline"
        calls = 0
        def complete_bounded(self, *args, **kwargs):
            self.calls += 1
            raise TimeoutError("socket lost after dispatch")
    model = Offline()
    calls = StudyCalls(tmp_path, model, 100000)
    with pytest.raises(TimeoutError):
        calls.call("planner", "system", {}, 100)
    with pytest.raises(UncertainStudyCall):
        StudyCalls(tmp_path, model, 100000).call("planner", "system", {}, 100)
    assert model.calls == 1 and calls.summary()["root"]["unknown_reserved_tokens"] > 0


def test_existing_outer_budget_is_reused_once(tmp_path):
    model = ScriptedModel({"status": "blocked"}, tool=False)
    outer = BudgetedModel(model, tmp_path / "outer", 500000)
    calls = StudyCalls(tmp_path / "study", outer, 100000)
    calls.call("query", "system", {}, 100)
    assert outer.summary()["model_calls"] == 1
    assert outer.summary()["charged_tokens"] == calls.summary()["study"]["charged_tokens"] == 20
    assert not (tmp_path / "study" / "budget" / "ledger.json").exists()


def test_all_roles_share_one_cumulative_turn_limit(tmp_path, design):
    states, store, _ = design
    model = ScriptedModel(design[2])
    root = tmp_path / "limited"
    calls = StudyCalls(root, model, 500000, max_model_turns=2)
    engine = StudyWorkflow(idea={"description": "Compare ranks"}, states=states, store=store,
        calls=calls, directory=root / "study", evidence=[{"chunk_id": "c1", "paper_id": "paper"}],
        retriever=HybridRetriever(store, embedding=HashingEmbeddingBackend(), reranker=LexicalReranker()))
    result = engine.run()
    assert result["stop_reason"] == "budget_exhausted"
    assert model.calls == calls.summary()["study"]["turns"] == 2
    assert not (engine.directory / "frozen_spec.json").exists()


def test_outside_chunk_and_reviewer_compile_tool_are_denied(tmp_path, design):
    class ProbingModel(ScriptedModel):
        def __init__(self, proposal):
            super().__init__(proposal, tool=False)
            self.observed_errors = []
        def complete_bounded(self, messages, tools, max_tokens, *, timeout):
            planner = "research planner for" in messages[0]["content"]
            tool_messages = [m for m in messages if m["role"] == "tool"]
            if not tool_messages:
                self.calls += 1
                return ModelTurn(tool_calls=(ToolCall("probe", "get_study_chunk" if planner else "compile_study_candidate",
                    {"chunk_id": "another-project-c1"} if planner else {"proposal": self.proposal}),), usage={"total_tokens": 20})
            observation = json.loads(tool_messages[-1]["content"])
            self.observed_errors.append(observation["status"])
            assert observation["status"] == ("error" if planner else "denied")
            return super().complete_bounded(messages, tools, max_tokens, timeout=timeout)
    model = ProbingModel(design[2])
    engine, _ = workflow(tmp_path, design, model)
    assert engine.run()["success"]
    assert model.observed_errors == ["error", "denied", "denied"]

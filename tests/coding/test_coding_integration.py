import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from researchops.coding import CodingApplication
from researchops.coding_plan import API_CONTRACT, validate_implementation
from researchops.coding_sandbox import project_execution_spec
from researchops.compression import PaperCompressor
from researchops.datasets import file_hash, write_json
from researchops.models import Chunk, Paper
from researchops.retrieval import HybridRetriever, SearchHit
from researchops.runtime import ModelTurn, ToolCall
from researchops.storage import ResearchStore


def test_filtered_retrieval_does_not_leak_fts_hits_from_other_papers(tmp_path):
    store = ResearchStore(tmp_path / "db.sqlite")
    for name in ("allowed", "foreign"):
        store.upsert_paper(Paper(name, name))
        store.replace_chunks(name, [Chunk(name + "_node", name, "Method", "low rank adaptation transformer")])
    hits = HybridRetriever(store).search("low rank adaptation", paper_ids={"allowed"})
    assert hits and {hit.paper_id for hit in hits} == {"allowed"}
    assert HybridRetriever(store).search("anything", paper_ids=set()) == []


def test_l3_model_consumes_retrieved_evidence_and_cache_changes_with_retrieval(tmp_path):
    store = ResearchStore(tmp_path / "db.sqlite")
    paper = Paper("paper", "Method")
    store.upsert_paper(paper)
    store.replace_chunks("paper", [Chunk("lexical", "paper", "Method", "objective objective objective"),
                                  Chunk("semantic", "paper", "Equation", "L = a + b")])
    class Retriever:
        embedding = SimpleNamespace(model_name="neural-test", model_revision="1")
        reranker = SimpleNamespace(model_name="cross-test")
        mmr_lambda = 0.75
        chosen = "semantic"
        def search(self, query, limit=8, candidate_limit=40, *, paper_ids=None):
            assert paper_ids == {"paper"}
            return [SearchHit(self.chosen, "paper", "Method", "Method", "unused", 1, 0, 1, 1, 1, "p1", "neural", "cross")]
    class Model:
        calls = 0
        def complete(self, messages, tools, max_tokens):
            self.calls += 1
            nodes = json.loads(messages[-1]["content"])["nodes"]
            return ModelTurn(json.dumps({"answer": nodes[0]["text"], "evidence": [{"node_id": nodes[0]["node_id"]}]}))
    retriever, model = Retriever(), Model()
    compressor = PaperCompressor(store, model, retriever)
    first = compressor.l3(paper, "objective")
    assert first["content"]["answer"] == "L = a + b"
    assert compressor.l3(paper, "objective")["cache_hit"]
    retriever.chosen = "lexical"
    changed = compressor.l3(paper, "objective")
    assert not changed["cache_hit"] and changed["source_hash"] != first["source_hash"]
    assert model.calls == 2


def plan_fixture(tmp_path):
    frozen = tmp_path / "artifacts" / "frozen.json"
    write_json(frozen, {"datasets": []})
    return {"plan_id": "coding_test", "idea_id": "toy", "objective": "Implement a toy module",
        "api_contract": {}, "finish_condition": "External verifier passes",
        "frozen_spec": {"path": str(frozen), "sha256": file_hash(frozen)}, "allowed_outputs": ["models.py"],
        "budget": {"max_model_turns": 10, "max_tool_calls": 15, "max_verification_rounds": 3,
                   "max_total_tokens": 20000, "max_wall_seconds": 60, "command_timeout_seconds": 5}}


class Scripted:
    model = "scripted"
    def __init__(self, calls):
        self.calls = iter(calls)
        self.index = 0
    def complete(self, messages, tools, max_tokens):
        self.index += 1
        name, args = next(self.calls)
        return ModelTurn("", (ToolCall(str(self.index), name, args),), {"total_tokens": 10})


def test_coding_loop_repairs_verifier_failures_and_cannot_self_declare_success(tmp_path, monkeypatch):
    class Sandbox:
        count = 0
        def __init__(self, *args, **kwargs): pass
        def verify(self, number):
            Sandbox.count += 1
            return {"passed": number == 2, "checks": [{"name": "behavior", "passed": number == 2, "error": "wrong value"}]}
    monkeypatch.setattr("researchops.coding.agent.CodingSandbox", Sandbox)
    plan = plan_fixture(tmp_path)
    model = Scripted([
        ("write_file", {"path": "../escape.py", "content": "BAD"}),
        ("write_file", {"path": "models.py", "content": "answer = 0\n"}),
        ("finish", {"summary": "I say done"}),
        ("run_verification", {}),
        ("write_file", {"path": "models.py", "content": "answer = 42\n"}),
        ("run_verification", {}),
        ("finish", {"summary": "Verified"}),
    ])
    result = CodingApplication(workspace=tmp_path, model=model).run(plan)
    assert result["status"] == "coding_verified" and result["verification_rounds"] == 2
    assert result["research_success"] is None
    assert result["scorecard"]["trace_completeness_rate"] == 1
    assert not (tmp_path / "coding" / "coding_test" / "escape.py").exists()
    assert file_hash(plan["frozen_spec"]["path"]) == plan["frozen_spec"]["sha256"]
    cached = CodingApplication(workspace=tmp_path, model=Scripted([])).run({**plan, "cache_hit": True}, resume=True)
    assert cached["cache_hit"]


def test_verification_is_invalidated_when_source_changes(tmp_path):
    app = CodingApplication(workspace=tmp_path, model=None)
    app.plan = plan_fixture(tmp_path)
    app.project = tmp_path / "project"
    app.project.mkdir()
    app.state = {"last_verification": {"passed": True}, "verified_source_hash": None}
    app._write_file("models.py", "answer = 42\n")
    app.state["verified_source_hash"] = app._source_hash()
    assert app._finish("done")["status"] == "coding_verified"
    app._write_file("models.py", "answer = -1\n")
    with pytest.raises(ValueError, match="current source hash"):
        app._finish("done again")


def test_container_path_projection_rejects_data_outside_workspace(tmp_path):
    with pytest.raises(ValueError):
        project_execution_spec({"datasets": [{"root": str(tmp_path / "outside"),
            "manifest_path": str(tmp_path / "outside" / "m.json"), "train_fractions": {}}]}, tmp_path / "datasets")


def test_coding_plan_rejects_cycles_and_incomplete_outputs():
    good = {"tasks": [{"id": "all", "description": "Implement", "depends_on": [],
                       "outputs": list(API_CONTRACT), "verification": ["external verifier"]}]}
    validate_implementation(good)
    with pytest.raises(ValueError, match="dependencies"):
        validate_implementation({"tasks": [{**good["tasks"][0], "depends_on": ["unknown"]}]})
    with pytest.raises(ValueError, match="all required"):
        validate_implementation({"tasks": [{**good["tasks"][0], "outputs": ["models.py"]}]})


def test_context_compaction_preserves_audit_and_current_failure(tmp_path):
    app = CodingApplication(workspace=tmp_path, model=None)
    full_code = "important_source = 123\n" * 1000
    call = {"id": "write", "type": "function", "function": {"name": "write_file",
        "arguments": json.dumps({"path": "models.py", "content": full_code})}}
    app.state = {"messages": [{"role": "system", "content": "contract"},
        {"role": "assistant", "tool_calls": [call]}, {"role": "tool", "tool_call_id": "write", "content": "saved"},
        {"role": "assistant", "content": "later"}, {"role": "user", "content": "continue"}],
        "last_verification": {"passed": False, "checks": [{"name": "metric", "passed": False, "error": "bad formula"}]}}
    compact = app._model_messages()
    assert "omitted" in compact[1]["tool_calls"][0]["function"]["arguments"]
    assert full_code in json.loads(app.state["messages"][1]["tool_calls"][0]["function"]["arguments"])["content"]
    assert "bad formula" in compact[-1]["content"]

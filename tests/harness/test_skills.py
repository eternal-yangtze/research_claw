from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from researchops.application.workflow import ResearchOpsApp, RESEARCH_LEAD
from researchops.harness.runtime import ModelTurn, Session, ToolCall, ToolContext


def context(app, session=None, agent=None):
    return ToolContext("test", session or Session("s", "t"), agent or RESEARCH_LEAD, app.executor)


def test_metadata_only_then_load_and_input_schema(tmp_path):
    app = ResearchOpsApp(tmp_path)
    entries = app.skills.catalog(RESEARCH_LEAD, app.executor.policy)
    assert len(entries["skills"]) == 4
    assert all("body" not in e and "input_schema" not in e for e in entries["skills"])
    c = context(app)
    loaded = app.skills.load("document-ingest", _context=c)
    assert loaded["body"] and loaded["input_schema"]["required"] == ["path"]
    assert c.session.metadata["loaded_skills"]["document-ingest"] == loaded["sha256"]


def test_skills_cannot_bypass_loading_scope_or_tool_policy(tmp_path):
    app = ResearchOpsApp(tmp_path)
    c = context(app)
    loaded = app.skills.loader.load("document-ingest")
    with pytest.raises(PermissionError):
        app.skills.invoke("document-ingest", loaded["sha256"], {"path": str(tmp_path)}, _context=c)
    denied = replace(RESEARCH_LEAD, denied_tools=("ingest_documents",))
    with pytest.raises(PermissionError):
        app.skills.load("document-ingest", _context=context(app, agent=denied))
    assert "document-ingest" not in [e["name"] for e in app.skills.catalog(denied, app.executor.policy)["skills"]]
    with pytest.raises(ValueError):
        app.skills.load("../document-ingest", _context=c)
    app.skills.load("document-ingest", _context=c)
    with pytest.raises(RuntimeError, match="Skill action denied"):
        app.skills.invoke("document-ingest", loaded["sha256"], {"path": str(tmp_path)},
                         _context=context(app, session=c.session, agent=denied))
    with pytest.raises(RuntimeError, match="outside configured roots"):
        app.skills.invoke("document-ingest", loaded["sha256"], {"path": str(tmp_path.parent / "outside")}, _context=c)
    with pytest.raises(RuntimeError, match="unknown argument"):
        app.skills.invoke("document-ingest", loaded["sha256"], {"path": str(tmp_path), "tool": "implement_experiment"}, _context=c)


def test_changed_skill_hash_requires_reload(tmp_path):
    app = ResearchOpsApp(tmp_path / "ws")
    # Isolated skill copies; never change the installed project definitions.
    import shutil
    copied = tmp_path / "skills"
    shutil.copytree(app.skills.loader.root, copied)
    app.skills.loader.root = copied.resolve()
    c = context(app)
    loaded = app.skills.load("document-ingest", _context=c)
    path = copied / "document-ingest" / "SKILL.md"
    path.write_text(path.read_text(encoding="utf-8") + "\nChanged version.\n", encoding="utf-8")
    with pytest.raises(PermissionError):
        app.skills.invoke("document-ingest", loaded["sha256"], {"path": str(app.workspace)}, _context=c)


def test_agent_loads_then_invokes_real_ingestion_and_hidden_tool_is_denied(tmp_path):
    document = tmp_path / "paper.md"
    document.write_text("Title: Test paper\n\n# Method\nLow rank adaptation preserves frozen weights.", encoding="utf-8")
    class Model:
        def complete(self, messages, tools, max_tokens):
            assert {t["function"]["name"] for t in tools} == {"search_skills", "load_skill", "invoke_skill"}
            results = [json.loads(m["content"]) for m in messages if m["role"] == "tool"]
            if not results:
                assert "# Ingest local research documents" not in messages[0]["content"]
                return ModelTurn(tool_calls=(ToolCall("1", "ingest_documents", {"path": str(document)}),))
            if len(results) == 1:
                assert results[-1]["status"] == "denied"
                return ModelTurn(tool_calls=(ToolCall("2", "load_skill", {"name": "document-ingest"}),))
            if len(results) == 2:
                return ModelTurn(tool_calls=(ToolCall("3", "invoke_skill", {"name": "document-ingest",
                    "sha256": results[-1]["output"]["sha256"], "arguments": {"path": str(document)}}),))
            return ModelTurn(text="Imported.", usage={"total_tokens": 10})
    app = ResearchOpsApp(tmp_path / "ws", allowed_roots=[document])
    result = app.run_skill_agent("Import this local paper", model=Model())
    assert result["skill_invocations"] == 1
    assert app.store.counts()["papers"] == 1
    assert result["budget"]["unknown_reserved_tokens"] > 0  # Never report missing usage as free.
    artifacts = list((app.artifact_root / "skills").glob("*/*.json"))
    invocation = json.loads(artifacts[0].read_text(encoding="utf-8"))
    direct = app.call("ingest_documents", {"path": str(document)})
    assert invocation["result"]["output"] == direct["output"]


def test_skill_budget_admission_happens_before_model_dispatch(tmp_path):
    class NeverCalled:
        def complete(self, *args):
            raise AssertionError("must not dispatch")
    from researchops.coding.budget import BudgetAdmissionError
    app = ResearchOpsApp(tmp_path)
    with pytest.raises(BudgetAdmissionError):
        app.run_skill_agent("Retrieve evidence", model=NeverCalled(), token_cap=1)
    failure = next((tmp_path / "skill_runs").glob("*/failure.json"))
    assert json.loads(failure.read_text(encoding="utf-8"))["budget"]["model_calls"] == 0


def test_plan_skill_snapshots_current_corpus_and_invalidates_on_growth(tmp_path, monkeypatch):
    from researchops.application import study
    from researchops.core.artifacts import file_hash
    from researchops.core.storage import ResearchStore
    app = ResearchOpsApp(tmp_path)
    with pytest.raises(ValueError, match="research_question or description"):
        app._plan_study_skill({"scientific_question": "not a recognized field"}, str(tmp_path))
    assert not (tmp_path / "skill_sources").exists()
    paper = tmp_path / "paper.md"
    paper.write_text("# Method\nA frozen weight has a low rank update.", encoding="utf-8")
    app.call("ingest_documents", {"path": str(paper)})
    def fake_plan(idea, *, source, output, **kwargs):
        assert source != app.workspace and source not in output.parents
        return {"source": str(source), "output": str(output),
                "papers": ResearchStore(source / "researchops.db").counts()["papers"],
                "hash": file_hash(source / "researchops.db")}
    monkeypatch.setattr(study, "plan_study", fake_plan)
    args = {"idea": {"description": "Compare ranks"}, "source_workspace": str(tmp_path)}
    initial = app._plan_study_skill(**args)
    assert initial["papers"] == 1
    assert app._plan_study_skill(**args) == initial
    paper2 = tmp_path / "paper2.md"
    paper2.write_text("# New evidence\nVisual prompts use learnable tokens.", encoding="utf-8")
    app.call("ingest_documents", {"path": str(paper2)})
    expanded = app._plan_study_skill(**args)
    assert expanded["papers"] == 2 and expanded["output"] != initial["output"]
    assert file_hash(Path(initial["source"]) / "researchops.db") == initial["hash"]
    states = tmp_path / "datasets" / "test" / "state.json"
    states.parent.mkdir(parents=True)
    states.write_text(json.dumps({"dataset_id": "test", "status": "verified"}), encoding="utf-8")
    assert app._plan_study_skill(**args)["output"] != expanded["output"]


def test_nested_skill_model_calls_share_outer_budget(tmp_path, monkeypatch):
    from researchops.application import study
    class Model:
        def complete(self, messages, tools, max_tokens):
            if messages[0]["content"] == "nested planning":
                return ModelTurn(text="Nested result", usage={"total_tokens": 11})
            results = [json.loads(m["content"]) for m in messages if m["role"] == "tool"]
            if not results:
                return ModelTurn(tool_calls=(ToolCall("1", "load_skill", {"name": "study-plan"}),), usage={"total_tokens": 10})
            if len(results) == 1:
                return ModelTurn(tool_calls=(ToolCall("2", "invoke_skill", {"name": "study-plan",
                    "sha256": results[-1]["output"]["sha256"], "arguments": {
                        "idea": {"description": "Compare ranks"}, "source_workspace": str(tmp_path)}}),),
                    usage={"total_tokens": 10})
            assert results[-1]["status"] == "success"
            return ModelTurn(text="Done", usage={"total_tokens": 10})
    def fake_plan(idea, *, model, **kwargs):
        for _ in range(2):
            model.complete([{"role": "user", "content": "nested planning"}], [], 100)
        return {"status": "proposed"}
    monkeypatch.setattr(study, "plan_study", fake_plan)
    result = ResearchOpsApp(tmp_path).run_skill_agent("Plan a study", model=Model(), token_cap=100000)
    assert result["skill_invocations"] == 1
    assert result["budget"]["model_calls"] == 5
    assert result["budget"]["known_actual_tokens"] == result["budget"]["charged_tokens"] == 52


def test_gateway_margin_is_reserved_without_requesting_extra_output(tmp_path):
    from researchops.harness.budget import BudgetedModel
    class Gateway:
        def complete(self, messages, tools, max_tokens):
            assert max_tokens == 100
            return ModelTurn(text="done", usage={"total_tokens": 20000})
    wrapper = BudgetedModel(Gateway(), tmp_path, 100000)
    wrapper.complete([{"role": "user", "content": "test"}], [], 100)
    assert wrapper.summary()["charged_tokens"] == 20000
    assert wrapper.state["call_ledger"][0]["provider_token_margin"] == 32768


def test_overreported_usage_preserves_response_and_still_stops(tmp_path):
    from researchops.harness.budget import BudgetedModel
    from researchops.coding.budget import BudgetAdmissionError
    class Gateway:
        def complete(self, *args):
            return ModelTurn(text="Retain this actual response", usage={"total_tokens": 90000})
    wrapper = BudgetedModel(Gateway(), tmp_path, 100000)
    with pytest.raises(BudgetAdmissionError, match="provider_usage_exceeded"):
        wrapper.complete([{"role": "user", "content": "test"}], [], 100)
    response = next((tmp_path / "calls").glob("*.response.json"))
    assert json.loads(response.read_text(encoding="utf-8"))["text"] == "Retain this actual response"
    assert wrapper.summary()["known_actual_tokens"] == 90000
    with pytest.raises(ValueError, match="must not be reset"):
        BudgetedModel(Gateway(), tmp_path, 100000)

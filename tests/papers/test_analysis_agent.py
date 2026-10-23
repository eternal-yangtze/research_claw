import json
from pathlib import Path

from researchops.core.models import Chunk, Domain, Idea, Paper
from researchops.core.storage import ResearchStore
from researchops.harness.runtime import ModelTurn, ToolCall
from researchops.knowledge.retrieval import (
    HashingEmbeddingBackend,
    HybridRetriever,
    LexicalReranker,
)
from researchops.papers.analysis import PaperAnalyzer, UNKNOWN
from researchops.harness.budget import BudgetedModel


def _candidate(citation=None):
    unknown = {"text": UNKNOWN, "citations": []}
    return {
        "paper_summary": {
            "text": "The method models graph relations.",
            "citations": [citation] if citation else [],
        },
        "research_problem": dict(unknown),
        "inputs_outputs": dict(unknown),
        "method_overview": dict(unknown),
        "architecture_components": [],
        "knowledge_or_graph_role": dict(unknown),
        "training_objective": dict(unknown),
        "inference_procedure": dict(unknown),
        "datasets": [],
        "data_splits_preprocessing": dict(unknown),
        "metrics": [],
        "paper_baselines": [],
        "reported_results": [],
        "implementation_availability": dict(unknown),
        "compute_requirements": dict(unknown),
        "limitations": [],
        "baseline_assessment": {
            "verdict": "insufficient_evidence",
            "rationale": dict(unknown),
            "task_alignment": dict(unknown),
            "input_output_compatibility": dict(unknown),
            "dataset_metric_comparability": dict(unknown),
            "reproduction_requirements": [],
            "adaptations_required": [],
        },
        "missing_information": ["Training details are not established."],
    }


def _unknown_candidate():
    value = _candidate()
    value["paper_summary"] = {"text": UNKNOWN, "citations": []}
    return value


class RepairingModel:
    model = "paper-analysis-test"

    def __init__(self):
        self.calls = 0

    def complete(self, messages, tools, max_tokens):
        self.calls += 1
        if self.calls == 1:
            return ModelTurn(text=json.dumps(_candidate()))
        if self.calls == 2:
            feedback = json.loads(messages[-1]["content"])
            assert feedback["type"] == "validation_feedback"
            assert "missing_citation" in {item["code"] for item in feedback["issues"]}
            return ModelTurn(tool_calls=(ToolCall(
                "search-1", "search_paper_chunks", {"query": "graph relations", "limit": 3}
            ),))
        if self.calls == 3:
            search = json.loads(messages[-1]["content"])["output"]
            assert search["paper_id"] == "p1"
            assert [item["chunk_id"] for item in search["results"]] == ["c1"]
            return ModelTurn(tool_calls=(ToolCall(
                "read-1", "get_paper_chunk", {"chunk_id": "c1"}
            ),))
        assert self.calls == 4
        chunk = json.loads(messages[-1]["content"])["output"]
        assert chunk["paper_id"] == "p1" and chunk["full_text"] is True
        return ModelTurn(text=json.dumps(_candidate("c1")))


def test_micro_paper_analysis_repairs_citations_and_resumes(tmp_path: Path):
    workspace = tmp_path / "workspace"
    store = ResearchStore(workspace / "researchops.db")
    paper = Paper("p1", "Graph Method", file_path="paper.md")
    other = Paper("p2", "Other Paper", file_path="other.md")
    store.upsert_paper(paper)
    store.upsert_paper(other)
    store.replace_chunks("p1", [
        Chunk("c1", "p1", "Method", "The method models graph relations.", source_hash="p1-v1"),
    ])
    store.replace_chunks("p2", [
        Chunk("c2", "p2", "Method", "Unrelated private evidence.", source_hash="p2-v1"),
    ])
    embedding = HashingEmbeddingBackend()
    retriever = HybridRetriever(
        store, embedding=embedding, reranker=LexicalReranker())
    model = RepairingModel()
    analyzer = PaperAnalyzer(
        model, store, workspace, mode="micro", retriever=retriever)
    idea = Idea(
        "Graph study", Domain.KNOWLEDGE_GRAPH, "Study graph relations.",
        "How are graph relations modeled?",
    )
    store.create_task("task-1", idea.title, idea.domain.value, {
        "title": idea.title,
        "domain": idea.domain.value,
        "description": idea.description,
        "research_question": idea.research_question,
    })

    first = analyzer.analyze("task-1", idea, paper, run_id="analysis-run")
    artifact = json.loads(Path(first["path"]).read_text(encoding="utf-8"))
    assert first["status"] == "completed"
    assert artifact["paper_summary"]["citations"] == ["c1"]
    assert set(artifact["citation_index"]) == {"c1"}
    assert artifact["micro_agent"]["model_turns"] == 4
    assert artifact["micro_agent"]["tool_calls"] == 2
    assert artifact["micro_agent"]["read_chunk_ids"] == ["c1"]

    second = analyzer.analyze("task-1", idea, paper, run_id="analysis-run")
    assert second["status"] == "completed" and model.calls == 4
    assert second["sha256"] == first["sha256"]


def test_micro_paper_analysis_without_full_text_is_explicitly_insufficient(tmp_path: Path):
    workspace = tmp_path / "workspace"
    store = ResearchStore(workspace / "researchops.db")
    paper = Paper("empty", "Metadata Only")
    store.upsert_paper(paper)
    model = RepairingModel()
    analyzer = PaperAnalyzer(model, store, workspace, mode="micro")
    idea = Idea("Question", Domain.AI_ML, "Description", "Question?")
    store.create_task("task-empty", idea.title, idea.domain.value, {
        "title": idea.title,
        "domain": idea.domain.value,
        "description": idea.description,
        "research_question": idea.research_question,
    })

    result = analyzer.analyze("task-empty", idea, paper)
    artifact = json.loads(Path(result["path"]).read_text(encoding="utf-8"))
    assert result["status"] == "insufficient_full_text"
    assert artifact["baseline_assessment"]["verdict"] == "insufficient_evidence"
    assert artifact["baseline_assessment"]["rationale"]["text"] == UNKNOWN
    assert model.calls == 0


def test_micro_paper_analysis_rejects_cross_paper_chunk_read(tmp_path: Path):
    class ScopeProbeModel:
        model = "scope-probe"

        def __init__(self):
            self.calls = 0

        def complete(self, messages, tools, max_tokens):
            self.calls += 1
            if self.calls == 1:
                return ModelTurn(tool_calls=(ToolCall(
                    "forbidden", "get_paper_chunk", {"chunk_id": "c2"}
                ),))
            result = json.loads(messages[-1]["content"])
            assert result["status"] == "error"
            assert result["error_type"] == "PermissionError"
            return ModelTurn(text=json.dumps(_unknown_candidate()))

    workspace = tmp_path / "workspace"
    store = ResearchStore(workspace / "researchops.db")
    paper = Paper("p1", "Bound Paper")
    store.upsert_paper(paper)
    store.upsert_paper(Paper("p2", "Forbidden Paper"))
    store.replace_chunks("p1", [Chunk("c1", "p1", "Method", "bound evidence")])
    store.replace_chunks("p2", [Chunk("c2", "p2", "Method", "private evidence")])
    idea = Idea("Scope", Domain.AI_ML, "Scope test", "What is supported?")
    store.create_task("task-scope", idea.title, idea.domain.value, {
        "title": idea.title,
        "domain": idea.domain.value,
        "description": idea.description,
        "research_question": idea.research_question,
    })
    model = ScopeProbeModel()
    retriever = HybridRetriever(
        store, embedding=HashingEmbeddingBackend(), reranker=LexicalReranker())
    analyzer = PaperAnalyzer(
        model, store, workspace, mode="micro", retriever=retriever)

    result = analyzer.analyze("task-scope", idea, paper)
    artifact = json.loads(Path(result["path"]).read_text(encoding="utf-8"))
    assert result["status"] == "completed"
    assert artifact["paper_summary"]["text"] == UNKNOWN
    assert "c2" not in artifact["citation_index"]
    assert model.calls == 2


def test_two_papers_share_root_budget_but_have_distinct_scopes(tmp_path: Path):
    from concurrent.futures import ThreadPoolExecutor

    class UnknownModel:
        model = "unknown-control"

        def complete(self, messages, tools, max_tokens):
            return ModelTurn(text=json.dumps(_unknown_candidate()), usage={"total_tokens": 10})

    store = ResearchStore(tmp_path / "researchops.db")
    idea = Idea("Scope", Domain.AI_ML, "Scope test", "What is supported?")
    store.create_task("two-papers", idea.title, idea.domain.value, {})
    papers = [Paper("p1", "First"), Paper("p2", "Second")]
    for paper in papers:
        store.upsert_paper(paper)
        store.replace_chunks(paper.paper_id, [
            Chunk("c-" + paper.paper_id, paper.paper_id, "Method", "Evidence text.")])
    budget = BudgetedModel(UnknownModel(), tmp_path / "budget", 500_000)
    analyzer = PaperAnalyzer(budget, store, tmp_path, mode="micro",
        retriever=HybridRetriever(store, embedding=HashingEmbeddingBackend(), reranker=LexicalReranker()))
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda p: analyzer.analyze("two-papers", idea, p, run_id="root"), papers))
    assert all(r["status"] == "completed" for r in results)
    summary = budget.summary()
    assert len(summary["scopes"]) == 2
    assert summary["charged_tokens"] == 20
    assert {scope["turns"] for scope in summary["scopes"].values()} == {1}


def test_completed_paper_hash_survives_downstream_root_spending(tmp_path: Path):
    class UnknownModel:
        model = "shared-root-stability"

        def complete(self, messages, tools, max_tokens):
            return ModelTurn(text=json.dumps(_unknown_candidate()), usage={"total_tokens": 10})

    store = ResearchStore(tmp_path / "researchops.db")
    paper = Paper("p1", "Bound paper")
    store.upsert_paper(paper)
    store.replace_chunks("p1", [Chunk("c1", "p1", "Method", "Eligible evidence.")])
    store.create_task("stable", "Study", "ai_ml", {})
    model = BudgetedModel(UnknownModel(), tmp_path / "budget", 500_000)
    analyzer = PaperAnalyzer(model, store, tmp_path, mode="micro", retriever=HybridRetriever(
        store, embedding=HashingEmbeddingBackend(), reranker=LexicalReranker()))
    idea = Idea("Study", Domain.AI_ML, "Compare", "What is established?")
    first = analyzer.analyze("stable", idea, paper)
    first_bytes = Path(first["path"]).read_bytes()
    snapshot = json.loads(first_bytes)["micro_agent"]["budget"]
    model.scope("downstream", max_model_turns=1, max_total_tokens=100_000).complete(
        [{"role": "user", "content": "A distinct downstream stage"}], [], 100)
    second = analyzer.analyze("stable", idea, paper)
    assert model.summary()["model_calls"] == 2
    assert model.summary()["charged_tokens"] == 20
    assert snapshot["charged_tokens"] == 10
    assert first == second
    assert Path(second["path"]).read_bytes() == first_bytes


def test_acquisition_analysis_node_recovers_without_calling_model_again(tmp_path: Path):
    from researchops.papers.acquisition import AcquisitionDAG, PaperState
    from researchops.papers.ingestion import DocumentIngestor
    from researchops.knowledge.compression import PaperCompressor
    from researchops.harness.runtime import TraceStore

    store = ResearchStore(tmp_path / "researchops.db")
    paper = Paper("p1", "Graph Method")
    store.upsert_paper(paper)
    store.replace_chunks("p1", [Chunk("c1", "p1", "Method", "The method models graph relations.")])
    idea = Idea("Graph", Domain.AI_ML, "Graph method", "How are relations modeled?")
    store.create_task("dag-task", idea.title, idea.domain.value, {})
    model = RepairingModel()
    retriever = HybridRetriever(store, embedding=HashingEmbeddingBackend(), reranker=LexicalReranker())
    analyzer = PaperAnalyzer(model, store, tmp_path, mode="micro", retriever=retriever)
    dag = AcquisitionDAG(store=store, ingestor=DocumentIngestor(store), analyzer=analyzer,
        compressor=PaperCompressor(store, model), workspace=tmp_path, traces=TraceStore(tmp_path / "runs"),
        downloader=lambda *args: {}, source_downloader=lambda *args: {})
    state = PaperState(paper, "selected", outputs={"parse_pdf": {"status": "ingested"}})
    first = dag._run_node("dag-task", idea, state, "analyze_full_text", "dag-root", "session")
    second = dag._run_node("dag-task", idea, state, "analyze_full_text", "dag-root", "session")
    assert first["output"]["status"] == "completed"
    assert second["recovered"] is True and model.calls == 4

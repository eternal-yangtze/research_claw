import json
from pathlib import Path

from researchops.acquisition import AcquisitionDAG
from researchops.batch_research import BatchResearchPipeline, _ranking_metrics, load_idea
from researchops.compression import PaperCompressor
from researchops.ingestion import DocumentIngestor
from researchops.literature import LiteratureSearch
from researchops.models import Domain, Idea, Paper
from researchops.paper_analysis import PaperAnalyzer
from researchops.runtime import ModelTurn, TraceStore
from researchops.storage import ResearchStore


class BatchModel:
    model = "batch-test-model"

    def complete(self, messages, tools, max_tokens):
        request = json.loads(messages[-1]["content"])
        if request.get("task", "").startswith("Select a diverse"):
            return ModelTurn(text=json.dumps({
                "shortlist": [
                    {"paper_id": item["paper_id"], "role": "competing_method", "reason": "Relevant."}
                    for item in request["papers"][:3]
                ]
            }))
        return ModelTurn(text="{}")


def test_idea_library_has_fixed_implementable_ideas() -> None:
    entry = load_idea("config/idea_library.json", "vit_peft_small_data")
    assert entry["difficulty"] == "beginner_to_intermediate"
    assert len(entry["subproblems"]) == 3
    assert len(entry["search_queries"]) >= 10


def test_l1_fan_in_consumes_thirty_records(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    store = ResearchStore(workspace / "researchops.db")
    model = BatchModel()
    pipeline = BatchResearchPipeline(
        model=model,
        store=store,
        literature=LiteratureSearch(store),
        ingestor=DocumentIngestor(store),
        workspace=workspace,
        traces=TraceStore(workspace / "runs"),
    )
    idea = Idea(
        "PEFT", Domain.AI_ML, "Efficient adaptation.",
        "Can adapters reduce trainable parameters?", ("Adapters are efficient.",), ("Compare accuracy.",),
    )
    papers = [
        Paper(f"paper_{index}", f"Adapter paper {index}", abstract="Vision adapter method.", pdf_url=f"https://x.test/{index}.pdf")
        for index in range(30)
    ]
    for paper in papers:
        store.upsert_paper(paper)
    l1 = [pipeline.compressor.l1(paper, idea) for paper in papers]

    result = pipeline._fan_in(idea, papers, l1, 3)

    assert result["input_count"] == 30
    assert result["shortlist_count"] == 3
    assert len({item["paper_id"] for item in result["shortlist"]}) == 3


def test_l1_fan_in_selects_only_verified_full_text(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    store = ResearchStore(workspace / "researchops.db")
    pipeline = BatchResearchPipeline(
        model=BatchModel(),
        store=store,
        literature=LiteratureSearch(store),
        ingestor=DocumentIngestor(store),
        workspace=workspace,
        traces=TraceStore(workspace / "runs"),
    )
    idea = Idea(
        "PEFT", Domain.AI_ML, "Efficient adaptation.",
        "Can adapters reduce trainable parameters?", ("Adapters are efficient.",), ("Compare accuracy.",),
    )
    papers = [
        Paper(f"paper_{index}", f"Adapter paper {index}", abstract="Vision adapter method.")
        for index in range(30)
    ]
    for paper in papers:
        store.upsert_paper(paper)
    l1 = [pipeline.compressor.l1(paper, idea) for paper in papers]
    eligible = {"paper_10", "paper_11", "paper_12"}

    result = pipeline._fan_in(idea, papers, l1, 3, eligible_ids=eligible)

    assert {item["paper_id"] for item in result["shortlist"]} == eligible


def test_dag_only_deep_analyzes_shortlist(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    store = ResearchStore(workspace / "researchops.db")
    idea = Idea("Test", Domain.AI_ML, "Test.", "Question?", ("Claim.",), ("Metric.",))
    task_id = "task_batch"
    store.create_task(task_id, idea.title, idea.domain.value, {"title": idea.title})
    papers = [Paper("p1", "First"), Paper("p2", "Second")]
    for paper in papers:
        store.upsert_paper(paper)
        store.link_task_paper(task_id, paper.paper_id, role="screened", acquired_status="pending")
    model = BatchModel()
    dag = AcquisitionDAG(
        store=store,
        ingestor=DocumentIngestor(store),
        analyzer=PaperAnalyzer(model, store, workspace),
        compressor=PaperCompressor(store, model),
        workspace=workspace,
        traces=TraceStore(workspace / "runs"),
        downloader=lambda *args: {},
        source_downloader=lambda *args: {},
        max_workers=2,
    )

    acquisitions, summary = dag.run(
        task_id, idea, [(paper, "screened") for paper in papers],
        "run_batch", "session_batch", deep_paper_ids={"p1"},
    )

    by_id = {item["paper_id"]: item for item in acquisitions}
    assert summary["deep_paper_count"] == 1
    assert len(by_id["p1"]["dag_stages"]) == 5
    assert len(by_id["p2"]["dag_stages"]) == 3
    assert by_id["p2"]["analysis_status"] == "not_selected_for_l2"


def test_provisional_judgments_are_not_reported_as_human_gold() -> None:
    judgments = [
        {"paper_id": f"p{index}", "relevance": 3 if index < 5 else 0,
         "review_status": "pending_human"}
        for index in range(40)
    ]
    metrics = _ranking_metrics(
        judgments,
        [f"p{index}" for index in range(30)],
        ["p0", "p1", "p10"],
    )
    assert metrics["relevant_count"] == 5
    assert metrics["shortlist_precision"] == 0.6667
    assert "Provisional" in metrics["note"]

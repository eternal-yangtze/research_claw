import json
from pathlib import Path

import pytest

from researchops.models import Chunk, Paper
from researchops.runtime import ModelTurn
from researchops.silver_evaluation import (
    GoldReviewManager,
    SilverEvaluationPipeline,
    _select_fulltext_evidence,
    silver_metrics,
)
from researchops.storage import ResearchStore


class SilverModel:
    model = "silver-test-model"

    def complete(self, messages, tools, max_tokens):
        request = json.loads(messages[-1]["content"])
        nodes = request["paper"]["evidence_nodes"]
        citations = [nodes[0]["chunk_id"]] if nodes else ["abstract"]
        return ModelTurn(
            text=json.dumps({
                "relevance": 2,
                "role": "method",
                "confidence": 0.85,
                "reason": "Useful related method with a task mismatch.",
                "counterargument": "The exact proposed task is not evaluated.",
                "citations": citations,
            }),
            usage={"prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140},
        )


def _idea() -> dict:
    return {
        "idea_id": "test_idea",
        "title": "Efficient vision adaptation",
        "research_question": "Can adapters improve small-data classification?",
        "atomic_claims": ["Adapters reduce trainable parameters."],
        "success_criteria": ["Compare accuracy and memory."],
    }


def test_fulltext_selection_covers_method_and_experiment() -> None:
    rows = [
        {
            "chunk_id": "method", "section": "Method", "page": 2,
            "node_type": "paragraph", "text": "Our adapter training objective minimizes classification loss.",
        },
        {
            "chunk_id": "experiment", "section": "Experiments", "page": 5,
            "node_type": "table", "text": "Dataset split baseline accuracy memory result comparison.",
        },
        {
            "chunk_id": "unrelated", "section": "References", "page": 9,
            "node_type": "paragraph", "text": "Citation list.",
        },
    ]

    selected = _select_fulltext_evidence(rows, _idea(), 2000)

    ids = {item["chunk_id"] for item in selected}
    assert {"method", "experiment"} <= ids
    assert any(item["node_type"] == "table" for item in selected)


def test_silver_pipeline_produces_thirty_labels_and_reserved_human_template(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    store = ResearchStore(workspace / "researchops.db")
    provisional = []
    acquisitions = []
    for index in range(30):
        paper_id = f"paper_{index}"
        paper = Paper(
            paper_id,
            f"Adapter paper {index}",
            abstract="A parameter-efficient vision adaptation method.",
        )
        store.upsert_paper(paper)
        store.replace_chunks(paper_id, [Chunk(
            f"chunk_{index}", paper_id, "Method and Experiments",
            "The adapter is evaluated on an image dataset with accuracy and memory metrics.",
            page=2,
        )])
        provisional.append({
            "idea_id": "test_idea", "paper_id": paper_id, "title": paper.title,
            "relevance": 1, "role": "weak", "reason": "Initial label.",
        })
        acquisitions.append({"paper_id": paper_id})
    provisional_path = workspace / "provisional.jsonl"
    provisional_path.write_text(
        "".join(json.dumps(item) + "\n" for item in provisional), encoding="utf-8"
    )
    report_path = workspace / "batch.json"
    report_path.write_text(json.dumps({
        "acquisitions": acquisitions,
        "fan_in": {"shortlist": [{"paper_id": f"paper_{index}"} for index in range(3)]},
    }), encoding="utf-8")
    pipeline = SilverEvaluationPipeline(
        model=SilverModel(), store=store, workspace=workspace, max_workers=3
    )

    report = pipeline.run(
        _idea(),
        provisional_path=provisional_path,
        batch_report_path=report_path,
        max_papers=30,
        max_fulltext_chars=2000,
    )

    assert report["completed_count"] == 30
    assert report["failed_count"] == 0
    assert report["full_text_available_count"] == 30
    assert report["valid_chunk_citation_count"] == 30
    assert report["usage"]["total_tokens"] == 4200
    assert Path(report["judgments_path"]).is_file()
    review_path = Path(report["human_review_template_path"])
    assert review_path.is_file()
    with pytest.raises(ValueError, match="human-verified"):
        GoldReviewManager.validate_verified(review_path)


def test_silver_metrics_report_agreement_without_claiming_gold() -> None:
    records = [
        {
            "paper_id": f"p{index}", "relevance": 3 if index < 3 else 0,
            "confidence": 0.9, "citations": ["abstract"],
            "previous_provisional": {"relevance": 3 if index < 2 else 0},
        }
        for index in range(10)
    ]

    metrics = silver_metrics(records, [f"p{i}" for i in range(10)], ["p0", "p5"])

    assert metrics["ndcg_at_10"] == 1.0
    assert metrics["shortlist_precision"] == 0.5
    assert metrics["provisional_exact_agreement"] == 0.9
    assert "not human-gold" in metrics["note"]


def test_evaluation_pool_prioritizes_screened_fulltext_corpus(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    store = ResearchStore(workspace / "researchops.db")
    model = SilverModel()
    pipeline = SilverEvaluationPipeline(
        model=model, store=store, workspace=workspace
    )
    for index in range(35):
        store.upsert_paper(Paper(f"p{index}", f"Paper {index}"))
    provisional = [
        {"paper_id": f"p{index}", "title": f"Paper {index}", "relevance": 0}
        for index in range(10, 35)
    ]

    pool = pipeline._evaluation_pool(
        provisional, [f"p{index}" for index in range(30)], 35
    )

    assert [item["paper_id"] for item in pool[:30]] == [f"p{index}" for index in range(30)]
    assert len(pool) == 35

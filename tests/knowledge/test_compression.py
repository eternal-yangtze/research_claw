import json
from pathlib import Path

from researchops.compression import PaperCompressor
from researchops.models import Chunk, Domain, Idea, Paper
from researchops.runtime import ModelTurn
from researchops.storage import ResearchStore


class CompressionModel:
    model = "test-model"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages, tools, max_tokens):
        self.calls += 1
        request = json.loads(messages[-1]["content"])
        node_id = request["nodes"][0]["node_id"]
        return ModelTurn(text=json.dumps({
            "answer": "The cited equation defines the objective.",
            "evidence": [{"node_id": node_id, "use": "objective"}],
            "formulas": [{"node_id": node_id, "latex": "L=x^2"}],
            "tables": [],
            "unknowns": [],
        }))


def test_l0_l1_l2_l3_are_versioned_and_cached(tmp_path: Path) -> None:
    store = ResearchStore(tmp_path / "research.db")
    paper = Paper(
        "paper_levels", "Graph Anomaly Detection", abstract="A graph method detects time series anomalies.",
        source="local", content_hash="pdf-hash",
    )
    store.upsert_paper(paper)
    formula = Chunk(
        "node_formula", paper.paper_id, "equation", "L=x^2", page=2,
        content_hash="formula-hash", node_type="formula",
        provenance={"parser": "docling", "page": 2}, source_hash="pdf-hash",
    )
    table = Chunk(
        "node_table", paper.paper_id, "table", "| model | F1 |", page=4,
        content_hash="table-hash", node_type="table",
        provenance={"parser": "docling", "page": 4}, source_hash="pdf-hash",
    )
    store.replace_chunks(paper.paper_id, [formula, table])
    idea = Idea(
        "Graph anomalies", Domain.TIME_SERIES, "Use graphs.",
        "Can graph structure improve time series anomaly detection?",
        ("Graph structure improves detection.",), ("Compare F1.",),
    )
    analysis_path = tmp_path / "analysis.json"
    analysis_path.write_text(json.dumps({
        "paper_id": paper.paper_id,
        "paper_title": paper.title,
        "paper_summary": {"text": "A method.", "citations": ["node_formula"]},
        "citation_index": {"node_formula": {"page": 2}},
    }), encoding="utf-8")
    model = CompressionModel()
    compressor = PaperCompressor(store, model)

    l0 = compressor.l0(paper)
    l1 = compressor.l1(paper, idea)
    l2 = compressor.l2_from_analysis(paper, analysis_path)
    l3 = compressor.l3(paper, "What is the training objective?")
    cached = compressor.l3(paper, "What is the training objective?")

    assert [l0["level"], l1["level"], l2["level"], l3["level"]] == ["L0", "L1", "L2", "L3"]
    assert all(item["prompt_hash"] and item["source_hash"] for item in (l0, l1, l2, l3))
    assert l2["content"]["formula_nodes"][0]["node_id"] == "node_formula"
    assert l2["content"]["table_nodes"][0]["node_id"] == "node_table"
    assert cached["cache_hit"] is True
    assert model.calls == 1
    assert store.counts()["paper_compressions"] == 4
    assert [row["level"] for row in store.paper_compressions(paper.paper_id)] == ["L0", "L1", "L2", "L3"]
    assert len(store.paper_compressions(paper.paper_id, "L2")) == 1


def test_chunk_node_provenance_round_trips_through_storage(tmp_path: Path) -> None:
    store = ResearchStore(tmp_path / "research.db")
    paper = Paper("paper_nodes", "Node Paper")
    store.upsert_paper(paper)
    chunk = Chunk(
        "node_1", paper.paper_id, "Methods", "x = y + 1", node_type="formula",
        parent_chunk_id="heading_1", provenance={"page": 3, "bbox": [1, 2, 3, 4]},
        source_hash="source-123",
    )
    store.replace_chunks(paper.paper_id, [chunk])

    row = store.paper_chunks(paper.paper_id)[0]
    assert row["node_type"] == "formula"
    assert row["parent_chunk_id"] == "heading_1"
    assert json.loads(row["provenance_json"])["page"] == 3
    assert row["source_hash"] == "source-123"

from pathlib import Path

from researchops.core.models import Paper, Chunk
from researchops.core.storage import ResearchStore
from researchops.knowledge.evidence_retrieval import EvidenceRetriever
from researchops.knowledge.retrieval import HashingEmbeddingBackend, LexicalReranker
from researchops.experiments.evidence_design import build_evidence_packet
from researchops.evaluation.references import load_reference


def test_incremental_and_text_correction_invalidate_only_changed_vectors(tmp_path: Path):
    store = ResearchStore(tmp_path / "corpus.db")
    class Embedding(HashingEmbeddingBackend):
        def __init__(self):
            self.count = 0
        def encode(self, texts):
            self.count += len(texts)
            return super().encode(texts)
    embedding = Embedding()
    store.upsert_paper(Paper("p1", "Graph methods"))
    store.replace_chunks("p1", [Chunk("c1", "p1", "Method", "graph anomaly mechanism", source_hash="same-pdf")])
    retriever = EvidenceRetriever(store, embedding=embedding, reranker=LexicalReranker())
    retriever.search("graph anomaly")
    assert retriever.last_diagnostics["refresh"]["encoded_passages"] == 1
    first = embedding.count
    retriever.search("graph anomaly")
    assert embedding.count == first + 1
    assert retriever.last_diagnostics["reranked_pairs"] == 0
    store.upsert_paper(Paper("p2", "New graph approach"))
    store.replace_chunks("p2", [Chunk("c2", "p2", "Method", "graph anomaly detection controls", source_hash="new-pdf")])
    retriever.search("graph anomaly")
    assert retriever.last_diagnostics["refresh"]["encoded_passages"] == 1
    assert retriever.last_diagnostics["refresh"]["reused_passages"] == 1
    store.replace_chunks("p1", [Chunk("c1", "p1", "Method", "corrected graph mechanism text", source_hash="same-pdf")])
    retriever.search("graph anomaly")
    assert retriever.last_diagnostics["refresh"]["encoded_passages"] == 1
    assert retriever.last_diagnostics["refresh"]["reused_passages"] == 1


def test_snapshot_filter_and_paper_diversity_survive_new_papers(tmp_path: Path):
    store = ResearchStore(tmp_path / "corpus.db")
    for index, count in [(1, 6), (2, 1), (3, 1)]:
        pid = f"p{index}"
        store.upsert_paper(Paper(pid, "graph methods"))
        store.replace_chunks(pid, [Chunk(f"c{index}_{n}", pid, "Method", f"graph anomaly mechanism {n}") for n in range(count)])
    retriever = EvidenceRetriever(store, embedding=HashingEmbeddingBackend(), reranker=LexicalReranker())
    hits = retriever.search("graph anomaly", limit=3)
    assert len({h.paper_id for h in hits}) == 3
    restricted = retriever.search("graph anomaly", paper_ids={"p2"}, limit=3)
    assert {h.paper_id for h in restricted} == {"p2"}


def test_quote_cards_reject_invented_quotes_and_preserve_fallback():
    text = "The frozen vision backbone uses low rank updates while the classifier is trained on a fixed split."
    hit = {"chunk_id": "c1", "paper_id": "p1", "paper_title": "Paper", "text": text}
    class Calls:
        def call(self, *args):
            return {"cards": [{"chunk_id": "c1", "quote": "Invented evidence " * 4,
                                "role": "empirical", "relevance": 3}], "gaps": []}
    result = build_evidence_packet("question", {}, [hit], Calls())
    assert len(result["rejected_cards"]) == 1
    assert result["cards"] == []
    assert result["passages"][0]["text"] == text


def test_gold_interface_reserved_while_silver_is_primary(tmp_path: Path):
    import pytest
    reference = tmp_path / "silver.json"
    reference.write_text("{}", encoding="utf-8")
    _, info = load_reference(silver=reference)
    assert info["role"] == "temporary_primary_reference"
    assert not info["human_gold"] and info["gold_interface"] is None
    with pytest.raises(ValueError, match="reserved"):
        load_reference(silver=reference, source="gold")

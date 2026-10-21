from pathlib import Path

from researchops.ingestion import DocumentIngestor
from researchops.novelty import EvidenceAnswerer
from researchops.retrieval import HybridRetriever
from researchops.storage import ResearchStore


def test_ingest_search_and_citation(tmp_path: Path) -> None:
    document = tmp_path / "paper.md"
    document.write_text(
        """# Paper Metadata
Title: Deterministic Bounds for Compressed Time Series
Authors: A. Researcher
Year: 2025
DOI (if available): 10.1000/example

# Core Idea
Segment residual norms provide deterministic error guarantees for range sum queries.
""",
        encoding="utf-8",
    )
    store = ResearchStore(tmp_path / "research.db")
    result = DocumentIngestor(store).ingest_file(document)
    assert result.chunks == 2

    retriever = HybridRetriever(store)
    hits = retriever.search("deterministic error guarantees range sum", limit=3)
    assert hits
    assert hits[0].paper_id == result.paper_id
    assert hits[0].locator.startswith("lines ")

    answer = EvidenceAnswerer(store, retriever).answer("How are range sum errors bounded?")
    assert answer["verified"] is True
    assert answer["unsupported_claims"] == 0
    assert answer["citations"][0]["citation"] in answer["answer"]


def test_no_evidence_refuses_answer(tmp_path: Path) -> None:
    store = ResearchStore(tmp_path / "empty.db")
    answer = EvidenceAnswerer(store, HybridRetriever(store)).answer("unknown")
    assert "证据不足" in answer["answer"]
    assert answer["citations"] == []


class FakeNeuralEmbedding:
    model_name = "fake-neural"
    model_revision = "r1"

    def __init__(self) -> None:
        self.encoded_texts = 0

    def encode(self, texts):
        self.encoded_texts += len(texts)
        output = []
        for text in texts:
            lowered = text.lower()
            vector = (
                float("graph" in lowered),
                float("anomaly" in lowered),
                float("compression" in lowered),
            )
            norm = sum(value * value for value in vector) ** 0.5 or 1.0
            output.append(tuple(value / norm for value in vector))
        return output


class FakeCrossEncoder:
    model_name = "fake-cross-encoder"

    def __init__(self) -> None:
        self.calls = 0

    def score(self, query, documents):
        self.calls += 1
        return [3.0 if "graph anomaly" in document.lower() else -2.0 for document in documents]


def test_neural_dense_reranker_mmr_and_embedding_cache(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "graph.md").write_text(
        "# Method\nGraph anomaly detection uses learned relations.\n\n# Duplicate\n"
        "Graph anomaly detection uses learned relations again.",
        encoding="utf-8",
    )
    (corpus / "compression.md").write_text(
        "# Method\nCompression preserves deterministic query error bounds.",
        encoding="utf-8",
    )
    store = ResearchStore(tmp_path / "research.db")
    DocumentIngestor(store).ingest_path(corpus)
    embedding = FakeNeuralEmbedding()
    reranker = FakeCrossEncoder()
    retriever = HybridRetriever(store, embedding=embedding, reranker=reranker, mmr_lambda=0.6)

    first = retriever.search("graph anomaly", limit=3)
    encoded_after_first = embedding.encoded_texts
    second = retriever.search("graph anomaly", limit=3)

    assert first[0].rerank_score > first[-1].rerank_score
    assert first[0].embedding_model == "fake-neural@r1"
    assert first[0].reranker_model == "fake-cross-encoder"
    assert all(hit.mmr_score <= hit.score for hit in first[1:])
    assert embedding.encoded_texts == encoded_after_first + 1  # only the second query, not documents
    assert [hit.chunk_id for hit in first] == [hit.chunk_id for hit in second]
    assert store.counts()["chunk_embeddings"] == store.counts()["chunks"]

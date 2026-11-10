from pathlib import Path
import uuid

import pytest

from researchclaw.core.models import Chunk, Paper
from researchclaw.core.storage import ResearchStore
from researchclaw.knowledge.retrieval import HybridRetriever, LexicalReranker
from researchclaw.knowledge.vector_backend import QdrantVectorBackend


class TinyEmbedding:
    model_name = "tiny-neural-test"
    model_revision = "r1"

    def encode(self, texts):
        values = []
        for text in texts:
            lowered = text.lower()
            vector = [
                float("graph" in lowered),
                float("vision" in lowered),
                float("adapter" in lowered),
            ]
            norm = sum(value * value for value in vector) ** 0.5 or 1.0
            values.append(tuple(value / norm for value in vector))
        return values


def _store(tmp_path: Path):
    store = ResearchStore(tmp_path / "corpus.db")
    store.upsert_paper(Paper("p1", "Graph Paper"))
    store.upsert_paper(Paper("p2", "Vision Paper"))
    store.replace_chunks("p1", [
        Chunk("c1", "p1", "Method", "graph relations", source_hash="h1"),
    ])
    store.replace_chunks("p2", [
        Chunk("c2", "p2", "Method", "vision adapters", source_hash="h2"),
    ])
    return store


def test_qdrant_sync_filter_update_delete_and_alias(tmp_path: Path):
    qdrant = pytest.importorskip("qdrant_client")
    client = qdrant.QdrantClient(location=":memory:")
    store = _store(tmp_path)
    embedding = TinyEmbedding()
    backend = QdrantVectorBackend(
        store,
        embedding,
        client=client,
        alias="test_chunks",
        project_id="project-test",
        corpus_version="corpus-v1",
    )
    retriever = HybridRetriever(
        store,
        embedding=embedding,
        reranker=LexicalReranker(),
        vector_backend=backend,
    )

    hits = retriever.search("graph", mode="dense", paper_ids={"p1"})
    assert [hit.chunk_id for hit in hits] == ["c1"]
    first = backend.last_status
    assert first.status == "ready" and first.indexed_points == 2
    assert first.upserted_points == 2 and first.deleted_points == 0
    assert client.collection_exists(first.version_collection)
    aliases = {item.alias_name: item.collection_name for item in client.get_aliases().aliases}
    assert aliases["test_chunks"] == first.version_collection

    candidate = QdrantVectorBackend(
        store,
        embedding,
        client=client,
        alias="test_chunks",
        project_id="project-test",
        corpus_version="corpus-v2",
    )
    candidate_retriever = HybridRetriever(
        store,
        embedding=embedding,
        reranker=LexicalReranker(),
        vector_backend=candidate,
    )
    candidate_retriever.search("graph", mode="dense")
    assert candidate.last_status.version_collection != first.version_collection
    aliases = {item.alias_name: item.collection_name for item in client.get_aliases().aliases}
    assert aliases["test_chunks"] == candidate.last_status.version_collection
    retriever.search("graph", mode="dense")
    aliases = {item.alias_name: item.collection_name for item in client.get_aliases().aliases}
    assert aliases["test_chunks"] == first.version_collection

    records, _ = client.scroll(first.version_collection, limit=10, with_payload=True)
    assert len(records) == 2
    assert all(uuid.UUID(str(record.id)).version == 5 for record in records)
    assert {record.payload["paper_id"] for record in records} == {"p1", "p2"}

    retriever.search("graph", mode="dense", paper_ids={"p1"})
    assert backend.last_status.upserted_points == 0
    store.replace_chunks("p1", [
        Chunk("c1", "p1", "Method", "graph adapter relations", source_hash="h1"),
    ])
    store.replace_chunks("p2", [])
    retriever.search("graph adapter", mode="dense")
    assert backend.last_status.upserted_points == 1
    assert backend.last_status.deleted_points == 1
    assert backend.sync_state()["status"] == "ready"
    store.replace_chunks("p1", [])
    assert retriever.search("graph", mode="dense") == []
    assert backend.last_status.indexed_points == 0
    assert backend.last_status.deleted_points == 1


def test_incremental_revision_streams_old_vectors_and_only_uploads_new_chunks(tmp_path: Path):
    qdrant = pytest.importorskip("qdrant_client")
    client = qdrant.QdrantClient(location=":memory:")
    store = _store(tmp_path)
    embedding = TinyEmbedding()
    rows = [dict(row) for row in store.all_chunks()]
    first_vectors = dict(zip(
        [row["chunk_id"] for row in rows],
        embedding.encode([row["text"] for row in rows]),
    ))
    first = QdrantVectorBackend(
        store, embedding, client=client, alias="incremental",
        project_id="project-incremental", corpus_version="v1",
    )
    first_status = first.sync(rows, first_vectors)

    store.upsert_paper(Paper("p3", "Adapter Paper"))
    store.replace_chunks("p3", [Chunk("c3", "p3", "Method", "adapter method", source_hash="h3")])
    next_rows = [dict(row) for row in store.all_chunks()]
    second = QdrantVectorBackend(
        store, embedding, client=client, alias="incremental",
        project_id="project-incremental", corpus_version="v2",
    )
    second_status = second.sync_incremental(
        next_rows, {"c3": embedding.encode(["adapter method"])[0]},
        reuse_collection=first_status.version_collection,
    )

    assert second_status.status == "ready"
    assert second_status.indexed_points == 3
    assert second_status.upserted_points == 1
    assert client.collection_exists(first_status.version_collection)
    records, _ = client.scroll(second_status.version_collection, limit=10, with_payload=True)
    assert {record.payload["chunk_id"] for record in records} == {"c1", "c2", "c3"}
    assert all(record.payload["corpus_version"] == "v2" for record in records)
    reused = [record for record in records if record.payload["chunk_id"] in {"c1", "c2"}]
    assert all(record.payload["reused_from_collection"] == first_status.version_collection
               for record in reused)
    aliases = {item.alias_name: item.collection_name for item in client.get_aliases().aliases}
    assert aliases["incremental"] == second_status.version_collection
    assert second.sync_state()["status"] == "ready"


def test_configured_vector_backend_failure_is_not_silently_local(tmp_path: Path):
    class BrokenBackend:
        name = "qdrant"
        last_status = None

        def search(self, *args, **kwargs):
            raise ConnectionError("qdrant unavailable")

    store = _store(tmp_path)
    retriever = HybridRetriever(
        store,
        embedding=TinyEmbedding(),
        reranker=LexicalReranker(),
        vector_backend=BrokenBackend(),
    )
    with pytest.raises(ConnectionError, match="qdrant unavailable"):
        retriever.search("graph", mode="dense")


def test_qdrant_incompatible_collection_is_audited_as_failed(tmp_path: Path):
    qdrant = pytest.importorskip("qdrant_client")
    from qdrant_client.http import models
    client = qdrant.QdrantClient(location=":memory:")
    store = _store(tmp_path)
    embedding = TinyEmbedding()
    backend = QdrantVectorBackend(store, embedding, client=client, alias="mismatch",
                                  project_id="bound-project")
    client.create_collection(backend._version_collection(3),
        vectors_config=models.VectorParams(size=3, distance=models.Distance.COSINE),
        metadata={"embedding_key": "wrong-model@v1"})
    rows = [dict(row) for row in store.all_chunks()]
    vectors = dict(zip([r["chunk_id"] for r in rows], embedding.encode([r["text"] for r in rows])))
    with pytest.raises(ValueError, match="metadata mismatch"):
        backend.sync(rows, vectors)
    assert backend.sync_state()["status"] == "failed"
    assert "metadata mismatch" in backend.sync_state()["error"]

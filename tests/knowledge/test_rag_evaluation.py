import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from researchops.core.models import Chunk, Paper
from researchops.core.storage import ResearchStore
from researchops.evaluation.rag import ConcurrentPaidCalls, ranking_metrics
from researchops.knowledge.passages import materialize_passages
from researchops.knowledge.retrieval import HashingEmbeddingBackend, HybridRetriever, LexicalReranker


def test_metrics_missing_results_dedup_and_no_relevant():
    result = ranking_metrics(["noise", "a", "a", "b"], {"a": 2, "b": 1, "c": 1}, 3)
    assert result["recall"] == pytest.approx(2 / 3)
    assert result["mrr"] == 0.5
    assert 0 < result["ndcg"] < 1
    assert ranking_metrics([], {"a": 1}, 10)["recall"] == 0
    assert ranking_metrics(["a"], {"a": 0}, 10)["recall"] is None


def test_passage_index_provenance_increment_and_changed_text(tmp_path):
    source, target = ResearchStore(tmp_path / "raw.db"), ResearchStore(tmp_path / "index.db")
    source.upsert_paper(Paper(paper_id="a", title="Visual adaptation", content_hash="pdf1"))
    source.replace_chunks("a", [Chunk(chunk_id=f"a{i}", paper_id="a", section="Page 1", page=1,
                                      text=f"line {i} about visual adaptation", source_hash="pdf1") for i in range(8)])
    materialize_passages(source, target, ["a"])
    first = target.all_chunks()
    assert len(first) < 8
    assert json.loads(first[0]["provenance_json"])["source_chunk_ids"] == [f"a{i}" for i in range(8)]
    assert materialize_passages(source, target, ["a"])["written_passages"] == 0
    source.replace_chunks("a", [Chunk(chunk_id="a0", paper_id="a", section="Page 1", page=1,
                                      text="changed content with the same PDF source hash", source_hash="pdf1")])
    materialize_passages(source, target, ["a"])
    assert target.all_chunks()[0]["chunk_id"] != first[0]["chunk_id"]
    assert "changed" in target.all_chunks()[0]["text"]
    source.upsert_paper(Paper(paper_id="b", title="New evidence", content_hash="pdf2"))
    source.replace_chunks("b", [Chunk(chunk_id="b0", paper_id="b", section="Method", text="new evidence")])
    result = materialize_passages(source, target, ["a", "b"])
    assert result["written_passages"] == 1
    assert result["unchanged_passages"] == 1


def test_retrieval_ablations_do_not_call_disabled_models(tmp_path):
    store = ResearchStore(tmp_path / "corpus.db")
    store.upsert_paper(Paper(paper_id="a", title="Visual adaptation"))
    store.replace_chunks("a", [Chunk(chunk_id="c", paper_id="a", section="Method", text="visual adaptation")])
    class NoModel:
        model_name, model_revision = "unused", "v1"
        def encode(self, texts):
            raise AssertionError("disabled embedding used")
        def score(self, query, docs):
            raise AssertionError("disabled reranker used")
    retriever = HybridRetriever(store, embedding=NoModel(), reranker=NoModel())
    assert retriever.search("visual", mode="bm25")[0].embedding_model == "none"
    retriever = HybridRetriever(store, embedding=HashingEmbeddingBackend(), reranker=NoModel())
    assert retriever.search("visual", mode="dense")
    assert retriever.search("visual", mode="hybrid")
    with pytest.raises(ValueError):
        retriever.search("visual", mode="unknown")


def test_concurrent_ledger_and_resume_do_not_duplicate_charges(tmp_path):
    class Model:
        model = "unit-test"
        def complete_bounded(self, messages, tools, output_tokens, timeout):
            return SimpleNamespace(text='{"ok":true}', usage={"total_tokens": 17})
    calls = ConcurrentPaidCalls(tmp_path, Model(), 100000)
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert all(pool.map(lambda i: calls.call("test", "return JSON", {"id": i}, 100), range(8)))
    assert calls.state["total_tokens"] == 8 * 17
    assert calls.state["turns"] == 8
    resumed = ConcurrentPaidCalls(tmp_path, Model(), 100000)
    assert resumed.call("test", "return JSON", {"id": 0}, 100)["ok"]
    assert resumed.state["total_tokens"] == 8 * 17
    assert resumed.state["turns"] == 8

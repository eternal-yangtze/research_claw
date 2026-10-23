"""Read-only snapshot of historical papers; verify Qdrant across real consumers."""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from pathlib import Path

from researchops.core.artifacts import canonical_hash, file_hash, progress, write_json
from researchops.core.storage import ResearchStore
from researchops.knowledge.evidence_retrieval import EvidenceRetriever, checkpoint_key
from researchops.knowledge.fusion_retrieval import FusionRetriever
from researchops.knowledge.retrieval import (
    CrossEncoderReranker, HybridRetriever, SentenceTransformerEmbedding, _row_source_hash,
)
from researchops.knowledge.vector_backend import LocalExactVectorBackend, QdrantVectorBackend


class SourceOnlyFusionControl:
    """Deterministic protocol control; this is not an LLM quality evaluation."""
    def call(self, label, system, payload, max_tokens):
        result = []
        for paper in payload["papers"]:
            passage = paper["passages"][0]
            quote = passage["text"][:100]
            result.append({"paper_id": paper["paper_id"], "utility": 1,
                           "chunk_id": passage["chunk_id"], "quote": quote,
                           "reason": "protocol verification only"})
        return {"papers": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("workspace_rag_cpu_v3/passages.db"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["RESEARCHOPS_EMBEDDING_LOCAL_ONLY"] = "true"
    os.environ["RESEARCHOPS_RERANKER_LOCAL_ONLY"] = "true"
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    torch.set_num_threads(4)
    root, source = args.output.resolve(), args.source.resolve()
    root.mkdir(parents=True, exist_ok=False)
    source_hash = file_hash(source)
    target = root / "passages.db"
    with sqlite3.connect(f"{source.as_uri()}?mode=ro", uri=True) as src:
        with sqlite3.connect(target) as dst:
            src.backup(dst)
    store = ResearchStore(target)
    embedding = SentenceTransformerEmbedding()
    resolved = checkpoint_key(embedding).rsplit("@", 1)[-1]
    embedding.model_revision = resolved
    reranker = CrossEncoderReranker()
    local_evidence = EvidenceRetriever(store, embedding=embedding, reranker=reranker,
                                      vector_backend=LocalExactVectorBackend())
    refresh = local_evidence._refresh()
    rows, vectors = local_evidence.rows, local_evidence.vectors
    # EvidenceRetriever checks text+source+model hashes before reusing each
    # historical vector. Seed Hybrid's namespace only from those checked values.
    store.save_chunk_embeddings(embedding.model_name, embedding.model_revision, [
        (row["chunk_id"], _row_source_hash(row), vectors[row["chunk_id"]]) for row in rows
    ])
    write_json(root / "corpus_manifest.json", {
        "source": str(source), "source_sha256": source_hash,
        "rows": [{"chunk_id": row["chunk_id"], "paper_id": row["paper_id"],
                  "text_hash": canonical_hash([row["section"], row["text"]])} for row in rows],
        "embedding_checkpoint": checkpoint_key(embedding),
    })
    backend = QdrantVectorBackend(
        store, embedding, alias="researchops_v03_real_" + root.name.lower(),
        project_id="v03-real-" + root.name.lower(), corpus_version="p0-94papers-v1")
    started = time.perf_counter()
    initial = backend.sync(rows, vectors)
    initial_seconds = time.perf_counter() - started
    progress(f"Qdrant indexed {len(rows)} passages from {len({r['paper_id'] for r in rows})} papers")
    indexed = 0
    for attempt in range(50):
        info = backend.client.get_collection(initial.version_collection)
        indexed = info.indexed_vectors_count or 0
        if indexed > 0:
            break
        time.sleep(1)
    frozen = json.loads(Path("config/benchmarks/v03_regression_manifest.json").read_text(encoding="utf-8"))
    split = next(track for track in frozen["tracks"] if track["name"] == "rag")
    questions = json.loads(Path(split["source"]).read_text(encoding="utf-8"))
    all_queries = {q["id"]: q["text"] for idea in questions["ideas"] for q in idea["queries"]}
    exact = LocalExactVectorBackend()
    results = []
    # ef settings are specified in advance. Regression is evaluated once at
    # ef=64; development observations do not change questions or answers.
    for group, query_ids in (("development", split["development_ids"]),
                             ("regression", split["regression_ids"])):
        for qid in query_ids:
            query_vector = embedding.encode([all_queries[qid]])[0]
            start = time.perf_counter()
            expected = exact.search(query_vector, rows=rows, vectors=vectors, limit=10)
            exact_ms = (time.perf_counter() - start) * 1000
            for ef in ((32, 64) if group == "development" else (64,)):
                backend.search_hnsw_ef = ef
                start = time.perf_counter()
                actual = backend.search(query_vector, rows=rows, vectors=vectors, limit=10)
                wall_ms = (time.perf_counter() - start) * 1000
                results.append({"id": qid, "split": group, "hnsw_ef": ef,
                                "exact_ids": list(expected), "ann_ids": list(actual),
                                "recall_at_10": len(set(actual) & set(expected)) / len(expected),
                                "exact_ms": exact_ms, "qdrant_ms_including_sync": wall_ms})
            progress(f"Compared {group}/{qid}")
    backend.search_hnsw_ef = 64
    question = all_queries[split["development_ids"][0]]
    hybrid = HybridRetriever(store, embedding=embedding, reranker=reranker, vector_backend=backend)
    hybrid_hits = hybrid.search(question, limit=10, mode="full")
    evidence = EvidenceRetriever(store, embedding=embedding, reranker=reranker, vector_backend=backend)
    evidence_hits = evidence.search(question, limit=10)
    fusion = FusionRetriever(store, SourceOnlyFusionControl(), embedding=embedding,
                             reranker=reranker, vector_backend=backend)
    fusion_hits = fusion.search(question, limit=10)
    pid = rows[0]["paper_id"]
    filtered = hybrid.search(question, limit=10, mode="dense", paper_ids={pid})
    checks = {
        "historical_source_untouched": source_hash == file_hash(source),
        "frozen_corpus_size": len(rows) == 7757 and len({r["paper_id"] for r in rows}) == 94,
        "hnsw_vectors_indexed": indexed > 0,
        "regression_ann_recall": all(r["recall_at_10"] >= 0.9 for r in results if r["split"] == "regression"),
        "hybrid_qdrant": bool(hybrid_hits) and hybrid.last_diagnostics["vector_backend"] == "qdrant",
        "evidence_qdrant": bool(evidence_hits) and evidence.last_diagnostics["vector_backend"] == "qdrant",
        "fusion_qdrant_both_branches": bool(fusion_hits)
            and fusion.cpu.last_diagnostics["vector_backend"] == "qdrant"
            and fusion.legacy.last_diagnostics["vector_backend"] == "qdrant",
        "paper_filter": bool(filtered) and {h.paper_id for h in filtered} == {pid},
    }
    report = {
        "passed": all(checks.values()), "checks": checks, "paid_model_calls": 0,
        "source": str(source), "source_sha256": source_hash,
        "papers": 94, "passages": len(rows), "embedding": checkpoint_key(embedding),
        "reranker": reranker.model_name, "initial_sync_seconds": initial_seconds,
        "initial_sync": initial.__dict__, "indexed_vectors_count": indexed,
        "refresh": refresh, "ann_comparison": results,
        "summary": {f"{group}_ef{ef}": {
            "queries": len(selected := [r for r in results if r["split"] == group and r["hnsw_ef"] == ef]),
            "mean_recall_at_10": sum(r["recall_at_10"] for r in selected) / len(selected),
            "mean_qdrant_ms_including_sync": sum(r["qdrant_ms_including_sync"] for r in selected) / len(selected),
        } for group, ef in (("development", 32), ("development", 64), ("regression", 64))},
        "consumers": {"hybrid": hybrid.last_diagnostics, "evidence": evidence.last_diagnostics,
                      "fusion": fusion.last_diagnostics},
        "query_audit": str(backend.audit_path),
        "limitations": ["Recall is ANN agreement with exact vectors, not semantic relevance accuracy.",
                        "Fusion uses a deterministic source-only control, not a paid LLM judge.",
                        "Latency includes consistency checks and sync; this is not a throughput benchmark."],
    }
    write_json(root / "report.json", report)
    progress(json.dumps({"passed": report["passed"], "summary": report["summary"],
                         "checks": checks, "report": str(root / "report.json")}, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

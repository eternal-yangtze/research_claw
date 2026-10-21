from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from researchops.ingestion import DocumentIngestor
from researchops.retrieval import HybridRetriever
from researchops.storage import ResearchStore


DOCUMENTS = {
    "graph_anomaly.md": """# Paper Metadata
Title: Graph-Constrained Multivariate Time-Series Anomaly Detection
Authors: Verification Author
Year: 2025

# Method
A learned dependency graph constrains multivariate time-series anomaly scores and identifies related sensors.

# Evaluation
The method reports F1, AUROC, localization accuracy, and an ablation without graph constraints.
""",
    "compression.md": """# Paper Metadata
Title: Deterministic Time-Series Compression
Authors: Verification Author
Year: 2024

# Method
Piecewise compression preserves deterministic error bounds for aggregate range queries.
""",
    "vision.md": """# Paper Metadata
Title: Image Classification with Convolutional Networks
Authors: Verification Author
Year: 2023

# Method
A convolutional image classifier is evaluated on object recognition benchmarks.
""",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default="workspace_neural_retrieval_smoke")
    parser.add_argument("--query", default="知识图谱约束的多变量时间序列异常检测")
    args = parser.parse_args()

    os.environ["RESEARCHOPS_RETRIEVAL_BACKEND"] = "neural"
    workspace = Path(args.workspace).resolve()
    corpus = workspace / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    for name, content in DOCUMENTS.items():
        (corpus / name).write_text(content, encoding="utf-8")

    store = ResearchStore(workspace / "researchops.db")
    DocumentIngestor(store).ingest_path(corpus)
    retriever = HybridRetriever(store)

    started = time.perf_counter()
    first = retriever.search(args.query, limit=4)
    first_seconds = time.perf_counter() - started
    embeddings_after_first = store.counts()["chunk_embeddings"]

    started = time.perf_counter()
    second = retriever.search(args.query, limit=4)
    second_seconds = time.perf_counter() - started
    embeddings_after_second = store.counts()["chunk_embeddings"]

    report = {
        "query": args.query,
        "embedding_model": first[0].embedding_model if first else None,
        "reranker_model": first[0].reranker_model if first else None,
        "chunk_count": store.counts()["chunks"],
        "embedding_count": embeddings_after_second,
        "first_search_seconds": round(first_seconds, 3),
        "cached_search_seconds": round(second_seconds, 3),
        "cache_reused": embeddings_after_first == embeddings_after_second,
        "stable_ranking": [hit.chunk_id for hit in first] == [hit.chunk_id for hit in second],
        "results": [hit.to_dict(include_text=False) for hit in second],
    }
    target = workspace / "artifacts" / "neural_retrieval_verification.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**report, "report_path": str(target)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

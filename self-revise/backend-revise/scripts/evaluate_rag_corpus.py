"""Evaluate a published RAG corpus with frozen query/relevant-paper labels."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(ROOT / "src"))

from researchclaw.core.qdrant_store import QdrantResearchStore
from researchclaw.knowledge.rag_retrieval import PublishedRAGRetriever
from researchclaw.knowledge.retrieval import SentenceTransformerEmbedding, CrossEncoderReranker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--corpus-version", required=True)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--embedding-revision", default="main")
    parser.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L6-v2")
    parser.add_argument("--reranker-revision", default="233902d25c440f23af6f7d6e94d2946bac0bee0a")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--candidate-limit", type=int, default=40)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--max-hits-per-paper", type=int, default=1)
    args = parser.parse_args()
    suite = json.loads(args.suite.read_text(encoding="utf-8"))
    store = QdrantResearchStore(args.workspace)
    embedding = SentenceTransformerEmbedding(args.embedding_model, revision=args.embedding_revision)
    reranker = CrossEncoderReranker(args.reranker_model, revision=args.reranker_revision)
    retriever = PublishedRAGRetriever(store, corpus_version=args.corpus_version,
                                      embedding=embedding, reranker=reranker,
                                      rrf_k=args.rrf_k,
                                      max_hits_per_paper=args.max_hits_per_paper)
    rows = []
    for case in suite["queries"]:
        hits = retriever.search(case["query"], limit=args.k,
                                candidate_limit=max(args.candidate_limit, args.k))
        ranked = []
        for hit in hits:
            if hit.paper_id not in ranked:
                ranked.append(hit.paper_id)
        relevant = set(case["relevant_paper_ids"])
        flags = [1 if value in relevant else 0 for value in ranked[:args.k]]
        recall = len(set(ranked[:args.k]) & relevant) / max(1, len(relevant))
        precision = sum(flags) / max(1, args.k)
        reciprocal = next((1 / index for index, flag in enumerate(flags, 1) if flag), 0.0)
        dcg = sum(flag / math.log2(index + 1) for index, flag in enumerate(flags, 1))
        ideal = sum(1 / math.log2(index + 1) for index in range(1, min(len(relevant), args.k) + 1))
        rows.append({"id": case["id"], "query": case["query"], "ranked_paper_ids": ranked,
                     f"recall@{args.k}": recall, f"precision@{args.k}": precision,
                     f"mrr@{args.k}": reciprocal, f"ndcg@{args.k}": dcg / ideal if ideal else 0.0})
    metrics = {key: sum(row[key] for row in rows) / len(rows) for key in
               (f"recall@{args.k}", f"precision@{args.k}", f"mrr@{args.k}", f"ndcg@{args.k}")}
    report = {"status": "passed", "suite": str(args.suite), "corpus_version": args.corpus_version,
              "embedding_model": args.embedding_model, "query_count": len(rows), "metrics": metrics,
              "candidate_limit": args.candidate_limit,
              "rrf_k": args.rrf_k, "max_hits_per_paper": args.max_hits_per_paper,
              "cases": rows, "scope": "Paper-level labeled retrieval; generation/RAGAS is a separate evaluation"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"status": "passed", **metrics}, ensure_ascii=False))


if __name__ == "__main__":
    main()

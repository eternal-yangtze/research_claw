"""Verify P2/P3 against Docker Qdrant and a deterministic paper MicroAgent."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from researchops.core.models import Chunk, Domain, Idea, Paper
from researchops.core.storage import ResearchStore
from researchops.harness.budget import BudgetedModel
from researchops.harness.runtime import ModelTurn, ToolCall, TraceStore
from researchops.knowledge.retrieval import (
    HybridRetriever,
    LexicalReranker,
    SentenceTransformerEmbedding,
)
from researchops.knowledge.vector_backend import (
    LocalExactVectorBackend,
    QdrantVectorBackend,
)
from researchops.papers.analysis import PaperAnalyzer, UNKNOWN


PROJECT_ID = "researchops-v03-p2-p3-verification"
CORPUS_VERSION = "v03-fixed-corpus-v1"
ALIAS = "researchops_v03_p2_verify"


def _candidate(citation=None):
    def unknown():
        return {"text": UNKNOWN, "citations": []}

    return {
        "paper_summary": {
            "text": "The paper models relational dependencies for anomaly detection.",
            "citations": [citation] if citation else [],
        },
        "research_problem": unknown(),
        "inputs_outputs": unknown(),
        "method_overview": unknown(),
        "architecture_components": [],
        "knowledge_or_graph_role": unknown(),
        "training_objective": unknown(),
        "inference_procedure": unknown(),
        "datasets": [],
        "data_splits_preprocessing": unknown(),
        "metrics": [],
        "paper_baselines": [],
        "reported_results": [],
        "implementation_availability": unknown(),
        "compute_requirements": unknown(),
        "limitations": [],
        "baseline_assessment": {
            "verdict": "insufficient_evidence",
            "rationale": unknown(),
            "task_alignment": unknown(),
            "input_output_compatibility": unknown(),
            "dataset_metric_comparability": unknown(),
            "reproduction_requirements": [],
            "adaptations_required": [],
        },
        "missing_information": ["Training details require additional evidence."],
    }


class ScriptedPaperModel:
    model = "v03-p3-deterministic-verifier"

    def __init__(self):
        self.calls = 0

    def complete_bounded(self, messages, tools, max_tokens, *, timeout):
        self.calls += 1
        usage = {"total_tokens": 20}
        if self.calls == 1:
            return ModelTurn(text=json.dumps(_candidate()), usage=usage)
        if self.calls == 2:
            feedback = json.loads(messages[-1]["content"])
            assert "missing_citation" in {item["code"] for item in feedback["issues"]}
            return ModelTurn(tool_calls=(ToolCall(
                "search-paper", "search_paper_chunks",
                {"query": "relational dependencies anomaly detection", "limit": 4},
            ),), usage=usage)
        if self.calls == 3:
            result = json.loads(messages[-1]["content"])
            assert result["status"] == "success"
            assert result["output"]["paper_id"] == "paper-graph"
            return ModelTurn(tool_calls=(ToolCall(
                "read-paper", "get_paper_chunk", {"chunk_id": "graph-anomaly"},
            ),), usage=usage)
        assert self.calls == 4
        result = json.loads(messages[-1]["content"])
        assert result["output"]["full_text"] is True
        return ModelTurn(text=json.dumps(_candidate("graph-anomaly")), usage=usage)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "resume"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _seed(store: ResearchStore):
    corpus = {
        "paper-graph": (
            "Relational Graph Anomaly Detection",
            [
                ("graph-message", "Method", "Graph neural networks aggregate neighboring node representations through message passing."),
                ("graph-anomaly", "Anomaly Detection", "Relational dependencies expose anomalous nodes whose neighborhood behavior differs from normal graph patterns."),
                ("graph-eval", "Evaluation", "The evaluation reports node anomaly ranking metrics on attributed graph benchmarks."),
            ],
        ),
        "paper-vision": (
            "Parameter Efficient Vision Adaptation",
            [
                ("vision-adapter", "Method", "Small bottleneck adapters tune a frozen vision transformer with few trainable parameters."),
                ("vision-prompt", "Prompt Tuning", "Learned visual prompt tokens adapt transformer representations while the backbone remains frozen."),
                ("vision-eval", "Evaluation", "Image classification accuracy is compared under equal data splits and parameter budgets."),
            ],
        ),
        "paper-series": (
            "Compressed Time Series Queries",
            [
                ("series-compress", "Compression", "Segment residual norms provide deterministic error bounds for compressed time series range queries."),
                ("series-forecast", "Forecasting", "Temporal attention predicts future values from multivariate time series windows."),
                ("series-eval", "Evaluation", "Forecasting error is reported with mean absolute error and root mean squared error."),
            ],
        ),
    }
    for paper_id, (title, rows) in corpus.items():
        store.upsert_paper(Paper(paper_id, title, file_path=f"{paper_id}.md"))
        store.replace_chunks(paper_id, [
            Chunk(chunk_id, paper_id, section, text, line_start=index * 10 + 1,
                  line_end=index * 10 + 2, source_hash=f"{chunk_id}-v1")
            for index, (chunk_id, section, text) in enumerate(rows)
        ])
    empty = Paper("paper-empty", "Metadata Only Paper")
    store.upsert_paper(empty)
    return corpus, empty


def _components(root: Path, *, resume_budget: bool):
    store = ResearchStore(root / "workspace" / "researchops.db")
    embedding = SentenceTransformerEmbedding()
    backend = QdrantVectorBackend(
        store,
        embedding,
        alias=ALIAS,
        project_id=f"{PROJECT_ID}-{root.name.lower()}",
        corpus_version=CORPUS_VERSION,
        hnsw_m=16,
        hnsw_ef_construct=100,
        search_hnsw_ef=64,
    )
    retriever = HybridRetriever(
        store,
        embedding=embedding,
        reranker=LexicalReranker(),
        vector_backend=backend,
    )
    scripted = ScriptedPaperModel()
    budget = BudgetedModel(
        scripted,
        root / "budget",
        500_000,
        max_model_turns=12,
        resume=resume_budget,
    )
    return store, embedding, backend, retriever, scripted, budget


def _ann_comparison(store, embedding, backend):
    rows = [dict(row) for row in store.all_chunks()]
    helper = HybridRetriever(
        store, embedding=embedding, reranker=LexicalReranker(),
        vector_backend=LocalExactVectorBackend())
    vectors = helper._vectors(rows)
    exact = LocalExactVectorBackend()
    queries = [
        "relational graph anomaly detection",
        "parameter efficient frozen vision transformer adapters",
        "visual prompt tokens for image classification",
        "deterministic error bounds compressed time series",
        "temporal attention multivariate forecasting",
    ]
    details = []
    exact_seconds = qdrant_seconds = 0.0
    recalls = []
    for query in queries:
        query_vector = embedding.encode([query])[0]
        started = time.perf_counter()
        local_scores = exact.search(
            query_vector, rows=rows, vectors=vectors, limit=3)
        exact_seconds += time.perf_counter() - started
        started = time.perf_counter()
        ann_scores = backend.search(
            query_vector, rows=rows, vectors=vectors, limit=3)
        qdrant_seconds += time.perf_counter() - started
        local_ids, ann_ids = list(local_scores), list(ann_scores)
        recall = len(set(local_ids) & set(ann_ids)) / max(1, len(local_ids))
        recalls.append(recall)
        details.append({
            "query": query,
            "exact_top3": local_ids,
            "ann_top3": ann_ids,
            "recall_at_3": recall,
        })
    return {
        "queries": details,
        "recall_at_3": sum(recalls) / len(recalls),
        "exact_total_ms": round(exact_seconds * 1000, 3),
        "qdrant_total_ms": round(qdrant_seconds * 1000, 3),
        "hnsw": {
            "m": backend.hnsw_m,
            "ef_construct": backend.hnsw_ef_construct,
            "search_ef": backend.search_hnsw_ef,
        },
        "note": "Small-corpus engineering comparison; not a production-scale latency claim.",
    }


def prepare(root: Path):
    root.mkdir(parents=True, exist_ok=False)
    store, embedding, backend, retriever, scripted, budget = _components(
        root, resume_budget=False)
    corpus, empty = _seed(store)
    first_hits = retriever.search("relational graph anomaly", mode="dense", limit=3)
    first_sync = backend.last_status
    retriever.search("relational graph anomaly", mode="dense", limit=3)
    idempotent_sync = backend.last_status

    temp = Paper("paper-temp", "Temporary Index Record")
    store.upsert_paper(temp)
    store.replace_chunks("paper-temp", [
        Chunk("temp-chunk", "paper-temp", "Method", "temporary graph retrieval text", source_hash="temp-v1")
    ])
    retriever.search("temporary graph", mode="dense", limit=3)
    added_sync = backend.last_status
    store.replace_chunks("paper-temp", [
        Chunk("temp-chunk", "paper-temp", "Method", "updated temporary graph retrieval text", source_hash="temp-v2")
    ])
    retriever.search("updated temporary graph", mode="dense", limit=3)
    updated_sync = backend.last_status
    store.replace_chunks("paper-temp", [])
    retriever.search("relational graph anomaly", mode="dense", limit=3)
    deleted_sync = backend.last_status

    filtered = retriever.search(
        "transformer", mode="dense", limit=6, paper_ids={"paper-vision"})
    ann_64 = _ann_comparison(store, embedding, backend)
    backend_32 = QdrantVectorBackend(
        store,
        embedding,
        alias=ALIAS,
        project_id=f"{PROJECT_ID}-{root.name.lower()}",
        corpus_version=CORPUS_VERSION,
        hnsw_m=16,
        hnsw_ef_construct=100,
        search_hnsw_ef=32,
    )
    ann_32 = _ann_comparison(store, embedding, backend_32)

    candidate_backend = QdrantVectorBackend(
        store,
        embedding,
        alias=ALIAS,
        project_id=f"{PROJECT_ID}-{root.name.lower()}",
        corpus_version=CORPUS_VERSION + "-candidate",
        hnsw_m=16,
        hnsw_ef_construct=100,
        search_hnsw_ef=64,
    )
    candidate_retriever = HybridRetriever(
        store,
        embedding=embedding,
        reranker=LexicalReranker(),
        vector_backend=candidate_backend,
    )
    candidate_retriever.search("relational graph anomaly", mode="dense", limit=3)
    candidate_collection = candidate_backend.last_status.version_collection
    aliases = {
        item.alias_name: item.collection_name
        for item in candidate_backend.client.get_aliases().aliases
    }
    alias_after_candidate = aliases.get(ALIAS)
    retriever.search("relational graph anomaly", mode="dense", limit=3)
    restored_collection = backend.last_status.version_collection
    aliases = {
        item.alias_name: item.collection_name
        for item in backend.client.get_aliases().aliases
    }
    alias_after_restore = aliases.get(ALIAS)

    idea = Idea(
        "Graph anomaly study",
        Domain.KNOWLEDGE_GRAPH,
        "Study relational evidence for graph anomalies.",
        "How do relational dependencies support anomaly detection?",
    )
    store.create_task("task-p3", idea.title, idea.domain.value, {
        "title": idea.title,
        "domain": idea.domain.value,
        "description": idea.description,
        "research_question": idea.research_question,
    })
    paper = Paper("paper-graph", corpus["paper-graph"][0], file_path="paper-graph.md")
    analyzer = PaperAnalyzer(
        budget,
        store,
        root / "workspace",
        mode="micro",
        retriever=retriever,
        traces=TraceStore(root / "workspace" / "runs"),
    )
    analysis_result = analyzer.analyze(
        "task-p3", idea, paper, run_id="p2-p3-verification")
    analysis = json.loads(Path(analysis_result["path"]).read_text(encoding="utf-8"))
    empty_result = analyzer.analyze(
        "task-p3", idea, empty, run_id="p2-p3-verification-empty")
    budget_summary = budget.summary()
    collection = backend.client.get_collection(backend.last_status.version_collection)
    checks = {
        "real_neural_embedding_384": len(embedding.encode(["dimension check"])[0]) == 384,
        "qdrant_initial_query": bool(first_hits),
        "qdrant_full_corpus_indexed": first_sync.indexed_points == 9,
        "qdrant_idempotent_upsert": idempotent_sync.upserted_points == 0,
        "qdrant_add": added_sync.upserted_points == 1,
        "qdrant_update": updated_sync.upserted_points == 1,
        "qdrant_delete": deleted_sync.deleted_points == 1,
        "server_paper_filter": bool(filtered) and {hit.paper_id for hit in filtered} == {"paper-vision"},
        "ann_recall_at_3": ann_64["recall_at_3"] >= 0.99,
        "hnsw_ef_comparison": ann_32["recall_at_3"] >= 0.99,
        "version_collection_switch": candidate_collection != restored_collection
        and alias_after_candidate == candidate_collection
        and alias_after_restore == restored_collection,
        "paper_agent_completed": analysis_result["status"] == "completed",
        "paper_agent_repaired_citation": analysis["paper_summary"]["citations"] == ["graph-anomaly"],
        "paper_agent_used_qdrant": retriever.last_diagnostics["vector_backend"] == "qdrant",
        "paper_agent_loop": analysis["micro_agent"]["model_turns"] == 4
        and analysis["micro_agent"]["tool_calls"] == 2,
        "no_evidence_blocks": empty_result["status"] == "insufficient_full_text",
        "budget_settled": budget_summary["charged_tokens"] == 80
        and budget_summary["unknown_reserved_tokens"] == 0,
    }
    report = {
        "schema_version": "researchops-v03-p2-p3-verification-v1",
        "phase": "prepare",
        "passed": all(checks.values()),
        "paid_model_calls": 0,
        "embedding": {
            "model": embedding.model_name,
            "revision": embedding.model_revision,
            "dimensions": 384,
        },
        "qdrant": {
            "health": backend.health(),
            "collection": backend.last_status.version_collection,
            "alias": ALIAS,
            "points_count": collection.points_count,
            "sync_state": backend.sync_state(),
            "initial_sync": first_sync.__dict__,
            "idempotent_sync": idempotent_sync.__dict__,
            "add_sync": added_sync.__dict__,
            "update_sync": updated_sync.__dict__,
            "delete_sync": deleted_sync.__dict__,
            "filtered_ids": [hit.chunk_id for hit in filtered],
            "ann_comparison": {
                "ef32": ann_32,
                "ef64": ann_64,
                "selected_search_ef": 64,
                "selection_note": (
                    "Both settings retained Recall@3 on this small corpus; ef=64 is kept "
                    "as the conservative default. Latency is descriptive only."
                ),
            },
            "version_switch": {
                "candidate_collection": candidate_collection,
                "alias_after_candidate": alias_after_candidate,
                "restored_collection": restored_collection,
                "alias_after_restore": alias_after_restore,
            },
            "corpus_manifest": [
                {
                    "paper_id": row["paper_id"],
                    "chunk_id": row["chunk_id"],
                    "source_hash": row["source_hash"],
                }
                for row in map(dict, store.all_chunks())
            ],
        },
        "paper_agent": {
            "result": analysis_result,
            "runtime": analysis["micro_agent"],
            "citation": analysis["paper_summary"],
            "no_evidence_result": empty_result,
            "scripted_model_calls": scripted.calls,
        },
        "budget": budget_summary,
        "checks": checks,
    }
    (root / "prepare.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


def resume(root: Path):
    prepare_report = json.loads((root / "prepare.json").read_text(encoding="utf-8"))
    store, embedding, backend, retriever, scripted, budget = _components(
        root, resume_budget=True)
    for attempt in range(20):
        try:
            backend.health()
            break
        except Exception:
            if attempt == 19:
                raise
            time.sleep(1)
    hits = retriever.search("relational graph anomaly", mode="dense", limit=3)
    post_restart_sync = backend.last_status
    idea = Idea(
        "Graph anomaly study",
        Domain.KNOWLEDGE_GRAPH,
        "Study relational evidence for graph anomalies.",
        "How do relational dependencies support anomaly detection?",
    )
    paper = Paper("paper-graph", "Relational Graph Anomaly Detection", file_path="paper-graph.md")
    analyzer = PaperAnalyzer(
        budget,
        store,
        root / "workspace",
        mode="micro",
        retriever=retriever,
        traces=TraceStore(root / "workspace" / "runs"),
    )
    before = budget.summary()
    analysis_result = analyzer.analyze(
        "task-p3", idea, paper, run_id="p2-p3-verification")
    after = budget.summary()
    checks = {
        "prepare_passed": prepare_report["passed"] is True,
        "query_after_container_restart": bool(hits),
        "persisted_index_required_no_upsert": post_restart_sync.upserted_points == 0,
        "paper_checkpoint_resume": analysis_result["status"] == "completed",
        "resume_made_no_model_call": scripted.calls == 0,
        "resume_did_not_recharge": before["charged_tokens"] == after["charged_tokens"] == 80,
    }
    report = {
        "schema_version": "researchops-v03-p2-p3-verification-v1",
        "phase": "complete-after-restart",
        "passed": all(checks.values()),
        "paid_model_calls": 0,
        "prepare": prepare_report,
        "post_restart": {
            "hits": [hit.chunk_id for hit in hits],
            "sync": post_restart_sync.__dict__,
            "analysis_result": analysis_result,
            "scripted_model_calls": scripted.calls,
            "budget_before": before,
            "budget_after": after,
        },
        "checks": checks,
    }
    (root / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


def main():
    args = parse_args()
    root = args.output.resolve()
    return prepare(root) if args.phase == "prepare" else resume(root)


if __name__ == "__main__":
    raise SystemExit(main())

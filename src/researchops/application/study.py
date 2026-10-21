"""Public fixed-corpus research planning, independent of benchmark task IDs."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from researchops.core.artifacts import canonical_hash, file_hash, write_json, workspace_lock
from researchops.core.storage import ResearchStore
from researchops.evaluation.precoding.runner import PaidCalls
from researchops.experiments.planning import HOST_PROTOCOL, dataset_catalog, read_verified_datasets
from researchops.harness.runtime import OpenAICompatibleModel
from researchops.knowledge.retrieval import HybridRetriever
from researchops.knowledge.query_planning import plan_retrieval_query
from researchops.knowledge.query_planning import plan_evidence_queries
from researchops.knowledge.evidence_retrieval import EvidenceRetriever
from researchops.knowledge.passages import materialize_passages
from researchops.experiments.evidence_design import build_evidence_packet, design_blueprint
from researchops.experiments.study_workflow import refine_study
from researchops.knowledge.fusion_retrieval import FusionRetriever
from researchops.experiments.fusion_design import build_fusion_packet
from researchops.papers.validation.automatic import formula_evidence_policy


def plan_study(idea: dict, *, source: Path, output: Path, token_cap=250000, model=None,
               retrieval_strategy="legacy", dataset_source: Path | None = None):
    """Natural-language task -> current-task RAG -> contract -> independent review.

    Does not open any evaluation manifest or require an old idea-specific silver report.
    """
    source, output = source.resolve(), output.resolve()
    if retrieval_strategy not in {"legacy", "cpu", "fusion"}:
        raise ValueError("retrieval_strategy must be legacy, cpu or fusion")
    dataset_source = (dataset_source or source).resolve()
    if source == output or source in output.parents:
        raise ValueError("Choose an independent output workspace")
    with workspace_lock(output):
        db = output / "study_corpus.db"
        snapshot_path = output / "study_snapshot.json"
        if db.exists():
            if not snapshot_path.exists():
                raise ValueError("Incomplete study snapshot; choose a new output workspace")
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            if Path(snapshot["source_workspace"]).resolve() != source:
                raise ValueError("Study workspace is pinned to a different source corpus")
            if snapshot.get("retrieval_strategy", "legacy") != retrieval_strategy:
                raise ValueError("Choose a new study workspace when changing retrieval strategy")
        if not db.exists():
            source_db = next((source / name for name in ["researchops.db", "raw.db", "passages.db"] if (source / name).is_file()), None)
            if source_db is None:
                raise ValueError("Source workspace has no research corpus database")
            with sqlite3.connect(f"{source_db.as_uri()}?mode=ro", uri=True) as src:
                with sqlite3.connect(db) as dst:
                    src.backup(dst)
            write_json(snapshot_path, {"source_workspace": str(source), "initial_database_sha256": file_hash(db),
                                       "retrieval_strategy": retrieval_strategy, "dataset_workspace": str(dataset_source),
                                       "mode": "fixed_corpus_snapshot; task relevance is recomputed"})
        store = ResearchStore(db)
        if retrieval_strategy in {"cpu", "fusion"}:
            # Materialize only when the input is raw nodes; preserve an existing passage index.
            rows = store.all_chunks()
            already_passages = bool(rows) and all(json.loads(r["provenance_json"] or "{}").get("index_version") for r in rows)
            if not already_passages:
                index = ResearchStore(output / "study_passages.db")
                index_report = materialize_passages(store, index, sorted({r["paper_id"] for r in rows}))
                write_json(output / "passage_index.json", index_report)
                store = index
        states = read_verified_datasets(dataset_source)
        query = idea.get("research_question") or idea.get("description")
        if not query:
            raise ValueError("A natural-language description or research_question is required")
        calls = PaidCalls(output, model or OpenAICompatibleModel(max_attempts=1), token_cap)
        query_plan = plan_retrieval_query(query, idea, calls)
        write_json(output / "retrieval_query_plan.json", query_plan)
        evidence_brief = None
        if retrieval_strategy in {"cpu", "fusion"}:
            query_plan = plan_evidence_queries(query, idea, calls, primary=query_plan)
            write_json(output / "evidence_query_plan.json", query_plan)
            retriever = FusionRetriever(store, calls) if retrieval_strategy == "fusion" else EvidenceRetriever(store)
            search_args = {"question": query} if retrieval_strategy == "fusion" else {}
            hits = retriever.search(query_plan["retrieval_query"], queries=[f["query"] for f in query_plan["facets"]], limit=40, **search_args)
            hits = [h for h in hits if formula_evidence_policy(store.get_chunk(h.chunk_id))["eligible_for_planning"]]
            write_json(output / "retrieval_diagnostics.json", retriever.last_diagnostics)
            packet_builder = build_fusion_packet if retrieval_strategy == "fusion" else build_evidence_packet
            packet = packet_builder(query, idea, hits, calls)
            evidence_brief = design_blueprint(query, idea, packet, calls)
            write_json(output / "evidence_packet.json", packet)
            write_json(output / "study_blueprint.json", evidence_brief)
            raw_evidence = [{"chunk_id": p["chunk_id"], "paper_id": p["paper_id"], "title": p["title"], "text": p["text"]} for p in packet["passages"]]
        else:
            hits = HybridRetriever(store).search(query_plan["retrieval_query"], limit=9, candidate_limit=36)
            raw_evidence = [{"chunk_id": h.chunk_id, "paper_id": h.paper_id, "title": h.paper_title, "text": h.text[:1000]} for h in hits]
        evidence = []
        for node in raw_evidence:
            paper = store.get_paper(node["paper_id"])
            policy = formula_evidence_policy(store.get_chunk(node["chunk_id"]))
            if not policy["eligible_for_planning"]:
                continue
            if not paper["file_path"] or file_hash(paper["file_path"]) != paper["content_hash"]:
                continue
            evidence.append({**node, "page": store.get_chunk(node["chunk_id"])["page"], "source_sha256": paper["content_hash"],
                             "formula_policy": policy})
        request = {"task": idea, "available_data": dataset_catalog(states),
                   "host_protocol": HOST_PROTOCOL, "fulltext_evidence": evidence,
                   "evidence_brief": evidence_brief, "retrieval_strategy": retrieval_strategy}
        task_hash = canonical_hash(request)[:20]
        directory = output / "studies" / task_hash
        workflow = refine_study(idea=idea, states=states, evidence=evidence, store=store,
                                calls=calls, directory=directory, evidence_brief=evidence_brief)
        ready = workflow["success"] and workflow["mechanical"].get("passed", False)
        result = {"status": "ready_for_coding" if ready else "blocked_before_coding",
                  **workflow,
                  "artifacts": str(directory), "charged_tokens_cumulative": calls.state["total_tokens"],
                  "retrieval_strategy": retrieval_strategy,
                  "contract_version": "researchops-experiment-contract-v2" if ready else None,
                  "coding_adapter": "Contract v2 requires arm-aware coding implementation; legacy v1 adapter is insufficient",
                  "training_executed": False, "research_success": None}
        write_json(directory / "result.json", result)
        return result

"""One bounded live provider paper analysis using a real historical paper."""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
from pathlib import Path

from researchops.core.artifacts import file_hash, write_json, progress
from researchops.core.config import load_project_env
from researchops.core.models import Chunk, Domain, Idea, Paper
from researchops.core.storage import ResearchStore
from researchops.harness.budget import BudgetedModel
from researchops.harness.runtime import OpenAICompatibleModel, TraceStore
from researchops.knowledge.evidence_retrieval import checkpoint_key
from researchops.knowledge.retrieval import HybridRetriever, LexicalReranker, SentenceTransformerEmbedding
from researchops.knowledge.vector_backend import QdrantVectorBackend
from researchops.papers.analysis import PaperAnalyzer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("workspace_rag_cpu_v3/passages.db"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transport", choices=("inprocess", "mcp"), default="inprocess")
    args = parser.parse_args()
    load_project_env()
    os.environ["RESEARCHOPS_PAPER_READ_TRANSPORT"] = args.transport
    os.environ["RESEARCHOPS_EMBEDDING_LOCAL_ONLY"] = "true"
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    torch.set_num_threads(4)
    provider = OpenAICompatibleModel(max_attempts=1)
    root, source = args.output.resolve(), args.source.resolve()
    root.mkdir(parents=True, exist_ok=True)
    source_hash = file_hash(source)
    with sqlite3.connect(f"{source.as_uri()}?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute("SELECT * FROM papers WHERE title LIKE '%AdaptFormer%' ORDER BY paper_id LIMIT 1").fetchone()
        if row is None:
            raise ValueError("Frozen corpus is missing AdaptFormer")
        chunks = list(connection.execute("SELECT * FROM chunks WHERE paper_id=? ORDER BY ordinal", (row["paper_id"],)))
    store = ResearchStore(root / "researchops.db")
    paper = Paper(row["paper_id"], row["title"], file_path=row["file_path"],
                  content_hash=row["content_hash"], doi=row["doi"])
    store.upsert_paper(paper)
    if not store.paper_chunks(paper.paper_id):
        store.replace_chunks(paper.paper_id, [Chunk(
            c["chunk_id"], c["paper_id"], c["section"], c["text"], page=c["page"],
            source_hash=c["source_hash"], content_hash=c["content_hash"],
            line_start=c["line_start"], line_end=c["line_end"], node_type=c["node_type"],
            provenance=json.loads(c["provenance_json"] or "{}")) for c in chunks])
    idea = Idea("Controlled visual adaptation", Domain.AI_ML,
                "Compare parallel bottleneck adapters for frozen vision transformers under matched budgets.",
                "Which mechanism and controls does this paper support for comparing parallel adapters?")
    store.create_task("p3-live", idea.title, idea.domain.value, {
        "title": idea.title, "domain": idea.domain.value, "description": idea.description,
        "research_question": idea.research_question})
    embedding = SentenceTransformerEmbedding()
    embedding.model_revision = checkpoint_key(embedding).rsplit("@", 1)[-1]
    backend = QdrantVectorBackend(store, embedding,
        alias="researchops_p3_live_" + root.name.lower(), corpus_version="adaptformer-v1")
    retriever = HybridRetriever(store, embedding=embedding, reranker=LexicalReranker(), vector_backend=backend)
    budget_root = root / "budget"
    budget = BudgetedModel(provider, budget_root, 300_000, max_model_turns=6,
                           resume=(budget_root / "ledger.json").is_file())
    analyzer = PaperAnalyzer(budget, store, root, mode="micro", retriever=retriever,
                             traces=TraceStore(root / "runs"))
    result = analyzer.analyze("p3-live", idea, paper, run_id="p3-live-provider")
    artifact = json.loads(Path(result["path"]).read_text(encoding="utf-8"))
    before = budget.summary()
    repeated = analyzer.analyze("p3-live", idea, paper, run_id="p3-live-provider")
    after = budget.summary()
    # A terminal checkpoint intentionally performs no new search. Verify the
    # original server query from its durable audit, not empty in-memory diagnostics.
    queries = [json.loads(line) for line in backend.audit_path.read_text(encoding="utf-8").splitlines()
               if line.strip()] if backend.audit_path.exists() else []
    qdrant_evidence = [q for q in queries if q.get("operation") == "query_points" and
        any(p.get("payload", {}).get("paper_id") == paper.paper_id and
            p.get("payload", {}).get("project_id") == backend.project_id and
            p.get("payload", {}).get("embedding_key") == backend.embedding_key
            for p in q.get("payload", {}).get("results", []))]
    checks = {
        "source_untouched": source_hash == file_hash(source),
        "completed": result["status"] in {"completed", "completed_with_gaps"},
        "provider_was_called": before["model_calls"] > 0,
        "used_read_tools": bool(artifact.get("micro_agent", {}).get("read_chunk_ids")),
        "used_qdrant": bool(qdrant_evidence),
        "resume_no_extra_calls": before["model_calls"] == after["model_calls"],
        "resume_no_extra_charge": before["charged_tokens"] == after["charged_tokens"],
    }
    if args.transport == "mcp":
        reads = artifact.get("micro_agent", {}).get("mcp_reads", [])
        checks["mcp_wire_recorded"] = bool(reads) and all(Path(r["audit_path"]).is_file() for r in reads)
        requests = [json.loads(p.read_text(encoding="utf-8")) for p in (budget_root / "calls").glob("*.request.json")]
        checks["mcp_result_consumed_in_later_model_request"] = any(
            m.get("role") == "tool" and "mcp_stdio" in str(m.get("content"))
            for r in requests for m in r["messages"])
    report = {"passed": all(checks.values()), "checks": checks, "model": provider.model,
              "paper_id": paper.paper_id, "paper_title": paper.title, "chunks": len(chunks),
              "source_sha256": source_hash, "result": result, "repeated": repeated,
              "micro_agent": artifact.get("micro_agent"), "budget": after,
              "query_audit": str(backend.audit_path),
              "durable_qdrant_queries": len(qdrant_evidence),
              "limitations": "One live engineering case; no statistical quality-improvement claim."}
    write_json(root / "report.json", report)
    progress(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

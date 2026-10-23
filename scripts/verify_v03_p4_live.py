"""Run the public LangGraph study entry against real corpus/data/provider."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from researchops.application.study import plan_study
from researchops.core.artifacts import file_hash, write_json, progress
from researchops.core.config import load_project_env
from researchops.harness.runtime import OpenAICompatibleModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", choices=("success", "boundary"), default="success")
    parser.add_argument("--resume", action="store_true", help="Resume the same pinned graph and budget after interruption")
    parser.add_argument("--token-cap", type=int, help="Root/study token cap, fixed for the lifetime of this output")
    parser.add_argument("--source", type=Path, default=Path("workspace_rag_cpu_v3"))
    parser.add_argument("--datasets", type=Path, default=Path("workspace_batch_vit_v2"))
    args = parser.parse_args()
    load_project_env()
    os.environ.update(RESEARCHOPS_VECTOR_BACKEND="qdrant", RESEARCHOPS_RETRIEVAL_BACKEND="neural",
        RESEARCHOPS_EMBEDDING_LOCAL_ONLY="true", RESEARCHOPS_RERANKER_LOCAL_ONLY="true", HF_HUB_OFFLINE="1",
        RESEARCHOPS_EMBEDDING_REVISION="e8f8c211226b894fcb81acc59f3b34ba3efd5f42")
    import torch
    torch.set_num_threads(4)
    root, source, datasets = args.output.resolve(), args.source.resolve(), args.datasets.resolve()
    root.mkdir(parents=True, exist_ok=args.resume)
    os.environ["RESEARCHOPS_QDRANT_COLLECTION"] = "researchops_p4_" + root.name.lower()
    source_db = next(source / n for n in ("researchops.db", "raw.db", "passages.db") if (source / n).exists())
    source_hash = file_hash(source_db)
    provider = OpenAICompatibleModel(max_attempts=1)
    idea = ({"description": "Design a prospective controlled ViT image classification study, no training execution.",
             "research_question": "On DTD and Oxford-IIIT Pet with 25% of training labels, compare LoRA rank 2 and 8 with alpha/r=1, dropout=0, Q/V targets in the last three zero-based blocks [9,10,11]. Use the same frozen ViT backbone, fixed data splits, seeds 42/43/44, equal two-LR validation-only tuning. Primary accuracy, secondary trainable_parameters. Estimate paired seed differences with a specified interval rule, report supported/refuted/inconclusive outcomes, and never presume an accuracy winner. Treat these configurations as proposed controls, not exact paper settings."}
            if args.case == "success" else {
             "description": "Plan a time-series forecasting study under the current installed task adapters.",
             "research_question": "Forecast multivariate sensor time series using temporal transformers. Preserve the time-series task and do not substitute image classification. If no verified time-series data and coding adapter exist, report the exact boundary and a separately marked alternative requiring constraint change."})
    cap = args.token_cap if args.token_cap is not None else (400000 if args.case == "success" else 150000)
    started = time.monotonic()
    try:
        result = plan_study(idea, source=source, output=root / "run", model=provider,
            dataset_source=datasets, retrieval_strategy="legacy", token_cap=cap, orchestration="langgraph")
        before = json.loads((root / "run" / "budget" / "ledger.json").read_text(encoding="utf-8"))
        repeated = plan_study(idea, source=source, output=root / "run", model=provider,
            dataset_source=datasets, retrieval_strategy="legacy", token_cap=cap, orchestration="langgraph")
        after = json.loads((root / "run" / "budget" / "ledger.json").read_text(encoding="utf-8"))
        audit_paths = list((root / "run" / "vector_audit").glob("*.jsonl"))
        query_count = sum('"operation": "query_points"' in line for p in audit_paths for line in p.read_text(encoding="utf-8").splitlines())
        expected = (result["status"] == "ready_for_coding" if args.case == "success" else
                    result["status"] == "blocked_before_coding" and result.get("boundary_handled") is True)
        checks = {"expected_domain_outcome": expected, "langgraph_business_entry": result["orchestration"] == "langgraph",
                  "real_provider_calls": before["turns"] > 0, "real_qdrant_queries": query_count > 0,
                  "source_untouched": file_hash(source_db) == source_hash,
                  "resume_no_calls": before["turns"] == after["turns"],
                  "resume_no_charge": before["total_tokens"] == after["total_tokens"],
                  "resume_same_result": result == repeated}
        report = {"passed": all(checks.values()), "case": args.case, "checks": checks,
                  "model": provider.model, "result": result, "source_sha256": source_hash,
                  "qdrant_queries": query_count, "query_audit": list(map(str, audit_paths)),
                  "seconds": time.monotonic() - started,
                  "limitations": "Engineering live case, no statistical quality gain or experimental research result."}
    except Exception as error:
        report = {"passed": False, "case": args.case, "error": f"{type(error).__name__}: {error}",
                  "seconds": time.monotonic() - started, "source_untouched": file_hash(source_db) == source_hash}
    report["phase"] = "resume" if args.resume else "fresh"
    previous_report = root / "report.json"
    if previous_report.exists():
        write_json(root / "report_history" / f"{time.time_ns()}.json",
                   json.loads(previous_report.read_text(encoding="utf-8")))
    write_json(previous_report, report)
    progress(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

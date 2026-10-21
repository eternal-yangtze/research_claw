"""Offline audit and readable report for preserved pre-coding benchmark artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from researchops.core.artifacts import canonical_hash, file_hash, write_json
from researchops.evaluation.precoding.runner import read_json
from researchops.evaluation.precoding.gates import mechanical_gate
from researchops.core.storage import ResearchStore


def audit(root, prior_roots=()):
    report = read_json(root / "report.json")
    store = ResearchStore(root / "researchops.db")
    checked, invalid, references = [], [], 0
    for path in sorted((root / "tasks").glob("*/frozen_spec.json")):
        spec = read_json(path)
        evidence = read_json(path.parent / "evidence.json")
        gate = mechanical_gate(spec, evidence, store)
        checked.append({"key": path.parent.name, "contract_sha256": file_hash(path), "gate": gate})
        if not gate["passed"]:
            invalid.append(path.parent.name)
        references += len(spec["baseline_citations"])
    ledger = read_json(root / "ledger.json")
    prior = [{"path": str(p), "ledger": read_json(p / "ledger.json")} for p in prior_roots]
    charged = ledger["total_tokens"] + sum(p["ledger"]["total_tokens"] for p in prior)
    all_leases = ledger["call_ledger"] + [c for p in prior for c in p["ledger"]["call_ledger"]]
    known = sum(c.get("actual_tokens", 0) for c in all_leases if c["status"] == "settled")
    result = {"checked_contracts": len(checked), "mechanically_valid_contracts": len(checked) - len(invalid),
              "invalid_contracts": invalid, "citation_references_checked": references,
              "ledger_with_prior_attempts": charged, "suite_cap": 300000,
              "actual_known_tokens_all_attempts": known,
              "unknown_usage_reserved_all_attempts": charged - known,
              "model_calls_all_attempts": len(all_leases),
              "model_wall_seconds_all_attempts": sum(c.get("wall_seconds", 0) for c in all_leases),
              "within_suite_cap": charged <= 300000, "prior_attempts": prior,
              "contracts": checked, "source_assets": "Read-only references to preserved original PDFs/data; copied database",
              "limitation": "Citation existence and file hashes do not establish semantic correctness"}
    for name, path in (("supplementary", root / "supplementary" / "report.json"),
                       ("recovery", root / "recovery.json")):
        if path.exists():
            result[name] = read_json(path)
    result["unexecuted_optional_checks"] = {
        "natural_language_contrastive_cases": 4, "additional_abstract_and_bm25_ablations": 8,
        "status": "not_run", "reason": "Remaining whole-suite budget could not admit additional planner/reviewer requests",
        "claim_limit": "Compiler input changes are unit-tested; no measured natural-language contrastive rate or RAG quality uplift is claimed"}
    write_json(root / "audit.json", result)
    # Produce compact machine-readable per-idea evidence without altering raw results.
    suite = read_json(Path("config/benchmarks/precoding_v1.json"))
    rows = []
    for track, group in report["tracks"].items():
        for record in group["records"]:
            key = record["key"]
            spec_path = root / "tasks" / key / "frozen_spec.json"
            spec = read_json(spec_path) if spec_path.exists() else None
            rows.append({"track": track, "key": key,
                         "split": "dev" if record.get("idea_id") in suite["dev_ids"] else "test" if track != "boundary" else "boundary",
                         "status": record["status"], "mechanical": record.get("mechanical_ready", False),
                         "silver": record.get("silver_accepted", False), "success": record.get("success", False),
                         "tokens": record.get("charged_tokens"), "wall_seconds": record.get("wall_seconds"),
                         "training_runs": spec["run_budget"]["training_runs"] if spec else None,
                         "test_evaluations": spec["run_budget"]["test_evaluations"] if spec else None,
                         "fractions": spec["design"]["fractions"] if spec else None,
                         "primary_metrics": record.get("primary_metrics"), "error": record.get("error")})
    write_json(root / "task_summary.json", rows)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("workspace_precoding_v2"))
    parser.add_argument("--prior", type=Path, nargs="*", default=[Path("workspace_precoding_v1"), Path("workspace_precoding_v1_live")])
    args = parser.parse_args()
    result = audit(args.workspace, args.prior)
    compact = {k: v for k, v in result.items() if k not in {"contracts", "prior_attempts", "supplementary", "recovery"}}
    compact["supplementary_tracks"] = result.get("supplementary", {}).get("tracks")
    compact["recovery_new_model_calls"] = result.get("recovery", {}).get("new_model_calls")
    print(json.dumps(compact, indent=2))
    return int(not result["within_suite_cap"])


if __name__ == "__main__":
    raise SystemExit(main())

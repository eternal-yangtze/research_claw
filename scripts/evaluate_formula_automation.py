"""Replay preserved real Docling outputs through production ingestion and gates.

No new OCR, model calls, downloads or gold labels. Historical pilot is read-only.
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from researchops.core.artifacts import file_hash, write_json
from researchops.core.models import Paper
from researchops.core.storage import ResearchStore
from researchops.knowledge.passages import materialize_passages
from researchops.papers.ingestion import DocumentIngestor
from researchops.papers.validation.automatic import formula_evidence_policy, formula_reproduction_gate, validation_fingerprint


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="workspace_formula_automation_v2")
    args = parser.parse_args()
    root = Path(args.output).resolve() / uuid.uuid4().hex
    root.mkdir(parents=True)
    store = ResearchStore(root / "raw.db")
    passages = ResearchStore(root / "passages.db")
    ingestor = DocumentIngestor(store)
    old = Path("workspace_formula_pilot_v1")
    old_final = old / "evaluation" / "40b21cd732bd4a309d2b78029c86a7f6" / "metrics.json"
    baseline = json.loads(old_final.read_text(encoding="utf-8"))
    manifest = json.loads(Path("config/benchmarks/formula_pilot_v1.json").read_text(encoding="utf-8"))
    rows, total, policy_counts = [], Counter(), Counter()
    for case in manifest["cases"]:
        pid = case["id"]
        original_parser = (next((old / "refinement").glob("*/refined_docling.json")) if pid == "explora"
                           else old / (pid + "_parse") / "docling.json")
        source = old / "sources" / ("arxiv_" + case["arxiv_id"] + ".tar.gz")
        seed = Paper("formula_" + pid, pid, source_archive_path=str(source.resolve()))
        first = ingestor.ingest_file(case["pdf"], seed, parser_artifact=original_parser)
        second = ingestor.ingest_file(case["pdf"], seed, parser_artifact=original_parser)
        check = first.formula_validation
        report = json.loads(Path(check["report_path"]).read_text(encoding="utf-8"))
        refs_path = Path(check["report_path"]).with_name("source_formulas.json")
        refs = json.loads(refs_path.read_text(encoding="utf-8"))
        policies = {r["chunk_id"]: formula_evidence_policy(r) for r in store.paper_chunks(seed.paper_id)
                    if r["node_type"] == "formula"}
        evidence = [{"chunk_id": r["chunk_id"]} for r in store.paper_chunks(seed.paper_id)
                    if formula_evidence_policy(r)["eligible_for_planning"]]
        prose = next(r["chunk_id"] for r in store.paper_chunks(seed.paper_id) if r["node_type"] != "formula")
        probe = {cid: {"formula_alone_passed": formula_reproduction_gate({"baseline_citations": [cid]}, evidence, store)["passed"],
                       "with_prose_passed": formula_reproduction_gate({"baseline_citations": [cid, prose]}, evidence, store)["passed"]}
                 for cid in policies}
        for p in policies.values():
            policy_counts["eligible" if p["eligible_for_planning"] else "withheld"] += 1
        for key in ["formula_regions", "accepted", "source_recovered", "disputed", "unverified", "regions_with_crops"]:
            total[key] += check["summary"][key]
        row = {"id": pid, "pdf_sha256": file_hash(case["pdf"]), "parser_sha256": file_hash(original_parser),
               "source_sha256": file_hash(source), "source_identity": report["source_identity"],
               "summary": check["summary"], "report_path": check["report_path"],
               "report_sha256": file_hash(check["report_path"]), "resume_cache_hit": second.formula_validation["cache_hit"],
               "expanded_macro_names": sorted({m for r in refs for m in r["macro_expansion"]["expanded_macros"]}),
               "source_formulas_with_unresolved_macros": sum(bool(r["unexpanded_macros"]) for r in refs),
               "formula_policies": policies}
        row["reproduction_gate_probes"] = probe
        rows.append(row)
    materialize_passages(store, passages, ["formula_" + c["id"] for c in manifest["cases"]])
    indexed = [r for r in passages.all_chunks() if r["node_type"] == "formula"]
    assertions = {
        "all_version_matched": all(r["source_identity"]["status"] == "matched_version" for r in rows),
        "all_resume_cache_hits": all(r["resume_cache_hit"] for r in rows),
        "all_formula_nodes_preserved_in_passages": len(indexed) == total["formula_regions"],
        "all_gate_states_preserved_in_passages": Counter("eligible" if formula_evidence_policy(r)["eligible_for_planning"] else "withheld" for r in indexed) == policy_counts,
        "all_original_parser_hashes_match_baseline": all(r["parser_sha256"] == file_hash(next((old / "refinement").glob("*/refined_docling.json"))) if r["id"] == "explora"
                else r["parser_sha256"] == next(b["parser_sha256"] for b in baseline["papers"] if b["id"] == r["id"]) for r in rows),
        "no_formula_claimed_as_sole_reproduction_truth": all(not formula_evidence_policy(r)["safe_as_sole_reproduction_evidence"] for r in indexed),
        "all_formula_only_plans_blocked": all(not p["formula_alone_passed"] for r in rows for p in r["reproduction_gate_probes"].values()),
        "prose_does_not_unblock_unresolved_formulas": all(
            p["with_prose_passed"] == r["formula_policies"][cid]["eligible_for_planning"]
            for r in rows for cid, p in r["reproduction_gate_probes"].items()),
    }
    report = {"schema_version": "formula-automation-pilot-v2", "historical_baseline_path": str(old_final.resolve()),
              "historical_baseline_sha256": file_hash(old_final), "baseline": baseline["after"],
              "after": dict(total), "papers": rows, "evidence_gate": dict(policy_counts), "checks": assertions,
              "implementation": validation_fingerprint(), "ingestion_sha256": file_hash(Path("src/researchops/papers/ingestion.py")),
              "supported_or_source_recovered_rate": (total["accepted"] + total["source_recovered"]) / total["formula_regions"],
              "recognition_accuracy": None, "detection_recall": None, "gold_interface": None,
              "human_gold": False, "paid_api_tokens": 0, "new_ocr_inference": False, "training": False,
              "limitations": ["same three development pages, preserved OCR replay, not a new held-out test",
                              "source recovery is inferred alignment, not accurate independent recognition",
                              "unknown/scoped/recursive/dynamic macros remain unresolved",
                              "text-only fallback is explicitly not assessed; tables/figures remain outside scope",
                              "no new paid-model end-to-end quality benchmark in this formula-only change"]}
    write_json(root / "metrics.json", report)
    print(json.dumps({"metrics_path": str(root / "metrics.json"), "baseline": report["baseline"],
                      "after": report["after"], "evidence_gate": report["evidence_gate"], "checks": assertions}, ensure_ascii=False, indent=2))
    if not all(assertions.values()):
        raise SystemExit("production replay check failed; artifacts retained")


if __name__ == "__main__":
    main()

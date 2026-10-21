"""Recompute the three-paper formula pilot without repeating neural inference."""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from researchops.core.artifacts import file_hash, write_json
from researchops.papers.validation import validate_formulas


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default="workspace_formula_pilot_v1")
    parser.add_argument("--manifest", default="config/benchmarks/formula_pilot_v1.json")
    args = parser.parse_args()
    workspace = Path(args.workspace).resolve()
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    evaluation = workspace / "evaluation" / uuid.uuid4().hex
    evaluation.mkdir(parents=True)
    rows = []
    for case in manifest["cases"]:
        case_id = case["id"]
        parsed = workspace / ("explora_parse_retry" if case_id == "explora" else case_id + "_parse") / "docling.json"
        packages = [p for p in (workspace / "sources").glob("arxiv_" + case["arxiv_id"] + ".tar*")
                    if p.name.endswith((".tar.gz", ".tar"))]
        if len(packages) != 1:
            raise ValueError("pilot requires exactly one version-matched source package per paper")
        source = packages[0]
        receipt = source.with_name(source.name + ".receipt.json")
        arguments = dict(source_package=source, source_receipt=receipt)
        before = validate_formulas(case["pdf"], evaluation / (case_id + "_baseline"), docling_json=parsed, **arguments)
        after = before
        refined_regions = 0
        if case_id == "explora":
            refinements = sorted((workspace / "refinement").glob("*/refined_docling.json"))
            if len(refinements) != 1:
                raise ValueError("pilot requires exactly one declared refinement artifact")
            refined = refinements[0]
            data = json.loads(refined.read_text(encoding="utf-8"))
            refined_regions = len(data["refinement"]["candidates"])
            after = validate_formulas(case["pdf"], evaluation / (case_id + "_refined"), docling_json=refined, **arguments)
        rows.append({"id": case_id, "arxiv_id": case["arxiv_id"], "pdf_sha256": file_hash(case["pdf"]),
                     "parser_sha256": file_hash(parsed), "pages": case["pages"], "before": before,
                     "after": after, "refined_regions": refined_regions,
                     "parser_elapsed_seconds": json.loads(parsed.read_text(encoding="utf-8"))["elapsed_seconds"]})
    def total(stage):
        counts = Counter()
        for row in rows:
            for key in ["formula_regions", "accepted", "source_recovered", "disputed", "unverified", "regions_with_crops"]:
                counts[key] += row[stage]["summary"][key]
        values = dict(counts)
        values["source_corroboration_rate"] = values["accepted"] / values["formula_regions"]
        values["source_supported_or_recovered_rate"] = (values["accepted"] + values["source_recovered"]) / values["formula_regions"]
        return values
    report = {"schema_version": "formula-pilot-results-v2", "manifest": str(manifest_path),
              "manifest_sha256": file_hash(manifest_path), "implementation_sha256": file_hash(Path(__file__)),
              "papers": rows, "before": total("before"), "after": total("after"),
              "version_matched_papers": sum(r["after"]["source_identity"]["status"] == "matched_version" for r in rows),
              "refined_regions": sum(r["refined_regions"] for r in rows),
              "recognition_accuracy": None, "detection_recall": None, "human_gold": False,
              "reference_type": "source_derived_silver", "paid_llm_calls": 0, "paid_api_tokens": 0,
              "local_generated_token_count": "not_recorded", "training_performed": False,
              "parser_metadata_correction": "The preserved initial parser artifacts contain max_formula_tokens=512; this was a requested model setting, not an effective bound. Installed Docling 2.140.0 code_formula_vlm_model.py sets per-input max_new_tokens=2048. The selective refinement input explicitly used 384. Current parse script reports the upstream limit instead.",
              "limitations": ["development pilot on three selected pages, not a held-out benchmark",
                              "same recognizer with a tighter crop is not an independent vote",
                              "source existence corroborates an expression but does not prove full region alignment or correctness",
                              "source_recovered restores missing symbols by inferred source alignment; never count it as OCR exact match or gold accuracy",
                              "no general semantic equivalence, exhaustive region annotation, public gold, tables or figures",
                              "initial 200-second Docling attempt timed out; empty output directory preserved; successful retry retained"]}
    write_json(evaluation / "metrics.json", report)
    print(json.dumps({"metrics_path": str(evaluation / "metrics.json"), "before": report["before"], "after": report["after"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

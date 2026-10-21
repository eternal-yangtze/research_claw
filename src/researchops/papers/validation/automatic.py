"""Automatic, cached formula checks shared by ingestion and study evidence."""
from __future__ import annotations

import json
from pathlib import Path

from researchops.core.artifacts import canonical_hash, file_hash, write_json
from .pipeline import VALIDATION_VERSION, implementation_hashes, validate_formulas


def validation_fingerprint() -> dict:
    return {**implementation_hashes(), "automatic.py": file_hash(Path(__file__))}


def automatic_formula_check(pdf: str | Path, parser_artifact: str | Path,
                            output: str | Path, *, source_package=None, source_receipt=None) -> dict:
    """Uses the existing parser output; never starts a second unbounded OCR pass."""
    pdf, parser_artifact, output = Path(pdf), Path(parser_artifact), Path(output)
    source = Path(source_package) if source_package else None
    receipt = Path(source_receipt) if source_receipt else (source.with_name(source.name + ".receipt.json") if source else None)
    inputs = {"pdf": file_hash(pdf), "parser": file_hash(parser_artifact),
              "source": file_hash(source) if source and source.is_file() else None,
              "receipt": file_hash(receipt) if receipt and receipt.is_file() else None,
              "implementation": validation_fingerprint(), "region_cap": 128}
    root = output.resolve() / canonical_hash(inputs)
    index = root / "index.json"
    if index.is_file():
        try:
            cached = json.loads(index.read_text(encoding="utf-8"))
            if (cached.get("inputs") == inputs and cached.get("status") != "validation_failed"
                    and cached.get("artifacts") and all(
                        Path(name).is_file() and file_hash(name) == digest for name, digest in cached["artifacts"].items())):
                return {**cached, "cache_hit": True}
        except (ValueError, OSError, TypeError, AttributeError):
            pass  # A broken cache index cannot establish validation success.
    root.mkdir(parents=True, exist_ok=True)
    try:
        payload = json.loads(parser_artifact.read_text(encoding="utf-8"))
        if (payload.get("pdf_sha256") or payload.get("source_sha256")) != inputs["pdf"]:
            raise ValueError("parser/PDF hash mismatch")
        document = payload.get("document") or payload.get("docling_document")
        if document is None:
            report = {"schema_version": VALIDATION_VERSION, "status": "not_assessed",
                      "reason": "parser has no formula regions; text extraction does not establish absence of formulas",
                      "pdf_sha256": inputs["pdf"], "regions": [],
                      "summary": {"formula_regions": 0, "recognition_accuracy": None, "detection_recall": None}}
            write_json(root / "not_assessed.json", report)
            result = {"status": "not_assessed", "report_path": str(root / "not_assessed.json"), "summary": report["summary"]}
        else:
            adapted = {**payload, "pdf_sha256": inputs["pdf"], "document": document}
            write_json(root / "parser_input.json", adapted)
            result = validate_formulas(pdf, root / "runs", docling_json=root / "parser_input.json",
                                       source_package=source if source and source.is_file() else None,
                                       source_receipt=receipt if receipt and receipt.is_file() else None)
            result["status"] = "checked_with_deferred_regions" if result["summary"]["deferred_regions"] else "checked"
        report_path = Path(result["report_path"])
        artifacts = {str(report_path): file_hash(report_path)}
        if (root / "parser_input.json").is_file():
            artifacts[str(root / "parser_input.json")] = file_hash(root / "parser_input.json")
        for artifact in report_path.parent.glob("*"):
            if artifact.is_file():
                artifacts[str(artifact)] = file_hash(artifact)
        result.update({"report_sha256": file_hash(report_path), "artifacts": artifacts})
    except Exception as error:
        result = {"status": "validation_failed", "reason": f"{type(error).__name__}: {error}", "artifacts": {}}
    result.update({"inputs": inputs, "cache_hit": False, "automatic": True,
                   "recognition_accuracy": None, "human_gold": False})
    write_json(index, result)
    return result


def formula_evidence_policy(row) -> dict:
    if row is None:
        return {"status": "missing_node", "eligible_for_planning": False,
                "safe_as_sole_reproduction_evidence": False}
    provenance = json.loads(row["provenance_json"] or "{}")
    validation = provenance.get("formula_validation", {})
    is_formula = row["node_type"] == "formula"
    status = validation.get("status", "not_assessed")
    # Unresolved formula nodes cannot independently support a plan's paper facts.
    allowed = not is_formula or status in {"accepted", "source_recovered"}
    if is_formula and allowed:
        path = validation.get("report_path")
        allowed = bool(path and Path(path).is_file() and file_hash(path) == validation.get("report_sha256"))
    return {"status": status, "eligible_for_planning": allowed,
            "safe_as_sole_reproduction_evidence": False,
            "selected_latex": validation.get("selected_latex") if is_formula else None,
            "reference_type": "source_derived_silver" if is_formula else None,
            "requires_original_page_review": is_formula,
            "report_path": validation.get("report_path")}


def formula_reproduction_gate(spec: dict | None, evidence: list[dict], store) -> dict:
    """Paper facts cannot rely on unresolved formulas or formulas alone."""
    allowed = {n["chunk_id"] for n in evidence}
    groups = [("baseline", (spec or {}).get("baseline_citations", []))]
    groups += [(f"paper_fact_{i}", fact.get("citations", []))
               for i, fact in enumerate((spec or {}).get("evidence_claims", [])) if fact.get("kind") == "paper_fact"]
    violations = []
    for label, refs in groups:
        rows = [store.get_chunk(ref) for ref in refs]
        for ref, row in zip(refs, rows):
            if row is None or ref not in allowed or not formula_evidence_policy(row)["eligible_for_planning"]:
                violations.append({"group": label, "chunk_id": ref, "reason": "missing or withheld evidence"})
            elif row["node_type"] == "formula" and not any(
                    other is not None and other["paper_id"] == row["paper_id"] and other["node_type"] != "formula"
                    and other["chunk_id"] in allowed for other in rows):
                violations.append({"group": label, "chunk_id": ref, "reason": "formula requires non-formula support from the same paper"})
    return {"passed": not violations, "violations": violations, "reference_type": "source_derived_silver",
            "independent_gold_accuracy": None}

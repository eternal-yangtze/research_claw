"""Versioned formula regions, auxiliary PDF text and conservative source checks."""
from __future__ import annotations

import json
import math
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from researchops.core.artifacts import canonical_hash, file_hash, write_json
from .formulas import compare_latex, normalize_latex, structure
from .sources import read_tex_package, source_formulas, source_identity


VALIDATION_VERSION = "formula-validation-v2"


def implementation_hashes() -> dict:
    return {name: file_hash(Path(__file__).with_name(name))
            for name in ["pipeline.py", "formulas.py", "sources.py", "macros.py"]}


def top_left_bbox(box: dict, width: float, height: float) -> list[float]:
    """Docling specifies its coordinate origin. Unknown origins must fail closed."""
    origin = str(box.get("coord_origin", "")).upper()
    left, top, right, bottom = [float(box[k]) for k in ("l", "t", "r", "b")]
    if origin == "BOTTOMLEFT":
        top, bottom = height - top, height - bottom
    elif origin != "TOPLEFT":
        raise ValueError("bbox coordinate origin is missing or unsupported")
    values = [left, top, right, bottom]
    if not all(math.isfinite(v) for v in values) or not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError("bbox is outside the original PDF page")
    return values


def decide_formula(latex: str, references: list[dict], identity: dict) -> dict:
    if len(latex) > 20000:
        return {"status": "unverified", "selected_latex": latex,
                "selection_reason": "candidate exceeds bounded comparison limit", "source_candidates": [],
                "safe_as_sole_reproduction_evidence": False}
    own = structure(latex)
    ranked = sorted(((compare_latex(latex, r.get("expanded_latex", r["latex"])), r) for r in references),
                    key=lambda pair: pair[0]["token_similarity"], reverse=True)
    candidates = [{"reference": r, "comparison": c} for c, r in ranked[:3]]
    base = {"status": "unverified", "selected_latex": latex,
            "selection_reason": "parser candidate retained; independent truth unavailable",
            "source_candidates": candidates, "structure": own,
            "confidence_basis": [], "safe_as_sole_reproduction_evidence": False}
    if not own["balanced"] or not normalize_latex(latex):
        return {**base, "status": "disputed", "selection_reason": "empty or structurally invalid parser candidate"}
    if not ranked or identity.get("status") != "matched_version":
        return base
    best, ref = ranked[0]
    exact = [(c, r) for c, r in ranked if c["normalized_exact_match"]]
    if exact and not own["unknown_commands"] and all(not r["unexpanded_macros"] and
            r.get("macro_expansion", {}).get("complete", True) for _, r in exact):
        return {**base, "status": "accepted",
                "selection_reason": "normalized parser expression corroborated by matched-version static source",
                "confidence_basis": ["matched_version", "normalized_exact_match", "balanced_structure", "no_unexpanded_macros"],
                "reference_type": "source_derived_silver"}
    if own["unknown_commands"]:
        return {**base, "selection_reason": "unresolved parser commands; source macro definitions are not applied to OCR guesses"}
    # A missing bold face can be restored from one unique matched-version source
    # expression. Keep this separate from OCR/source exact corroboration.
    recoverable = []
    for comparison, reference in ranked:
        diagnostic = comparison["symbol_diagnostics"]
        source_structure = comparison["right_structure"]
        if (diagnostic["bold_only_alignment"] and diagnostic["style_difference"]
                and not any(diagnostic["parser_styles"].values())
                and any(diagnostic["source_styles"][k] for k in ["mathbf", "boldsymbol", "bm"])
                and not reference["unexpanded_macros"] and reference.get("macro_expansion", {}).get("complete", True)
                and not own["unknown_commands"] and source_structure["balanced"]
                and not source_structure["unknown_commands"]):
            recoverable.append(reference)
    unique = {r["normalized"] for r in recoverable}
    if len(unique) == 1:
        reference = recoverable[0]
        return {**base, "status": "source_recovered",
                "selected_latex": reference.get("expanded_latex", reference["latex"]),
                "selection_reason": "unique matched-version source restores missing bold symbols; inferred source alignment, not OCR accuracy",
                "recovery_reference": reference, "reference_type": "source_derived_silver",
                "confidence_basis": ["matched_version", "unique_bold_only_alignment", "bounded_static_expansion"],
                "requires_original_page_review": True}
    if best["token_similarity"] >= 0.65 and not best["normalized_exact_match"]:
        return {**base, "status": "disputed",
                "selection_reason": "possible source match differs; alignment and semantics need review"}
    return {**base, "selection_reason": "no conclusive source match, or unresolved source macros"}


def validate_formulas(pdf: str | Path, output_root: str | Path, *,
                      docling_json: str | Path, source_package: str | Path | None = None,
                      source_receipt: str | Path | None = None, max_regions: int = 128) -> dict:
    """Keep all input artifacts untouched; each invocation gets a new run directory."""
    import pymupdf

    if not 1 <= max_regions <= 1024:
        raise ValueError("formula region cap must be between 1 and 1024")
    started = time.perf_counter()
    pdf = Path(pdf).resolve()
    parser_path = Path(docling_json).resolve()
    payload = json.loads(parser_path.read_text(encoding="utf-8"))
    digest = file_hash(pdf)
    if payload.get("pdf_sha256") != digest:
        raise ValueError("Docling output does not belong to this PDF hash")
    document = payload["document"]
    root = Path(output_root).resolve() / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:10])
    root.mkdir(parents=True, exist_ok=False)
    refs = []
    identity = {"status": "unverified", "reference_type": "none"}
    errors = []
    with pymupdf.open(pdf) as original:
        if source_package:
            source = Path(source_package).resolve()
            receipt = json.loads(Path(source_receipt).read_text(encoding="utf-8")) if source_receipt else None
            identity = source_identity(original[0].get_text(), source, receipt)
            try:
                refs = source_formulas(read_tex_package(source))
            except (ValueError, OSError) as error:
                errors.append({"stage": "static_source", "error": str(error)})
                identity["status"] = "unverified"
        regions = []
        formula_items = [item for item in document.get("texts", []) if item.get("label") in {"formula", "equation"}]
        for item in formula_items[:max_regions]:
            latex = str(item.get("text") or item.get("orig") or "")
            region = {"region_id": item.get("self_ref"), "parser": payload.get("parser"),
                      "parser_version": payload.get("parser_version"), "latex": latex,
                      "candidate_sha256": canonical_hash(latex), "locations": [],
                      "decision": decide_formula(latex, refs, identity)}
            for ordinal, location in enumerate(item.get("prov") or []):
                saved = {"original_provenance": location}
                try:
                    page_number = int(location["page_no"])
                    page = original[page_number - 1] if 1 <= page_number <= len(original) else None
                    if page is None:
                        raise ValueError("page number outside PDF")
                    # Docling can export resized pages; reject rather than crop the wrong area.
                    page_meta = document.get("pages", {}).get(str(page_number), {})
                    size = page_meta.get("size") or {}
                    if size and (abs(float(size["width"]) - page.rect.width) > 1 or abs(float(size["height"]) - page.rect.height) > 1):
                        raise ValueError("parser and PDF page dimensions differ")
                    bbox = top_left_bbox(location["bbox"], page.rect.width, page.rect.height)
                    clip = pymupdf.Rect(bbox)
                    # Keep a little visual context without including an entire page.
                    crop_clip = pymupdf.Rect(clip.x0 - 3, clip.y0 - 3, clip.x1 + 3, clip.y1 + 3) & page.rect
                    filename = f"region_{len(regions):04d}_{ordinal}.png"
                    page.get_pixmap(matrix=pymupdf.Matrix(2, 2), clip=crop_clip).save(root / filename)
                    saved.update({"page": page_number, "bbox_top_left": bbox,
                                  "crop_bbox_top_left": list(crop_clip), "crop_path": str(root / filename),
                                  "crop_sha256": file_hash(root / filename),
                                  "pymupdf_text": page.get_text("text", clip=clip, sort=True),
                                  "auxiliary_role": "text_and_location_only_not_independent_latex_ocr"})
                except (KeyError, ValueError, TypeError) as error:
                    saved["error"] = str(error)
                region["locations"].append(saved)
            if not region["locations"] or any("error" in loc for loc in region["locations"]):
                region["decision"].update(status="unverified", selection_reason="original PDF location could not be verified")
            regions.append(region)
    counts = Counter(r["decision"]["status"] for r in regions)
    summary = {"formula_regions": len(regions), "accepted": counts["accepted"],
               "source_recovered": counts["source_recovered"],
               "disputed": counts["disputed"], "unverified": counts["unverified"],
               "detected_formula_regions": len(formula_items), "deferred_regions": max(0, len(formula_items) - len(regions)),
               "regions_with_crops": sum(any("crop_path" in l for l in r["locations"]) for r in regions),
               "source_display_environments": len(refs),
               "source_corroboration_rate": counts["accepted"] / len(regions) if regions else None,
               "source_supported_or_recovered_rate": (counts["accepted"] + counts["source_recovered"]) / len(formula_items) if formula_items else None,
               "recognition_accuracy": None, "detection_recall": None,
               "metric_limitation": "source-derived silver; no independent gold or exhaustively annotated PDF regions"}
    report = {"schema_version": VALIDATION_VERSION, "pdf_path": str(pdf), "pdf_sha256": digest,
              "parser_artifact": str(parser_path), "parser_sha256": file_hash(parser_path),
              "implementation_sha256": implementation_hashes(),
              "parser_page_range": payload.get("page_range"), "parser_status": payload.get("status"),
              "source_package": str(Path(source_package).resolve()) if source_package else None,
              "source_sha256": file_hash(source_package) if source_package else None,
              "source_identity": identity, "summary": summary, "regions": regions,
              "errors": errors, "elapsed_seconds": time.perf_counter() - started,
              "llm_calls": 0, "token_usage": 0}
    write_json(root / "report.json", report)
    write_json(root / "source_formulas.json", refs)
    return {"report_path": str(root / "report.json"), "source_identity": identity, "summary": summary}

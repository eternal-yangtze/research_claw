"""Phase 3.1: 5 fixed failures + 3 deterministic controls; read-only PDF/TEI."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time
import tracemalloc
from xml.etree import ElementTree as ET

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(ROOT / "src"))
from researchclaw.knowledge.coordinate_alignment import align_coordinates, tokens
from researchclaw.knowledge.hybrid_structure import hybrid_chunks
from researchclaw.knowledge.rag_pipeline import extract_pages


def coverage(rows, eligible=False):
    selected = [r for r in rows if not eligible or len(tokens(r["text"])) >= 4]
    return sum(len(r["text"]) for r in selected if r["paragraph_id"]) / max(1, sum(len(r["text"]) for r in selected))


def score_gold(labels, predictions):
    """Score only independently reviewed rows, never fill labels from predictions."""
    reviewed = [r for r in labels if r.get("reviewer") and isinstance(r.get("is_alignable"), bool)]
    tp = accepted = positives = correct_sections = section_cases = 0
    for label in reviewed:
        key = (label["paper_id"], label["page"], label["line_ordinal"])
        row = predictions[key]
        if row["text"] != label["text"]:
            raise ValueError("Gold source text changed")
        if label["is_alignable"] and not label.get("expected_paragraph_id"):
            raise ValueError("Alignable gold requires an independently assigned paragraph ID")
        positive = label["is_alignable"]
        prediction = row["paragraph_id"] is not None
        positives += positive
        accepted += prediction
        tp += bool(positive and row["paragraph_id"] == label.get("expected_paragraph_id"))
        if label.get("expected_section_id") is not None:
            section_cases += 1
            correct_sections += row["section_id"] == label["expected_section_id"]
    precision = tp / accepted if accepted else None
    recall = tp / positives if positives else None
    f1 = (2 * precision * recall / (precision + recall) if precision + recall else 0) if precision is not None and recall is not None else None
    return {"reviewed_rows": len(reviewed), "alignment_precision": precision,
            "alignment_recall": recall, "alignment_f1": f1,
            "chapter_attribution_accuracy": correct_sections / section_cases if section_cases else None,
            "scope": "Representative audit sample, not an unbiased population estimate"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=ROOT / "self-revise/backend-revise/rag-evidence/coordinate-phase3-1")
    ap.add_argument("--geometry", choices=("line-box", "word-union"), default="word-union")
    ap.add_argument("--gold", type=Path, help="Independently completed human-review JSON")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = ROOT / "self-revise/backend-revise/rag-evidence/grobid-phase2-sample"
    previous = json.loads((ROOT / "self-revise/backend-revise/rag-evidence/hybrid-phase3-sample-v2/report.json").read_text(encoding="utf8"))
    failures = set(previous["alignment_gate_failures"])
    controls = []
    for group in ("double_column", "single_column", "formula_table_heavy"):
        controls.append(sorted(r["paper_id"] for r in previous["papers"]
                               if r["stratum"] == group and r["paper_id"] not in failures)[0])
    manifest = json.loads((source / "sample-manifest.json").read_text(encoding="utf8"))
    selected = [x for x in manifest if x["paper_id"] in failures | set(controls)]
    config = {"sample": selected, "control_ids": controls, "geometry": args.geometry,
              "thresholds": {"eligible_char_coverage_per_failure": .80,
                             "maximum_control_coverage_drop": .01,
                             "minimum_spatial_containment": .50,
                             "minimum_text_overlap": .65, "minimum_score_margin": .10},
              "gold_policy": "Independent human labels only; missing labels mean pending, never pass",
              "formal_database_access": False}
    (args.output / "manifest-and-gates.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf8")
    reports, reviews, predictions = [], [], {}
    for item in selected:
        pid = item["paper_id"]
        path = Path(item["path"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
        doc = json.loads((source / "schema" / (pid + ".json")).read_text(encoding="utf8"))
        root = ET.parse(next((source / "tei").glob(pid + "-*.tei.xml"))).getroot()
        pages = extract_pages(path)
        tracemalloc.start()
        started, cpu = time.perf_counter(), time.process_time()
        baseline, aligned, geometry = align_coordinates(path, pages, doc, root, geometry_mode=args.geometry)
        predictions.update({(pid, r["page"], r["line_ordinal"]): r for r in aligned})
        seconds, cpu_seconds = time.perf_counter() - started, time.process_time() - cpu
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert [(r["page"], r["line_ordinal"], r["text"]) for r in baseline] == [(r["page"], r["line_ordinal"], r["text"]) for r in aligned]
        a_chunks, c_chunks = hybrid_chunks(baseline), hybrid_chunks(aligned)
        assert " ".join(r["text"] for r in c_chunks).split() == " ".join(p.text for p in pages).split()
        changes = Counter("unchanged" if a["paragraph_id"] == b["paragraph_id"] else
                          "new" if not a["paragraph_id"] and b["paragraph_id"] else
                          "rejected" if a["paragraph_id"] and not b["paragraph_id"] else "changed_paragraph"
                          for a, b in zip(baseline, aligned))
        row = {"paper_id": pid, "title": doc["paper"]["title"], "control": pid in controls,
               "baseline_eligible_coverage": coverage(baseline, True),
               "coordinate_eligible_coverage": coverage(aligned, True),
               "baseline_all_coverage": coverage(baseline), "coordinate_all_coverage": coverage(aligned),
               "baseline_short_line_coverage": coverage([r for r in baseline if len(tokens(r["text"])) < 4]),
               "coordinate_short_line_coverage": coverage([r for r in aligned if len(tokens(r["text"])) < 4]),
               "changes": dict(changes), "source_retention": 1.0,
               "child_counts": {"baseline": len(a_chunks), "coordinate": len(c_chunks)},
               "known_section_children": sum(r["section_id"] is not None for r in c_chunks),
               "unsupported_geometry_pages": sum(not g["supported"] for g in geometry),
               "wall_seconds": seconds, "cpu_seconds": cpu_seconds,
               "python_tracemalloc_peak_bytes": peak,
               "memory_scope": "Python allocations only; excludes native PDF allocations and process RSS"}
        reports.append(row)
        evidence = {"metrics": row, "page_geometry": geometry, "lines": aligned}
        (args.output / (pid + ".json")).write_text(json.dumps(evidence, ensure_ascii=False), encoding="utf8")
        # Representative audit sample, not a probability sample. Algorithm
        # predictions are kept in evidence files, not prefilled as truth.
        pools = [ [r for r in aligned if r["paragraph_id"] and not r["baseline_paragraph_id"]],
                  [r for r in aligned if r["paragraph_id"] and r["baseline_paragraph_id"] and r["paragraph_id"] != r["baseline_paragraph_id"]],
                  [r for r in aligned if len(tokens(r["text"])) < 4],
                  [r for r in aligned if not r["paragraph_id"]],
                  [r for r in aligned if r["paragraph_id"]] ]
        chosen = {}
        for pool in pools:
            for r in sorted(pool, key=lambda v: hashlib.sha256(f'{pid}:{v["page"]}:{v["line_ordinal"]}'.encode()).hexdigest())[:3]:
                chosen[(r["page"], r["line_ordinal"])] = r
        for r in chosen.values():
            reviews.append({"paper_id": pid, "pdf_path": str(path), "page": r["page"],
                            "line_ordinal": r["line_ordinal"], "text": r["text"],
                            "expected_paragraph_id": None, "expected_section_id": None,
                            "is_alignable": None, "reviewer": None, "review_notes": None})
        print(json.dumps(row, ensure_ascii=True), flush=True)
    failed = [r["paper_id"] for r in reports if not r["control"] and r["coordinate_eligible_coverage"] < .8]
    regressions = [r["paper_id"] for r in reports if r["control"] and r["coordinate_eligible_coverage"] < r["baseline_eligible_coverage"] - .01]
    report = {"papers": reports, "coverage_gate_failures": failed, "control_regressions": regressions,
              "geometry": args.geometry,
              "alignment_precision": None, "alignment_recall": None, "alignment_f1": None,
              "chapter_attribution_accuracy": None, "gold_status": "pending_independent_human_labels",
              "full_corpus_gate_passed": False, "formal_migration": False}
    if args.gold:
        report.update(score_gold(json.loads(args.gold.read_text(encoding="utf8")), predictions))
        report["gold_status"] = "provided_review_labels_scored"
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")
    review_path = args.output / "human-review-template.json"
    if not review_path.exists():
        review_path.write_text(json.dumps(reviews, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps({k: v for k, v in report.items() if k != "papers"}, indent=2))


if __name__ == "__main__":
    main()

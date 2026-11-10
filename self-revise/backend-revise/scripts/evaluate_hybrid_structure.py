"""Phase 3A/B sample gate: no database writes, vector creation or migration."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from xml.etree import ElementTree as ET

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(ROOT / "src"))
from researchclaw.knowledge.hybrid_structure import (
    align_cleaned_pages, paragraph_chunks, hybrid_chunks, recover_hierarchy, tokens)
from researchclaw.knowledge.rag_pipeline import extract_pages, _parent_child_chunks_v1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, default=ROOT / "self-revise/backend-revise/rag-evidence/grobid-phase2-sample")
    parser.add_argument("--output", type=Path, default=ROOT / "self-revise/backend-revise/rag-evidence/hybrid-phase3-sample")
    parser.add_argument("--packing", choices=("strict", "bounded"), default="bounded")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    # Freeze before processing. These measure source integrity and annotation
    # availability, not human accuracy. Failure stops expansion to 194 papers.
    gates = {"minimum_per_paper_eligible_char_alignment": .80,
             "minimum_source_line_retention": 1.0,
             "maximum_duplicate_source_lines": 0,
             "human_structure_review": "required_before_full_corpus"}
    (args.output / "gates.json").write_text(json.dumps(gates, indent=2), encoding="utf-8")
    manifest = json.loads((args.sample / "sample-manifest.json").read_text(encoding="utf-8"))
    output = []
    for item in manifest:
        started = time.perf_counter()
        path = Path(item["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"PDF changed: {path}")
        doc = json.loads((args.sample / "schema" / (item["paper_id"] + ".json")).read_text(encoding="utf-8"))
        tei = next((args.sample / "tei").glob(item["paper_id"] + "-*.tei.xml"))
        sections = recover_hierarchy(doc, ET.fromstring(tei.read_bytes()))
        pages = extract_pages(path)
        a = [c for c in _parent_child_chunks_v1(item["paper_id"], pages, "A", "time-series") if c.node_type == "child"]
        b_units = [{"text": p["text"], "section_id": p["section_id"],
                    "page": p["page_start"] if p["page_start"] == p["page_end"] else None,
                    "paragraph_id": p["paragraph_id"], "coordinates": p["coordinates"]}
                   for p in doc["paragraphs"]]
        # Abstract is a TEI unit with unknown page unless independently mapped.
        if doc["paper"].get("abstract"):
            b_units.insert(0, {"text": doc["paper"]["abstract"], "section_id": "abstract", "page": None})
        b = paragraph_chunks(b_units)
        lines = align_cleaned_pages(pages, doc)
        c = hybrid_chunks(lines) if args.packing == "bounded" else paragraph_chunks(lines)
        expected = [(p.page, i, text) for p in pages for i, text in enumerate(p.text.splitlines()) if text.strip()]
        actual = [(r["page"], r["line_ordinal"], r["text"]) for r in lines]
        assert actual == expected, "Hybrid altered or reordered cleaned source lines"
        # Whitespace-normalized child stream must preserve all authoritative text.
        assert " ".join(x["text"] for x in c).split() == " ".join(p.text for p in pages).split()
        eligible = [r for r in lines if len(tokens(r["text"])) >= 4]
        total_chars = sum(len(r["text"]) for r in eligible)
        aligned = sum(len(r["text"]) for r in eligible if r["paragraph_id"])
        counts = {name: len(chunks) for name, chunks in (("A", a), ("B", b), ("C", c))}
        row = {"paper_id": item["paper_id"], "title": doc["paper"]["title"], "stratum": item["stratum"],
               "child_counts": counts, "source_line_retention": 1.0,
               "duplicate_source_lines": len(actual) - len({(r[0], r[1]) for r in actual}),
               "eligible_char_alignment": aligned / max(1, total_chars),
               "all_char_alignment": sum(len(r["text"]) for r in lines if r["paragraph_id"]) / max(1, sum(len(r["text"]) for r in lines)),
               "fallback_lines": sum(not r["paragraph_id"] for r in lines),
               "sections": len(sections),
               "known_hierarchy": sum(s["section_level"] is not None for s in sections),
               "inferred_parent_links": sum(s["parent_section_id"] is not None for s in sections),
               "B_unknown_page_children": sum(x["page"] is None for x in b),
               "C_unknown_page_children": sum(x["page"] is None for x in c),
               "C_known_section_children": sum(x["section_id"] is not None for x in c),
               "seconds": time.perf_counter() - started}
        candidate = {"paper_id": item["paper_id"], "sections": sections,
                     "formulas": doc["formulas"], "figures_tables": doc["figures_tables"],
                     "A": [{"text": x.text, "page": x.page, "section": x.section} for x in a],
                     "B": b, "C": c, "aligned_source_lines": lines}
        (args.output / (item["paper_id"] + ".json")).write_text(json.dumps(candidate, ensure_ascii=False), encoding="utf-8")
        output.append(row)
        print(json.dumps(row, ensure_ascii=True), flush=True)
    failed = [r["paper_id"] for r in output if r["eligible_char_alignment"] < .80]
    ratio = sum(r["child_counts"]["C"] for r in output) / sum(r["child_counts"]["A"] for r in output)
    report = {"scope": "20-paper parser/chunking feasibility; not retrieval A/B or accuracy",
              "gates": gates, "packing": args.packing, "papers": output, "alignment_gate_failures": failed,
              "mean_eligible_char_alignment": sum(r["eligible_char_alignment"] for r in output) / len(output),
              "child_counts": {k: sum(r["child_counts"][k] for r in output) for k in "ABC"},
              "child_ratio_vs_A": ratio,
              "automatic_quality_gate_passed": not failed,
              "full_corpus_gate_passed": False,
              "human_review": "pending", "formal_migration": False}
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "papers"}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()

"""Import user gold1, score frozen predictions and general refinement."""
import hashlib
import json
from pathlib import Path
import re
import sys
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from researchclaw.knowledge.alignment_refinement import refine_lines
from evaluate_coordinate_alignment import score_gold, coverage


def main():
    base = ROOT / "self-revise/backend-revise/rag-evidence"
    review = base / "alignment-human-review"
    output = base / "alignment-gold1-refinement"
    output.mkdir(exist_ok=True)
    gold_path = ROOT / "self-revise/preserved-specs/金标1.md"
    text = gold_path.read_text(encoding="utf8")
    cases = {c["case_id"]: c for c in json.loads((review / "cases.json").read_text(encoding="utf8"))}
    labels = []
    for match in re.finditer(r"^## (R\d{2}-\d)\s*\n(.*?)(?=^## R|\Z)", text, re.M | re.S):
        cid, block = match.groups()
        def field(name):
            found = re.search(r"^- " + name + r"：\s*(.+)$", block, re.M)
            if not found:
                raise ValueError(f"Missing {cid} {name}")
            return found[1].strip().strip("*`")
        alignable = field("可对齐")
        if alignable not in ("true", "false"):
            raise ValueError("Invalid gold Boolean")
        p, s = field("正确 paragraph_id"), field("正确 section_id")
        p = None if p == "null" else p.split("`")[0]
        s = None if s == "null" else s.split("`")[0]
        case = cases[cid]
        labels.append({"case_id": cid, "paper_id": case["paper_id"], "page": case["page"],
                       "line_ordinal": case["line_ordinal"], "text": case["text"],
                       "is_alignable": alignable == "true", "expected_paragraph_id": p,
                       "expected_section_id": s, "reviewer": "user_gold1",
                       "source": str(gold_path.relative_to(ROOT)), "review_notes": block,
                       "mapping_extent": "partial_or_majority" if cid in ("R02-3", "R03-3", "R08-3") else "as_reviewed"})
    assert len(labels) == len(cases) == 24
    predictions, refined, lexical = {}, {}, {}
    per_paper = []
    for path in sorted((base / "coordinate-phase3-1-rawline").glob("rag_*.json")):
        d = json.loads(path.read_text(encoding="utf8"))
        doc = json.loads((base / "grobid-phase2-sample/schema" / path.name).read_text(encoding="utf8"))
        rows = refine_lines(d["lines"], doc, d["page_geometry"])
        assert [(r["page"],r["line_ordinal"],r["text"]) for r in rows] == [(r["page"],r["line_ordinal"],r["text"]) for r in d["lines"]]
        for old, new in zip(d["lines"], rows):
            key = (path.stem, old["page"], old["line_ordinal"])
            predictions[key], refined[key] = old, new
            p = next((p for p in doc["paragraphs"] if p["paragraph_id"] == old["baseline_paragraph_id"]), None)
            lexical[key] = dict(old, paragraph_id=old["baseline_paragraph_id"], section_id=p["section_id"] if p else None)
        per_paper.append({"paper_id": path.stem, "old_coverage": coverage(d["lines"], True),
                          "new_coverage": coverage(rows, True), "changed_rows": sum(a["paragraph_id"] != b["paragraph_id"] for a,b in zip(d["lines"],rows))})
        (output / path.name).write_text(json.dumps({**d,"lines":rows},ensure_ascii=False),encoding="utf8")
    results = {"source_gold_sha256": hashlib.sha256(gold_path.read_bytes()).hexdigest(),
               "scope": "24 user-reviewed diagnostic cases; development regression, NOT held-out evaluation",
               "metric_definition": "User-selected primary paragraph, including partial/majority attribution; not full-span accuracy",
               "lexical": score_gold(labels,lexical), "coordinate":score_gold(labels,predictions),
               "refined":score_gold(labels,refined), "papers":per_paper,
               "cases":[{"case_id":g["case_id"],"expected":g["expected_paragraph_id"],
                         "old":predictions[(g["paper_id"],g["page"],g["line_ordinal"])]["paragraph_id"],
                         "new":refined[(g["paper_id"],g["page"],g["line_ordinal"])]["paragraph_id"]} for g in labels],
               "formal_migration":False,"holdout_human_review":"required"}
    (output / "gold1.json").write_text(json.dumps(labels,ensure_ascii=False,indent=2),encoding="utf8")
    (output / "report.json").write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding="utf8")
    print(json.dumps(results,ensure_ascii=True,indent=2))


if __name__ == "__main__":
    main()

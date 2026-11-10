"""Evaluate GROBID structure extraction on a deterministic 20-paper sample."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from difflib import SequenceMatcher
from xml.etree import ElementTree as ET

import fitz
import requests

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(ROOT / "src"))

from researchclaw.core.qdrant_store import QdrantResearchStore
from researchclaw.knowledge.rag_pipeline import extract_pages

NS = {"tei": "http://www.tei-c.org/ns/1.0"}
HEADING = re.compile(r"^(?:\d+(?:\.\d+)*|[IVXLC]+)[.)]?\s+\S", re.I)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pdf_profile(path: Path) -> dict:
    with fitz.open(path) as doc:
        pages = len(doc)
        chars = 0
        narrow_left = narrow_right = blocks_seen = headings = formula_tables = 0
        for page in doc:
            text = page.get_text("text") or ""
            chars += len(re.sub(r"\s+", "", text))
            headings += sum(bool(HEADING.match(line.strip())) for line in text.splitlines())
            formula_tables += len(re.findall(r"\b(?:table|equation|theorem|algorithm|lemma)\b|[=∑∫≤≥]", text, re.I))
            width = page.rect.width
            for x0, _y0, x1, _y1, value, *_ in page.get_text("blocks"):
                if len(str(value).strip()) < 80:
                    continue
                blocks_seen += 1
                if x1 - x0 < width * .62:
                    if (x0 + x1) / 2 < width / 2:
                        narrow_left += 1
                    else:
                        narrow_right += 1
        column_score = min(narrow_left, narrow_right) / max(1, blocks_seen)
    return {"pages": pages, "pymupdf_chars": chars, "column_score": column_score,
            "heading_score": headings / max(1, pages),
            "formula_table_score": formula_tables / max(1, pages)}


def select_sample(rows: list[dict]) -> list[dict]:
    candidates = []
    for row in rows:
        path = Path(row["file_path"])
        if not path.is_file() or path.suffix.lower() != ".pdf":
            continue
        profile = pdf_profile(path)
        candidates.append({**row, **profile, "path": str(path), "sha256": file_hash(path)})
    selected: list[dict] = []
    used: set[str] = set()
    strata = [
        ("double_column", lambda x: (x["column_score"], x["pages"])),
        ("single_column", lambda x: (-x["column_score"], x["pages"])),
        ("complex_headings", lambda x: (x["heading_score"], x["pages"])),
        ("formula_table_heavy", lambda x: (x["formula_table_score"], x["pages"])),
    ]
    for label, score in strata:
        available = [item for item in candidates if item["sha256"] not in used]
        for item in sorted(available, key=lambda x: (*score(x), x["sha256"]), reverse=True)[:5]:
            used.add(item["sha256"])
            selected.append({**item, "stratum": label})
    return selected


def memory_bytes(container: str) -> int:
    try:
        raw = subprocess.check_output(
            ["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", container],
            text=True, stderr=subprocess.DEVNULL, timeout=10).split("/")[0].strip()
        value, unit = re.match(r"([0-9.]+)([KMG]i?B)", raw).groups()
        factors = {"KB": 1000, "KiB": 1024, "MB": 10**6, "MiB": 1024**2,
                   "GB": 10**9, "GiB": 1024**3}
        return int(float(value) * factors[unit])
    except Exception:
        return 0


def parse_coords(value: str | None) -> list[dict]:
    result = []
    for part in (value or "").split(";"):
        fields = part.split(",")
        if len(fields) == 5:
            try:
                result.append({"page": int(fields[0]), "x": float(fields[1]), "y": float(fields[2]),
                               "width": float(fields[3]), "height": float(fields[4])})
            except ValueError:
                pass
    return result


def normalized_tei(root: ET.Element, item: dict) -> tuple[dict, dict]:
    title = " ".join("".join(root.find(".//tei:titleStmt/tei:title", NS).itertext()).split()) \
        if root.find(".//tei:titleStmt/tei:title", NS) is not None else ""
    abstract_node = root.find(".//tei:profileDesc/tei:abstract", NS)
    abstract = " ".join("".join(abstract_node.itertext()).split()) if abstract_node is not None else ""
    sections = []
    paragraphs = []
    formulas = []
    figures = []
    section_index = paragraph_index = 0

    def visit_div(div: ET.Element, parent_id: str | None, level: int) -> None:
        nonlocal section_index, paragraph_index
        head = div.find("tei:head", NS)
        section_index += 1
        section_id = f"{item['paper_id']}:section:{section_index}"
        heading = " ".join("".join(head.itertext()).split()) if head is not None else ""
        coords = parse_coords(head.get("coords") if head is not None else None)
        section_paragraphs = div.findall("tei:p", NS)
        page_values = [coord["page"] for node in section_paragraphs
                       for coord in parse_coords(node.get("coords"))]
        sections.append({"section_id": section_id, "paper_id": item["paper_id"],
                         "section_title": heading, "section_type": "unknown",
                         "section_level": level, "source_number": div.get("n"),
                         "parent_section_id": parent_id,
                         "page_start": min(page_values) if page_values else (coords[0]["page"] if coords else None),
                         "page_end": max(page_values) if page_values else (coords[-1]["page"] if coords else None),
                         "structure_confidence": None})
        for node in section_paragraphs:
            text = " ".join("".join(node.itertext()).split())
            if not text:
                continue
            paragraph_index += 1
            locations = parse_coords(node.get("coords"))
            pages = [value["page"] for value in locations]
            paragraphs.append({"paragraph_id": f"{item['paper_id']}:paragraph:{paragraph_index}",
                               "section_id": section_id, "text": text,
                               "page_start": min(pages) if pages else None,
                               "page_end": max(pages) if pages else None,
                               "source_element_id": node.get("{http://www.w3.org/XML/1998/namespace}id"),
                               "coordinates": locations})
        for child in div.findall("tei:div", NS):
            visit_div(child, section_id, level + 1)

    body = root.find(".//tei:body", NS)
    if body is not None:
        for div in body.findall("tei:div", NS):
            visit_div(div, None, 1)
    for index, node in enumerate(root.findall(".//tei:formula", NS), 1):
        formulas.append({"formula_id": f"{item['paper_id']}:formula:{index}",
                         "text": " ".join("".join(node.itertext()).split()),
                         "coordinates": parse_coords(node.get("coords"))})
    for index, node in enumerate(root.findall(".//tei:figure", NS), 1):
        figures.append({"figure_id": f"{item['paper_id']}:figure:{index}",
                        "type": node.get("type", "figure"),
                        "coordinates": parse_coords(node.get("coords"))})
    tei_chars = sum(len(re.sub(r"\s+", "", row["text"])) for row in paragraphs)
    mapped = sum(row["page_start"] is not None for row in paragraphs)
    document = {"paper": {"paper_id": item["paper_id"], "title": title,
                           "language": root.get("{http://www.w3.org/XML/1998/namespace}lang"),
                           "parser": "grobid", "parser_version": "0.8.2",
                           "abstract": abstract},
                "sections": sections, "paragraphs": paragraphs,
                "formulas": formulas, "figures_tables": figures}
    metrics = {"title_present": bool(title), "abstract_present": bool(abstract),
               "title_agreement_with_catalog": SequenceMatcher(
                   None, title.casefold(), str(item.get("title") or "").casefold()).ratio(),
               "sections": len(sections), "titled_sections": sum(bool(x["section_title"]) for x in sections),
               "nested_sections": sum(x["parent_section_id"] is not None for x in sections),
               "paragraphs": len(paragraphs), "formulas": len(formulas), "figures_tables": len(figures),
               "tei_body_chars": tei_chars,
               "paragraph_coverage_vs_pymupdf": tei_chars / max(1, item["pymupdf_chars"]),
                "body_abstract_coverage_vs_current_cleaner": (
                    tei_chars + len(re.sub(r"\s+", "", abstract))) /
                    max(1, item.get("current_clean_chars", item["pymupdf_chars"])),
               "paragraph_page_mapping_ratio": mapped / max(1, len(paragraphs))}
    return document, metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--grobid-url", default="http://127.0.0.1:8070")
    parser.add_argument("--container", default="researchclaw-grobid-experiment")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    raw_dir, schema_dir = args.output / "tei", args.output / "schema"
    raw_dir.mkdir(exist_ok=True); schema_dir.mkdir(exist_ok=True)
    store = QdrantResearchStore(args.workspace)
    with store.database.connect(read_only=True) as db:
        rows = [dict(row) for row in db.execute(
            "SELECT paper_id,title,file_path,quality_status FROM papers ORDER BY paper_id")]
    sample = select_sample(rows)
    for item in sample:
        current_pages = extract_pages(Path(item["path"]))
        item["current_clean_chars"] = sum(len(re.sub(r"\s+", "", page.text)) for page in current_pages)
    (args.output / "sample-manifest.json").write_text(
        json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8")
    results = []
    for index, item in enumerate(sample, 1):
        tei_path = raw_dir / f"{item['paper_id']}-{item['sha256'][:12]}.tei.xml"
        cache_hit = tei_path.exists()
        started = time.perf_counter(); peak = None; parse_elapsed = None; error = None
        if not cache_hit:
            try:
                peak = memory_bytes(args.container)
                parse_started = time.perf_counter()
                with Path(item["path"]).open("rb") as stream:
                    response = requests.post(
                        args.grobid_url.rstrip("/") + "/api/processFulltextDocument",
                        files={"input": (Path(item["path"]).name, stream, "application/pdf")},
                        data={"consolidateHeader": "0", "consolidateCitations": "0",
                              "segmentSentences": "1", "teiCoordinates": ["head", "p", "s", "formula", "figure"]},
                        timeout=300)
                response.raise_for_status(); tei_path.write_bytes(response.content)
                parse_elapsed = time.perf_counter() - parse_started
                peak = max(peak, memory_bytes(args.container))
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - started
        row = {"ordinal": index, "paper_id": item["paper_id"], "title": item["title"],
               "path": item["path"], "sha256": item["sha256"], "stratum": item["stratum"],
               "pages": item["pages"], "cache_hit": cache_hit,
               "processing_wall_seconds": elapsed, "parse_wall_seconds": parse_elapsed,
               "container_memory_bytes": peak, "error": error}
        if error is None:
            try:
                document, metrics = normalized_tei(ET.fromstring(tei_path.read_bytes()), item)
                schema_path = schema_dir / f"{item['paper_id']}.json"
                schema_path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
                row.update(metrics); row["tei_path"] = str(tei_path); row["schema_path"] = str(schema_path)
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
        results.append(row)
        print(json.dumps({"paper": index, "of": len(sample), "paper_id": item["paper_id"],
                          "status": "failed" if row["error"] else "passed",
                          "seconds": round(elapsed, 2)}, ensure_ascii=False), flush=True)
    passed = [row for row in results if not row["error"]]
    mean = lambda key: sum(float(row.get(key) or 0) for row in passed) / max(1, len(passed))
    parsed = [row for row in passed if row["parse_wall_seconds"] is not None]
    report = {"status": "passed" if len(passed) == len(results) else "completed_with_failures",
              "parser": "grobid", "parser_version": "0.8.2", "sample_count": len(results),
              "passed": len(passed), "failed": len(results) - len(passed),
              "automatic_metrics": {"title_presence_rate": mean("title_present"),
                  "abstract_presence_rate": mean("abstract_present"),
                  "mean_title_agreement_with_catalog": mean("title_agreement_with_catalog"),
                  "mean_sections": mean("sections"), "mean_titled_sections": mean("titled_sections"),
                  "mean_nested_sections": mean("nested_sections"),
                  "mean_paragraph_coverage_vs_pymupdf": mean("paragraph_coverage_vs_pymupdf"),
                  "mean_body_abstract_coverage_vs_current_cleaner": mean(
                      "body_abstract_coverage_vs_current_cleaner"),
                  "mean_paragraph_page_mapping_ratio": mean("paragraph_page_mapping_ratio"),
                  "mean_processing_wall_seconds": mean("processing_wall_seconds"),
                  "mean_parse_wall_seconds": (
                      sum(row["parse_wall_seconds"] for row in parsed) / len(parsed) if parsed else None),
                  "cache_hits": sum(row["cache_hit"] for row in results),
                  "peak_container_memory_bytes": max(
                      (row["container_memory_bytes"] for row in results
                       if row["container_memory_bytes"] is not None), default=None)},
              "manual_metrics_status": "pending_human_review",
              "manual_review_required": ["title_accuracy", "abstract_completeness",
                  "section_precision_recall", "hierarchy_accuracy", "page_mapping_accuracy"],
              "papers": results}
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "papers"}, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

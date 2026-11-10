"""Opt-in structure candidate; never writes a corpus or changes default parsing.

PyMuPDF cleaned lines are the authoritative text stream. GROBID paragraphs
annotate that stream only after page-constrained lexical matching. No TEI
paragraph is appended to the stream, so fallback cannot duplicate source text.
"""
from __future__ import annotations

from collections import defaultdict
import re
from xml.etree import ElementTree as ET

from researchclaw.knowledge.rag_pipeline import repair_mojibake

NS = {"tei": "http://www.tei-c.org/ns/1.0"}


def tokens(text):
    return set(re.findall(r"[a-z0-9]+", repair_mojibake(text).casefold()))


def recover_hierarchy(document, tei_root):
    """Only recover explicit numeric prefixes with an already observed parent.

    Flat unnumbered headings remain unknown. This is rule inference, never a
    claim of native TEI hierarchy or a calibrated confidence score.
    """
    heads = tei_root.findall(".//tei:body//tei:div/tei:head", NS)
    numbered = {}
    sections = []
    for original in document["sections"]:
        row = dict(original)
        row.update(section_level=None, parent_section_id=None,
                   hierarchy_source="unknown", structure_confidence=None)
        # Align by actual heading text, not index (some divs have no head).
        matches = [h for h in heads if " ".join("".join(h.itertext()).split()) == row["section_title"]]
        raw = matches[0].get("n", "") if len(matches) == 1 else ""
        match = re.fullmatch(r"\s*(\d+(?:\.\d+)*)[.)]?\s*", raw)
        if match:
            number = match.group(1)
            parts = number.split(".")
            parent = numbered.get(".".join(parts[:-1])) if len(parts) > 1 else None
            if len(parts) == 1 or parent:
                row.update(section_level=len(parts), parent_section_id=parent,
                           hierarchy_source="explicit_number_rule", source_number=number)
                numbered[number] = row["section_id"]
        sections.append(row)
    return sections


def align_cleaned_pages(pages, document, *, minimum_overlap=.65, minimum_margin=.10):
    by_page = defaultdict(list)
    for para in document["paragraphs"]:
        # Use actual coordinate pages, never a fabricated range between endpoints.
        for page in sorted({c["page"] for c in para["coordinates"]}):
            by_page[page].append((para, tokens(para["text"])))
    rows = []
    for page in pages:
        for ordinal, text in enumerate(page.text.splitlines()):
            if not text.strip():
                continue
            words = tokens(text)
            scored = sorted(((len(words & vocab) / max(1, len(words)), para)
                             for para, vocab in by_page[page.page]),
                            key=lambda x: (-x[0], x[1]["paragraph_id"])) if len(words) >= 4 else []
            best = scored[0][0] if scored else 0
            second = scored[1][0] if len(scored) > 1 else 0
            matched = bool(scored and best >= minimum_overlap and best - second >= minimum_margin)
            para = scored[0][1] if matched else {}
            rows.append({"text": text, "page": page.page, "line_ordinal": ordinal,
                         "source": "pymupdf_cleaned_line", "section_id": para.get("section_id"),
                         "paragraph_id": para.get("paragraph_id"),
                         "alignment_overlap": best, "alignment_margin": best - second,
                         "structure_source": "grobid_matched" if matched else "unknown"})
    return rows


def paragraph_chunks(units, *, max_words=220):
    """Pack consecutive paragraph units within one section/page, with a hard cap.

    Oversized paragraphs split only at a word safety bound. All source units
    remain traceable. This is paragraph-aware packing, not embedding semantics.
    """
    result, current = [], None
    for unit in units:
        words = unit["text"].split()
        for start in range(0, len(words), max_words):
            text = " ".join(words[start:start + max_words])
            key = (unit.get("section_id"), unit.get("page"))
            if current is None or current["key"] != key or len(current["text"].split()) + len(text.split()) > max_words:
                current = {"key": key, "section_id": key[0], "page": key[1], "text": "", "sources": []}
                result.append(current)
            current["text"] = (current["text"] + " " + text).strip()
            current["sources"].append({k: v for k, v in unit.items() if k != "text"})
    for row in result:
        row.pop("key")
    return result


def hybrid_chunks(lines, *, max_words=220, minimum_section_words=80):
    """Bound fragmentation without inventing a section for fallback lines.

    A confident section transition can end a sufficiently large chunk. Short
    unknown runs do not force boundaries. Mixed/unknown chunks keep unknown
    section metadata; exact line annotations remain available in sources.
    """
    result, current = [], None
    for line in lines:
        words = line["text"].split()
        for start in range(0, len(words), max_words):
            fragment = " ".join(words[start:start + max_words])
            section = line.get("section_id")
            size = len(current["text"].split()) if current else 0
            boundary = bool(current and section and current["section_id"] and
                            section != current["section_id"] and size >= minimum_section_words)
            if (current is None or current["page"] != line["page"] or boundary
                    or size + len(fragment.split()) > max_words):
                current = {"text": "", "page": line["page"], "section_id": section, "sources": []}
                result.append(current)
            elif section != current["section_id"]:
                current["section_id"] = None
            current["text"] = (current["text"] + " " + fragment).strip()
            current["sources"].append({k: v for k, v in line.items() if k != "text"})
    return result

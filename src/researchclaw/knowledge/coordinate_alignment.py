"""Experimental spatial alignment. No corpus writes or default-parser hooks."""
from collections import defaultdict
import math
import re

import pymupdf

from researchclaw.knowledge.hybrid_structure import align_cleaned_pages, tokens
from researchclaw.knowledge.rag_pipeline import repair_mojibake


def normalized(text):
    return re.sub(r"\s+", " ", repair_mojibake(text).replace("\u00ad", "")).strip()


def containment(line_box, paragraph_box):
    line = pymupdf.Rect(line_box)
    intersection = line & pymupdf.Rect(paragraph_box)
    return max(0, intersection.get_area()) / max(.001, line.get_area())


def spatial_support(location, boxes):
    # TEI may emit several boxes for one PDF line (citations / math / spans).
    # Count word centres covered by ANY box rather than requiring one TEI box
    # to cover most of the whole PDF line. Never take the enclosing rectangle
    # across columns, which would cover unrelated text between them.
    line = location["bbox"]
    boxes = [b for b in boxes if b[0] - 1.5 < line[2] and b[2] + 1.5 > line[0]
             and b[1] - 1.5 < line[3] and b[3] + 1.5 > line[1]]
    words = location["words"]
    if words:
        covered = 0
        for word in words:
            x, y = (word[0] + word[2]) / 2, (word[1] + word[3]) / 2
            if any(b[0] - 1.5 <= x <= b[2] + 1.5 and b[1] - 1.5 <= y <= b[3] + 1.5 for b in boxes):
                covered += 1
        return covered / len(words)
    return max((containment(location["bbox"], b) for b in boxes), default=0)


def align_coordinates(path, pages, document, tei_root, *, geometry_mode="word-union"):
    """Coordinates narrow candidates; unchanged lexical threshold confirms them.

    Only unrotated, uncropped pages whose TEI surface agrees with PDF media
    size are supported. Other transforms are explicitly unknown, never guessed.
    Raw text identity links cleaned lines to PDF line/word boxes. Repeated raw
    text is ambiguous and is not resolved by a positional guess.
    """
    baseline = align_cleaned_pages(pages, document)
    surfaces = {int(e.get("n")): e.attrib for e in tei_root.findall(".//{*}surface")}
    by_page = defaultdict(list)
    for index, para in enumerate(document["paragraphs"]):
        locations = defaultdict(list)
        for c in para["coordinates"]:
            locations[c["page"]].append((c["x"], c["y"], c["x"] + c["width"], c["y"] + c["height"]))
        for page, boxes in locations.items():
            by_page[page].append((index, para, tokens(para["text"]), boxes))
    bands = defaultdict(dict)
    for page, entries in by_page.items():
        for entry in entries:
            for box in entry[3]:
                for band in range(math.floor(box[1] / 20), math.floor(box[3] / 20) + 1):
                    bands[(page, band)][entry[0]] = entry
    geometry = []
    raw_lines = {}
    with pymupdf.open(path) as pdf:
        for page in pdf:
            number = page.number + 1
            surface = surfaces.get(number, {})
            width = float(surface.get("lrx", 0)) - float(surface.get("ulx", 0))
            height = float(surface.get("lry", 0)) - float(surface.get("uly", 0))
            supported = (page.rotation == 0 and abs(page.cropbox.x0) < .01
                         and abs(page.cropbox.y0) < .01
                         and abs(page.cropbox.width - page.mediabox.width) < .01
                         and abs(page.cropbox.height - page.mediabox.height) < .01
                         and abs(width - page.mediabox.width) < .1
                         and abs(height - page.mediabox.height) < .1
                         and float(surface.get("ulx", -1)) == 0
                         and float(surface.get("uly", -1)) == 0)
            geometry.append({"page": number, "rotation": page.rotation,
                             "cropbox": list(page.cropbox), "mediabox": list(page.mediabox),
                             "tei_surface": surface, "supported": supported})
            mapping = defaultdict(list)
            if supported:
                # WORDS and DICT extraction can assign different line indices.
                # Derive words from the SAME raw line's character boxes instead
                # of joining unrelated extraction outputs by numeric IDs.
                for block_index, block in enumerate(page.get_text("rawdict")["blocks"]):
                    for line_index, line in enumerate(block.get("lines", [])):
                        chars = [c for span in line["spans"] for c in span["chars"]]
                        value = normalized("".join(c["c"] for c in chars))
                        words, current = [], []
                        for char in [*chars, {"c": " "}]:
                            if char["c"].isspace():
                                if current:
                                    bbox = (min(c["bbox"][0] for c in current), min(c["bbox"][1] for c in current),
                                            max(c["bbox"][2] for c in current), max(c["bbox"][3] for c in current))
                                    words.append([*bbox, "".join(c["c"] for c in current)])
                                    current = []
                            else:
                                current.append(char)
                        mapping[value].append({"bbox": list(line["bbox"]), "block": block_index,
                                               "raw_line": line_index,
                                               "words": words})
            raw_lines[number] = mapping
    output = []
    previous = {}
    for old in baseline:
        row = dict(old)
        row.update(section_id=None, paragraph_id=None, structure_source="unknown",
                   coordinate_candidates=[], coordinate_reason="no_unique_raw_line")
        locations = raw_lines[old["page"]].get(normalized(old["text"]), [])
        if len(locations) == 1:
            location = locations[0]
            row.update(pdf_bbox=location["bbox"], pdf_block=location["block"],
                       pdf_raw_line=location["raw_line"], pdf_words=location["words"])
            vocab = tokens(old["text"])
            candidates = {}
            nearby = {}
            for band in range(math.floor(location["bbox"][1] / 20), math.floor(location["bbox"][3] / 20) + 1):
                nearby.update(bands[(old["page"], band)])
            for index, para, para_vocab, boxes in nearby.values():
                # Cheap rejection before word/box comparisons.
                line_box = location["bbox"]
                if not any(b[0] < line_box[2] and b[2] > line_box[0] and b[1] < line_box[3] and b[3] > line_box[1] for b in boxes):
                    continue
                spatial = (spatial_support(location, boxes) if geometry_mode == "word-union"
                           else max((containment(location["bbox"], box) for box in boxes), default=0))
                if spatial < .50:
                    continue
                lexical = len(vocab & para_vocab) / max(1, len(vocab))
                score = .75 * lexical + .25 * spatial
                candidate = {"paragraph_id": para["paragraph_id"], "section_id": para["section_id"],
                             "index": index, "lexical": lexical, "spatial": spatial, "score": score}
                pid = para["paragraph_id"]
                if pid not in candidates or score > candidates[pid]["score"]:
                    candidates[pid] = candidate
            ranked = sorted(candidates.values(), key=lambda x: (-x["score"], x["paragraph_id"]))
            row["coordinate_candidates"] = ranked[:3]
            row["coordinate_reason"] = "no_spatial_candidate"
            if ranked:
                first = ranked[0]
                margin = first["score"] - (ranked[1]["score"] if len(ranked) > 1 else 0)
                # Short lines require stronger corroboration. One-word/formula
                # lines remain unknown even if a bounding box overlaps them.
                text_ok = (len(vocab) >= 4 and first["lexical"] >= .65
                           or 2 <= len(vocab) < 4 and first["lexical"] >= .90 and first["spatial"] >= .70)
                key = (old["page"], location["block"])
                order_ok = first["index"] >= previous.get(key, -1)
                row["coordinate_reason"] = "ambiguous_or_weak_text_or_order"
                if text_ok and margin >= .10 and order_ok:
                    row.update(section_id=first["section_id"], paragraph_id=first["paragraph_id"],
                               structure_source="coordinate_text_order", coordinate_reason="accepted",
                               alignment_overlap=first["lexical"], alignment_margin=margin)
                    previous[key] = first["index"]
        row["baseline_paragraph_id"] = old["paragraph_id"]
        output.append(row)
    return baseline, output, geometry

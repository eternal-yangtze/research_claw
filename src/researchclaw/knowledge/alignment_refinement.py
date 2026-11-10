"""Opt-in evidence refinement; no gold IDs or paper-specific rules."""
import re


def sequence(text):
    # Normalize punctuation, soft hyphen and extraction-inserted line breaks.
    return re.sub(r"[^a-z0-9]", "", text.casefold())


def refine_lines(rows, document, geometry):
    paragraphs = document["paragraphs"]
    heights = {g["page"]: float(g["tei_surface"].get("lry", 0)) for g in geometry}
    supported = {g["page"] for g in geometry if g.get("supported", False)}
    output = []
    for original in rows:
        row = dict(original)
        row["refinement"] = "unchanged"
        row["mapping_extent"] = "unverified"
        box = row.get("pdf_bbox")
        value = sequence(row["text"])
        # Refinement must not bypass the original coordinate safety gate.
        if row["page"] not in supported:
            output.append(row)
            continue
        # Publication folios are content furniture even when native TEI p
        # contains them. Preserve source text, only reject paragraph attribution.
        if box and box[1] < heights.get(row["page"], 0) * .10 and re.fullmatch(r"\d{1,4}:\d{1,4}", row["text"].strip()):
            row.update(paragraph_id=None, section_id=None, refinement="folio_excluded",
                       content_type="furniture", mapping_extent="none")
            output.append(row)
            continue
        if row["paragraph_id"]:
            p = next(p for p in paragraphs if p["paragraph_id"] == row["paragraph_id"])
            row["mapping_extent"] = "full_text" if value and value in sequence(p["text"]) else "partial_or_math"
            output.append(row)
            continue
        if len(value) < 10:
            output.append(row)
            continue
        matches = []
        for p in paragraphs:
            coords = [c for c in p["coordinates"] if c["page"] == row["page"]]
            if not coords or value not in sequence(p["text"]):
                continue
            # Require either genuine local geometry (sparse boxes accepted only
            # with complete continuous text) or adjacent-source context.
            spatial = bool(box and any(c["x"] < box[2] and c["x"]+c["width"] > box[0]
                                      and c["y"] < box[3] and c["y"]+c["height"] > box[1] for c in coords))
            neighbors = [r for r in rows if r["page"] == row["page"] and abs(r["line_ordinal"]-row["line_ordinal"]) <= 2
                         and r["line_ordinal"] != row["line_ordinal"]]
            context = any(len(sequence(n["text"])) >= 12 and sequence(n["text"]) in sequence(p["text"]) for n in neighbors)
            if spatial or context:
                matches.append((p, spatial))
        if len(matches) == 1:
            p, spatial = matches[0]
            row.update(paragraph_id=p["paragraph_id"], section_id=p["section_id"],
                       refinement="continuous_text_with_geometry" if spatial else "continuous_text_with_context",
                       structure_source="refined_evidence", mapping_extent="full_text")
        output.append(row)
    return output


def refine_instance_evidence(rows, document, geometry, raw_lines):
    """v3: character-box coverage and unique, spatially adjacent source anchors.

    A neighbor never assigns a paragraph. It can only select an actual source
    occurrence, whose own geometry and text must identify a unique paragraph.
    """
    from researchclaw.knowledge.coordinate_alignment import normalized, spatial_support

    locations = {key: [r["bbox"] for r in records] for key, records in raw_lines.items()}
    output = refine_spatial_consensus(rows, document, geometry, locations)
    supported = {g["page"] for g in geometry if g.get("supported", False)}
    by_page = {}
    for source in rows:
        by_page.setdefault(source["page"], {})[source["line_ordinal"]] = source
    for row in output:
        if row["paragraph_id"] or row["page"] not in supported or row.get("content_type") == "furniture":
            continue
        records = raw_lines.get((row["page"], normalized(row["text"])), [])
        value = sequence(row["text"])
        if not records or len(value) < 6:
            continue
        candidates = []
        for record in records:
            choices = set()
            for p in document["paragraphs"]:
                if value not in sequence(p["text"]):
                    continue
                boxes = [(c["x"], c["y"], c["x"]+c["width"], c["y"]+c["height"])
                         for c in p["coordinates"] if c["page"] == row["page"]]
                if spatial_support(record, boxes) >= .7:
                    choices.add(p["paragraph_id"])
            candidates.append(choices)
        consensus = all(len(c) == 1 for c in candidates) and len(set.union(*candidates)) == 1
        selected = list(range(len(records))) if consensus else []
        anchors = []
        if not consensus and len(records) > 1:
            supported_instances = set()
            for delta in (-2, -1, 1, 2):
                neighbor = by_page[row["page"]].get(row["line_ordinal"] + delta)
                if not neighbor or len(sequence(neighbor["text"])) < 12:
                    continue
                anchor = raw_lines.get((row["page"], normalized(neighbor["text"])), [])
                if len(anchor) != 1:
                    continue
                a = anchor[0]["bbox"]
                for index, record in enumerate(records):
                    b = record["bbox"]
                    overlap = max(0, min(a[2],b[2])-max(a[0],b[0])) / max(.001,min(a[2]-a[0],b[2]-b[0]))
                    dy = (a[1]+a[3]-b[1]-b[3])/2
                    if overlap >= .5 and 0 < abs(dy) <= 30 and dy * delta > 0:
                        supported_instances.add(index)
                        anchors.append({"line_ordinal":neighbor["line_ordinal"], "raw_instance":index})
            # Conflicting nearby anchors leave the instance unresolved.
            if len(supported_instances) == 1:
                selected = list(supported_instances)
        if not selected or any(len(candidates[i]) != 1 for i in selected):
            continue
        pids = set.union(*(candidates[i] for i in selected))
        if len(pids) != 1:
            continue
        pid = next(iter(pids))
        p = next(p for p in document["paragraphs"] if p["paragraph_id"] == pid)
        row.update(paragraph_id=pid, section_id=p["section_id"],
                   refinement="character_coverage_consensus" if consensus else "unique_spatial_source_anchor",
                   structure_source="instance_geometry_text", mapping_extent="normalized_text",
                   raw_occurrence_count=len(records), raw_instance_resolved=len(selected)==1,
                   source_instance_boxes=[records[i]["bbox"] for i in selected], source_anchors=anchors)
    return output


def refine_spatial_consensus(rows, document, geometry, raw_locations):
    """Experimental v2: no context-only rescue; all raw occurrences must agree.

    raw_locations maps (page, normalized raw text) to actual PDF line boxes.
    Agreement resolves paragraph identity, never the identity of a raw instance.
    """
    from researchclaw.knowledge.coordinate_alignment import normalized, containment

    output = refine_lines(rows, document, geometry)
    supported = {g["page"] for g in geometry if g.get("supported", False)}
    for row in output:
        if row["refinement"] == "continuous_text_with_context":
            row.update(paragraph_id=None, section_id=None, structure_source="unknown",
                       refinement="context_without_geometry_rejected", mapping_extent="unverified")
        if row["paragraph_id"] or row["page"] not in supported or row.get("content_type") == "furniture":
            continue
        locations = raw_locations.get((row["page"], normalized(row["text"])), [])
        value = sequence(row["text"])
        if not locations or len(value) < 6:
            continue
        choices = []
        for box in locations:
            candidates = set()
            for p in document["paragraphs"]:
                if value not in sequence(p["text"]):
                    continue
                boxes = [(c["x"], c["y"], c["x"]+c["width"], c["y"]+c["height"])
                         for c in p["coordinates"] if c["page"] == row["page"]]
                if max((containment(box, b) for b in boxes), default=0) >= .5:
                    candidates.add(p["paragraph_id"])
            choices.append(candidates)
        if all(len(c) == 1 for c in choices) and len(set.union(*choices)) == 1:
            pid = next(iter(choices[0]))
            p = next(p for p in document["paragraphs"] if p["paragraph_id"] == pid)
            row.update(paragraph_id=pid, section_id=p["section_id"],
                       structure_source="spatial_occurrence_consensus",
                       refinement="all_raw_occurrences_agree", mapping_extent="normalized_text",
                       raw_occurrence_count=len(locations), raw_instance_resolved=len(locations) == 1)
    return output

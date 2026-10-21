"""A provenance-preserving passage index over immutable parser output.

Tiny adjacent PDF blocks are joined within a page. This is text organization,
not OCR correction; original nodes, source hashes and page locations survive.
"""
from __future__ import annotations

import json
from dataclasses import fields

from researchops.core.artifacts import canonical_hash
from researchops.core.models import Chunk, Paper
from researchops.core.storage import ResearchStore
from researchops.pipelines.batch import _paper_from_dict

VERSION = "page-passages-v1-1200"


def paper_from_row(row) -> Paper:
    value = dict(row)
    for key in ("authors", "identity", "source_records"):
        raw = value.get(key + "_json")
        if raw:
            value[key] = json.loads(raw)
    return _paper_from_dict({f.name: value[f.name] for f in fields(Paper) if f.name in value})


def materialize_passages(source: ResearchStore, target: ResearchStore, paper_ids: list[str],
                         max_chars: int = 1200) -> dict:
    if source.path == target.path:
        raise ValueError("Passage index must not overwrite the raw source database")
    added, reused, input_nodes = 0, 0, 0
    for pid in sorted(paper_ids):
        paper = dict(source.get_paper(pid))
        rows = [dict(row) for row in source.paper_chunks(pid)]
        signature = canonical_hash([VERSION, max_chars, [(r["chunk_id"], r["text"], r["provenance_json"]) for r in rows]])
        current = target.paper_chunks(pid)
        if current and json.loads(current[0]["provenance_json"]).get("index_signature") == signature:
            reused += len(current)
            continue
        target.upsert_paper(paper_from_row(paper))
        chunks, parts, ids, page, bbox = [], [], [], None, None

        def flush():
            if not parts:
                return
            text = " ".join(parts)
            cid = "passage_" + canonical_hash([pid, paper["content_hash"], page, ids, text, VERSION])[:24]
            chunks.append(Chunk(chunk_id=cid, paper_id=pid, section=f"Page {page}" if page else "Text",
                                text=text, page=page, bbox=tuple(bbox) if bbox else None,
                                content_hash=canonical_hash(text), source_hash=paper["content_hash"],
                                provenance={"index_version": VERSION, "index_signature": signature,
                                            "source_chunk_ids": list(dict.fromkeys(ids)),
                                            "source_sha256": paper["content_hash"]}))
            parts.clear()
            ids.clear()

        for row in rows:
            input_nodes += 1
            if row["node_type"] == "formula":
                # Do not hide formula quality by merging it into a paragraph.
                flush()
                provenance = json.loads(row["provenance_json"] or "{}")
                provenance.update(index_version=VERSION, index_signature=signature,
                                  source_chunk_ids=[row["chunk_id"]], source_sha256=paper["content_hash"])
                chunks.append(Chunk(chunk_id="passage_" + canonical_hash([row["chunk_id"], row["text"], VERSION])[:24],
                    paper_id=pid, section=row["section"], text=row["text"], page=row["page"],
                    bbox=tuple(json.loads(row["bbox_json"])) if row["bbox_json"] else None,
                    node_type="formula", content_hash=canonical_hash(row["text"]), source_hash=paper["content_hash"],
                    provenance=provenance))
                continue
            text = " ".join(row["text"].split())
            while text:
                fragment, text = text[:max_chars], text[max_chars:]
                if parts and (row["page"] != page or sum(map(len, parts)) + len(parts) + len(fragment) > max_chars):
                    flush()
                if not parts:
                    page = row["page"]
                    bbox = json.loads(row["bbox_json"]) if row["bbox_json"] else None
                elif bbox and row["bbox_json"]:
                    other = json.loads(row["bbox_json"])
                    bbox = [min(bbox[0], other[0]), min(bbox[1], other[1]),
                            max(bbox[2], other[2]), max(bbox[3], other[3])]
                parts.append(fragment)
                ids.append(row["chunk_id"])
        flush()
        target.replace_chunks(pid, chunks)
        added += len(chunks)
    return {"version": VERSION, "input_nodes_processed": input_nodes,
            "written_passages": added, "unchanged_passages": reused,
            "total_passages": len(target.all_chunks())}

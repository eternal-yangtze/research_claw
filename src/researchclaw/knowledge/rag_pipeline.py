"""Versioned offline paper corpus construction for the RAG baseline.

The builder never mutates source PDFs. It writes a new PostgreSQL/Qdrant
workspace and keeps parser, cleaning, quality and chunking decisions in the
chunk provenance so the corpus can be rebuilt and compared.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Iterable

from researchclaw.core.models import Chunk, Paper
from researchclaw.core.qdrant_store import QdrantResearchStore
from researchclaw.knowledge.retrieval import SentenceTransformerEmbedding
from researchclaw.knowledge.vector_backend import QdrantVectorBackend, VectorBackendStatus, _row_hash


PARSER_FINGERPRINT = "pymupdf-clean-v3"
PAGE_CHUNK_POLICY = "page-parent-word-child-v1"
STRUCTURE_CHUNK_POLICY = "paper-structure-semantic-v2.1"
SUPPORTED_CHUNK_POLICIES = (PAGE_CHUNK_POLICY, STRUCTURE_CHUNK_POLICY)
CHUNK_POLICY_FINGERPRINT = PAGE_CHUNK_POLICY

_SPACE = re.compile(r"[ \t\f\v]+")
_PAGE_NUMBER = re.compile(r"^(?:page\s+)?\d+(?:\s*/\s*\d+)?$", re.I)
_REFERENCE_HEADING = re.compile(r"^\s*(?:\d+(?:\.\d+)*\s+)?(?:references|bibliography)\s*$", re.I)
_MOJIBAKE = re.compile(r"(?:[\u00c2\u00c3\u00e2][\x80-\xff]|[\u9300\u9230\u984e\u7d01\ufffd])")
_TITLE_NOISE = re.compile(
    r"(?:latest updates|contents lists available|available online|review article|purdue university|"
    r"^i\.?\s+introduction$|journal|https?://|doi\b|paper title|"
    r"^pattern recognition\s+\d+)", re.I)
_NUMBERED_HEADING = re.compile(
    r"^(?:(?:\d+(?:\.\d+){0,4}|[IVXLC]+)[.)]?\s+)([A-Z][^.!?]{1,100})$", re.I)
_KNOWN_HEADING = re.compile(
    r"^(?:abstract|introduction|related work|literature review|background|preliminaries|"
    r"problem(?: formulation| statement)?|method(?:ology)?|proposed (?:method|approach)|"
    r"model|algorithm|implementation|experimental setup|experiments?|evaluation|datasets?|"
    r"results?|ablation(?: study)?|discussion|limitations?|conclusions?|future work|appendix)$", re.I)
_CAPTION = re.compile(r"^(?:fig(?:ure)?|table)\s+[A-Z0-9]+[.:]\s+", re.I)
_MOJIBAKE_REPLACEMENTS = {
    "\u00e2\u20ac\u201d": "\u2014", "\u00e2\u20ac\u201c": "\u2013",
    "\u00e2\u20ac\u2122": "\u2019", "\u00e2\u20ac\u02dc": "\u2018",
    "\u00e2\u20ac\u0153": "\u201c", "\u00e2\u20ac\u009d": "\u201d",
    "\u00e2\u02c6\u2020": "\u0394", "\u00e2\u02c6\u2019": "\u2212",
    "\u00e2\u2030\u00a4": "\u2264", "\u00e2\u2030\u00a5": "\u2265",
}


def repair_mojibake(value: str) -> str:
    """Repair common UTF-8-decoded-as-Western-text failures conservatively."""
    for broken, repaired in _MOJIBAKE_REPLACEMENTS.items():
        value = value.replace(broken, repaired)
    def repair_pair(match: re.Match) -> str:
        try:
            return match.group(0).encode("cp1252").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return match.group(0)
    value = re.sub(r"[\u00c2\u00c3][\x80-\xff]", repair_pair, value)
    if not re.search(r"[\u00c2\u00c3\u00e2]", value):
        return value
    best = value
    for encoding in ("cp1252", "latin-1"):
        try:
            candidate = value.encode(encoding).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
        if len(_MOJIBAKE.findall(candidate)) < len(_MOJIBAKE.findall(best)):
            best = candidate
    return best


@dataclass(frozen=True)
class CleanedPage:
    page: int
    text: str
    quality: dict


@dataclass(frozen=True)
class BuildResult:
    status: str
    corpus_version: str
    workspace: str
    papers_seen: int
    papers_indexed: int
    papers_encoded: int
    papers_reused: int
    papers_skipped_duplicate: int
    parents: int
    children: int
    quality_counts: dict[str, int]
    embedding_model: str
    embedding_revision: str
    report_path: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clean_lines(lines: list[str]) -> list[str]:
    values = []
    for raw in lines:
        value = repair_mojibake(_SPACE.sub(" ", raw.replace("\u00ad", "").strip()))
        if not value or _PAGE_NUMBER.fullmatch(value):
            continue
        values.append(value)
    # Preserve evidence but remove repeated page furniture by frequency.
    frequency: dict[str, int] = {}
    for value in values:
        if len(value) <= 140:
            frequency[value] = frequency.get(value, 0) + 1
    repeated = {value for value, count in frequency.items() if count >= 3}
    return [value for value in values if value not in repeated]


def extract_pages(path: Path) -> list[CleanedPage]:
    import fitz

    pages: list[CleanedPage] = []
    with fitz.open(path) as document:
        raw_pages = [[repair_mojibake(_SPACE.sub(" ", value.replace("\u00ad", "").strip()))
                      for value in (page.get_text("text") or "").splitlines()]
                     for page in document]
        furniture: dict[str, int] = {}
        for lines in raw_pages:
            visible = [value for value in lines if value]
            for value in [*visible[:2], *visible[-2:]]:
                if len(value) <= 160:
                    furniture[value] = furniture.get(value, 0) + 1
        threshold = max(3, math.ceil(len(raw_pages) * 0.30))
        repeated = {value for value, count in furniture.items() if count >= threshold}
        references_started = False
        for number, raw_lines in enumerate(raw_pages, 1):
            lines = _clean_lines([value for value in raw_lines if value not in repeated])
            if references_started:
                continue
            reference_index = next((index for index, line in enumerate(lines)
                                    if _REFERENCE_HEADING.fullmatch(line)), None)
            if reference_index is not None:
                lines = lines[:reference_index]
                references_started = True
            text = "\n".join(lines).strip()
            chars = len(text)
            replacement = len(_MOJIBAKE.findall(text))
            quality = {
                "parser": "pymupdf",
                "text_chars": chars,
                "replacement_chars": replacement,
                "ocr_required": chars < 40,
                "page_quality": "A" if chars >= 200 and replacement == 0 else "B" if chars >= 40 else "C",
            }
            if text:
                pages.append(CleanedPage(number, text, quality))
    return pages


def quality_grade(pages: list[CleanedPage]) -> tuple[str, dict]:
    if not pages:
        return "D", {"reason": "no_extractable_text", "pages": 0}
    chars = sum(item.quality["text_chars"] for item in pages)
    replacement = sum(item.quality["replacement_chars"] for item in pages)
    low = sum(item.quality["ocr_required"] for item in pages)
    ratio = replacement / max(1, chars)
    if ratio > 0.02 or low / len(pages) > 0.30:
        grade = "C"
    elif low or any(item.quality["page_quality"] == "B" for item in pages):
        grade = "B"
    else:
        grade = "A"
    return grade, {"pages": len(pages), "text_chars": chars, "replacement_chars": replacement,
                   "low_text_pages": low, "replacement_ratio": ratio}


def _title(path: Path, pages: list[CleanedPage]) -> str:
    try:
        import fitz
        with fitz.open(path) as document:
            metadata_title = repair_mojibake((document.metadata or {}).get("title", "").strip())
        if 8 <= len(metadata_title) <= 240 and not _TITLE_NOISE.search(metadata_title):
            return metadata_title
    except Exception:
        pass
    for line in (pages[0].text.splitlines()[:30] if pages else []):
        if (8 <= len(line) <= 240 and not line.endswith((".", ":"))
                and not _TITLE_NOISE.search(line) and len(line.split()) >= 3):
            return line
    return path.stem.replace("_", " ").replace("-", " ")


def _parent_child_chunks_v1(paper_id: str, pages: list[CleanedPage], grade: str, domain: str) -> list[Chunk]:
    parents: list[Chunk] = []
    children: list[Chunk] = []
    # A parent is a page-sized semantic unit for the first baseline. Child
    # blocks are bounded by paragraphs and then by a conservative word cap.
    for p_index, page in enumerate(pages):
        parent_id = hashlib.sha1(f"parent:{paper_id}:{page.page}".encode()).hexdigest()
        parent_text = page.text
        parents.append(Chunk(parent_id, paper_id, f"Page {page.page}", parent_text,
                             page=page.page, content_hash=hashlib.sha256(parent_text.encode()).hexdigest(),
                             node_type="parent", provenance={"quality_grade": grade, "page": page.page,
                                                             "page_start": page.page, "page_end": page.page,
                                                             "domain": domain,
                                                             "chunk_policy": PAGE_CHUNK_POLICY},
                             source_hash=None))
        units = [x.strip() for x in re.split(r"(?<=[.!?])\s+(?=[A-Z][a-z])|\n+", parent_text) if x.strip()]
        paragraphs, current, word_count = [], [], 0
        for unit in units:
            size = len(unit.split())
            if current and word_count + size > 260:
                paragraphs.append(" ".join(current))
                current, word_count = [], 0
            current.append(unit)
            word_count += size
            if word_count >= 180:
                paragraphs.append(" ".join(current))
                current, word_count = [], 0
        if current:
            paragraphs.append(" ".join(current))
        if len(paragraphs) > 1 and len(paragraphs[-1].split()) < 50:
            tail = paragraphs.pop()
            paragraphs[-1] += " " + tail
        ordinal = 0
        for paragraph in paragraphs:
            words = paragraph.split()
            for start in range(0, len(words), 220):
                text = " ".join(words[start:start + 220]).strip()
                if not text:
                    continue
                child_id = hashlib.sha1(f"child:{paper_id}:{page.page}:{ordinal}:{text}".encode()).hexdigest()
                children.append(Chunk(child_id, paper_id, f"Page {page.page}", text,
                                      page=page.page, content_hash=hashlib.sha256(text.encode()).hexdigest(),
                                      node_type="child", parent_chunk_id=parent_id,
                                      provenance={"quality_grade": grade, "page": page.page,
                                                  "page_start": page.page, "page_end": page.page,
                                                  "domain": domain, "parent_id": parent_id,
                                                  "chunk_policy": PAGE_CHUNK_POLICY},
                                      source_hash=None))
                ordinal += 1
    return [*parents, *children]


def _heading(line: str) -> tuple[str, str] | None:
    value = _SPACE.sub(" ", line).strip().strip(":")
    if not value or len(value) > 120 or len(value.split()) > 14 or _CAPTION.match(value):
        return None
    match = _NUMBERED_HEADING.match(value)
    label = match.group(1).strip() if match else value
    if not (match or _KNOWN_HEADING.fullmatch(label)
            or value.isupper() and len(value.split()) <= 8):
        return None
    lowered = label.lower()
    if "abstract" in lowered:
        kind = "abstract"
    elif any(word in lowered for word in ("method", "model", "algorithm", "approach")):
        kind = "method"
    elif any(word in lowered for word in ("experiment", "evaluation", "dataset", "implementation")):
        kind = "experimental_setup"
    elif any(word in lowered for word in ("result", "ablation", "discussion")):
        kind = "result"
    elif any(word in lowered for word in ("limit", "conclusion", "future")):
        kind = "conclusion"
    elif any(word in lowered for word in ("background", "related", "prelim", "literature")):
        kind = "background"
    elif any(word in lowered for word in ("problem", "introduction")):
        kind = "problem_background"
    else:
        kind = "section"
    return value, kind


def _semantic_blocks(text: str, minimum: int = 120, target: int = 190,
                     maximum: int = 240) -> list[str]:
    units = [unit.strip() for unit in re.split(r"\n{2,}|(?<=[.!?])\s+(?=[A-Z])|\n", text)
             if unit.strip()]
    blocks: list[str] = []
    current: list[str] = []
    count = 0
    for unit in units:
        words = unit.split()
        if len(words) > maximum:
            if current:
                blocks.append(" ".join(current))
                current, count = [], 0
            blocks.extend(" ".join(words[start:start + maximum])
                          for start in range(0, len(words), maximum))
            continue
        if current and count + len(words) > maximum:
            blocks.append(" ".join(current))
            current, count = [], 0
        current.append(unit)
        count += len(words)
        if count >= target:
            blocks.append(" ".join(current))
            current, count = [], 0
    if current:
        tail = " ".join(current)
        if blocks and len(tail.split()) < minimum:
            blocks[-1] += " " + tail
        else:
            blocks.append(tail)
    return [block.strip() for block in blocks if block.strip()]


def _parent_child_chunks_v2(paper_id: str, pages: list[CleanedPage], grade: str,
                            domain: str) -> list[Chunk]:
    sections: list[dict] = []
    current = {"heading": "Front matter", "kind": "front_matter", "lines": [], "pages": []}
    headings_found = 0
    for page in pages:
        for line in page.text.splitlines():
            detected = _heading(line)
            if detected and current["lines"]:
                sections.append(current)
                current = {"heading": detected[0], "kind": detected[1], "lines": [], "pages": []}
                headings_found += 1
                continue
            if detected:
                current["heading"], current["kind"] = detected
                headings_found += 1
                continue
            current["lines"].append(line)
            current["pages"].append(page.page)
    if current["lines"]:
        sections.append(current)
    if headings_found < 2:
        nodes = _parent_child_chunks_v1(paper_id, pages, grade, domain)
        return [Chunk(**{**node.__dict__, "provenance": {
            **node.provenance, "chunk_policy": STRUCTURE_CHUNK_POLICY,
            "fallback_used": True}}) for node in nodes]

    parents: list[Chunk] = []
    children: list[Chunk] = []
    for section_index, section in enumerate(sections):
        text = "\n".join(section["lines"]).strip()
        if not text or not section["pages"]:
            continue
        page_start, page_end = min(section["pages"]), max(section["pages"])
        for part_index, parent_text in enumerate(
                _semantic_blocks(text, minimum=300, target=850, maximum=1100)):
            heading = section["heading"]
            section_name = heading if len(text.split()) <= 1100 else f"{heading} (part {part_index + 1})"
            parent_id = hashlib.sha1(
                f"parent:{STRUCTURE_CHUNK_POLICY}:{paper_id}:{section_index}:{part_index}:{parent_text}".encode()
            ).hexdigest()
            provenance = {"quality_grade": grade, "domain": domain,
                          "heading_path": [heading], "structure_type": section["kind"],
                          "page_start": page_start, "page_end": page_end,
                          "chunk_policy": STRUCTURE_CHUNK_POLICY, "fallback_used": False}
            parents.append(Chunk(parent_id, paper_id, section_name, parent_text, page=page_start,
                                 content_hash=hashlib.sha256(parent_text.encode()).hexdigest(),
                                 node_type="parent", provenance=provenance, source_hash=None))
            for ordinal, child_text in enumerate(_semantic_blocks(parent_text)):
                child_id = hashlib.sha1(
                    f"child:{STRUCTURE_CHUNK_POLICY}:{paper_id}:{section_index}:{part_index}:{ordinal}:{child_text}".encode()
                ).hexdigest()
                children.append(Chunk(child_id, paper_id, section_name, child_text, page=page_start,
                                      content_hash=hashlib.sha256(child_text.encode()).hexdigest(),
                                      node_type="child", parent_chunk_id=parent_id,
                                      provenance={**provenance, "parent_id": parent_id}, source_hash=None))
    return [*parents, *children]


def _parent_child_chunks(paper_id: str, pages: list[CleanedPage], grade: str, domain: str,
                         chunk_policy: str = PAGE_CHUNK_POLICY) -> list[Chunk]:
    if chunk_policy == PAGE_CHUNK_POLICY:
        return _parent_child_chunks_v1(paper_id, pages, grade, domain)
    if chunk_policy == STRUCTURE_CHUNK_POLICY:
        return _parent_child_chunks_v2(paper_id, pages, grade, domain)
    raise ValueError(f"Unsupported chunk policy: {chunk_policy}")


def build_corpus(workspace: str | Path, sources: Iterable[tuple[str, Path]], *, corpus_version: str,
                 model_name: str | None = None, model_revision: str | None = None,
                 batch_size: int = 32, report_path: str | Path | None = None,
                 excluded_sources: Iterable[str] = (),
                 base_corpus_version: str | None = None,
                 chunk_policy: str = PAGE_CHUNK_POLICY, progress=None) -> BuildResult:
    if chunk_policy not in SUPPORTED_CHUNK_POLICIES:
        raise ValueError(f"Unsupported chunk policy: {chunk_policy}")
    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    source_rows = [(domain, Path(path).resolve()) for domain, path in sources]
    files = [(domain, path) for domain, path in source_rows if path.is_file() and path.suffix.lower() == ".pdf"]
    store = QdrantResearchStore(workspace)
    embedding = SentenceTransformerEmbedding(model_name, revision=model_revision)
    embedding.batch_size = batch_size
    dense = QdrantVectorBackend(store, embedding, corpus_version=corpus_version)
    seen_hashes: set[str] = set()
    papers_indexed = papers_encoded = papers_reused = duplicate = parent_count = child_count = encoded_children = 0
    quality_counts: dict[str, int] = {}
    report_rows = []
    indexed_vectors: dict[str, tuple[float, ...]] = {}
    embedding_key = f"{embedding.model_name}@{embedding.model_revision}"
    def notify(stage, **details):
        if progress:
            progress(stage, **details)
    with store.database.connect(read_only=True) as db:
        prior = db.execute("""SELECT * FROM vector_sync_state WHERE project_id=? AND corpus_version=?
            AND embedding_key=? AND status='ready' ORDER BY updated_at DESC LIMIT 1""",
            (store.project_id, corpus_version, embedding_key)).fetchone()
        if base_corpus_version:
            base = db.execute("""SELECT * FROM vector_sync_state WHERE project_id=?
                AND corpus_version=? AND embedding_key=? AND status='ready'
                ORDER BY updated_at DESC LIMIT 1""",
                (store.project_id, base_corpus_version, embedding_key)).fetchone()
        else:
            base = db.execute("""SELECT * FROM vector_sync_state WHERE project_id=?
                AND corpus_version<>? AND embedding_key=? AND status='ready'
                ORDER BY updated_at DESC LIMIT 1""",
                (store.project_id, corpus_version, embedding_key)).fetchone()
    notify('inventory', total_files=len(files), completed_files=0)
    for file_index, (domain, path) in enumerate(files):
        notify('parse', document=path.name, completed_files=file_index, total_files=len(files))
        source_hash = sha256(path)
        if source_hash in seen_hashes:
            duplicate += 1
            notify('file_done', completed_files=file_index + 1, reused=True)
            continue
        seen_hashes.add(source_hash)
        paper_id = "rag_" + source_hash[:24]
        existing = store.get_paper(paper_id)
        existing_chunks = store.paper_chunks(paper_id) if existing else []
        chunk_policy_matches = bool(existing_chunks) and all(
            json.loads(row.get("provenance_json") or "{}").get("chunk_policy")
            == chunk_policy
            for row in existing_chunks if row.get("node_type") == "child"
        )
        if (existing and existing.get("content_hash") == source_hash
                and existing.get("parser_version") == PARSER_FINGERPRINT
                and chunk_policy_matches):
            papers_indexed += 1
            papers_reused += 1
            reused_grade = existing["quality_status"]
            quality_counts[reused_grade] = quality_counts.get(reused_grade, 0) + 1
            report_rows.append({"paper_id": paper_id, "path": str(path), "domain": domain,
                                "title": existing["title"], "quality_grade": existing["quality_status"],
                                "incremental_reuse": True, "source_sha256": source_hash})
            notify('file_done', completed_files=file_index + 1, reused=True)
            continue
        pages = extract_pages(path)
        grade, quality = quality_grade(pages)
        quality_counts[grade] = quality_counts.get(grade, 0) + 1
        title = _title(path, pages)
        if grade == "D":
            report_rows.append({"paper_id": paper_id, "path": str(path), "domain": domain,
                                "title": title, "quality_grade": grade, "quality": quality,
                                "published": False, "reason": "quality_grade_D", "source_sha256": source_hash})
            notify('file_done', completed_files=file_index + 1, rejected=True)
            continue
        paper = Paper(paper_id=paper_id, title=title, source="rag_staging", file_path=str(path),
                      content_hash=source_hash, parser_version="pymupdf-clean-v3",
                      quality_status=grade, identity=None)
        nodes = _parent_child_chunks(paper_id, pages, grade, domain, chunk_policy)
        # Only Child nodes are indexed; parents remain authoritative context.
        children = [node for node in nodes if node.node_type == "child"]
        child_vectors = []
        for offset in range(0, len(children), batch_size):
            notify('embed', document=path.name, vectors_done=offset, vectors_total=len(children))
            child_vectors.extend(embedding.encode([f"{node.section}\n{node.text}"
                for node in children[offset:offset + batch_size]]))
        encoded_children += len(child_vectors)
        children = [Chunk(**{**node.__dict__, "vector": vector, "source_hash": source_hash})
                    for node, vector in zip(children, child_vectors)]
        parents = [Chunk(**{**node.__dict__, "source_hash": source_hash}) for node in nodes if node.node_type == "parent"]
        store.upsert_paper(paper)
        store.replace_chunks(paper_id, [*parents, *children])
        indexed_vectors.update({node.chunk_id: tuple(vector)
                                for node, vector in zip(children, child_vectors)})
        papers_indexed += 1
        papers_encoded += 1
        parent_count += len(parents)
        child_count += len(children)
        report_rows.append({"paper_id": paper_id, "path": str(path), "domain": domain,
                            "title": title, "quality_grade": grade, "quality": quality,
                            "parents": len(parents), "children": len(children), "source_sha256": source_hash})
        notify('file_done', completed_files=file_index + 1, vectors_done=len(children), vectors_total=len(children))
    notify('dense_sync', completed_files=len(files))
    store.flush_outbox()
    # PostgreSQL is the complete Child authority; Qdrant carries all vectors.
    # Existing vectors were loaded from the published collection, while new
    # or changed children were encoded above.
    with store.database.connect(read_only=True) as db:
        indexed_rows = [dict(row) for row in db.execute("""SELECT c.*,p.title AS paper_title
            FROM chunks c JOIN papers p ON p.paper_id=c.paper_id WHERE c.node_type='child'""")]
    base_collection = None
    base_version = None
    if base and base["version_collection"] and store.client.collection_exists(base["version_collection"]):
        base_collection = base["version_collection"]
        base_version = base["corpus_version"]
    if prior and prior["version_collection"] and store.client.collection_exists(prior["version_collection"]):
        # Verify the published content inventory WITHOUT downloading/rehashing
        # normalized float vectors. Rehashing round-tripped cosine vectors used
        # to rewrite the entire collection even when no source changed.
        offset = None
        published = {}
        while True:
            points, offset = store.client.scroll(prior['version_collection'], limit=512, offset=offset,
                with_payload=['chunk_id', 'content_hash'], with_vectors=False)
            published.update({str(p.payload.get('chunk_id')): p.payload.get('content_hash') for p in points})
            if offset is None:
                break
        if not encoded_children and published == {r['chunk_id']: _row_hash(r) for r in indexed_rows}:
            info = store.client.get_collection(prior['version_collection'])
            vector_status = VectorBackendStatus(dense.name, dense.alias, prior['version_collection'],
                store.project_id, corpus_version, embedding_key, info.config.params.vectors.size,
                'ready', indexed_points=len(indexed_rows))
        else:
            vector_status = dense.sync_incremental(indexed_rows, indexed_vectors,
                reuse_collection=prior['version_collection'])
    elif base_collection:
        vector_status = dense.sync_incremental(
            indexed_rows, indexed_vectors, reuse_collection=base_collection)
    else:
        missing_vectors = [row["chunk_id"] for row in indexed_rows
                           if row["chunk_id"] not in indexed_vectors]
        if missing_vectors:
            raise RuntimeError(
                f"Build lacks {len(missing_vectors)} vector(s) and no compatible base revision exists"
            )
        vector_status = dense.sync(indexed_rows, indexed_vectors)
    with store.database.connect(read_only=True) as db:
        parent_count = db.execute("SELECT count(*) FROM chunks WHERE node_type='parent'").fetchone()[0]
        child_count = db.execute("SELECT count(*) FROM chunks WHERE node_type='child'").fetchone()[0]
        postgres_vector_copies = db.execute("SELECT count(*) FROM chunk_embeddings").fetchone()[0]
        parent_provenance = [json.loads(row[0] or "{}") for row in db.execute(
            "SELECT provenance_json FROM chunks WHERE node_type='parent'")]
    structured_parents = sum(bool(item.get("heading_path")) for item in parent_provenance)
    multipage_parents = sum(item.get("page_end", 0) > item.get("page_start", 0)
                            for item in parent_provenance)
    fallback_parents = sum(bool(item.get("fallback_used")) for item in parent_provenance)
    from researchclaw.knowledge.qdrant_lexical import build_lexical_index
    notify('bm25', completed_files=len(files))
    lexical_status = build_lexical_index(store, corpus_version=corpus_version, embedding_key=embedding_key)
    report = {"status": "passed", "corpus_version": corpus_version, "workspace": str(workspace),
              "lexical_index": lexical_status,
              "source_files": len(files), "papers_seen": len(files), "papers_indexed": papers_indexed,
              "papers_excluded_by_policy": list(excluded_sources),
              "papers_encoded": papers_encoded, "papers_reused": papers_reused,
              "papers_skipped_duplicate": duplicate, "parents": parent_count, "children": child_count,
              "quality_counts": quality_counts, "embedding_model": embedding.model_name,
              "embedding_revision": embedding.model_revision,
              "parser_fingerprint": PARSER_FINGERPRINT,
              "chunk_policy_fingerprint": chunk_policy,
              "structure_stats": {"structured_parents": structured_parents,
                                  "multi_page_parents": multipage_parents,
                                  "fallback_parents": fallback_parents,
                                  "structured_parent_ratio": structured_parents / max(1, parent_count)},
              "base_corpus_version": base_version,
              "vectors_encoded": encoded_children,
              "vectors_reused": child_count - encoded_children,
              "dense_points_written": vector_status.upserted_points,
              "vector_status": vector_status.__dict__, "postgres_vector_copies": postgres_vector_copies,
              "papers": report_rows}
    target = Path(report_path or workspace / "artifacts" / "rag-build-report.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    notify('published', completed_files=len(files), report_path=str(target))
    return BuildResult("passed", corpus_version, str(workspace), len(files), papers_indexed,
                       papers_encoded, papers_reused, duplicate,
                       parent_count, child_count, quality_counts, embedding.model_name,
                       embedding.model_revision, str(target))

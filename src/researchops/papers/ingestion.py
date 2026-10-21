from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

from researchops.core.models import Chunk, Paper
from researchops.knowledge.retrieval import hashing_vector
from researchops.core.storage import ResearchStore


TITLE_PATTERN = re.compile(r"^Title:\s*(.+)$", re.MULTILINE | re.IGNORECASE)
AUTHORS_PATTERN = re.compile(r"^Authors:\s*(.+)$", re.MULTILINE | re.IGNORECASE)
YEAR_PATTERN = re.compile(r"^Year:\s*.*?((?:19|20)\d{2})", re.MULTILINE | re.IGNORECASE)
DOI_PATTERN = re.compile(r"^DOI[^:]*:\s*(.+)$", re.MULTILINE | re.IGNORECASE)
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


@dataclass(frozen=True)
class IngestResult:
    paper_id: str
    title: str
    chunks: int
    parser: str
    quality_status: str
    path: str
    parsed_path: str | None = None
    formula_validation: dict | None = None

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class DocumentIngestor:
    def __init__(self, store: ResearchStore, max_chunk_chars: int = 1800) -> None:
        self.store = store
        self.max_chunk_chars = max_chunk_chars

    def ingest_path(self, path: str | Path, recursive: bool = True) -> list[IngestResult]:
        target = Path(path).resolve()
        if not target.exists():
            raise FileNotFoundError(target)
        if target.is_file():
            return [self.ingest_file(target)]
        iterator: Iterable[Path] = target.rglob("*") if recursive else target.glob("*")
        supported = [item for item in iterator if item.is_file() and item.suffix.lower() in {".md", ".txt", ".pdf"}]
        results: list[IngestResult] = []
        for item in sorted(supported):
            if item.name.startswith("_") or item.name.startswith("00_"):
                continue
            results.append(self.ingest_file(item))
        return results

    def ingest_file(self, path: str | Path, paper: Paper | None = None, *,
                    parser_artifact: str | Path | None = None) -> IngestResult:
        source = Path(path).resolve()
        suffix = source.suffix.lower()
        if suffix in {".md", ".txt"}:
            return self._ingest_text(source)
        if suffix == ".pdf":
            if parser_artifact is not None:
                from docling_core.types.doc import DoclingDocument
                payload = json.loads(Path(parser_artifact).read_text(encoding="utf-8"))
                if payload.get("pdf_sha256") != _hash_bytes(source.read_bytes()):
                    raise ValueError("cached Docling artifact/PDF hash mismatch")
                document = DoclingDocument.model_validate(payload["document"])
                result = self._store_docling_document(source, document, paper)
                return self._attach_formula_validation(result, paper)
            return self._ingest_pdf(source, paper)
        raise ValueError(f"Unsupported document type: {suffix}")

    def _ingest_text(self, path: Path) -> IngestResult:
        raw = path.read_text(encoding="utf-8", errors="replace")
        content_hash = _hash_bytes(raw.encode("utf-8"))
        metadata = _metadata(raw, path)
        paper_id = _paper_id(metadata["doi"], metadata["title"], content_hash)
        chunks = _markdown_chunks(raw, paper_id, self.max_chunk_chars)
        paper = Paper(
            paper_id=paper_id,
            title=metadata["title"],
            authors=metadata["authors"],
            year=metadata["year"],
            doi=metadata["doi"],
            source="local",
            file_path=str(path),
            content_hash=content_hash,
            parser_version="markdown-lines-v1",
            quality_status="accepted",
        )
        self.store.upsert_paper(paper)
        self.store.replace_chunks(paper_id, chunks)
        return IngestResult(paper_id, paper.title, len(chunks), paper.parser_version or "", paper.quality_status, str(path))

    def _ingest_pdf(self, path: Path, paper: Paper | None = None) -> IngestResult:
        parser = os.getenv("RESEARCHOPS_PDF_PARSER", "auto").strip().lower()
        if parser not in {"auto", "docling", "pymupdf"}:
            raise ValueError("RESEARCHOPS_PDF_PARSER must be auto, docling, or pymupdf")
        if parser == "pymupdf":
            result = self._ingest_pdf_pymupdf(path, paper)
        else:
            try:
                result = self._ingest_pdf_docling(path, paper)
            except Exception as error:
                if parser == "docling":
                    raise RuntimeError(f"Docling PDF conversion failed: {type(error).__name__}: {error}") from error
                result = self._ingest_pdf_pymupdf(path, paper, fallback_error=error)
        return self._attach_formula_validation(result, paper)

    def _attach_formula_validation(self, result: IngestResult, seed: Paper | None = None) -> IngestResult:
        from researchops.papers.validation.automatic import automatic_formula_check
        check = automatic_formula_check(result.path, result.parsed_path,
                self.store.path.parent / "artifacts" / "formula_validation",
                source_package=seed.source_archive_path if seed else None)
        regions = {}
        if check.get("report_path"):
            report = json.loads(Path(check["report_path"]).read_text(encoding="utf-8"))
            regions = {r["region_id"]: r for r in report.get("regions", [])}
        original = json.loads(Path(result.parsed_path).read_text(encoding="utf-8"))
        chunks = []
        for row in self.store.paper_chunks(result.paper_id):
            provenance = json.loads(row["provenance_json"] or "{}")
            region = regions.get(provenance.get("self_ref"))
            decision = region["decision"] if region else {}
            provenance["formula_validation"] = {
                "status": decision.get("status", "deferred" if row["node_type"] == "formula" and check["status"].startswith("checked") else check["status"]),
                "report_path": check.get("report_path"), "report_sha256": check.get("report_sha256"),
                "selected_latex": decision.get("selected_latex"),
                "original_latex": region.get("latex") if region else None,
                "safe_as_sole_reproduction_evidence": False}
            chunks.append(Chunk(chunk_id=row["chunk_id"], paper_id=row["paper_id"], section=row["section"],
                text=row["text"], page=row["page"], bbox=tuple(json.loads(row["bbox_json"])) if row["bbox_json"] else None,
                line_start=row["line_start"], line_end=row["line_end"], content_hash=row["content_hash"],
                vector=tuple(json.loads(row["vector_json"])), node_type=row["node_type"],
                parent_chunk_id=row["parent_chunk_id"], provenance=provenance, source_hash=row["source_hash"]))
        self.store.replace_chunks(result.paper_id, chunks)
        parsed = self._write_pdf_sidecar(result.paper_id, Path(result.path), result.parser,
                    original["source_sha256"], chunks, original.get("docling_document"))
        return replace(result, parsed_path=parsed, formula_validation=check)

    def _ingest_pdf_docling(self, path: Path, seed: Paper | None = None) -> IngestResult:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        pipeline_options = PdfPipelineOptions(
            do_ocr=True,
            do_table_structure=True,
            do_formula_enrichment=True,
        )
        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )
        result = converter.convert(str(path))
        return self._store_docling_document(path, result.document, seed)

    def _store_docling_document(self, path: Path, document, seed: Paper | None = None) -> IngestResult:
        markdown = document.export_to_markdown()
        content_hash = _hash_bytes(path.read_bytes())
        metadata = _metadata(markdown, path)
        paper_id = seed.paper_id if seed else _paper_id(metadata["doi"], metadata["title"], content_hash)
        chunks: list[Chunk] = []
        active_heading_id: str | None = None
        for ordinal, (item, _level) in enumerate(document.iterate_items()):
            label_value = getattr(item, "label", "document") or "document"
            label = str(getattr(label_value, "value", label_value)).casefold()
            node_type = _docling_node_type(label)
            text = _docling_item_text(item, document, node_type)
            if not text:
                continue
            provenance = list(getattr(item, "prov", ()) or ())
            first = provenance[0] if provenance else None
            box = getattr(first, "bbox", None)
            bbox = None
            if box is not None:
                bbox = tuple(round(float(getattr(box, name)), 2) for name in ("l", "t", "r", "b"))
            item_hash = _hash_bytes(text.encode("utf-8"))
            chunk_id = _stable_id("node", paper_id, str(ordinal), node_type, item_hash)
            provenance_value = {
                "parser": "docling",
                "self_ref": getattr(item, "self_ref", None),
                "item_index": ordinal,
                "label": label,
                "locations": [_docling_provenance(value) for value in provenance],
            }
            chunks.append(Chunk(
                chunk_id=chunk_id,
                paper_id=paper_id,
                section=label,
                text=text,
                page=int(getattr(first, "page_no")) if first is not None and getattr(first, "page_no", None) is not None else None,
                bbox=bbox,
                content_hash=item_hash,
                vector=hashing_vector(f"{label} {text}"),
                node_type=node_type,
                parent_chunk_id=active_heading_id if node_type != "heading" else None,
                provenance=provenance_value,
                source_hash=content_hash,
            ))
            if node_type == "heading":
                active_heading_id = chunk_id
        if not chunks:
            chunks = _markdown_chunks(markdown, paper_id, self.max_chunk_chars)
        paper = Paper(
            paper_id=paper_id,
            title=seed.title if seed else metadata["title"],
            authors=seed.authors if seed and seed.authors else metadata["authors"],
            year=seed.year if seed else metadata["year"],
            doi=seed.doi if seed else metadata["doi"],
            source=seed.source if seed else "local_pdf",
            source_url=seed.source_url if seed else None,
            file_path=str(path),
            content_hash=content_hash,
            parser_version="docling",
            quality_status="review_required",
            abstract=seed.abstract if seed else None,
            pdf_url=seed.pdf_url if seed else None,
            external_id=seed.external_id if seed else None,
            identity=seed.identity if seed else None,
            source_records=seed.source_records if seed else (),
            source_version=seed.source_version if seed else None,
            source_archive_url=seed.source_archive_url if seed else None,
            source_archive_path=seed.source_archive_path if seed else None,
        )
        self.store.upsert_paper(paper)
        self.store.replace_chunks(paper_id, chunks)
        parsed_path = self._write_pdf_sidecar(
            paper_id, path, "docling", content_hash, chunks, document.export_to_dict()
        )
        return IngestResult(
            paper_id, paper.title, len(chunks), "docling", paper.quality_status, str(path), parsed_path
        )

    def _ingest_pdf_pymupdf(
        self,
        path: Path,
        seed: Paper | None = None,
        fallback_error: Exception | None = None,
    ) -> IngestResult:
        try:
            import pymupdf as fitz
        except ImportError as error:
            try:
                import fitz
            except ImportError:
                raise RuntimeError("PDF ingestion requires Docling or PyMuPDF; install the 'pdf' or 'docling' extra") from error

        content_hash = _hash_bytes(path.read_bytes())
        document = fitz.open(path)
        chunks: list[Chunk] = []
        title = seed.title if seed else str(document.metadata.get("title") or path.stem)
        for page_number, page in enumerate(document, start=1):
            blocks = page.get_text("blocks", sort=True)
            for block_number, block in enumerate(blocks):
                text = " ".join(str(block[4]).split())
                if not text:
                    continue
                chunk_id = _stable_id("chunk", content_hash, str(page_number), str(block_number), text)
                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    paper_id="pending",
                    section=f"Page {page_number}",
                    text=text,
                    page=page_number,
                    bbox=tuple(round(float(value), 2) for value in block[:4]),
                    content_hash=_hash_bytes(text.encode("utf-8")),
                    vector=hashing_vector(text),
                    node_type="paragraph",
                    provenance={
                        "parser": "pymupdf",
                        "page": page_number,
                        "block": block_number,
                        "bbox": [round(float(value), 2) for value in block[:4]],
                    },
                    source_hash=content_hash,
                ))
        paper_id = seed.paper_id if seed else _paper_id(None, title, content_hash)
        if not chunks:
            raise RuntimeError(
                "PyMuPDF extracted no text; this PDF requires OCR and cannot be marked as ingested"
            )
        chunks = [Chunk(**{**chunk.__dict__, "paper_id": paper_id}) for chunk in chunks]
        parser_version = f"pymupdf-{getattr(fitz, '__version__', 'unknown')}"
        if fallback_error is not None:
            parser_version += f"-fallback-from-{type(fallback_error).__name__}"
        paper = Paper(
            paper_id=paper_id,
            title=title,
            authors=seed.authors if seed else (),
            year=seed.year if seed else None,
            doi=seed.doi if seed else None,
            source=seed.source if seed else "local_pdf",
            source_url=seed.source_url if seed else None,
            file_path=str(path),
            content_hash=content_hash,
            parser_version=parser_version,
            quality_status="review_required",
            abstract=seed.abstract if seed else None,
            pdf_url=seed.pdf_url if seed else None,
            external_id=seed.external_id if seed else None,
            identity=seed.identity if seed else None,
            source_records=seed.source_records if seed else (),
            source_version=seed.source_version if seed else None,
            source_archive_url=seed.source_archive_url if seed else None,
            source_archive_path=seed.source_archive_path if seed else None,
        )
        self.store.upsert_paper(paper)
        self.store.replace_chunks(paper_id, chunks)
        parsed_path = self._write_pdf_sidecar(
            paper_id, path, paper.parser_version or "pymupdf", content_hash, chunks
        )
        return IngestResult(
            paper_id, title, len(chunks), paper.parser_version or "", paper.quality_status, str(path), parsed_path
        )

    def _write_pdf_sidecar(
        self,
        paper_id: str,
        source_path: Path,
        parser: str,
        content_hash: str,
        chunks: list[Chunk],
        parser_document: dict | None = None,
    ) -> str:
        target_dir = self.store.path.parent / "artifacts" / "parsed"
        target_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "paper_id": paper_id,
            "source_path": str(source_path),
            "source_sha256": content_hash,
            "parser": parser,
            "quality_status": "review_required",
            "chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "section": chunk.section,
                    "page": chunk.page,
                    "bbox": chunk.bbox,
                    "content_hash": chunk.content_hash,
                    "text": chunk.text,
                    "node_type": chunk.node_type,
                    "parent_chunk_id": chunk.parent_chunk_id,
                    "provenance": chunk.provenance,
                    "source_hash": chunk.source_hash,
                }
                for chunk in chunks
            ],
        }
        if parser_document is not None:
            payload["docling_document"] = parser_document
        # Preserve parser revisions alongside the historical compatibility sidecar.
        revision = _hash_bytes(json.dumps(payload, sort_keys=True, default=str).encode("utf-8"))
        target = target_dir / f"{paper_id}.{revision}.json"
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        legacy = target_dir / f"{paper_id}.json"
        if not legacy.exists():
            legacy.write_text(target.read_text(encoding="utf-8"), encoding="utf-8")
            return str(legacy)
        return str(target)


def _markdown_chunks(text: str, paper_id: str, max_chars: int) -> list[Chunk]:
    lines = text.splitlines()
    heading_path: list[str] = []
    blocks: list[tuple[str, int, int, str]] = []
    buffer: list[str] = []
    start = 1

    def flush(end: int) -> None:
        nonlocal buffer, start
        value = "\n".join(buffer).strip()
        if value:
            section = " > ".join(heading_path) or "Document"
            blocks.append((section, start, end, value))
        buffer = []

    for number, line in enumerate(lines, start=1):
        heading = HEADING_PATTERN.match(line)
        if heading:
            flush(number - 1)
            level = len(heading.group(1))
            heading_path[:] = heading_path[:level - 1]
            heading_path.append(heading.group(2).strip())
            start = number + 1
            continue
        projected = sum(len(item) + 1 for item in buffer) + len(line)
        if buffer and projected > max_chars:
            flush(number - 1)
            start = number
        if not buffer and line.strip():
            start = number
        buffer.append(line)
    flush(len(lines))

    chunks: list[Chunk] = []
    for ordinal, (section, line_start, line_end, value) in enumerate(blocks):
        content_hash = _hash_bytes(value.encode("utf-8"))
        chunk_id = _stable_id("chunk", paper_id, str(ordinal), content_hash)
        chunks.append(Chunk(
            chunk_id=chunk_id,
            paper_id=paper_id,
            section=section,
            text=value,
            line_start=line_start,
            line_end=line_end,
            content_hash=content_hash,
            vector=hashing_vector(f"{section} {value}"),
            node_type="paragraph",
            provenance={
                "parser": "markdown-lines-v1",
                "line_start": line_start,
                "line_end": line_end,
            },
            source_hash=content_hash,
        ))
    return chunks


def _metadata(text: str, path: Path) -> dict:
    title_match = TITLE_PATTERN.search(text)
    authors_match = AUTHORS_PATTERN.search(text)
    year_match = YEAR_PATTERN.search(text)
    doi_match = DOI_PATTERN.search(text)
    title = title_match.group(1).strip() if title_match else path.stem.replace("_", " ")
    authors = tuple(part.strip() for part in re.split(r",|;| and ", authors_match.group(1)) if part.strip()) if authors_match else ()
    doi = doi_match.group(1).strip() if doi_match else None
    if doi and doi.lower() in {"n/a", "none", "not available"}:
        doi = None
    return {"title": title, "authors": authors, "year": int(year_match.group(1)) if year_match else None, "doi": doi}


def _paper_id(doi: str | None, title: str, content_hash: str) -> str:
    seed = doi.lower() if doi else f"{title.lower()}:{content_hash}"
    return _stable_id("paper", seed)


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:20]
    return f"{prefix}_{digest}"


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _docling_node_type(label: str) -> str:
    if "formula" in label or "equation" in label:
        return "formula"
    if "table" in label:
        return "table"
    if "title" in label or "heading" in label or "section_header" in label:
        return "heading"
    if "code" in label:
        return "code"
    if "list" in label:
        return "list_item"
    if "picture" in label or "figure" in label:
        return "figure_caption"
    return "paragraph"


def _docling_item_text(item: object, document: object, node_type: str) -> str:
    if node_type == "table" and hasattr(item, "export_to_markdown"):
        try:
            value = item.export_to_markdown(doc=document)
            if str(value or "").strip():
                return str(value).strip()
        except (AttributeError, TypeError, ValueError):
            pass
    for attribute in ("text", "orig", "name"):
        value = getattr(item, attribute, None)
        if str(value or "").strip():
            return str(value).strip()
    code = getattr(item, "code", None)
    value = getattr(code, "text", None) if code is not None else None
    return str(value or "").strip()


def _docling_provenance(value: object) -> dict[str, object]:
    box = getattr(value, "bbox", None)
    bbox = None
    if box is not None:
        bbox = [round(float(getattr(box, name)), 2) for name in ("l", "t", "r", "b")]
    return {
        "page": int(getattr(value, "page_no")) if getattr(value, "page_no", None) is not None else None,
        "bbox": bbox,
        "coord_origin": str(getattr(getattr(box, "coord_origin", None), "value", getattr(box, "coord_origin", "unknown"))),
        "charspan": list(getattr(value, "charspan")) if getattr(value, "charspan", None) else None,
    }

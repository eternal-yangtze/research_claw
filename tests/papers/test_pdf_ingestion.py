from pathlib import Path

import pytest

from researchops.ingestion import DocumentIngestor
from researchops.storage import ResearchStore


def test_pdf_fallback_preserves_page_and_bbox(tmp_path: Path, monkeypatch) -> None:
    fitz = pytest.importorskip("fitz")
    monkeypatch.setenv("RESEARCHOPS_PDF_PARSER", "pymupdf")
    pdf_path = tmp_path / "paper.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Deterministic error bound for a compressed time series")
    document.save(pdf_path)
    document.close()

    store = ResearchStore(tmp_path / "research.db")
    result = DocumentIngestor(store).ingest_file(pdf_path)
    assert result.quality_status == "review_required"
    rows = store.all_chunks()
    assert rows[0]["page"] == 1
    assert rows[0]["bbox_json"] is not None
    assert (tmp_path / "artifacts" / "parsed" / f"{result.paper_id}.json").exists()


def test_changed_parser_sidecar_preserves_old_artifact(tmp_path: Path) -> None:
    import json
    store = ResearchStore(tmp_path / "research.db")
    ingestor = DocumentIngestor(store)
    source = tmp_path / "unused.pdf"
    first = ingestor._write_pdf_sidecar("p1", source, "old-parser", "pdfhash", [])
    original = Path(first).read_bytes()
    second = ingestor._write_pdf_sidecar("p1", source, "new-parser", "pdfhash", [])
    assert first != second
    assert Path(first).read_bytes() == original
    assert json.loads(Path(second).read_text(encoding="utf-8"))["parser"] == "new-parser"

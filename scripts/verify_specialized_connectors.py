from __future__ import annotations

import json
import os
from pathlib import Path

from researchops.compression import PaperCompressor
from researchops.ingestion import DocumentIngestor
from researchops.literature import LiteratureSearch, download_pdf
from researchops.models import Domain, Idea, Paper, PaperIdentity, PaperSourceRecord
from researchops.storage import ResearchStore


def _paper(value: dict) -> Paper:
    return Paper(**{
        **value,
        "authors": tuple(value.get("authors") or ()),
        "identity": PaperIdentity(**value["identity"]) if value.get("identity") else None,
        "source_records": tuple(PaperSourceRecord(**item) for item in value.get("source_records") or ()),
    })


def main() -> int:
    workspace = Path("workspace_acl_cvf_smoke").resolve()
    os.environ["RESEARCHOPS_PDF_PARSER"] = "pymupdf"
    store = ResearchStore(workspace / "researchops.db")
    search = LiteratureSearch(store, timeout=90)
    ingestor = DocumentIngestor(store)
    compressor = PaperCompressor(store)
    idea = Idea(
        "Graph neural network methods",
        Domain.AI_ML,
        "Study graph neural network methods and their evaluation.",
        "Which graph neural network methods and evaluation designs are useful?",
        ("Graph structure can guide learned representations.",),
        ("Compare against non-graph baselines.",),
    )
    output = []
    for source in ("acl", "cvf"):
        result = search.search(
            "graph neural network", f"run_{source}_download_verification", 1, sources=[source]
        )
        paper = _paper(result["papers"][0])
        downloaded = download_pdf(
            paper.pdf_url or "", workspace / "artifacts" / "papers", f"{source}_{paper.paper_id}.pdf"
        )
        ingested = ingestor.ingest_file(downloaded["path"], paper=paper)
        updated = Paper(**{**paper.__dict__, "file_path": downloaded["path"], "content_hash": downloaded["sha256"]})
        l0 = compressor.l0(updated)
        l1 = compressor.l1(updated, idea)
        output.append({
            "source": source,
            "paper_id": paper.paper_id,
            "title": paper.title,
            "external_id": paper.external_id,
            "pdf_url": paper.pdf_url,
            "download": downloaded,
            "ingestion": ingested.to_dict(),
            "compressions": [l0["compression_id"], l1["compression_id"]],
        })
    report = {
        "query": "graph neural network",
        "papers": output,
        "database": str(store.path),
        "counts": store.counts(),
    }
    target = workspace / "artifacts" / "specialized_connector_verification.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**report, "report_path": str(target)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run unmodified reference connectors without user keys or contact emails.

This diagnostic does not load the project's .env and does not copy connector code.
Run one case per process so upstream calls and failures remain isolated.
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[2]
LITERATURE = ROOT / "references2_structure" / "2文献研究-检索、精读、综述与引用网络"
OUTPUT = ROOT / "researchops_mvp" / "workspace" / "reference_access_verification"
CASES = {
    "paper_crossref": ("paper", "crossref", "CrossRefSearcher", "search", {"query": "Attention Is All You Need", "max_results": 1}),
    "paper_openalex": ("paper", "openalex", "OpenAlexSearcher", "search", {"query": "Attention Is All You Need", "max_results": 1}),
    "paper_semantic": ("paper", "semantic", "SemanticSearcher", "search", {"query": "Attention Is All You Need", "max_results": 1}),
    "paper_unpaywall": ("paper", "unpaywall", "UnpaywallResolver", "get_paper_by_doi", {"doi": "10.1145/3299869.3320211"}),
    "paper_arxiv": ("paper", "arxiv", "ArxivSearcher", "search", {"query": "id:1706.03762", "max_results": 1}),
    "nature_crossref": ("nature", "crossref", "CrossRefSource", "search", {"query": "Attention Is All You Need", "rows": 1}),
    "nature_openalex": ("nature", "openalex", "OpenAlexSource", "search", {"query": "Attention Is All You Need", "rows": 1}),
    "nature_semantic": ("nature", "semantic_scholar", "SemanticScholarSource", "search", {"query": "Attention Is All You Need", "rows": 1}),
    "nature_arxiv": ("nature", "arxiv", "ArxivSource", "search", {"query": 'ti:"Attention Is All You Need"', "rows": 1}),
    "paper_arxiv_download": ("paper", "arxiv", "ArxivSearcher", "download_pdf", {"paper_id": "1706.03762", "save_path": str(OUTPUT / "downloads")}),
    "paper_arxiv_read": ("paper", "arxiv", "ArxivSearcher", "read_paper", {"paper_id": "1706.03762", "save_path": str(OUTPUT / "downloads")}),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=CASES)
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "downloads").mkdir(exist_ok=True)

    # Blank process-local values prevent inherited credentials being used.
    # An explicit nonexistent config path prevents the reference loading a user's .env.
    for name in ("OPENALEX_API_KEY", "OPENALEX_EMAIL", "SEMANTIC_SCHOLAR_API_KEY", "UNPAYWALL_EMAIL", "CROSSREF_MAILTO", "PUBMED_EMAIL", "NCBI_API_KEY"):
        os.environ[name] = ""
        os.environ["PAPER_SEARCH_MCP_" + name] = ""
    os.environ["PAPER_SEARCH_MCP_ENV_FILE"] = str(OUTPUT / "not_configured.env")
    os.environ["NATURE_ACADEMIC_SEARCH_CONFIG"] = str(OUTPUT / "not_configured.toml")
    logging.basicConfig(level=logging.ERROR)
    log_errors: list[str] = []

    class CaptureErrors(logging.Handler):
        def emit(self, event: logging.LogRecord) -> None:
            if event.levelno >= logging.ERROR:
                log_errors.append(event.getMessage())

    logging.getLogger().addHandler(CaptureErrors())

    record = {
        "case": args.case,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "user_api_key_supplied": False,
        "user_email_supplied": False,
        "connector_modified": False,
    }
    began = time.perf_counter()
    package, module_name, class_name, method, parameters = CASES[args.case]
    if package == "paper":
        base = LITERATURE / "paper-search-mcp-main"
        sys.path.insert(0, str(base))
        module = f"paper_search_mcp.academic_platforms.{module_name}"
    else:
        base = LITERATURE / "nature-academic-search-main"
        sys.path.insert(0, str(base / "src"))
        module = f"nature_academic_search.sources.{module_name}"
    try:
        connector_module = importlib.import_module(module)
        record["source_file"] = str(Path(connector_module.__file__).resolve())
        connector = getattr(connector_module, class_name)()
        result = getattr(connector, method)(**parameters)
        if method == "download_pdf":
            from pypdf import PdfReader
            path = Path(result)
            pages = len(PdfReader(path).pages)
            record.update(status="verified_success" if pages > 0 else "failed_validation", pdf_pages=pages, pdf_bytes=path.stat().st_size, pdf_path=str(path))
        elif method == "read_paper":
            text = str(result or "")
            record.update(status="verified_success" if len(text.strip()) > 100 and "attention" in text.lower() else "failed_validation", extracted_characters=len(text), contains_attention="attention" in text.lower())
        else:
            if isinstance(result, dict):
                items = result.get("results") or result.get("data") or []
            elif isinstance(result, list):
                items = result
            elif result is None:
                items = []
            else:
                items = [result]
            titles = [str(item.get("title") or "") if isinstance(item, dict) else str(getattr(item, "title", "")) for item in items]
            titles = [title for title in titles if title.strip()]
            record.update(status="verified_success" if titles else "not_verified", result_count=len(items), sample_titles=titles[:2])
            if not titles:
                record["note"] = "Empty/None output is not proof of successful search; see connector logs or credential gates."
    except Exception as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error))
    if log_errors:
        record["connector_error_logs"] = log_errors
    record["duration_seconds"] = round(time.perf_counter() - began, 3)
    (OUTPUT / f"{args.case}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import gzip
import io
import tarfile
from pathlib import Path

import pytest

import researchops.literature as literature_module
from researchops.literature import (
    LiteratureSearch,
    _arxiv_query,
    download_arxiv_source,
    ensure_paper_identity,
    merge_papers,
)
from researchops.models import Paper
from researchops.storage import ResearchStore


def test_source_failures_are_not_reported_as_zero_results(tmp_path: Path) -> None:
    store = ResearchStore(tmp_path / "research.db")
    search = LiteratureSearch(store)
    search.connectors = {
        "working": lambda query, limit: [Paper("p1", "A real result", source="working")],
        "broken": lambda query, limit: (_ for _ in ()).throw(RuntimeError("upstream unavailable")),
    }
    result = search.search("time series", "run-test", 5)
    statuses = {item["source"]: item for item in result["sources"]}
    assert statuses["working"]["status"] == "success"
    assert statuses["working"]["result_count"] == 1
    assert statuses["broken"]["status"] == "failed"
    assert statuses["broken"]["result_count"] == 0
    assert "upstream unavailable" in statuses["broken"]["error"]


def test_openalex_reconstructs_abstract_and_direct_pdf(tmp_path: Path, monkeypatch) -> None:
    search = LiteratureSearch(ResearchStore(tmp_path / "research.db"))
    monkeypatch.setattr(literature_module, "_get_json", lambda *args, **kwargs: {
        "results": [{
            "id": "https://openalex.org/W123",
            "display_name": "A Unified Paper",
            "doi": "https://doi.org/10.1000/TEST",
            "publication_year": 2025,
            "authorships": [{"author": {"display_name": "Ada Author"}}],
            "abstract_inverted_index": {"hello": [0], "world": [1]},
            "primary_location": {
                "landing_page_url": "https://example.org/article",
                "pdf_url": "https://example.org/article.pdf",
            },
        }]
    })

    paper = search._openalex("unified", 3)[0]
    assert paper.abstract == "hello world"
    assert paper.doi == "10.1000/test"
    assert paper.pdf_url == "https://example.org/article.pdf"
    assert paper.identity.openalex_id == "W123"


def test_openreview_accepts_only_public_root_papers(tmp_path: Path, monkeypatch) -> None:
    search = LiteratureSearch(ResearchStore(tmp_path / "research.db"))
    notes = [
        {
            "id": "paper1", "forum": "paper1", "cdate": 1735689600000,
            "content": {
                "title": {"value": "Public Paper"},
                "authors": {"value": ["A. Author", {"fullname": "B. Author"}]},
                "abstract": {"value": "The method."},
                "pdf": {"value": "/pdf?id=paper1"},
            },
        },
        {"id": "review1", "forum": "paper1", "replyto": "paper1", "content": {"title": "Review"}},
        {"id": "deleted", "forum": "deleted", "ddate": 1, "content": {"title": "Deleted", "abstract": "x"}},
    ]
    monkeypatch.setattr(literature_module, "_get_json", lambda *args, **kwargs: {"notes": notes})

    papers = search._openreview("method", 10)
    assert len(papers) == 1
    assert papers[0].external_id == "paper1"
    assert papers[0].authors == ("A. Author", "B. Author")
    assert papers[0].pdf_url == "https://openreview.net/pdf?id=paper1"
    assert papers[0].identity.openreview_id == "paper1"


def test_dblp_parses_json_metadata_without_inventing_pdf(tmp_path: Path, monkeypatch) -> None:
    search = LiteratureSearch(ResearchStore(tmp_path / "research.db"))
    monkeypatch.setattr(literature_module, "_get_json", lambda *args, **kwargs: {
        "result": {"hits": {"hit": [{"info": {
            "title": "A &amp; B",
            "authors": {"author": [{"text": "First Author"}, "Second Author"]},
            "year": "2024",
            "key": "conf/test/Paper24",
            "url": "https://dblp.org/rec/conf/test/Paper24",
            "ee": "https://doi.org/10.5555/example",
        }}]}}
    })

    paper = search._dblp("paper", 5)[0]
    assert paper.title == "A & B"
    assert paper.authors == ("First Author", "Second Author")
    assert paper.doi == "10.5555/example"
    assert paper.pdf_url is None
    assert paper.identity.dblp_key == "conf/test/Paper24"


def test_identity_merge_keeps_provenance_and_richer_fields() -> None:
    first = ensure_paper_identity(Paper(
        "p1", "Same Paper", doi="10.1/ABC", source="dblp", external_id="conf/x/1"
    ), dblp_key="conf/x/1")
    second = ensure_paper_identity(Paper(
        "p2", "Same Paper", doi="https://doi.org/10.1/abc", source="openalex",
        external_id="W1", abstract="A substantially richer abstract.", pdf_url="https://x.test/p.pdf",
    ), openalex_id="W1")

    merged = merge_papers(first, second)
    assert merged.doi == "10.1/abc"
    assert merged.abstract == second.abstract
    assert merged.pdf_url == second.pdf_url
    assert {item.source for item in merged.source_records} == {"dblp", "openalex"}
    assert merged.identity.dblp_key == "conf/x/1"
    assert merged.identity.openalex_id == "W1"


def test_arxiv_natural_language_query_uses_bounded_term_conjunction() -> None:
    query = _arxiv_query("parameter efficient fine tuning vision transformer image classification")
    assert query == "all:parameter AND all:efficient AND all:fine AND all:tuning AND all:vision"
    assert '"' not in query


def test_identity_merge_prefers_reliable_public_pdf() -> None:
    journal = ensure_paper_identity(Paper(
        "p1", "Same Paper", source="openalex",
        pdf_url="https://publisher.test/download.pdf",
    ))
    arxiv = ensure_paper_identity(Paper(
        "p2", "Same Paper", source="arxiv", external_id="2401.00001",
        pdf_url="https://arxiv.org/pdf/2401.00001",
    ))

    assert merge_papers(journal, arxiv).pdf_url == arxiv.pdf_url


def test_arxiv_source_download_validates_tar_without_extracting(tmp_path: Path, monkeypatch) -> None:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        content = b"\\documentclass{article}\\begin{document}ok\\end{document}"
        info = tarfile.TarInfo("main.tex")
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))
    monkeypatch.setattr(literature_module, "_get_bytes", lambda *args, **kwargs: buffer.getvalue())

    result = download_arxiv_source("1706.03762v7", tmp_path / "sources")
    assert Path(result["path"]).is_file()
    assert result["archive_format"] == "tar_gzip"
    assert result["member_count"] == 1
    assert result["url"].endswith("1706.03762v7")
    assert result["source_version"] == "v7"
    assert Path(result["receipt_path"]).is_file()


def test_arxiv_source_rejects_invalid_id_and_html(tmp_path: Path, monkeypatch) -> None:
    with pytest.raises(ValueError, match="Invalid arXiv"):
        download_arxiv_source("../escape", tmp_path)
    monkeypatch.setattr(literature_module, "_get_bytes", lambda *args, **kwargs: b"<html>rate limited</html>")
    with pytest.raises(ValueError, match="markup"):
        download_arxiv_source("1706.03762", tmp_path)


def test_acl_connector_returns_official_pdf_links(tmp_path: Path, monkeypatch) -> None:
    catalog = b'''@inproceedings{2024.emnlp-main.554,
      title = {A Useful NLP Method},
      author = {Doe, Jane and Roe, John},
      year = {2024},
      url = {https://aclanthology.org/2024.emnlp-main.554/},
    }
    @inproceedings{2023.acl-long.12,
      title = {Another Paper},
      author = {Smith, Ada},
      year = {2023},
      url = {https://aclanthology.org/2023.acl-long.12/},
    }'''
    monkeypatch.setattr(literature_module, "_get_bytes", lambda *args, **kwargs: gzip.compress(catalog))
    search = LiteratureSearch(ResearchStore(tmp_path / "research.db"))

    paper = search._acl("useful method", 1)[0]
    assert paper.identity.acl_id == "2024.emnlp-main.554"
    assert paper.authors == ("Doe, Jane", "Roe, John")
    assert paper.pdf_url == "https://aclanthology.org/2024.emnlp-main.554.pdf"
    assert paper.source_url == "https://aclanthology.org/2024.emnlp-main.554/"


def test_cvf_connector_filters_catalog_and_builds_identity(tmp_path: Path, monkeypatch) -> None:
    html = b'''<html><dl>
    <dt class="ptitle"><a href="/content/CVPR2025/html/Doe_Graph_Time_Series_CVPR_2025_paper.html">Graph Time Series Vision</a></dt>
    <dd>Jane Doe, John Roe</dd><dd>[<a href="/content/CVPR2025/papers/Doe_Graph_Time_Series_CVPR_2025_paper.pdf">pdf</a>]</dd>
    <dt class="ptitle"><a href="/content/CVPR2025/html/Other_Paper_CVPR_2025_paper.html">Unrelated Rendering</a></dt>
    <dd>A. Author</dd><dd>[<a href="/content/CVPR2025/papers/Other_Paper_CVPR_2025_paper.pdf">pdf</a>]</dd>
    </dl></html>'''
    monkeypatch.setenv("RESEARCHOPS_CVF_VENUES", "CVPR2025")
    monkeypatch.setattr(literature_module, "_get_bytes", lambda *args, **kwargs: html)
    search = LiteratureSearch(ResearchStore(tmp_path / "research.db"))

    papers = search._cvf("graph time series", 5)
    assert len(papers) == 1
    assert papers[0].authors == ("Jane Doe", "John Roe")
    assert papers[0].identity.cvf_id.startswith("CVPR2025:")
    assert papers[0].pdf_url.endswith("_paper.pdf")

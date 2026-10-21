from __future__ import annotations

import hashlib
import gzip
import html
import io
import json
import os
import re
import ssl
import tarfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable

import certifi

from researchops.core.models import Paper, PaperIdentity, PaperSourceRecord, SourceStatus
from researchops.core.storage import ResearchStore


USER_AGENT = "researchops-mvp/0.2 (auditable academic search)"
ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_SOURCE = "https://export.arxiv.org/e-print/"
OPENALEX_API = "https://api.openalex.org/works"
OPENREVIEW_API = "https://api2.openreview.net"
DBLP_API = "https://dblp.org/search/publ/api"
ACL_SEARCH = "https://aclanthology.org/search/"
ACL_CATALOG = "https://aclanthology.org/anthology.bib.gz"
CVF_ROOT = "https://openaccess.thecvf.com"
ATOM = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


class LiteratureSearch:
    def __init__(self, store: ResearchStore, timeout: float = 12.0) -> None:
        self.store = store
        self.timeout = timeout
        self.connectors: dict[str, Callable[[str, int], list[Paper]]] = {
            "arxiv": self._arxiv,
            "openalex": self._openalex,
            "openreview": self._openreview,
            "dblp": self._dblp,
            "acl": self._acl,
            "cvf": self._cvf,
            "crossref": self._crossref,
        }
        self._arxiv_lock = threading.Lock()
        self._last_arxiv_request = 0.0
        self._acl_catalog_cache: list[dict[str, object]] | None = None

    def search(
        self,
        query: str,
        run_id: str,
        limit_per_source: int = 10,
        sources: list[str] | tuple[str, ...] | None = None,
    ) -> dict:
        if not query.strip():
            raise ValueError("query must not be empty")
        papers: dict[str, Paper] = {}
        statuses: list[SourceStatus] = []
        selected = tuple(sources or self.connectors)
        unknown = sorted(set(selected) - set(self.connectors))
        if unknown:
            raise ValueError("unsupported literature source(s): " + ", ".join(unknown))
        with ThreadPoolExecutor(max_workers=max(1, len(selected))) as pool:
            futures = {
                pool.submit(self.connectors[name], query, limit_per_source): name
                for name in selected
            }
            for future in as_completed(futures):
                source = futures[future]
                try:
                    found = future.result()
                    statuses.append(SourceStatus(source, "success", query, len(found), searched_at=_now()))
                    for paper in found:
                        paper = ensure_paper_identity(paper)
                        matching = next((key for key, current in papers.items() if papers_match(current, paper)), None)
                        if matching is None:
                            papers[paper.paper_id] = paper
                        else:
                            papers[matching] = merge_papers(papers[matching], paper)
                except Exception as error:
                    statuses.append(SourceStatus(source, "failed", query, 0, f"{type(error).__name__}: {error}", _now()))
        for paper in papers.values():
            self.store.upsert_paper(paper)
        for status in statuses:
            self.store.record_source_status(run_id, status)
        return {
            "papers": [asdict(paper) for paper in papers.values()],
            "sources": [status.__dict__ for status in sorted(statuses, key=lambda item: item.source)],
        }

    def _arxiv(self, query: str, limit: int) -> list[Paper]:
        """Search arXiv's public Atom API.

        Query quoting, pacing, metadata mapping, and PDF URL handling are
        adapted from paper-search-mcp's MIT-licensed ArxivSearcher.
        """

        search_query = _arxiv_query(query)
        url = ARXIV_API + "?" + urllib.parse.urlencode({
            "search_query": search_query,
            "start": 0,
            "max_results": min(max(1, limit), 100),
            "sortBy": "relevance",
            "sortOrder": "descending",
        })
        with self._arxiv_lock:
            elapsed = time.monotonic() - self._last_arxiv_request
            if elapsed < 3.0:
                time.sleep(3.0 - elapsed)
            try:
                payload = _get_bytes(url, self.timeout, accept="application/atom+xml")
            finally:
                self._last_arxiv_request = time.monotonic()
        root = ET.fromstring(payload)
        papers: list[Paper] = []
        for entry in root.findall("atom:entry", ATOM):
            title = _xml_text(entry, "atom:title")
            if not title:
                continue
            entry_url = _xml_text(entry, "atom:id")
            raw_external_id = entry_url.rsplit("/abs/", 1)[-1].strip()
            external_id, version = normalize_arxiv_id(raw_external_id)
            authors = tuple(
                _xml_text(author, "atom:name")
                for author in entry.findall("atom:author", ATOM)
                if _xml_text(author, "atom:name")
            )
            pdf_url = next(
                (
                    str(link.attrib.get("href") or "")
                    for link in entry.findall("atom:link", ATOM)
                    if link.attrib.get("type") == "application/pdf"
                ),
                f"https://arxiv.org/pdf/{raw_external_id}.pdf" if raw_external_id else "",
            )
            doi = _xml_text(entry, "arxiv:doi") or None
            published = _xml_text(entry, "atom:published")
            paper = Paper(
                paper_id=_external_paper_id(doi, title),
                title=_clean_space(title),
                authors=authors,
                year=int(published[:4]) if published[:4].isdigit() else None,
                doi=doi,
                source="arxiv",
                source_url=entry_url,
                quality_status="metadata_only",
                abstract=_clean_space(_xml_text(entry, "atom:summary")) or None,
                pdf_url=pdf_url or None,
                external_id=external_id or None,
                source_version=version,
                source_archive_url=ARXIV_SOURCE + external_id if external_id else None,
            )
            papers.append(ensure_paper_identity(paper, arxiv_id=external_id))
        return papers

    def _crossref(self, query: str, limit: int) -> list[Paper]:
        params = {"query.bibliographic": query, "rows": limit, "sort": "relevance", "order": "desc"}
        mailto = os.getenv("CROSSREF_MAILTO", "").strip()
        if mailto:
            params["mailto"] = mailto
        url = "https://api.crossref.org/works?" + urllib.parse.urlencode(params)
        payload = _get_json(url, self.timeout)
        papers: list[Paper] = []
        for item in payload.get("message", {}).get("items", []):
            title = " ".join(item.get("title") or []).strip()
            if not title:
                continue
            doi = str(item.get("DOI") or "").strip() or None
            authors = tuple(
                " ".join(filter(None, (author.get("given"), author.get("family"))))
                for author in item.get("author") or []
            )
            year = _date_year(item.get("published"))
            papers.append(ensure_paper_identity(Paper(
                paper_id=_external_paper_id(doi, title),
                title=title,
                authors=authors,
                year=year,
                doi=doi,
                source="crossref",
                source_url=item.get("URL"),
                quality_status="metadata_only",
                abstract=_strip_markup(str(item.get("abstract") or "")) or None,
                pdf_url=_crossref_pdf_url(item),
                external_id=doi,
            )))
        return papers

    def _openalex(self, query: str, limit: int) -> list[Paper]:
        params: dict[str, str | int] = {"search": query, "per-page": min(max(1, limit), 100)}
        email = os.getenv("PAPER_SEARCH_MCP_OPENALEX_EMAIL", "").strip()
        if email:
            params["mailto"] = email
        headers = {}
        api_key = os.getenv("PAPER_SEARCH_MCP_OPENALEX_API_KEY", "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        url = OPENALEX_API + "?" + urllib.parse.urlencode(params)
        payload = _get_json(url, self.timeout, headers=headers)
        if not isinstance(payload.get("results"), list):
            raise RuntimeError("OpenAlex response is missing its results array")
        papers: list[Paper] = []
        for item in payload.get("results", []):
            if not isinstance(item, dict):
                continue
            title = str(item.get("display_name") or "").strip()
            if not title:
                continue
            doi = normalize_doi(str(item.get("doi") or ""))
            authors = tuple(
                str((entry.get("author") or {}).get("display_name") or "").strip()
                for entry in item.get("authorships") or []
                if isinstance(entry, dict) and (entry.get("author") or {}).get("display_name")
            )
            primary = item.get("primary_location") or {}
            best_oa = item.get("best_oa_location") or {}
            pdf_url = _https_url(primary.get("pdf_url")) or _https_url(best_oa.get("pdf_url"))
            source_url = (
                _https_url(primary.get("landing_page_url"))
                or _https_url(best_oa.get("landing_page_url"))
                or _https_url(item.get("id"))
            )
            openalex_id = str(item.get("id") or "").rsplit("/", 1)[-1] or None
            paper = Paper(
                paper_id=_external_paper_id(doi, title),
                title=title,
                authors=authors,
                year=item.get("publication_year"),
                doi=doi,
                source="openalex",
                source_url=source_url,
                quality_status="metadata_only",
                abstract=_openalex_abstract(item.get("abstract_inverted_index")) or None,
                pdf_url=pdf_url,
                external_id=openalex_id,
            )
            papers.append(ensure_paper_identity(paper, openalex_id=openalex_id))
        return papers

    def _openreview(self, query: str, limit: int) -> list[Paper]:
        params = {
            "term": query,
            "content": "all",
            "group": "all",
            "source": "forum",
            "limit": min(max(1, limit), 100),
            "offset": 0,
        }
        payload = _get_json(OPENREVIEW_API + "/notes/search?" + urllib.parse.urlencode(params), self.timeout)
        notes = payload.get("notes")
        if not isinstance(notes, list):
            raise RuntimeError("OpenReview response is missing its notes array")
        papers: list[Paper] = []
        for note in notes:
            paper = _openreview_paper(note)
            if paper is not None:
                papers.append(paper)
                if len(papers) >= limit:
                    break
        return papers

    def _dblp(self, query: str, limit: int) -> list[Paper]:
        params = {"q": query, "format": "json", "h": min(max(1, limit), 1000)}
        payload = _get_json(DBLP_API + "?" + urllib.parse.urlencode(params), self.timeout)
        hits = ((payload.get("result") or {}).get("hits") or {}).get("hit")
        if hits is None:
            raise RuntimeError("DBLP response is missing its hit array")
        if isinstance(hits, dict):
            hits = [hits]
        if not isinstance(hits, list):
            raise RuntimeError("DBLP response contains a malformed hit array")
        papers: list[Paper] = []
        for hit in hits:
            paper = _dblp_paper(hit)
            if paper is not None:
                papers.append(paper)
        return papers

    def _acl(self, query: str, limit: int) -> list[Paper]:
        catalog = self._load_acl_catalog()
        terms = _search_terms(query)
        normalized_query = _clean_space(query).casefold()
        ranked: list[tuple[int, dict[str, object]]] = []
        for item in catalog:
            searchable = f"{item['title']} {' '.join(item['authors'])}".casefold()
            matched = sum(term in searchable for term in terms)
            if terms and not matched:
                continue
            score = matched + (len(terms) + 2 if normalized_query in searchable else 0)
            ranked.append((score, item))
        ranked.sort(key=lambda pair: (-pair[0], -int(pair[1].get("year") or 0), str(pair[1]["title"])))
        papers: list[Paper] = []
        for _score, item in ranked[:limit]:
            acl_id = str(item["acl_id"])
            title = str(item["title"])
            paper = Paper(
                paper_id=_external_paper_id(None, title),
                title=title,
                authors=tuple(str(author) for author in item["authors"]),
                year=int(item["year"]) if item.get("year") else None,
                source="acl",
                source_url=f"https://aclanthology.org/{acl_id}/",
                quality_status="metadata_only",
                pdf_url=f"https://aclanthology.org/{acl_id}.pdf",
                external_id=acl_id,
            )
            papers.append(ensure_paper_identity(paper, acl_id=acl_id))
        return papers

    def _load_acl_catalog(self) -> list[dict[str, object]]:
        if self._acl_catalog_cache is not None:
            return self._acl_catalog_cache
        cache_dir = self.store.path.parent / "cache" / "literature"
        cache_dir.mkdir(parents=True, exist_ok=True)
        archive_path = cache_dir / "acl_anthology.bib.gz"
        if not archive_path.is_file():
            data = _get_bytes(
                ACL_CATALOG,
                max(self.timeout, 60),
                accept="application/gzip",
                max_bytes=20_000_000,
            )
            if not data.startswith(b"\x1f\x8b"):
                raise RuntimeError("ACL Anthology catalog is not a gzip archive")
            archive_path.write_bytes(data)
        try:
            bibtex = gzip.decompress(archive_path.read_bytes()).decode("utf-8", errors="replace")
        except (OSError, EOFError) as error:
            raise RuntimeError("ACL Anthology cached catalog is corrupt") from error
        catalog: list[dict[str, object]] = []
        for bibkey, block in _bibtex_entries(bibtex):
            title = _clean_bibtex(_bibtex_field(block, "title"))
            if not title:
                continue
            official_url = _bibtex_field(block, "url")
            parsed_url = urllib.parse.urlparse(official_url)
            acl_id = (
                parsed_url.path.strip("/")
                if parsed_url.scheme == "https" and parsed_url.hostname == "aclanthology.org"
                else bibkey
            )
            author_text = _bibtex_field(block, "author")
            year_text = _bibtex_field(block, "year")
            authors = tuple(
                _clean_bibtex(author)
                for author in re.split(r"\s+and\s+", author_text)
                if _clean_bibtex(author)
            )
            catalog.append({
                "acl_id": acl_id,
                "title": title,
                "authors": authors,
                "year": int(year_text) if year_text.strip().isdigit() else None,
            })
        if not catalog:
            raise RuntimeError("ACL Anthology catalog contained no parseable papers")
        self._acl_catalog_cache = catalog
        return catalog

    def _cvf(self, query: str, limit: int) -> list[Paper]:
        venues = [
            item.strip()
            for item in os.getenv("RESEARCHOPS_CVF_VENUES", "CVPR2025,ICCV2025,WACV2025").split(",")
            if re.fullmatch(r"(?:CVPR|ICCV|WACV|ECCV)\d{4}", item.strip(), re.I)
        ]
        query_terms = _search_terms(query)
        candidates: list[tuple[int, Paper]] = []
        for venue in venues:
            url = f"{CVF_ROOT}/{venue.upper()}?day=all"
            payload = _get_bytes(url, max(self.timeout, 30), accept="text/html", max_bytes=20_000_000)
            parser = _CVFProceedingsParser(venue.upper())
            parser.feed(payload.decode("utf-8", errors="replace"))
            for item in parser.results:
                searchable = f"{item['title']} {' '.join(item['authors'])}"
                score = sum(term in searchable.casefold() for term in query_terms)
                if query_terms and not score:
                    continue
                cvf_id = item["cvf_id"]
                paper = Paper(
                    paper_id=_external_paper_id(None, item["title"]),
                    title=item["title"],
                    authors=tuple(item["authors"]),
                    year=int(venue[-4:]),
                    source="cvf",
                    source_url=urllib.parse.urljoin(CVF_ROOT, item["paper_url"]),
                    quality_status="metadata_only",
                    pdf_url=urllib.parse.urljoin(CVF_ROOT, item["pdf_url"]),
                    external_id=cvf_id,
                )
                candidates.append((score, ensure_paper_identity(paper, cvf_id=cvf_id)))
            if len(candidates) >= limit:
                break
        candidates.sort(key=lambda pair: (-pair[0], -(pair[1].year or 0), pair[1].title))
        return [paper for _, paper in candidates[:limit]]


def download_pdf(url: str, output_dir: str | Path, filename: str | None = None, max_bytes: int = 50_000_000) -> dict:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("Only explicit HTTPS URLs are allowed")
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    safe_name = Path(filename or Path(parsed.path).name or "paper.pdf").name
    if not safe_name.lower().endswith(".pdf"):
        safe_name += ".pdf"
    target = (root / safe_name).resolve()
    if root != target.parent:
        raise ValueError("download target escaped artifact directory")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/pdf"})
    tls_context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(request, timeout=30, context=tls_context) as response:
        content_length = int(response.headers.get("Content-Length") or 0)
        if content_length > max_bytes:
            raise ValueError(f"PDF exceeds {max_bytes} byte limit")
        data = response.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError(f"PDF exceeds {max_bytes} byte limit")
    if not data.startswith(b"%PDF-"):
        raise ValueError("downloaded response is not a PDF")
    target.write_bytes(data)
    return {"path": str(target), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "url": url}


def download_arxiv_source(
    arxiv_id: str,
    output_dir: str | Path,
    max_bytes: int = 100_000_000,
) -> dict:
    """Download and validate an arXiv source package without extracting it."""
    base_id, source_version = normalize_arxiv_id(arxiv_id)
    if not base_id:
        raise ValueError("Invalid arXiv identifier")
    # A versionless request returns the latest source, which need not match the PDF.
    requested_id = base_id + (source_version or "")
    url = ARXIV_SOURCE + requested_id
    data = _get_bytes(
        url,
        60,
        accept="application/gzip, application/x-tar, text/plain",
        max_bytes=max_bytes,
    )
    prefix = data.lstrip()[:64].lower()
    if prefix.startswith((b"<!doctype html", b"<html", b"<?xml")):
        raise ValueError("arXiv source endpoint returned markup instead of source")

    archive_format = "plain_tex"
    member_count: int | None = None
    suffix = ".tex"
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as archive:
            members = archive.getmembers()
            for member in members:
                member_path = Path(member.name.replace("\\", "/"))
                if member_path.is_absolute() or ".." in member_path.parts:
                    raise ValueError("arXiv source archive contains an unsafe member path")
            member_count = len(members)
            archive_format = "tar_gzip" if data.startswith(b"\x1f\x8b") else "tar"
            suffix = ".tar.gz" if archive_format == "tar_gzip" else ".tar"
    except tarfile.ReadError:
        if data.startswith(b"\x1f\x8b"):
            with gzip.GzipFile(fileobj=io.BytesIO(data)) as stream:
                sample = stream.read(64_001)
            if len(sample) > 64_000:
                sample = sample[:64_000]
            if not _looks_like_tex(sample):
                raise ValueError("unrecognized gzip arXiv source payload")
            archive_format = "gzip_tex"
            suffix = ".tex.gz"
        elif not _looks_like_tex(data[:64_000]):
            raise ValueError("unrecognized arXiv source payload")

    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    safe_id = requested_id.replace("/", "_")
    digest = hashlib.sha256(data).hexdigest()
    target = (root / f"arxiv_{safe_id}{suffix}").resolve()
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != digest:
        target = (root / f"arxiv_{safe_id}_{digest}{suffix}").resolve()
    if target.parent != root:
        raise ValueError("download target escaped artifact directory")
    target.write_bytes(data)
    receipt = {
        "path": str(target),
        "bytes": len(data),
        "sha256": digest,
        "url": url,
        "arxiv_id": base_id,
        "source_version": source_version,
        "retrieved_at": _now(),
        "archive_format": archive_format,
        "member_count": member_count,
    }
    receipt_path = target.with_name(target.name + ".receipt.json")
    if not receipt_path.exists():
        receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**receipt, "receipt_path": str(receipt_path)}


def _get_json(url: str, timeout: float, *, headers: dict[str, str] | None = None) -> dict:
    data = _get_bytes(url, timeout, accept="application/json", headers=headers)
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"{urllib.parse.urlparse(url).netloc} returned a non-JSON response"
        ) from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"{urllib.parse.urlparse(url).netloc} returned a non-object JSON response")
    return payload


def _get_bytes(
    url: str,
    timeout: float,
    *,
    accept: str,
    headers: dict[str, str] | None = None,
    max_bytes: int | None = None,
) -> bytes:
    request_headers = {"User-Agent": USER_AGENT, "Accept": accept, **(headers or {})}
    request = urllib.request.Request(url, headers=request_headers)
    try:
        tls_context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(request, timeout=timeout, context=tls_context) as response:
            content_length = int(response.headers.get("Content-Length") or 0)
            if max_bytes is not None and content_length > max_bytes:
                raise ValueError(f"response exceeds {max_bytes} byte limit")
            data = response.read(max_bytes + 1 if max_bytes is not None else -1)
            if max_bytes is not None and len(data) > max_bytes:
                raise ValueError(f"response exceeds {max_bytes} byte limit")
            return data
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"HTTP {error.code} from {urllib.parse.urlparse(url).netloc}") from error


def _external_paper_id(doi: str | None, title: str) -> str:
    seed = normalize_doi(doi) or title_fingerprint(title)
    return "paper_" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:20]


def _date_year(value) -> int | None:
    parts = (value or {}).get("date-parts") or []
    return int(parts[0][0]) if parts and parts[0] else None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _arxiv_query(query: str) -> str:
    normalized = _clean_space(query)
    if not normalized:
        raise ValueError("query must not be empty")
    if '"' in normalized or re.search(r"\b(?:all|ti|au|abs|cat|co|jr|id):", normalized, re.I):
        return normalized
    if re.search(r"\b(?:AND|OR|ANDNOT)\b", normalized):
        return normalized
    terms = [
        term
        for term in re.findall(r"[A-Za-z0-9][A-Za-z0-9+._-]*", normalized)
        if term.casefold() not in {"a", "an", "and", "for", "in", "of", "on", "the", "to", "with"}
    ]
    # arXiv treats all:"a long natural-language query" as one exact phrase,
    # which commonly returns zero papers. A small conjunction preserves intent.
    terms = list(dict.fromkeys(terms))[:5]
    return " AND ".join(f"all:{term}" for term in terms)


def _xml_text(element: ET.Element, path: str) -> str:
    child = element.find(path, ATOM)
    return str(child.text or "").strip() if child is not None else ""


def _clean_space(value: str) -> str:
    return " ".join(value.split())


def _strip_markup(value: str) -> str:
    return _clean_space(re.sub(r"<[^>]+>", " ", value))


def _crossref_pdf_url(item: dict) -> str | None:
    for link in item.get("link") or []:
        if str(link.get("content-type") or "").lower() == "application/pdf":
            url = str(link.get("URL") or "")
            if url.startswith("https://"):
                return url
    return None


def normalize_doi(value: str | None) -> str | None:
    doi = str(value or "").strip().lower()
    doi = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", doi, flags=re.I)
    return doi.rstrip(" .") or None


def normalize_arxiv_id(value: str | None) -> tuple[str | None, str | None]:
    candidate = str(value or "").strip()
    candidate = re.sub(r"^arxiv:\s*", "", candidate, flags=re.I)
    candidate = re.sub(r"^https?://(?:export\.)?arxiv\.org/(?:abs|pdf|e-print)/", "", candidate, flags=re.I)
    candidate = re.sub(r"\.pdf$", "", candidate, flags=re.I)
    match = re.fullmatch(r"((?:\d{4}\.\d{4,5})|(?:[a-z-]+(?:\.[A-Z]{2})?/\d{7}))(v\d+)?", candidate, re.I)
    if not match:
        return None, None
    return match.group(1), match.group(2)


def title_fingerprint(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24] if normalized else ""


def ensure_paper_identity(
    paper: Paper,
    *,
    arxiv_id: str | None = None,
    openalex_id: str | None = None,
    openreview_id: str | None = None,
    dblp_key: str | None = None,
    acl_id: str | None = None,
    cvf_id: str | None = None,
) -> Paper:
    current = paper.identity if isinstance(paper.identity, PaperIdentity) else PaperIdentity()
    inferred_arxiv = arxiv_id
    if inferred_arxiv is None and paper.source == "arxiv":
        inferred_arxiv, _ = normalize_arxiv_id(paper.external_id)
    identity = PaperIdentity(
        doi=normalize_doi(paper.doi) or current.doi,
        arxiv_id=inferred_arxiv or current.arxiv_id,
        openalex_id=openalex_id or current.openalex_id or (paper.external_id if paper.source == "openalex" else None),
        openreview_id=openreview_id or current.openreview_id or (paper.external_id if paper.source == "openreview" else None),
        dblp_key=dblp_key or current.dblp_key or (paper.external_id if paper.source == "dblp" else None),
        acl_id=acl_id or current.acl_id or (paper.external_id if paper.source == "acl" else None),
        cvf_id=cvf_id or current.cvf_id or (paper.external_id if paper.source == "cvf" else None),
        title_fingerprint=current.title_fingerprint or title_fingerprint(paper.title),
    )
    record = PaperSourceRecord(
        source=paper.source,
        external_id=paper.external_id,
        source_url=paper.source_url,
        pdf_url=paper.pdf_url,
        version=paper.source_version,
    )
    records = list(paper.source_records)
    if not any((item.source, item.external_id) == (record.source, record.external_id) for item in records):
        records.append(record)
    return replace(paper, doi=identity.doi, identity=identity, source_records=tuple(records))


def papers_match(left: Paper, right: Paper) -> bool:
    left = ensure_paper_identity(left)
    right = ensure_paper_identity(right)
    for field in ("doi", "arxiv_id", "openalex_id", "openreview_id", "dblp_key", "acl_id", "cvf_id"):
        first = getattr(left.identity, field)
        second = getattr(right.identity, field)
        if first and second and first.casefold() == second.casefold():
            return True
    return bool(
        left.identity.title_fingerprint
        and left.identity.title_fingerprint == right.identity.title_fingerprint
    )


def merge_papers(left: Paper, right: Paper) -> Paper:
    left = ensure_paper_identity(left)
    right = ensure_paper_identity(right)
    if not papers_match(left, right):
        raise ValueError("cannot merge papers with different identities")
    identity = PaperIdentity(**{
        field: getattr(left.identity, field) or getattr(right.identity, field)
        for field in PaperIdentity.__dataclass_fields__
    })
    records: list[PaperSourceRecord] = []
    for record in (*left.source_records, *right.source_records):
        if not any((item.source, item.external_id) == (record.source, record.external_id) for item in records):
            records.append(record)
    richer = right if len(right.abstract or "") > len(left.abstract or "") else left
    pdf_owner = max(
        (left, right),
        key=lambda paper: (_pdf_url_reliability(paper.pdf_url), bool(paper.pdf_url)),
    )
    return replace(
        left,
        title=richer.title or left.title,
        authors=richer.authors or left.authors,
        year=left.year or right.year,
        doi=identity.doi,
        source_url=left.source_url or right.source_url,
        abstract=richer.abstract,
        pdf_url=pdf_owner.pdf_url,
        identity=identity,
        source_records=tuple(records),
        source_archive_url=left.source_archive_url or right.source_archive_url,
        source_archive_path=left.source_archive_path or right.source_archive_path,
    )


def _pdf_url_reliability(url: str | None) -> int:
    hostname = (urllib.parse.urlparse(url).hostname or "").casefold() if url else ""
    if hostname in {"arxiv.org", "export.arxiv.org"}:
        return 5
    if hostname in {
        "openreview.net", "aclanthology.org", "openaccess.thecvf.com",
    }:
        return 4
    if url and str(url).casefold().endswith(".pdf"):
        return 2
    return int(bool(url))


def _openalex_abstract(value: object) -> str:
    if not isinstance(value, dict):
        return ""
    positions: list[tuple[int, str]] = []
    for word, indexes in value.items():
        if isinstance(indexes, list):
            positions.extend((index, str(word)) for index in indexes if isinstance(index, int))
    return " ".join(word for _, word in sorted(positions))


def _content_value(content: dict, key: str, default=None):
    value = content.get(key, default)
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def _openreview_author(value: object) -> str:
    if isinstance(value, dict):
        for key in ("fullname", "name", "value"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        return ""
    return str(value).strip()


def _openreview_paper(note: object) -> Paper | None:
    if not isinstance(note, dict) or note.get("ddate"):
        return None
    note_id = str(note.get("id") or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", note_id):
        return None
    if note.get("forum") != note_id or note.get("replyto"):
        return None
    content = note.get("content")
    if not isinstance(content, dict):
        return None
    title = _content_value(content, "title", "")
    authors = _content_value(content, "authors", [])
    abstract = _content_value(content, "abstract", "")
    pdf = _content_value(content, "pdf", "")
    if not isinstance(title, str) or not title.strip():
        return None
    if not authors and not abstract and not pdf:
        return None
    doi = normalize_doi(_content_value(content, "doi", ""))
    timestamp = note.get("pdate") or note.get("cdate") or note.get("tcdate")
    year = None
    if isinstance(timestamp, (int, float)):
        year = datetime.fromtimestamp(timestamp / 1000, timezone.utc).year
    paper = Paper(
        paper_id=_external_paper_id(doi, title),
        title=_clean_space(title),
        authors=tuple(
            name for name in (_openreview_author(author) for author in authors) if name
        ) if isinstance(authors, list) else (),
        year=year,
        doi=doi,
        source="openreview",
        source_url=f"https://openreview.net/forum?id={note_id}",
        quality_status="metadata_only",
        abstract=_clean_space(abstract) if isinstance(abstract, str) and abstract.strip() else None,
        pdf_url=f"https://openreview.net/pdf?id={note_id}" if isinstance(pdf, str) and pdf.strip() else None,
        external_id=note_id,
    )
    return ensure_paper_identity(paper, openreview_id=note_id)


def _dblp_paper(hit: object) -> Paper | None:
    if not isinstance(hit, dict) or not isinstance(hit.get("info"), dict):
        return None
    info = hit["info"]
    raw_title = str(info.get("title") or "")
    title = _clean_space(html.unescape(re.sub(r"<[^>]+>", " ", raw_title)))
    if not title:
        return None
    author_values = (info.get("authors") or {}).get("author", []) if isinstance(info.get("authors"), dict) else []
    if isinstance(author_values, (str, dict)):
        author_values = [author_values]
    authors = tuple(
        str(author.get("text") or author.get("name") or "").strip() if isinstance(author, dict) else str(author).strip()
        for author in author_values
        if (str(author.get("text") or author.get("name") or "").strip() if isinstance(author, dict) else str(author).strip())
    )
    electronic = info.get("ee") or []
    if isinstance(electronic, (str, dict)):
        electronic = [electronic]
    doi = None
    for value in electronic:
        candidate = str(value.get("text") or value.get("url") or "") if isinstance(value, dict) else str(value)
        match = re.search(r"(?:doi\.org/)?(10\.\d{4,9}/[^\s<>\"]+)", candidate, re.I)
        if match:
            doi = normalize_doi(match.group(1))
            break
    record_url = _https_url(info.get("url"))
    dblp_key = str(info.get("key") or "").strip()
    if not dblp_key and record_url and "/rec/" in record_url:
        dblp_key = record_url.split("/rec/", 1)[1].removesuffix(".html")
    year_text = str(info.get("year") or "")
    paper = Paper(
        paper_id=_external_paper_id(doi, title),
        title=title,
        authors=authors,
        year=int(year_text) if year_text.isdigit() else None,
        doi=doi,
        source="dblp",
        source_url=record_url,
        quality_status="metadata_only",
        external_id=dblp_key or None,
    )
    return ensure_paper_identity(paper, dblp_key=dblp_key or None)


def _https_url(value: object) -> str | None:
    candidate = str(value or "").strip()
    parsed = urllib.parse.urlparse(candidate)
    return candidate if parsed.scheme == "https" and parsed.netloc else None


def _looks_like_tex(value: bytes) -> bool:
    sample = value.decode("utf-8", errors="ignore")
    return any(marker in sample for marker in ("\\documentclass", "\\begin{document}", "\\input{", "\\def\\"))


def _search_terms(value: str) -> tuple[str, ...]:
    stop = {"the", "and", "for", "with", "from", "into", "using", "based", "this", "that"}
    return tuple(
        term for term in re.findall(r"[a-z0-9]{3,}", value.casefold())
        if term not in stop
    )


def _bibtex_entries(value: str):
    start_pattern = re.compile(r"@\w+\s*\{\s*([^,\s]+)\s*,", re.I)
    position = 0
    while True:
        match = start_pattern.search(value, position)
        if not match:
            return
        depth = 1
        index = match.end()
        quoted = False
        escaped = False
        while index < len(value) and depth:
            char = value[index]
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = not quoted
            elif not quoted and char == "{":
                depth += 1
            elif not quoted and char == "}":
                depth -= 1
            index += 1
        if depth:
            return
        yield match.group(1), value[match.end():index - 1]
        position = index


def _bibtex_field(block: str, name: str) -> str:
    match = re.search(rf"(?im)^\s*{re.escape(name)}\s*=\s*", block)
    if not match:
        return ""
    index = match.end()
    while index < len(block) and block[index].isspace():
        index += 1
    if index >= len(block):
        return ""
    opener = block[index]
    if opener not in {'{', '"'}:
        end = block.find(",", index)
        return block[index:end if end >= 0 else len(block)].strip()
    closer = "}" if opener == "{" else '"'
    depth = 1
    index += 1
    start = index
    escaped = False
    while index < len(block):
        char = block[index]
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif opener == "{" and char == "{":
            depth += 1
        elif char == closer:
            depth -= 1
            if not depth:
                return block[start:index].strip()
        index += 1
    return ""


def _clean_bibtex(value: str) -> str:
    text = re.sub(r"[{}]", "", value)
    text = re.sub(r"\\[a-zA-Z]+\s*", "", text)
    text = text.replace("~", " ")
    return _clean_space(html.unescape(text))


class _ACLSearchParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[tuple[str, str]] = []
        self._active_id: str | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        href = dict(attrs).get("href") or ""
        match = re.fullmatch(r"/(\d{4}\.[A-Za-z0-9_.-]+\.\d+)/", href)
        if match:
            self._active_id = match.group(1)
            self._text = []

    def handle_data(self, data: str) -> None:
        if self._active_id:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._active_id:
            title = _clean_space(" ".join(self._text))
            if title and not any(item[0] == self._active_id for item in self.results):
                self.results.append((self._active_id, title))
            self._active_id = None
            self._text = []


class _CVFProceedingsParser(HTMLParser):
    def __init__(self, venue: str) -> None:
        super().__init__(convert_charrefs=True)
        self.venue = venue
        self.results: list[dict[str, object]] = []
        self._in_title = False
        self._active_href: str | None = None
        self._active_text: list[str] = []
        self._current: dict[str, object] | None = None
        self._in_dd = False
        self._dd_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        classes = set((values.get("class") or "").split())
        if tag == "dt" and "ptitle" in classes:
            self._in_title = True
        elif tag == "dd" and self._current is not None:
            self._in_dd = True
            self._dd_text = []
        elif tag == "a" and self._in_title:
            self._active_href = values.get("href")
            self._active_text = []
        elif tag == "a" and self._current is not None:
            href = values.get("href") or ""
            if href.lower().endswith(".pdf") and "/papers/" in href:
                self._current["pdf_url"] = href

    def handle_data(self, data: str) -> None:
        if self._active_href is not None:
            self._active_text.append(data)
        elif self._in_dd:
            self._dd_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._active_href is not None:
            title = _clean_space(" ".join(self._active_text))
            if title:
                paper_url = self._active_href
                stem = Path(urllib.parse.urlparse(paper_url).path).stem.removesuffix("_paper")
                self._current = {
                    "cvf_id": f"{self.venue}:{stem}",
                    "title": title,
                    "authors": [],
                    "paper_url": paper_url,
                    "pdf_url": paper_url.replace("/html/", "/papers/").removesuffix("_paper.html") + "_paper.pdf",
                }
            self._active_href = None
            self._active_text = []
        elif tag == "dt":
            self._in_title = False
        elif tag == "dd" and self._current is not None:
            text = _clean_space(" ".join(self._dd_text))
            if text and not self._current["authors"] and "pdf" not in text.casefold():
                self._current["authors"] = [part.strip() for part in text.split(",") if part.strip()]
            self._in_dd = False
            self._dd_text = []
            if self._current.get("pdf_url") and self._current not in self.results:
                self.results.append(self._current)

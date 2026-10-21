from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from researchops.papers.acquisition import AcquisitionDAG
from researchops.knowledge.compression import PaperCompressor
from researchops.papers.ingestion import DocumentIngestor
from researchops.papers.connectors import (
    LiteratureSearch,
    download_arxiv_source,
    download_pdf,
    ensure_paper_identity,
    merge_papers,
    papers_match,
)
from researchops.core.models import BOUNDARIES, Domain, Idea, Paper, PaperIdentity, PaperSourceRecord, TaskStatus
from researchops.papers.analysis import PaperAnalyzer, ResearchExecutionPlanner
from researchops.harness.runtime import ModelPort, TraceStore
from researchops.core.storage import ResearchStore


SUPPORTED_SOURCES = ("arxiv", "crossref", "openalex", "openreview", "dblp", "acl", "cvf")


@dataclass(frozen=True)
class SearchQuery:
    source: str
    query: str
    rationale: str


@dataclass(frozen=True)
class ResearchPlan:
    idea: Idea
    queries: tuple[SearchQuery, ...]
    ambiguities: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "idea": {
                **asdict(self.idea),
                "domain": self.idea.domain.value,
                "atomic_claims": list(self.idea.atomic_claims),
                "success_criteria": list(self.idea.success_criteria),
            },
            "search_queries": [asdict(query) for query in self.queries],
            "ambiguities": list(self.ambiguities),
        }


class IdeaResearchPlanner:
    """Use the configured LLM to structure an Idea and plan bounded searches.

    The source-specific portfolio and bounded-query concepts are adapted from
    PaperSeek's Apache-2.0 QueryPlanner and QueryLoopPipeline.
    """

    def __init__(self, model: ModelPort) -> None:
        self.model = model

    def plan(self, idea_text: str, *, max_queries: int = 7) -> ResearchPlan:
        value = idea_text.strip()
        if not value:
            raise ValueError("idea_text must not be empty")
        prompt = {
            "task": "Structure the research idea and create source-specific literature searches.",
            "idea_text": value,
            "allowed_domains": [domain.value for domain in Domain],
            "sources": list(SUPPORTED_SOURCES),
            "requirements": [
                "Preserve the user's intended contribution; do not invent claimed results.",
                "Choose exactly one allowed domain. If outside scope, set domain to out_of_scope.",
                "Write atomic, falsifiable contribution claims and measurable success criteria.",
                "Queries must be concise English scholarly search expressions.",
                "Use all complementary sources when the query budget permits: arXiv for preprints/source, OpenAlex for OA discovery, OpenReview for public ML submissions, DBLP for CS bibliography, Crossref for DOI metadata, ACL for NLP, and CVF for computer vision.",
                f"Return at most {max_queries} search queries.",
            ],
            "output_schema": {
                "title": "string",
                "domain": "time_series|ai_ml|knowledge_graph|out_of_scope",
                "description": "string",
                "research_question": "string",
                "atomic_claims": ["string"],
                "success_criteria": ["string"],
                "ambiguities": ["string"],
                "search_queries": [
                    {"source": "arxiv|openalex|openreview|dblp|crossref|acl|cvf", "query": "string", "rationale": "string"}
                ],
            },
        }
        turn = self.model.complete(
            [
                {
                    "role": "system",
                    "content": (
                        "You are the intake and literature-search planner for an auditable research agent. "
                        "Return one JSON object only, without markdown."
                    ),
                },
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            [],
            3000,
        )
        payload = _parse_json_object(turn.text)
        domain_value = str(payload.get("domain") or "").strip()
        if domain_value not in {domain.value for domain in Domain}:
            raise ValueError(
                "The Idea is outside the current domains or its domain could not be determined: "
                f"{domain_value or 'missing'}"
            )
        idea = Idea.from_dict(payload)
        queries = _validated_queries(payload.get("search_queries"), idea, max_queries)
        return ResearchPlan(
            idea=idea,
            queries=queries,
            ambiguities=tuple(_strings(payload.get("ambiguities"))),
        )

    def select_papers(
        self,
        plan: ResearchPlan,
        papers: list[Paper],
        *,
        max_papers: int,
    ) -> list[tuple[Paper, str]]:
        if max_papers <= 0 or not papers:
            return []
        candidates = [
            {
                "paper_id": paper.paper_id,
                "title": paper.title,
                "year": paper.year,
                "source": paper.source,
                "doi": paper.doi,
                "abstract": (paper.abstract or "")[:1600],
                "full_text_available": bool(paper.pdf_url),
            }
            for paper in papers[:60]
        ]
        request = {
            "research_question": plan.idea.research_question,
            "atomic_claims": list(plan.idea.atomic_claims),
            "max_selected": max_papers,
            "selection_priority": [
                "Direct relevance to one or more claims",
                "Method or baseline comparability",
                "Availability of full text",
                "Coverage diversity rather than title duplication",
                "Method, dataset, evaluation, or implementation ideas that could improve the research design",
            ],
            "selection_rules": [
                "Use only the supplied title and abstract.",
                "Do not claim architecture, decomposition, results, or implementation details absent from the abstract.",
                "When only a title is available, state that the reason is title-level and provisional.",
                "Retain direct evidence, baseline candidates, and genuinely useful inspiration; exclude merely keyword-adjacent papers.",
            ],
            "candidates": candidates,
            "output_schema": {"selected": [{
                "paper_id": "string",
                "role": "direct|baseline|method_inspiration|evaluation_inspiration",
                "reason": "string",
            }]},
        }
        try:
            turn = self.model.complete(
                [
                    {
                        "role": "system",
                        "content": (
                            "Select literature for evidence review. Never invent paper contents from a title. "
                            "Return one JSON object only."
                        ),
                    },
                    {"role": "user", "content": json.dumps(request, ensure_ascii=False)},
                ],
                [],
                2200,
            )
            selected = _parse_json_object(turn.text).get("selected") or []
        except Exception:
            selected = []
        by_id = {paper.paper_id: paper for paper in papers}
        output: list[tuple[Paper, str]] = []
        seen: set[str] = set()
        for item in selected:
            if not isinstance(item, dict):
                continue
            paper_id = str(item.get("paper_id") or "")
            if paper_id not in by_id or paper_id in seen:
                continue
            seen.add(paper_id)
            role = str(item.get("role") or "direct")
            reason = str(item.get("reason") or "LLM relevance selection")
            output.append((by_id[paper_id], f"{role}: {reason}"))
            if len(output) >= max_papers:
                break
        if output:
            return _prioritize_open_full_text(output, papers, plan, max_papers)
        ranked = sorted(
            papers,
            key=lambda paper: (
                -_term_overlap(plan.idea.research_question, f"{paper.title} {paper.abstract or ''}"),
                not bool(paper.pdf_url),
                -(paper.year or 0),
                paper.title,
            ),
        )
        fallback = [(paper, "Deterministic relevance fallback") for paper in ranked[:max_papers]]
        return _prioritize_open_full_text(fallback, papers, plan, max_papers)


class LiteratureAcquisitionPipeline:
    def __init__(
        self,
        *,
        model: ModelPort,
        store: ResearchStore,
        literature: LiteratureSearch,
        ingestor: DocumentIngestor,
        workspace: Path,
        traces: TraceStore,
        downloader: Callable[..., dict] = download_pdf,
        source_downloader: Callable[..., dict] = download_arxiv_source,
    ) -> None:
        self.planner = IdeaResearchPlanner(model)
        self.store = store
        self.literature = literature
        self.ingestor = ingestor
        self.analyzer = PaperAnalyzer(model, store, workspace)
        self.compressor = PaperCompressor(store, model)
        self.execution_planner = ResearchExecutionPlanner(model, workspace)
        self.workspace = workspace
        self.traces = traces
        self.downloader = downloader
        self.source_downloader = source_downloader
        self._last_dag_summary: dict[str, Any] = {}

    def run(
        self,
        idea_text: str,
        *,
        max_queries: int = 7,
        limit_per_query: int = 8,
        max_downloads: int = 12,
        run_id: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        max_queries = _bounded(max_queries, 1, 12, "max_queries")
        limit_per_query = _bounded(limit_per_query, 1, 25, "limit_per_query")
        max_downloads = _bounded(max_downloads, 0, 50, "max_downloads")
        plan = self.planner.plan(idea_text, max_queries=max_queries)
        plan_data = plan.to_dict()
        task_id = _stable_id("task", plan.idea.title, plan.idea.description)
        run_id = run_id or f"run_{uuid.uuid4().hex}"
        session_id = session_id or f"session_{uuid.uuid4().hex}"
        self.store.create_task(task_id, plan.idea.title, plan.idea.domain.value, plan_data["idea"])
        self.store.start_run(run_id, task_id, session_id)
        self.store.update_task_status(task_id, TaskStatus.PROBLEM_REFINED)
        self.traces.emit(run_id, "research.idea.structured", session_id, {
            "task_id": task_id,
            "domain": plan.idea.domain.value,
            "query_count": len(plan.queries),
        })
        self.store.update_task_status(task_id, TaskStatus.SEARCH_PLANNED)
        try:
            candidates, search_runs = self._search(plan, run_id, session_id, limit_per_query)
            statuses = [status for search in search_runs for status in search["statuses"]]
            if statuses and not any(status["status"] == "success" for status in statuses):
                raise RuntimeError("All configured literature sources failed; acquisition cannot continue")
            self.store.update_task_status(task_id, TaskStatus.SEARCH_COMPLETED)
            selected = self.planner.select_papers(plan, candidates, max_papers=max_downloads)
            selection_reasons = {paper.paper_id: reason for paper, reason in selected}
            selected_ids = set(selection_reasons)
            for paper in candidates:
                self.store.link_task_paper(
                    task_id,
                    paper.paper_id,
                    role="selected" if paper.paper_id in selected_ids else "candidate",
                    acquired_status="pending" if paper.paper_id in selected_ids else "metadata_only",
                    selection_reason=selection_reasons.get(paper.paper_id, ""),
                )
            if selected:
                self.store.update_task_status(task_id, TaskStatus.PAPERS_SELECTED)
            acquisitions = self._acquire(task_id, plan.idea, selected, run_id, session_id)
            ingested_count = sum(item["status"] == "ingested" for item in acquisitions)
            if selected and ingested_count == len(selected):
                task_status = TaskStatus.FULL_TEXT_READY
            elif ingested_count:
                task_status = TaskStatus.FULL_TEXT_PARTIAL
            elif selected:
                task_status = TaskStatus.PAPERS_SELECTED
            else:
                task_status = TaskStatus.SEARCH_COMPLETED
            self.store.update_task_status(task_id, task_status)
            analysis_records = [
                {
                    "status": item.get("analysis_status"),
                    "path": item.get("analysis_path"),
                }
                for item in acquisitions
            ]
            try:
                execution_plan = self.execution_planner.build(task_id, plan.idea, analysis_records)
            except Exception as planning_error:
                execution_plan = self.execution_planner.failed(task_id, plan.idea, planning_error)
            report = {
                "task_id": task_id,
                "run_id": run_id,
                "task_status": task_status.value,
                "input_idea": idea_text,
                "structured_plan": plan_data,
                "boundaries": list(BOUNDARIES),
                "searches": search_runs,
                "candidate_count": len(candidates),
                "selected_papers": [
                    {
                        "paper_id": paper.paper_id,
                        "title": paper.title,
                        "source": paper.source,
                        "year": paper.year,
                        "doi": paper.doi,
                        "pdf_url": paper.pdf_url,
                        "identity": asdict(paper.identity) if paper.identity else None,
                        "sources": [asdict(record) for record in paper.source_records],
                        "reason": reason,
                        "selection_evidence_level": "abstract" if paper.abstract else "title_only",
                    }
                    for paper, reason in selected
                ],
                "acquisitions": acquisitions,
                "acquisition_dag": self._last_dag_summary,
                "execution_plan": execution_plan,
                "pipeline_counts": {
                    "candidates_found": len(candidates),
                    "papers_selected": len(selected),
                    "pdfs_downloaded": sum(bool(item.get("pdf_path")) for item in acquisitions),
                    "papers_parsed": ingested_count,
                    "papers_analyzed": sum(
                        item.get("analysis_status") in {"completed", "completed_with_gaps"}
                        for item in acquisitions
                    ),
                    "plans_generated": int(execution_plan["status"] in {
                        "completed", "completed_with_gaps", "completed_with_fallback",
                    }),
                    "metadata_only": sum(item["status"] == "metadata_only" for item in acquisitions),
                },
                "storage": {
                    "database": str(self.store.path),
                    "pdf_directory": str(self.workspace / "artifacts" / "papers"),
                    "parsed_directory": str(self.workspace / "artifacts" / "parsed"),
                    "source_directory": str(self.workspace / "artifacts" / "sources"),
                    "analysis_directory": str(self.workspace / "artifacts" / "analyses" / task_id),
                    "plan_path": execution_plan["path"],
                },
                "semantic_scholar_used": False,
                "limitations": [
                    "Crossref records are metadata unless a valid open PDF was acquired.",
                    "Downloaded PDF equations and tables remain review_required until checked against the original pages.",
                    "Search coverage is bounded by the recorded queries and successful sources.",
                ],
            }
            artifact = self._write_artifact(task_id, run_id, report)
            report["artifact"] = artifact
            self.store.finish_run(run_id, "completed", {
                "candidate_count": len(candidates),
                "selected_count": len(selected),
                "acquired_count": ingested_count,
                "plan_status": execution_plan["status"],
                "artifact": artifact,
            })
            self.traces.emit(run_id, "research.acquisition.completed", session_id, {
                "task_id": task_id,
                "candidate_count": len(candidates),
                "acquired_count": ingested_count,
                "plan_status": execution_plan["status"],
            })
            return report
        except Exception as error:
            self.store.finish_run(run_id, "failed", {"error_type": type(error).__name__, "error": str(error)})
            self.traces.emit(run_id, "research.acquisition.failed", session_id, {
                "task_id": task_id,
                "error_type": type(error).__name__,
                "error": str(error),
            })
            raise

    def _search(
        self,
        plan: ResearchPlan,
        run_id: str,
        session_id: str,
        limit_per_query: int,
    ) -> tuple[list[Paper], list[dict[str, Any]]]:
        papers: dict[str, Paper] = {}
        searches: list[dict[str, Any]] = []
        for query in plan.queries:
            result = self.literature.search(
                query.query,
                run_id,
                limit_per_query,
                sources=[query.source],
            )
            searches.append({
                "source": query.source,
                "query": query.query,
                "rationale": query.rationale,
                "statuses": result["sources"],
            })
            if query.source == "arxiv" and not result["papers"]:
                refined = _broaden_arxiv_query(query.query)
                if refined and refined.casefold() != query.query.casefold():
                    retry = self.literature.search(
                        refined,
                        run_id,
                        limit_per_query,
                        sources=["arxiv"],
                    )
                    searches.append({
                        "source": "arxiv",
                        "query": refined,
                        "rationale": "Automatic broadening after a successful zero-result arXiv query",
                        "refines_query": query.query,
                        "statuses": retry["sources"],
                    })
                    result = {
                        "papers": [*result["papers"], *retry["papers"]],
                        "sources": [*result["sources"], *retry["sources"]],
                    }
            for value in result["papers"]:
                paper = ensure_paper_identity(_paper_from_dict(value))
                matching = next((key for key, current in papers.items() if papers_match(current, paper)), None)
                if matching is None:
                    papers[paper.paper_id] = paper
                else:
                    papers[matching] = merge_papers(papers[matching], paper)
            self.traces.emit(run_id, "research.search.completed", session_id, {
                "source": query.source,
                "query": query.query,
                "result_count": len(result["papers"]),
                "statuses": result["sources"],
            })
        return list(papers.values()), searches

    def _acquire(
        self,
        task_id: str,
        idea: Idea,
        selected: list[tuple[Paper, str]],
        run_id: str,
        session_id: str,
    ) -> list[dict[str, Any]]:
        dag = AcquisitionDAG(
            store=self.store,
            ingestor=self.ingestor,
            analyzer=self.analyzer,
            compressor=self.compressor,
            workspace=self.workspace,
            traces=self.traces,
            downloader=self.downloader,
            source_downloader=self.source_downloader,
        )
        acquisitions, summary = dag.run(
            task_id, idea, selected, run_id, session_id
        )
        self._last_dag_summary = summary
        return acquisitions

    def _write_artifact(self, task_id: str, run_id: str, report: dict[str, Any]) -> dict[str, str]:
        data = json.dumps(report, ensure_ascii=False, indent=2, default=str).encode("utf-8")
        digest = hashlib.sha256(data).hexdigest()
        artifact_id = f"artifact_{digest[:20]}"
        target = self.workspace / "artifacts" / f"{artifact_id}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        self.store.save_artifact(
            artifact_id,
            task_id,
            run_id,
            "literature_acquisition",
            str(target),
            digest,
            {"format": "json", "schema_version": 2},
        )
        return {"artifact_id": artifact_id, "path": str(target), "sha256": digest}


def _validated_queries(raw: Any, idea: Idea, maximum: int) -> tuple[SearchQuery, ...]:
    output: list[SearchQuery] = []
    seen: set[tuple[str, str]] = set()
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source") or "").strip().lower()
        query = " ".join(str(item.get("query") or "").split())
        if source not in SUPPORTED_SOURCES or not query or len(query) > 500:
            continue
        signature = (source, query.casefold())
        if signature in seen:
            continue
        seen.add(signature)
        output.append(SearchQuery(source, query, str(item.get("rationale") or "")))
        if len(output) >= maximum:
            break
    if not output:
        fallback = " ".join(idea.research_question.split())[:500]
        output = [
            SearchQuery(source, fallback, "Fallback from the structured research question")
            for source in SUPPORTED_SOURCES[:maximum]
        ]
    covered = {item.source for item in output}
    base_query = output[0].query if output else " ".join(idea.research_question.split())[:500]
    for source in SUPPORTED_SOURCES:
        if source in covered or len(output) >= maximum:
            continue
        output.append(SearchQuery(
            source,
            base_query,
            "Added to ensure complementary independent-source coverage",
        ))
        covered.add(source)
    return tuple(output)


def _parse_json_object(text: str) -> dict[str, Any]:
    value = str(text or "").strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.I)
        value = re.sub(r"\s*```$", "", value)
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        start, end = value.find("{"), value.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("model did not return a JSON object")
        payload = json.loads(value[start:end + 1])
    if not isinstance(payload, dict):
        raise ValueError("model output must be a JSON object")
    return payload


def _strings(value: Any) -> list[str]:
    return [str(item).strip() for item in value if str(item).strip()] if isinstance(value, list) else []


def _paper_from_dict(value: dict[str, Any]) -> Paper:
    identity = value.get("identity")
    records = value.get("source_records") or ()
    return Paper(**{
        **value,
        "authors": tuple(value.get("authors") or ()),
        "identity": PaperIdentity(**identity) if isinstance(identity, dict) else identity,
        "source_records": tuple(
            PaperSourceRecord(**record) if isinstance(record, dict) else record
            for record in records
        ),
    })


def _normalize_title(value: str) -> str:
    return re.sub(r"[^a-z0-9\u3400-\u9fff]+", " ", value.casefold()).strip()


def _prioritize_open_full_text(
    selected: list[tuple[Paper, str]],
    candidates: list[Paper],
    plan: ResearchPlan,
    maximum: int,
) -> list[tuple[Paper, str]]:
    if maximum <= 0:
        return []
    selected_by_id = {paper.paper_id: (paper, reason) for paper, reason in selected}
    downloadable = sorted(
        (paper for paper in candidates if paper.pdf_url),
        key=lambda paper: (
            -_term_overlap(plan.idea.research_question, f"{paper.title} {paper.abstract or ''}"),
            -(paper.year or 0),
            paper.title,
        ),
    )
    output: list[tuple[Paper, str]] = []
    seen: set[str] = set()
    for paper in downloadable:
        if paper.paper_id in selected_by_id:
            item = selected_by_id[paper.paper_id]
        else:
            item = (
                paper,
                "Open full text is available; selected for evidence-backed method and baseline analysis.",
            )
        output.append(item)
        seen.add(paper.paper_id)
        if len(output) >= maximum:
            return output
    for paper, reason in selected:
        if paper.paper_id in seen:
            continue
        output.append((paper, reason))
        seen.add(paper.paper_id)
        if len(output) >= maximum:
            break
    return output


def _broaden_arxiv_query(value: str) -> str:
    normalized = " ".join(value.replace('"', " ").split()).casefold()
    if "time series" in normalized:
        focus = next(
            (
                term
                for term in ("anomaly", "compression", "forecasting", "classification", "representation")
                if term in normalized
            ),
            None,
        )
        if focus:
            method = next(
                (
                    term
                    for term in ("transformer", "attention", "graph", "diffusion", "autoencoder")
                    if term in normalized
                ),
                None,
            )
            suffix = f" AND all:{method}" if method else ""
            return f'all:"time series" AND all:{focus}{suffix}'
    stop = {
        "using", "based", "with", "from", "into", "toward", "towards", "for", "and", "the",
        "method", "approach", "model", "research", "study", "enhanced", "improved",
    }
    terms = [
        term
        for term in re.findall(r"[a-z][a-z0-9-]{2,}", normalized)
        if term not in stop
    ]
    unique = list(dict.fromkeys(terms))[:4]
    return " AND ".join(f"all:{term}" for term in unique)


def _term_overlap(left: str, right: str) -> float:
    terms = set(re.findall(r"[a-zA-Z][a-zA-Z0-9-]{2,}|[\u3400-\u9fff]{2,}", left.casefold()))
    if not terms:
        return 0.0
    other = set(re.findall(r"[a-zA-Z][a-zA-Z0-9-]{2,}|[\u3400-\u9fff]{2,}", right.casefold()))
    return len(terms & other) / len(terms)


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:20]
    return f"{prefix}_{digest}"


def _bounded(value: int, minimum: int, maximum: int, name: str) -> int:
    number = int(value)
    if number < minimum or number > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return number

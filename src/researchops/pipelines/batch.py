from __future__ import annotations

import hashlib
import json
import math
import os
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

from researchops.papers.acquisition import AcquisitionDAG
from researchops.knowledge.compression import PaperCompressor
from researchops.papers.ingestion import DocumentIngestor
from researchops.papers.connectors import (
    LiteratureSearch,
    _pdf_url_reliability,
    download_arxiv_source,
    download_pdf,
    ensure_paper_identity,
    merge_papers,
    papers_match,
)
from researchops.core.models import Idea, Paper, PaperIdentity, PaperSourceRecord
from researchops.papers.analysis import PaperAnalyzer
from researchops.knowledge.retrieval import HybridRetriever, tokenize
from researchops.harness.runtime import ModelPort, TraceStore
from researchops.core.storage import ResearchStore


BATCH_SCHEMA_VERSION = "batch-research-v1"
SYNTHESIS_VERSION = "multi-paper-map-reduce-rag-v2"


class BatchResearchPipeline:
    """Thirty-paper screening, shortlist deep reading, map-reduce synthesis, and evaluation."""

    def __init__(
        self,
        *,
        model: ModelPort,
        store: ResearchStore,
        literature: LiteratureSearch,
        ingestor: DocumentIngestor,
        workspace: Path,
        traces: TraceStore,
    ) -> None:
        self.model = model
        self.store = store
        self.literature = literature
        self.ingestor = ingestor
        self.workspace = workspace
        self.traces = traces
        self.retriever = HybridRetriever(store)
        self.compressor = PaperCompressor(store, model, retriever=self.retriever)
        self.analyzer = PaperAnalyzer(model, store, workspace)

    def run(
        self,
        idea_entry: dict[str, Any],
        *,
        target_papers: int = 30,
        shortlist_size: int = 6,
        limit_per_query: int = 12,
        evaluation_size: int = 40,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        if not 30 <= target_papers <= 50:
            raise ValueError("target_papers must be between 30 and 50")
        if not 3 <= shortlist_size <= min(12, target_papers):
            raise ValueError("shortlist_size must be between 3 and 12")
        idea = _idea_from_entry(idea_entry)
        idea_id = str(idea_entry["idea_id"])
        task_id = _stable_id("batch_task", idea_id, idea.research_question)
        run_id = run_id or f"run_batch_{uuid.uuid4().hex}"
        session_id = f"session_batch_{uuid.uuid4().hex}"
        self.store.create_task(task_id, idea.title, idea.domain.value, {
            **asdict(idea), "domain": idea.domain.value, "idea_id": idea_id,
        })
        self.store.start_run(run_id, task_id, session_id)
        try:
            candidates, searches = self._search(idea_entry, run_id, limit_per_query)
            selected = _rank_papers(idea, candidates, target_papers)
            if len(selected) < target_papers:
                raise RuntimeError(
                    f"Only {len(selected)} unique papers were found; {target_papers} are required"
                )
            for paper in selected:
                self.store.upsert_paper(paper)
                self.store.link_task_paper(
                    task_id, paper.paper_id, role="screened", acquired_status="pending"
                )
            l1_records = [self.compressor.l1(paper, idea) for paper in selected]
            selected_pairs = [(paper, "L1 screened corpus.") for paper in selected]
            dag = AcquisitionDAG(
                store=self.store,
                ingestor=self.ingestor,
                analyzer=self.analyzer,
                compressor=self.compressor,
                workspace=self.workspace,
                traces=self.traces,
                downloader=download_pdf,
                source_downloader=download_arxiv_source,
            )
            shallow_acquisitions, shallow_summary = dag.run(
                task_id,
                idea,
                selected_pairs,
                run_id,
                session_id,
                deep_paper_ids=set(),
            )
            full_text_ids = {
                item["paper_id"] for item in shallow_acquisitions
                if item["status"] == "ingested"
            }
            if len(full_text_ids) < shortlist_size:
                raise RuntimeError(
                    f"Only {len(full_text_ids)} of 30 L1 papers yielded parsed full text; "
                    f"{shortlist_size} are required for the L2 shortlist"
                )
            fan_in = self._fan_in(
                idea, selected, l1_records, shortlist_size, eligible_ids=full_text_ids
            )
            shortlist_ids = {item["paper_id"] for item in fan_in["shortlist"]}
            selected_pairs = [
                (paper, _shortlist_reason(paper.paper_id, fan_in)) for paper in selected
            ]
            acquisitions, dag_summary = dag.run(
                task_id,
                idea,
                selected_pairs,
                run_id,
                session_id,
                deep_paper_ids=shortlist_ids,
            )
            acquired_by_id = {item["paper_id"]: item for item in acquisitions}
            shortlist_items = {
                item["paper_id"]: item for item in fan_in["shortlist"]
            }
            successful_items = [
                shortlist_items[paper.paper_id] for paper in selected
                if paper.paper_id in shortlist_ids
                and acquired_by_id.get(paper.paper_id, {}).get("analysis_status")
                in {"completed", "completed_with_gaps"}
            ]
            backfill_runs = []
            attempted_ids = set(shortlist_ids)
            for paper in selected:
                if len(successful_items) >= shortlist_size:
                    break
                if paper.paper_id in attempted_ids or paper.paper_id not in full_text_ids:
                    continue
                attempted_ids.add(paper.paper_id)
                reason = "Automatic L2 backfill after a shortlisted full-text analysis failed."
                replacement, replacement_summary = dag.run(
                    task_id,
                    idea,
                    [(paper, reason)],
                    run_id,
                    session_id,
                    deep_paper_ids={paper.paper_id},
                )
                backfill_runs.append(replacement_summary)
                acquired_by_id[paper.paper_id] = replacement[0]
                if replacement[0].get("analysis_status") in {
                    "completed", "completed_with_gaps",
                }:
                    successful_items.append({
                        "paper_id": paper.paper_id,
                        "title": paper.title,
                        "role": "method_inspiration",
                        "reason": reason,
                    })
            fan_in["shortlist"] = successful_items[:shortlist_size]
            fan_in["shortlist_count"] = len(fan_in["shortlist"])
            shortlist_ids = {item["paper_id"] for item in fan_in["shortlist"]}
            acquisitions = [acquired_by_id[paper.paper_id] for paper in selected]
            dag_summary["backfill_runs"] = backfill_runs
            dag_summary["deep_paper_attempt_count"] = len(attempted_ids)
            usable = [paper for paper in selected if paper.paper_id in shortlist_ids]
            retrieval_checks = self._index_and_retrieve(idea_entry)
            synthesis = self._map_reduce(idea, idea_entry, usable)
            evaluation = self._build_and_evaluate_benchmark(
                idea_entry,
                candidates[: max(evaluation_size, target_papers)],
                selected,
                fan_in,
                evaluation_size,
            )
            report = {
                "schema_version": BATCH_SCHEMA_VERSION,
                "task_id": task_id,
                "run_id": run_id,
                "idea_id": idea_id,
                "idea": {**asdict(idea), "domain": idea.domain.value},
                "searches": searches,
                "candidate_count": len(candidates),
                "l1_fan_in_count": len(l1_records),
                "l1_token_estimate": sum(record["token_estimate"] for record in l1_records),
                "fan_in": fan_in,
                "shortlist_requested": shortlist_size,
                "shortlist_usable": len(usable),
                "acquisition_dag": dag_summary,
                "shallow_acquisition_dag": shallow_summary,
                "pipeline_counts": {
                    "papers_screened_l1": len(selected),
                    "pdfs_downloaded": sum(bool(item.get("pdf_path")) for item in acquisitions),
                    "papers_parsed": sum(item["status"] == "ingested" for item in acquisitions),
                    "papers_analyzed_l2": sum(
                        item["analysis_status"] in {"completed", "completed_with_gaps"}
                        for item in acquisitions
                    ),
                    "l3_maps": synthesis["map_count"],
                    "l3_reductions": len(synthesis["subproblem_syntheses"]),
                },
                "retrieval_checks": retrieval_checks,
                "synthesis": synthesis,
                "evaluation": evaluation,
                "acquisitions": acquisitions,
            }
            artifact = self._write_artifact(task_id, run_id, "batch_research", report)
            report["artifact"] = artifact
            self.store.finish_run(run_id, "completed", {
                "artifact": artifact,
                "pipeline_counts": report["pipeline_counts"],
                "gold_ready": evaluation["gold_ready"],
            })
            return report
        except Exception as error:
            self.store.finish_run(run_id, "failed", {
                "error_type": type(error).__name__, "error": str(error),
            })
            raise

    def _search(
        self,
        idea_entry: dict[str, Any],
        run_id: str,
        limit_per_query: int,
    ) -> tuple[list[Paper], list[dict[str, Any]]]:
        papers: dict[str, Paper] = {}
        searches: list[dict[str, Any]] = []
        for item in idea_entry.get("search_queries") or []:
            source = str(item["source"])
            query = str(item["query"])
            result = self.literature.search(
                query, run_id, limit_per_query, sources=[source]
            )
            searches.append({
                "source": source, "query": query, "statuses": result["sources"],
                "result_count": len(result["papers"]),
            })
            for value in result["papers"]:
                paper = ensure_paper_identity(_paper_from_dict(value))
                match = next(
                    (key for key, current in papers.items() if papers_match(current, paper)),
                    None,
                )
                if match is None:
                    papers[paper.paper_id] = paper
                else:
                    papers[match] = merge_papers(papers[match], paper)
        ranked = sorted(
            papers.values(),
            key=lambda paper: (not bool(paper.pdf_url), -(paper.year or 0), paper.title),
        )
        return ranked, searches

    def _fan_in(
        self,
        idea: Idea,
        papers: list[Paper],
        l1_records: list[dict[str, Any]],
        shortlist_size: int,
        eligible_ids: set[str] | None = None,
    ) -> dict[str, Any]:
        by_id = {paper.paper_id: paper for paper in papers}
        eligible = set(by_id) if eligible_ids is None else set(eligible_ids)
        fan_in_records = []
        for record in l1_records:
            content = dict(record["content"])
            content["full_text_available"] = record["paper_id"] in eligible
            fan_in_records.append(content)
        request = {
            "task": "Select a diverse full-text shortlist from exactly thirty L1 screening records.",
            "research_question": idea.research_question,
            "claims": list(idea.atomic_claims),
            "shortlist_size": shortlist_size,
            "rules": [
                "Use only supplied L1 metadata and summaries.",
                "Prefer papers with open full text and direct baseline or method relevance.",
                "Select only records where full_text_available is true.",
                "Cover methods, baselines, evaluation protocols, and efficiency trade-offs.",
                "Do not infer results absent from an abstract.",
            ],
            "papers": fan_in_records,
            "output_schema": {"shortlist": [{
                "paper_id": "string",
                "role": "primary_baseline|competing_method|evaluation_protocol|method_inspiration",
                "reason": "string",
            }]},
        }
        try:
            turn = self.model.complete(
                [
                    {"role": "system", "content": "Return one grounded JSON object only."},
                    {"role": "user", "content": json.dumps(request, ensure_ascii=False)},
                ],
                [],
                2600,
            )
            raw = _parse_json(turn.text).get("shortlist") or []
        except Exception:
            raw = []
        output = []
        seen: set[str] = set()
        for item in raw:
            if not isinstance(item, dict):
                continue
            paper_id = str(item.get("paper_id") or "")
            if paper_id not in by_id or paper_id not in eligible or paper_id in seen:
                continue
            output.append({
                "paper_id": paper_id,
                "title": by_id[paper_id].title,
                "role": str(item.get("role") or "method_inspiration"),
                "reason": str(item.get("reason") or "L1 fan-in selection"),
            })
            seen.add(paper_id)
            if len(output) >= shortlist_size:
                break
        if len(output) < shortlist_size:
            l1_by_id = {record["paper_id"]: record for record in l1_records}
            fallback = sorted(
                (paper for paper in papers if paper.paper_id in eligible and paper.paper_id not in seen),
                key=lambda paper: (
                    -float(l1_by_id[paper.paper_id]["content"].get("relevance_score") or 0),
                    -(paper.year or 0),
                    paper.title,
                ),
            )
            for paper in fallback:
                output.append({
                    "paper_id": paper.paper_id,
                    "title": paper.title,
                    "role": "method_inspiration",
                    "reason": "Deterministic open-full-text fallback after L1 fan-in.",
                })
                if len(output) >= shortlist_size:
                    break
        return {
            "input_count": len(l1_records),
            "shortlist": output,
            "shortlist_count": len(output),
        }

    def _index_and_retrieve(self, idea_entry: dict[str, Any]) -> list[dict[str, Any]]:
        retriever = HybridRetriever(self.store)
        checks = []
        for subproblem in (idea_entry.get("subproblems") or [])[:3]:
            hits = retriever.search(str(subproblem), limit=8)
            checks.append({
                "subproblem": subproblem,
                "hit_count": len(hits),
                "paper_ids": list(dict.fromkeys(hit.paper_id for hit in hits)),
                "top_hits": [hit.to_dict(include_text=False) for hit in hits[:3]],
            })
        return checks

    def _map_reduce(
        self,
        idea: Idea,
        idea_entry: dict[str, Any],
        papers: list[Paper],
    ) -> dict[str, Any]:
        subproblems = [str(value) for value in idea_entry.get("subproblems") or []]
        mapped: list[dict[str, Any]] = []
        reductions: list[dict[str, Any]] = []
        for subproblem in subproblems:
            current_maps = []
            workers = min(3, max(1, int(os.getenv("RESEARCHOPS_SYNTHESIS_WORKERS", "3"))))
            def map_paper(paper):
                return paper, self.compressor.l3(paper, subproblem, max_nodes=20)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(map_paper, papers))
            for paper, value in results:
                record = {
                    "paper_id": paper.paper_id,
                    "paper_title": paper.title,
                    "subproblem": subproblem,
                    "compression_id": value["compression_id"],
                    "content": value["content"],
                }
                mapped.append(record)
                current_maps.append(record)
            if current_maps:
                reductions.append(self._reduce_subproblem(idea, subproblem, current_maps))
        final = self._reduce_final(idea, reductions) if reductions else {
            "status": "insufficient_full_text",
            "answer": "No usable shortlisted full-text papers were available.",
        }
        target_dir = self.workspace / "artifacts" / "syntheses"
        target_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(json.dumps(mapped, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        target = target_dir / f"synthesis_{digest[:20]}.json"
        payload = {
            "schema_version": SYNTHESIS_VERSION,
            "idea": idea.title,
            "maps": mapped,
            "subproblem_syntheses": reductions,
            "final_synthesis": final,
        }
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            "status": "completed" if reductions else "insufficient_full_text",
            "map_count": len(mapped),
            "subproblem_syntheses": reductions,
            "final_synthesis": final,
            "path": str(target),
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        }

    def _reduce_subproblem(
        self,
        idea: Idea,
        subproblem: str,
        maps: list[dict[str, Any]],
    ) -> dict[str, Any]:
        request = {
            "task": "Reduce paper-level L3 maps into one cross-paper evidence synthesis.",
            "idea": idea.title,
            "subproblem": subproblem,
            "rules": [
                "Use only supplied maps.",
                "Preserve disagreements and missing evidence.",
                "Every evidence entry must retain paper_id and node_id.",
            ],
            "maps": maps,
            "output_schema": {
                "answer": "string",
                "consensus": ["string"],
                "disagreements": ["string"],
                "baseline_implications": ["string"],
                "evidence": [{"paper_id": "string", "node_id": "string", "claim": "string"}],
                "unknowns": ["string"],
            },
        }
        turn = self.model.complete(
            [
                {"role": "system", "content": "Return one evidence-grounded JSON object only."},
                {"role": "user", "content": json.dumps(request, ensure_ascii=False)},
            ],
            [],
            2600,
        )
        value = _parse_json(turn.text)
        valid = {
            (record["paper_id"], str(item.get("node_id")))
            for record in maps
            for item in record["content"].get("evidence", [])
            if isinstance(item, dict) and item.get("node_id")
        }
        evidence = []
        for item in value.get("evidence") or []:
            if not isinstance(item, dict):
                continue
            pair = (str(item.get("paper_id") or ""), str(item.get("node_id") or ""))
            if pair in valid:
                evidence.append(item)
        value["evidence"] = evidence
        value["subproblem"] = subproblem
        return value

    def _reduce_final(self, idea: Idea, reductions: list[dict[str, Any]]) -> dict[str, Any]:
        request = {
            "task": "Create the final research decision memo from subproblem syntheses.",
            "idea": {**asdict(idea), "domain": idea.domain.value},
            "subproblem_syntheses": reductions,
            "rules": [
                "Do not claim novelty or experimental success.",
                "Separate supported baseline choices from proposed experiment decisions.",
                "Retain unresolved questions and evidence limitations.",
            ],
            "output_schema": {
                "idea_restatement": "string",
                "recommended_baseline_families": ["string"],
                "experiment_design": ["string"],
                "coding_requirements": ["string"],
                "risks": ["string"],
                "decision": "proceed|refine|insufficient_evidence",
            },
        }
        turn = self.model.complete(
            [
                {"role": "system", "content": "Return one conservative JSON object only."},
                {"role": "user", "content": json.dumps(request, ensure_ascii=False)},
            ],
            [],
            3000,
        )
        return _parse_json(turn.text)

    def _build_and_evaluate_benchmark(
        self,
        idea_entry: dict[str, Any],
        candidates: list[Paper],
        selected: list[Paper],
        fan_in: dict[str, Any],
        evaluation_size: int,
    ) -> dict[str, Any]:
        pool = candidates[: min(50, max(30, evaluation_size), len(candidates))]
        request = {
            "task": "Produce provisional relevance labels for later human adjudication.",
            "idea": {
                "idea_id": idea_entry["idea_id"],
                "research_question": idea_entry["research_question"],
                "claims": idea_entry.get("atomic_claims") or [],
            },
            "label_definition": {
                "3": "direct method or strong reproducible baseline",
                "2": "useful competing method or evaluation protocol",
                "1": "weak inspiration only",
                "0": "not relevant",
            },
            "papers": [{
                "paper_id": paper.paper_id,
                "title": paper.title,
                "abstract": (paper.abstract or "")[:1200],
            } for paper in pool],
            "output_schema": {"judgments": [{
                "paper_id": "string", "relevance": "0|1|2|3",
                "role": "baseline|method|evaluation|weak|irrelevant", "reason": "string",
            }]},
        }
        try:
            turn = self.model.complete(
                [
                    {"role": "system", "content": "Return one JSON object. These are provisional labels, not human gold."},
                    {"role": "user", "content": json.dumps(request, ensure_ascii=False)},
                ],
                [],
                3500,
            )
            raw = _parse_json(turn.text).get("judgments") or []
        except Exception:
            raw = []
        by_id = {paper.paper_id: paper for paper in pool}
        judgments = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            paper_id = str(item.get("paper_id") or "")
            if paper_id not in by_id:
                continue
            relevance = int(item.get("relevance") or 0)
            judgments.append({
                "idea_id": idea_entry["idea_id"],
                "paper_id": paper_id,
                "title": by_id[paper_id].title,
                "relevance": min(3, max(0, relevance)),
                "role": str(item.get("role") or "weak"),
                "reason": str(item.get("reason") or ""),
                "annotation_source": "model_provisional",
                "review_status": "pending_human",
            })
        judged = {item["paper_id"] for item in judgments}
        for paper in pool:
            if paper.paper_id in judged:
                continue
            judgments.append({
                "idea_id": idea_entry["idea_id"],
                "paper_id": paper.paper_id,
                "title": paper.title,
                "relevance": 0,
                "role": "unjudged",
                "reason": "Model did not return a valid judgment; human review required.",
                "annotation_source": "fallback_unjudged",
                "review_status": "pending_human",
            })
        target_dir = self.workspace / "artifacts" / "evaluation"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"judgments_{idea_entry['idea_id']}.jsonl"
        target.write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in judgments),
            encoding="utf-8",
        )
        selected_ids = [paper.paper_id for paper in selected]
        shortlist_ids = [item["paper_id"] for item in fan_in["shortlist"]]
        metrics = _ranking_metrics(judgments, selected_ids, shortlist_ids)
        human_count = sum(item["review_status"] == "human_verified" for item in judgments)
        return {
            "schema_version": "researchops-gold-v1",
            "fixed_idea_count": 5,
            "judgment_count": len(judgments),
            "human_verified_count": human_count,
            "gold_ready": human_count >= 30,
            "provisional_metrics": metrics,
            "path": str(target),
            "review_instructions": (
                "A human reviewer must inspect title/abstract/full text as needed, edit relevance/role/reason, "
                "and set review_status=human_verified. Provisional labels are excluded from formal gold claims."
            ),
        }

    def _write_artifact(
        self,
        task_id: str,
        run_id: str,
        kind: str,
        value: dict[str, Any],
    ) -> dict[str, str]:
        data = json.dumps(value, ensure_ascii=False, indent=2, default=str).encode("utf-8")
        digest = hashlib.sha256(data).hexdigest()
        target = self.workspace / "artifacts" / f"{kind}_{digest[:20]}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        artifact_id = f"artifact_{digest[:20]}"
        self.store.save_artifact(
            artifact_id, task_id, run_id, kind, str(target), digest,
            {"schema_version": BATCH_SCHEMA_VERSION},
        )
        return {"artifact_id": artifact_id, "path": str(target), "sha256": digest}


def load_idea(library_path: str | Path, idea_id: str) -> dict[str, Any]:
    value = json.loads(Path(library_path).read_text(encoding="utf-8"))
    for idea in value.get("ideas") or []:
        if idea.get("idea_id") == idea_id:
            return idea
    raise KeyError(f"Unknown idea_id: {idea_id}")


def _idea_from_entry(value: dict[str, Any]) -> Idea:
    return Idea.from_dict(value)


def _rank_papers(idea: Idea, papers: list[Paper], limit: int) -> list[Paper]:
    idea_terms = set(tokenize(
        f"{idea.title} {idea.research_question} {' '.join(idea.atomic_claims)}"
    ))

    def score(paper: Paper) -> tuple[int, float, int, int, str]:
        terms = set(tokenize(f"{paper.title} {paper.abstract or ''}"))
        overlap = len(idea_terms & terms) / max(1, len(idea_terms))
        return (
            _pdf_url_reliability(paper.pdf_url),
            overlap,
            int(bool(paper.abstract)),
            paper.year or 0,
            paper.title,
        )

    return sorted(papers, key=score, reverse=True)[:limit]


def _shortlist_reason(paper_id: str, fan_in: dict[str, Any]) -> str:
    item = next(
        (value for value in fan_in["shortlist"] if value["paper_id"] == paper_id),
        None,
    )
    return item["reason"] if item else "L1 screened corpus; not selected for L2."


def _ranking_metrics(
    judgments: list[dict[str, Any]],
    selected_ids: list[str],
    shortlist_ids: list[str],
) -> dict[str, Any]:
    relevance = {item["paper_id"]: int(item["relevance"]) for item in judgments}

    def ndcg(ranking: list[str], k: int) -> float:
        gains = [relevance.get(paper_id, 0) for paper_id in ranking[:k]]
        dcg = sum((2 ** gain - 1) / math.log2(index + 2) for index, gain in enumerate(gains))
        ideal = sorted(relevance.values(), reverse=True)[:k]
        idcg = sum((2 ** gain - 1) / math.log2(index + 2) for index, gain in enumerate(ideal))
        return round(dcg / idcg, 4) if idcg else 0.0

    relevant = {paper_id for paper_id, value in relevance.items() if value >= 2}
    shortlist_relevant = relevant & set(shortlist_ids)
    return {
        "ndcg_at_10": ndcg(selected_ids, 10),
        "ndcg_at_30": ndcg(selected_ids, 30),
        "shortlist_precision": round(len(shortlist_relevant) / max(1, len(shortlist_ids)), 4),
        "shortlist_recall": round(len(shortlist_relevant) / max(1, len(relevant)), 4),
        "relevant_count": len(relevant),
        "note": "Provisional only until at least 30 judgments are human_verified.",
    }


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


def _parse_json(text: str) -> dict[str, Any]:
    value = str(text or "").strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*|\s*```$", "", value, flags=re.I)
    try:
        result = json.loads(value)
    except json.JSONDecodeError:
        start, end = value.find("{"), value.rfind("}")
        if start < 0 or end <= start:
            raise
        result = json.loads(value[start:end + 1])
    if not isinstance(result, dict):
        raise ValueError("model output must be a JSON object")
    return result


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:20]
    return f"{prefix}_{digest}"

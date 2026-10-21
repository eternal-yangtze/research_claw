from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

from researchops.knowledge.compression import COMPRESSOR_VERSION, PaperCompressor
from researchops.papers.ingestion import DocumentIngestor
from researchops.papers.connectors import normalize_arxiv_id
from researchops.core.models import Idea, Paper
from researchops.papers.analysis import ANALYZER_VERSION, PaperAnalyzer
from researchops.harness.runtime import TraceStore
from researchops.core.storage import ResearchStore
from researchops.core.layout import ensure_workspace_assets


DAG_VERSION = "acquisition-dag-v1"
DAG_LAYERS = (
    ("compress_l0_l1", "source_archive", "download_pdf"),
    ("parse_pdf",),
    ("analyze_full_text",),
    ("compress_l2",),
)


@dataclass
class PaperState:
    paper: Paper
    reason: str
    outputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    nodes: dict[str, dict[str, Any]] = field(default_factory=dict)


class AcquisitionDAG:
    """Bounded parallel paper-stage DAG with content-addressed stage recovery."""

    def __init__(
        self,
        *,
        store: ResearchStore,
        ingestor: DocumentIngestor,
        analyzer: PaperAnalyzer,
        compressor: PaperCompressor,
        workspace: Path,
        traces: TraceStore,
        downloader: Callable[..., dict],
        source_downloader: Callable[..., dict],
        max_workers: int | None = None,
    ) -> None:
        self.store = store
        self.ingestor = ingestor
        self.analyzer = analyzer
        self.compressor = compressor
        self.workspace = workspace
        ensure_workspace_assets(self.workspace)
        self.traces = traces
        configured = max_workers or int(os.getenv("RESEARCHOPS_ACQUIRE_WORKERS", "4"))
        self.max_workers = min(max(1, configured), 16)
        self.downloader = downloader
        self.source_downloader = source_downloader

    def run(
        self,
        task_id: str,
        idea: Idea,
        selected: list[tuple[Paper, str]],
        run_id: str,
        session_id: str,
        deep_paper_ids: set[str] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        states = [PaperState(paper, reason) for paper, reason in selected]
        deep_ids = deep_paper_ids if deep_paper_ids is not None else {
            state.paper.paper_id for state in states
        }
        for layer in DAG_LAYERS:
            jobs = []
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                for state in states:
                    for stage in layer:
                        if not self._applies(stage, state.paper, deep_ids):
                            continue
                        jobs.append((state, stage, pool.submit(
                            self._run_node, task_id, idea, state, stage, run_id, session_id
                        )))
                for state, stage, future in jobs:
                    node = future.result()
                    state.nodes[stage] = node
                    if node.get("output") is not None:
                        state.outputs[stage] = node["output"]

        acquisitions = [self._report(task_id, state) for state in states]
        nodes = [node for state in states for node in state.nodes.values()]
        summary = {
            "dag_version": DAG_VERSION,
            "max_workers": self.max_workers,
            "layers": [list(layer) for layer in DAG_LAYERS],
            "nodes_total": len(nodes),
            "nodes_executed": sum(not node["recovered"] for node in nodes),
            "nodes_recovered": sum(node["recovered"] for node in nodes),
            "nodes_failed": sum(node["status"] == "failed" for node in nodes),
            "deep_paper_count": len(deep_ids),
        }
        return acquisitions, summary

    @staticmethod
    def _applies(stage: str, paper: Paper, deep_paper_ids: set[str]) -> bool:
        if stage in {"analyze_full_text", "compress_l2"}:
            return paper.paper_id in deep_paper_ids
        if stage != "source_archive":
            return True
        arxiv_id = paper.identity.arxiv_id if paper.identity else None
        if not arxiv_id and paper.source == "arxiv":
            arxiv_id, _ = normalize_arxiv_id(paper.external_id)
        return bool(arxiv_id)

    def _run_node(
        self,
        task_id: str,
        idea: Idea,
        state: PaperState,
        stage: str,
        run_id: str,
        session_id: str,
    ) -> dict[str, Any]:
        input_hash = self._input_hash(stage, idea, state)
        existing = self.store.acquisition_stage(task_id, state.paper.paper_id, stage)
        if (
            existing is not None
            and existing["status"] == "completed"
            and existing["input_hash"] == input_hash
        ):
            output = json.loads(existing["output_json"] or "{}")
            if self._valid_cached_output(stage, output, state.paper.paper_id):
                self.traces.emit(run_id, "research.stage.recovered", session_id, {
                    "task_id": task_id, "paper_id": state.paper.paper_id, "stage": stage,
                })
                return {
                    "stage": stage,
                    "status": "completed",
                    "input_hash": input_hash,
                    "attempt_count": existing["attempt_count"],
                    "recovered": True,
                    "error": None,
                    "output": output,
                }

        self.store.start_acquisition_stage(
            task_id, state.paper.paper_id, stage, input_hash, run_id
        )
        self.traces.emit(run_id, "research.stage.started", session_id, {
            "task_id": task_id, "paper_id": state.paper.paper_id, "stage": stage,
            "input_hash": input_hash,
        })
        try:
            output = self._execute(stage, task_id, idea, state, run_id)
            self.store.finish_acquisition_stage(
                task_id, state.paper.paper_id, stage, output
            )
            row = self.store.acquisition_stage(task_id, state.paper.paper_id, stage)
            self.traces.emit(run_id, "research.stage.completed", session_id, {
                "task_id": task_id, "paper_id": state.paper.paper_id, "stage": stage,
            })
            return {
                "stage": stage,
                "status": "completed",
                "input_hash": input_hash,
                "attempt_count": row["attempt_count"] if row else 1,
                "recovered": False,
                "error": None,
                "output": output,
            }
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            self.store.fail_acquisition_stage(
                task_id, state.paper.paper_id, stage, message
            )
            row = self.store.acquisition_stage(task_id, state.paper.paper_id, stage)
            self.traces.emit(run_id, "research.stage.failed", session_id, {
                "task_id": task_id, "paper_id": state.paper.paper_id,
                "stage": stage, "error": message,
            })
            return {
                "stage": stage,
                "status": "failed",
                "input_hash": input_hash,
                "attempt_count": row["attempt_count"] if row else 1,
                "recovered": False,
                "error": message,
                "output": None,
            }

    def _execute(
        self,
        stage: str,
        task_id: str,
        idea: Idea,
        state: PaperState,
        run_id: str,
    ) -> dict[str, Any]:
        paper = state.paper
        if stage == "compress_l0_l1":
            return {
                "compressions": [
                    _compression_summary(self.compressor.l0(paper)),
                    _compression_summary(self.compressor.l1(paper, idea)),
                ]
            }
        if stage == "source_archive":
            arxiv_id = paper.identity.arxiv_id if paper.identity else None
            if not arxiv_id:
                arxiv_id, _ = normalize_arxiv_id(paper.external_id)
            if not arxiv_id:
                return {"status": "not_applicable"}
            result = self.source_downloader(
                arxiv_id + (paper.source_version or ""), self.workspace / "artifacts" / "sources"
            )
            updated = replace(paper, source_archive_path=str(result["path"]))
            self.store.upsert_paper(updated)
            self.store.save_artifact(
                f"arxiv_source_{result['sha256'][:20]}",
                task_id,
                run_id,
                "arxiv_source",
                str(result["path"]),
                str(result["sha256"]),
                {
                    "paper_id": paper.paper_id,
                    "arxiv_id": arxiv_id,
                    "archive_format": result.get("archive_format"),
                    "member_count": result.get("member_count"),
                },
            )
            return {"status": "downloaded", **result}
        if stage == "download_pdf":
            if not paper.pdf_url:
                return {
                    "status": "metadata_only",
                    "detail": "No direct open PDF URL was supplied by the source.",
                }
            result = self.downloader(
                paper.pdf_url,
                self.workspace / "artifacts" / "papers",
                _paper_filename(paper),
            )
            return {"status": "downloaded", **result}
        if stage == "parse_pdf":
            download = state.outputs.get("download_pdf") or {}
            if download.get("status") != "downloaded":
                return {
                    "status": "skipped",
                    "reason": download.get("detail") or "PDF download did not complete.",
                }
            seeded = replace(
                paper,
                file_path=str(download["path"]),
                content_hash=str(download["sha256"]),
                source_archive_path=(state.outputs.get("source_archive") or {}).get("path") or paper.source_archive_path,
            )
            result = self.ingestor.ingest_file(download["path"], paper=seeded)
            return {
                "status": "ingested",
                "paper_id": result.paper_id,
                "path": result.path,
                "parsed_path": result.parsed_path,
                "parser": result.parser,
                "quality_status": result.quality_status,
                "chunks": result.chunks,
                "source_hash": download["sha256"],
                "parsed_sha256": hashlib.sha256(Path(result.parsed_path).read_bytes()).hexdigest(),
                "formula_validation": result.formula_validation,
            }
        if stage == "analyze_full_text":
            parsed = state.outputs.get("parse_pdf") or {}
            if parsed.get("status") != "ingested":
                reason = parsed.get("reason") or "No parsed full text is available."
                return self.analyzer.mark_insufficient(task_id, paper, reason)
            try:
                return self.analyzer.analyze(task_id, idea, paper)
            except Exception as error:
                return self.analyzer.failed(task_id, paper, error)
        if stage == "compress_l2":
            analysis = state.outputs.get("analyze_full_text") or {}
            if analysis.get("status") not in {"completed", "completed_with_gaps"}:
                return {
                    "status": "skipped",
                    "reason": "A completed full-text analysis is required for L2.",
                }
            path = analysis.get("path")
            if not path:
                return {"status": "skipped", "reason": "Analysis artifact is missing."}
            return {
                "status": "completed",
                "compression": _compression_summary(
                    self.compressor.l2_from_analysis(paper, path)
                ),
            }
        raise ValueError(f"Unknown acquisition stage: {stage}")

    def _input_hash(self, stage: str, idea: Idea, state: PaperState) -> str:
        paper = state.paper
        base: dict[str, Any] = {
            "dag_version": DAG_VERSION,
            "stage": stage,
            "paper": asdict(paper),
        }
        if stage in {"compress_l0_l1", "analyze_full_text"}:
            base["idea"] = asdict(idea)
        if stage == "source_archive":
            base["source_request_contract"] = "explicit-arxiv-version-v2"
        if stage == "compress_l0_l1":
            base["compressor_version"] = COMPRESSOR_VERSION
        elif stage == "parse_pdf":
            from researchops.papers.validation.automatic import validation_fingerprint
            base["download"] = state.outputs.get("download_pdf")
            base["source_archive"] = state.outputs.get("source_archive")
            base["formula_validation"] = validation_fingerprint()
            base["pdf_parser"] = os.getenv("RESEARCHOPS_PDF_PARSER", "auto")
        elif stage == "analyze_full_text":
            base["parse"] = state.outputs.get("parse_pdf")
            base["analyzer_version"] = ANALYZER_VERSION
        elif stage == "compress_l2":
            base["analysis"] = state.outputs.get("analyze_full_text")
            base["compressor_version"] = COMPRESSOR_VERSION
        return hashlib.sha256(
            json.dumps(base, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

    def _valid_cached_output(self, stage: str, output: dict[str, Any], paper_id: str) -> bool:
        if stage == "source_archive" and output.get("status") == "downloaded":
            return _matches_sha256(output.get("path"), output.get("sha256"))
        if stage == "download_pdf" and output.get("status") == "downloaded":
            return _matches_sha256(output.get("path"), output.get("sha256"))
        if stage == "parse_pdf" and output.get("status") == "ingested":
            validation = output.get("formula_validation") or {}
            return (_matches_sha256(output.get("parsed_path"), output.get("parsed_sha256"))
                    and bool(self.store.paper_chunks(paper_id)) and bool(validation)
                    and validation.get("status") != "validation_failed"
                    and all(_matches_sha256(path, digest) for path, digest in validation.get("artifacts", {}).items()))
        if stage == "analyze_full_text":
            return (
                output.get("status") in {"completed", "completed_with_gaps"}
                and _matches_sha256(output.get("path"), output.get("sha256"))
            )
        return True

    def _report(self, task_id: str, state: PaperState) -> dict[str, Any]:
        download_node = state.nodes.get("download_pdf") or {}
        download = state.outputs.get("download_pdf") or {}
        parsed_node = state.nodes.get("parse_pdf") or {}
        parsed = state.outputs.get("parse_pdf") or {}
        analysis = state.outputs.get("analyze_full_text") or {}
        source_node = state.nodes.get("source_archive") or {}
        source = state.outputs.get("source_archive") or {}
        compression = state.outputs.get("compress_l0_l1") or {}
        l2 = state.outputs.get("compress_l2") or {}

        if download.get("status") == "metadata_only":
            status = "metadata_only"
            detail: Any = download.get("detail")
        elif download_node.get("status") == "failed" or parsed_node.get("status") == "failed":
            status = "failed"
            detail = download_node.get("error") or parsed_node.get("error")
        elif parsed.get("status") == "ingested":
            status = "ingested"
            detail = {
                "download": download,
                "chunks": parsed.get("chunks", 0),
                "parser": parsed.get("parser"),
                "quality_status": parsed.get("quality_status"),
            }
        else:
            status = "failed"
            detail = parsed.get("reason") or "Acquisition did not produce parsed full text."

        compressions = list(compression.get("compressions") or [])
        if l2.get("compression"):
            compressions.append(l2["compression"])
        compression_errors = [
            node["error"]
            for name, node in state.nodes.items()
            if name.startswith("compress_") and node.get("status") == "failed"
        ]
        source_status = "not_applicable"
        source_error = None
        if source_node:
            source_status = source.get("status") or source_node.get("status")
            source_error = source_node.get("error")
        self.store.link_task_paper(
            task_id,
            state.paper.paper_id,
            role="selected",
            acquired_status=status,
            selection_reason=state.reason,
        )
        return {
            "paper_id": state.paper.paper_id,
            "title": state.paper.title,
            "status": status,
            "detail": detail,
            "pdf_path": download.get("path"),
            "parsed_path": parsed.get("parsed_path"),
            "parser": parsed.get("parser"),
            "chunk_count": parsed.get("chunks", 0),
            "analysis_status": analysis.get("status", "not_selected_for_l2"),
            "analysis_path": analysis.get("path"),
            "baseline_verdict": analysis.get("baseline_verdict", "insufficient_evidence"),
            "source_archive_status": source_status,
            "source_archive_path": source.get("path"),
            "source_archive_receipt_path": source.get("receipt_path"),
            "source_archive_error": source_error,
            "formula_validation": parsed.get("formula_validation"),
            "compressions": compressions,
            "compression_error": "; ".join(compression_errors) or None,
            "dag_stages": [state.nodes[name] for layer in DAG_LAYERS for name in layer if name in state.nodes],
        }


def _compression_summary(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.get(key)
        for key in (
            "compression_id", "paper_id", "level", "scope_key", "schema_version",
            "compressor_version", "prompt_hash", "source_hash", "model_name",
            "token_estimate", "status", "cache_hit",
        )
    }


def _paper_filename(paper: Paper) -> str:
    identity = paper.identity
    arxiv_id = identity.arxiv_id if identity else None
    if arxiv_id:
        return f"{arxiv_id.replace('/', '_')}.pdf"
    return f"{paper.paper_id}.pdf"


def _is_file(value: object) -> bool:
    return bool(value) and Path(str(value)).is_file()


def _matches_sha256(path_value: object, expected: object) -> bool:
    if not _is_file(path_value) or not expected:
        return False
    return hashlib.sha256(Path(str(path_value)).read_bytes()).hexdigest() == str(expected)

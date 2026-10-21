from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
from pathlib import Path

from researchops.acquisition import AcquisitionDAG
from researchops.compression import PaperCompressor
from researchops.ingestion import DocumentIngestor
from researchops.models import Domain, Idea, Paper, PaperIdentity
from researchops.paper_analysis import PaperAnalyzer
from researchops.runtime import ModelTurn, TraceStore
from researchops.storage import ResearchStore


class GroundedVerificationModel:
    model = "deterministic-verification-model"

    def complete(self, messages, tools, max_tokens):
        request = json.loads(messages[-1]["content"])
        chunk_id = request["chunks"][0]["chunk_id"]
        field = {"text": "Statement retained from a parsed full-text node.", "citations": [chunk_id]}
        return ModelTurn(text=json.dumps({
            "paper_summary": field,
            "research_problem": field,
            "inputs_outputs": field,
            "method_overview": field,
            "architecture_components": [field],
            "knowledge_or_graph_role": field,
            "training_objective": field,
            "inference_procedure": field,
            "datasets": [field],
            "data_splits_preprocessing": field,
            "metrics": [field],
            "paper_baselines": [field],
            "reported_results": [field],
            "implementation_availability": field,
            "compute_requirements": field,
            "limitations": [field],
            "baseline_assessment": {
                "verdict": "partial",
                "rationale": field,
                "task_alignment": field,
                "input_output_compatibility": field,
                "dataset_metric_comparability": field,
                "reproduction_requirements": [field],
                "adaptations_required": [field],
            },
            "missing_information": [],
        }))


def main() -> int:
    os.environ["RESEARCHOPS_PDF_PARSER"] = "pymupdf"
    root = Path(__file__).resolve().parents[1]
    source_root = root / "workspace_acl_cvf_smoke" / "artifacts" / "papers"
    sources = {
        "acl": source_root / "acl_paper_6179bc76e7f483f38e3e.pdf",
        "cvf": source_root / "cvf_paper_e92d0c564798d41a7dd2.pdf",
    }
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing verified source PDFs: " + ", ".join(missing))

    workspace = root / "workspace_dag_smoke"
    store = ResearchStore(workspace / "researchops.db")
    idea = Idea(
        "Graph methods", Domain.AI_ML, "Evaluate graph-based machine-learning methods.",
        "Which graph methods provide reproducible baselines?",
        ("Graph structure changes model behavior.",),
        ("Compare with a non-graph baseline.",),
    )
    task_id = "task_dag_real_pdf_verification"
    store.create_task(task_id, idea.title, idea.domain.value, {
        "title": idea.title, "domain": idea.domain.value,
        "description": idea.description, "research_question": idea.research_question,
    })
    papers = [
        Paper(
            "paper_dag_acl", "ACL graph paper", source="acl",
            pdf_url="https://aclanthology.org/2026.nlpcss-1.20.pdf",
            identity=PaperIdentity(acl_id="2026.nlpcss-1.20"),
        ),
        Paper(
            "paper_dag_cvf", "CVF graph paper", source="cvf",
            pdf_url="https://openaccess.thecvf.com/verification.pdf",
            identity=PaperIdentity(cvf_id="CVPR2025:verification"),
        ),
    ]
    for paper in papers:
        store.upsert_paper(paper)
        store.link_task_paper(task_id, paper.paper_id, role="selected", acquired_status="pending")

    download_calls = 0
    lock = threading.Lock()

    def downloader(url, output_dir, filename):
        nonlocal download_calls
        source = sources["acl" if "aclanthology" in url else "cvf"]
        target = Path(output_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        data = target.read_bytes()
        with lock:
            download_calls += 1
        return {
            "path": str(target), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(), "url": url,
        }

    model = GroundedVerificationModel()
    dag = AcquisitionDAG(
        store=store,
        ingestor=DocumentIngestor(store),
        analyzer=PaperAnalyzer(model, store, workspace),
        compressor=PaperCompressor(store, model),
        workspace=workspace,
        traces=TraceStore(workspace / "runs"),
        downloader=downloader,
        source_downloader=lambda *_: {},
        max_workers=2,
    )
    selected = [(paper, "real PDF DAG verification") for paper in papers]
    first, first_summary = dag.run(task_id, idea, selected, "run_dag_first", "session_dag")
    second, second_summary = dag.run(task_id, idea, selected, "run_dag_resume", "session_dag")
    report = {
        "paper_count": len(papers),
        "download_calls": download_calls,
        "first_run": first_summary,
        "resumed_run": second_summary,
        "papers": [{
            "paper_id": item["paper_id"], "status": item["status"],
            "chunks": item["chunk_count"], "analysis_status": item["analysis_status"],
            "compression_levels": [value["level"] for value in item["compressions"]],
        } for item in second],
        "database_counts": store.counts(),
    }
    target = workspace / "artifacts" / "dag_verification.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**report, "report_path": str(target)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

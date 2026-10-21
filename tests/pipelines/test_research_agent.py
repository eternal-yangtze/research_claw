import json
import hashlib
from pathlib import Path

import pytest

from researchops.ingestion import DocumentIngestor
from researchops.literature import LiteratureSearch
from researchops.models import Paper
from researchops.research import LiteratureAcquisitionPipeline, _broaden_arxiv_query
from researchops.runtime import ModelTurn, ToolCall
from researchops.runtime import TraceStore
from researchops.storage import ResearchStore
from researchops.workflow import ResearchOpsApp


class ScriptedResearchModel:
    def __init__(self) -> None:
        self.calls = 0
        self.first_messages = None

    def complete(self, messages, tools, max_tokens):
        self.calls += 1
        if self.calls == 1:
            self.first_messages = list(messages)
            return ModelTurn(tool_calls=(ToolCall(
                "start-1",
                "start_research",
                {
                    "idea_text": "Use graph structure to detect anomalies in multivariate time series.",
                    "max_queries": 2,
                    "limit_per_query": 3,
                    "max_downloads": 0,
                },
            ),))
        if self.calls == 2:
            return ModelTurn(text=json.dumps({
                "title": "Graph-aware multivariate time-series anomaly detection",
                "domain": "time_series",
                "description": "Use learned graph structure in multivariate time-series anomaly detection.",
                "research_question": "Does learned graph structure improve multivariate time-series anomaly detection?",
                "atomic_claims": ["A learned graph captures cross-series dependencies for anomaly detection."],
                "success_criteria": ["Compare against a non-graph baseline on fixed splits."],
                "ambiguities": ["The graph-learning mechanism is not yet specified."],
                "search_queries": [
                    {
                        "source": "arxiv",
                        "query": "graph multivariate time series anomaly detection",
                        "rationale": "Find methods and baselines.",
                    },
                    {
                        "source": "crossref",
                        "query": "graph multivariate time series anomaly detection",
                        "rationale": "Verify publication metadata.",
                    },
                ],
            }))
        return ModelTurn(text="Research intake and bounded literature acquisition completed.")


def test_agent_runs_new_idea_through_structuring_and_search(tmp_path: Path) -> None:
    app = ResearchOpsApp(tmp_path / "workspace")
    paper = Paper(
        paper_id="paper_graph_ts",
        title="Graph Neural Networks for Multivariate Time Series Anomaly Detection",
        source="arxiv",
        source_url="https://arxiv.org/abs/0000.00000",
        abstract="A graph model captures dependencies in multivariate time series.",
        external_id="0000.00000",
        quality_status="metadata_only",
    )
    app.literature.connectors = {
        "arxiv": lambda query, limit: [paper],
        "crossref": lambda query, limit: [],
    }
    model = ScriptedResearchModel()

    result = app.run_agent("Please research this new Idea.", model=model, run_id="agent-run")

    assert result["answer"].startswith("Research intake")
    assert result["tool_calls"] == 1
    assert model.first_messages[0]["role"] == "system"
    assert "start_research" in model.first_messages[0]["content"]
    with app.store.connect() as connection:
        task = connection.execute("SELECT status,domain FROM tasks").fetchone()
        sources = {row[0] for row in connection.execute("SELECT source FROM source_runs")}
        linked = connection.execute("SELECT role,acquired_status FROM task_papers").fetchone()
    assert tuple(task) == ("SEARCH_COMPLETED", "time_series")
    assert sources == {"arxiv", "crossref"}
    assert tuple(linked) == ("candidate", "metadata_only")
    artifacts = list((tmp_path / "workspace" / "artifacts").glob("artifact_*.json"))
    assert len(artifacts) == 1
    report = json.loads(artifacts[0].read_text(encoding="utf-8"))
    assert report["semantic_scholar_used"] is False
    assert report["structured_plan"]["idea"]["title"].startswith("Graph-aware")
    assert report["pipeline_counts"] == {
        "candidates_found": 1,
        "papers_selected": 0,
        "pdfs_downloaded": 0,
        "papers_parsed": 0,
        "papers_analyzed": 0,
        "plans_generated": 0,
        "metadata_only": 0,
    }
    assert report["run_id"] == "agent-run"


class AcquisitionModel:
    def __init__(self, paper_id: str) -> None:
        self.paper_id = paper_id
        self.calls = 0

    def complete(self, messages, tools, max_tokens):
        self.calls += 1
        if self.calls == 1:
            return ModelTurn(text=json.dumps({
                "title": "Query-aware compression",
                "domain": "time_series",
                "description": "Allocate compression error from query-level tolerances.",
                "research_question": "Can query tolerances determine segment compression budgets?",
                "atomic_claims": ["Query tolerances can be mapped to segment error budgets."],
                "success_criteria": ["All measured errors remain within deterministic bounds."],
                "ambiguities": [],
                "search_queries": [{
                    "source": "arxiv",
                    "query": "query aware time series compression",
                    "rationale": "Find direct methods.",
                }],
            }))
        if self.calls == 2:
            return ModelTurn(text=json.dumps({
                "selected": [{"paper_id": self.paper_id, "reason": "Direct method evidence"}],
            }))
        request = json.loads(messages[-1]["content"])
        if "paper_analyses" in request:
            analysis = request["paper_analyses"][0]
            chunk_id = next(iter(analysis["citation_index"]))
            paper_id = analysis["paper_id"]
            return ModelTurn(text=json.dumps({
                "idea_restatement": "Implement and evaluate the proposed method.",
                "paper_grounding_summary": ["One parsed paper supports a partial baseline."],
                "baseline_selection": [{
                    "paper_id": paper_id,
                    "role": "primary",
                    "justification": "The parsed method is comparable.",
                    "evidence": [{"paper_id": paper_id, "chunk_id": chunk_id}],
                    "adaptation_required": ["Use the target data interface."],
                }],
                "research_plan": {
                    "objective": "Evaluate the method.",
                    "hypotheses": ["The method improves the target metric."],
                    "datasets": ["A fixed public dataset."],
                    "data_protocol": ["Use fixed splits."],
                    "baselines": ["Parsed paper baseline."],
                    "metrics": ["Task metric."],
                    "ablations": ["Remove the proposed component."],
                    "experiment_matrix": ["Baseline versus proposed."],
                    "reproducibility_controls": ["Fixed seeds."],
                    "expected_artifacts": ["Metrics JSON."],
                    "decision_rules": ["Compare repeated runs."],
                },
                "coding_plan": {
                    "architecture": ["Configuration-driven experiment runner."],
                    "modules": [{"path": "src/model.py", "responsibility": "Model implementation."}],
                    "implementation_steps": ["Implement baseline."],
                    "interfaces": ["fit and predict."],
                    "tests": ["Smoke test."],
                    "commands": ["python -m pytest"],
                    "outputs": ["metrics.json"],
                },
                "open_questions": [],
                "risks": [],
            }))
        chunk_id = request["chunks"][0]["chunk_id"]
        field = {"text": "Supported by the parsed paper text.", "citations": [chunk_id]}
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


def test_pipeline_downloads_and_ingests_selected_pdf(tmp_path: Path, monkeypatch) -> None:
    fitz = pytest.importorskip("pymupdf")
    monkeypatch.setenv("RESEARCHOPS_PDF_PARSER", "pymupdf")
    workspace = tmp_path / "workspace"
    store = ResearchStore(workspace / "researchops.db")
    literature = LiteratureSearch(store)
    paper = Paper(
        paper_id="paper_selected",
        title="Query-aware compression for time series",
        source="arxiv",
        source_url="https://arxiv.org/abs/1706.03762",
        pdf_url="https://arxiv.org/pdf/1706.03762",
        external_id="1706.03762",
        abstract="Compression budgets are optimized for a query workload.",
        quality_status="metadata_only",
    )
    literature.connectors = {
        "arxiv": lambda query, limit: [paper],
        "crossref": lambda query, limit: [],
    }

    def fake_download(url, output_dir, filename):
        target = Path(output_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        document = fitz.open()
        page = document.new_page()
        page.insert_text((72, 72), "Query-aware compression with deterministic error bounds")
        document.save(target)
        document.close()
        data = target.read_bytes()
        return {"path": str(target), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "url": url}

    def fake_source_download(arxiv_id, output_dir):
        target = Path(output_dir) / f"arxiv_{arxiv_id}.tar.gz"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"source archive")
        return {
            "path": str(target), "bytes": target.stat().st_size,
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "url": f"https://export.arxiv.org/e-print/{arxiv_id}",
            "archive_format": "tar_gzip", "member_count": 2,
        }

    pipeline = LiteratureAcquisitionPipeline(
        model=AcquisitionModel(paper.paper_id),
        store=store,
        literature=literature,
        ingestor=DocumentIngestor(store),
        workspace=workspace,
        traces=TraceStore(workspace / "runs"),
        downloader=fake_download,
        source_downloader=fake_source_download,
    )
    report = pipeline.run("Allocate compression error from query tolerances.", max_queries=2, max_downloads=1)

    assert {search["source"] for search in report["searches"]} == {"arxiv", "crossref"}
    assert report["acquisitions"][0]["status"] == "ingested"
    assert report["acquisitions"][0]["pdf_path"].endswith("1706.03762.pdf")
    assert Path(report["acquisitions"][0]["parsed_path"]).name.startswith("paper_selected.")
    assert report["acquisitions"][0]["formula_validation"]["status"] == "not_assessed"
    assert report["acquisitions"][0]["analysis_status"] == "completed"
    assert report["acquisitions"][0]["source_archive_status"] == "downloaded"
    assert report["acquisitions"][0]["source_archive_path"].endswith("arxiv_1706.03762.tar.gz")
    assert report["acquisitions"][0]["baseline_verdict"] == "partial"
    assert report["execution_plan"]["status"] == "completed"
    assert Path(report["execution_plan"]["path"]).exists()
    linked = store.task_papers(report["task_id"])
    assert linked[0]["acquired_status"] == "ingested"
    assert linked[0]["file_path"].endswith("1706.03762.pdf")
    assert store.counts()["chunks"] == 1
    assert store.counts()["paper_analyses"] == 1
    assert report["acquisition_dag"]["nodes_total"] == 6
    assert report["acquisition_dag"]["nodes_failed"] == 0


def test_acquisition_dag_recovers_completed_paper_stages(tmp_path: Path, monkeypatch) -> None:
    fitz = pytest.importorskip("fitz")
    monkeypatch.setenv("RESEARCHOPS_PDF_PARSER", "pymupdf")
    workspace = tmp_path / "workspace"
    store = ResearchStore(workspace / "researchops.db")
    literature = LiteratureSearch(store)
    paper = Paper(
        paper_id="paper_resume",
        title="Resumable acquisition",
        source="arxiv",
        source_url="https://arxiv.org/abs/1706.03762",
        abstract="Query-aware compression.",
        pdf_url="https://arxiv.org/pdf/1706.03762.pdf",
        external_id="1706.03762",
        quality_status="metadata_only",
    )
    literature.connectors = {
        "arxiv": lambda query, limit: [paper],
        "crossref": lambda query, limit: [],
    }
    calls = {"pdf": 0, "source": 0}

    def fake_download(url, output_dir, filename):
        calls["pdf"] += 1
        target = Path(output_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        document = fitz.open()
        page = document.new_page()
        page.insert_text((72, 72), "Resumable query-aware compression")
        document.save(target)
        document.close()
        data = target.read_bytes()
        return {"path": str(target), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "url": url}

    def fake_source(arxiv_id, output_dir):
        calls["source"] += 1
        target = Path(output_dir) / f"arxiv_{arxiv_id}.tar.gz"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"source")
        return {
            "path": str(target), "bytes": 6,
            "sha256": hashlib.sha256(b"source").hexdigest(),
            "url": "https://export.arxiv.org/e-print/1706.03762",
            "archive_format": "tar_gzip", "member_count": 1,
        }

    def make_pipeline():
        return LiteratureAcquisitionPipeline(
            model=AcquisitionModel(paper.paper_id),
            store=store,
            literature=literature,
            ingestor=DocumentIngestor(store),
            workspace=workspace,
            traces=TraceStore(workspace / "runs"),
            downloader=fake_download,
            source_downloader=fake_source,
        )

    first = make_pipeline().run(
        "Allocate compression error from query tolerances.", max_queries=2, max_downloads=1
    )
    second = make_pipeline().run(
        "Allocate compression error from query tolerances.", max_queries=2, max_downloads=1
    )

    assert first["acquisition_dag"]["nodes_executed"] == 6
    assert second["acquisition_dag"]["nodes_recovered"] == 6
    assert second["acquisition_dag"]["nodes_executed"] == 0
    assert calls == {"pdf": 1, "source": 1}
    first_attempts = {
        node["stage"]: node["attempt_count"]
        for node in first["acquisitions"][0]["dag_stages"]
    }
    second_attempts = {
        node["stage"]: node["attempt_count"]
        for node in second["acquisitions"][0]["dag_stages"]
    }
    assert second_attempts == first_attempts


def test_metadata_only_selection_is_not_baseline_ready(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    store = ResearchStore(workspace / "researchops.db")
    literature = LiteratureSearch(store)
    paper = Paper(
        paper_id="paper_metadata",
        title="Metadata only paper",
        source="crossref",
        abstract="An abstract without an open full text.",
        quality_status="metadata_only",
    )
    literature.connectors = {
        "arxiv": lambda query, limit: [],
        "crossref": lambda query, limit: [paper],
    }
    pipeline = LiteratureAcquisitionPipeline(
        model=AcquisitionModel(paper.paper_id),
        store=store,
        literature=literature,
        ingestor=DocumentIngestor(store),
        workspace=workspace,
        traces=TraceStore(workspace / "runs"),
    )

    report = pipeline.run("Allocate compression error from query tolerances.", max_queries=2, max_downloads=1)

    acquisition = report["acquisitions"][0]
    assert acquisition["status"] == "metadata_only"
    assert acquisition["pdf_path"] is None
    assert acquisition["parsed_path"] is None
    assert acquisition["analysis_status"] == "insufficient_full_text"
    assert acquisition["baseline_verdict"] == "insufficient_evidence"
    assert Path(acquisition["analysis_path"]).exists()
    assert report["execution_plan"]["status"] == "insufficient_full_text"
    with store.connect() as connection:
        status = connection.execute("SELECT status FROM tasks").fetchone()[0]
    assert status == "PAPERS_SELECTED"


def test_zero_result_arxiv_query_is_broadened_once(tmp_path: Path) -> None:
    assert _broaden_arxiv_query(
        "Transformer multivariate time series anomaly detection attention"
    ) == 'all:"time series" AND all:anomaly AND all:transformer'
    workspace = tmp_path / "workspace"
    store = ResearchStore(workspace / "researchops.db")
    literature = LiteratureSearch(store)
    arxiv_queries = []

    def empty_arxiv(query, limit):
        arxiv_queries.append(query)
        return []

    literature.connectors = {
        "arxiv": empty_arxiv,
        "crossref": lambda query, limit: [],
    }
    pipeline = LiteratureAcquisitionPipeline(
        model=AcquisitionModel("unused"),
        store=store,
        literature=literature,
        ingestor=DocumentIngestor(store),
        workspace=workspace,
        traces=TraceStore(workspace / "runs"),
    )

    report = pipeline.run("Allocate compression error from query tolerances.", max_queries=2, max_downloads=0)

    assert arxiv_queries == [
        "query aware time series compression",
        'all:"time series" AND all:compression',
    ]
    assert any(search.get("refines_query") for search in report["searches"])
    assert report["task_status"] == "SEARCH_COMPLETED"


def test_legacy_acquired_status_is_migrated_by_actual_full_text_state(tmp_path: Path) -> None:
    database = tmp_path / "research.db"
    store = ResearchStore(database)
    store.create_task("task_old", "Old task", "time_series", {"title": "Old task"})
    store.upsert_paper(Paper("paper_old", "Old metadata", source="crossref", quality_status="metadata_only"))
    store.link_task_paper(
        "task_old",
        "paper_old",
        role="selected",
        acquired_status="metadata_only",
    )
    with store.connect() as connection:
        connection.execute("UPDATE tasks SET status='PAPERS_ACQUIRED' WHERE task_id='task_old'")

    ResearchStore(database)

    with store.connect() as connection:
        status = connection.execute(
            "SELECT status FROM tasks WHERE task_id='task_old'"
        ).fetchone()[0]
    assert status == "PAPERS_SELECTED"

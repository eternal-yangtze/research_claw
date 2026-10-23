from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from researchops.core.config import load_project_env
from researchops.application.workflow import RESEARCH_LEAD, ResearchOpsApp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="researchops", description="Auditable research agent")
    parser.add_argument("--workspace", default="knowledge_base/default", help="SQLite, trace, and artifact directory")
    parser.add_argument("--allow-root", action="append", default=[], help="Explicit readable input root for skills; repeat as needed")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init", help="Initialize the persistent store")
    subparsers.add_parser("skills", help="List project skill metadata without loading all instructions")
    agent = subparsers.add_parser("agent", help="Run the research agent through discover/load/invoke skills")
    agent.add_argument("task", help="Natural-language task with necessary input paths")
    agent.add_argument("--token-cap", type=int, default=250000)
    agent.add_argument("--context-mode", choices=("lazy", "eager"), default="lazy")

    ingest = subparsers.add_parser("ingest", help="Ingest Markdown, text, or PDF documents")
    ingest.add_argument("path")

    validate = subparsers.add_parser("validate-formulas", help="Validate PDF formula candidates against version-matched static source")
    validate.add_argument("--pdf", required=True)
    validate.add_argument("--docling-json", required=True)
    validate.add_argument("--source-package")
    validate.add_argument("--source-receipt")

    search = subparsers.add_parser("search", help="Run hybrid retrieval")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=8)

    ask = subparsers.add_parser("ask", help="Return extractive evidence with citations")
    ask.add_argument("question")
    ask.add_argument("--limit", type=int, default=5)

    audit = subparsers.add_parser("audit", help="Audit the configured idea against the indexed corpus")
    audit.add_argument("--idea", default="config/seed_idea.json")

    online = subparsers.add_parser(
        "online-search",
        help="Search arXiv, OpenAlex, OpenReview, DBLP, Crossref, ACL, and CVF",
    )
    online.add_argument("query")
    online.add_argument("--limit", type=int, default=5)

    demo = subparsers.add_parser("demo", help="Run ingestion, novelty audit, and evidence QA")
    demo.add_argument("--idea", default="config/seed_idea.json")
    demo.add_argument("--corpus", required=True)

    research = subparsers.add_parser(
        "research",
        help="Run the LLM agent on a new Idea and autonomously acquire literature",
    )
    research.add_argument("idea", nargs="?", help="Natural-language research Idea")
    research.add_argument("--idea-file", help="UTF-8 text file containing the Idea")
    research.add_argument("--max-queries", type=int, default=7)
    research.add_argument("--limit-per-query", type=int, default=8)
    research.add_argument("--max-downloads", type=int, default=12)
    research.add_argument("--pdf-parser", choices=("auto", "docling", "pymupdf"))
    research.add_argument("--legacy-tools", action="store_true", help="Use the historical direct tool agent")
    research.add_argument("--token-cap", type=int, default=250000)

    batch = subparsers.add_parser(
        "batch-research",
        help="Run 30-paper L1 screening, L2 shortlist, L3 map-reduce, and benchmark preparation",
    )
    batch.add_argument("--idea-id", default="vit_peft_small_data")
    batch.add_argument("--idea-library", default="config/idea_library.json")
    batch.add_argument("--target-papers", type=int, default=30)
    batch.add_argument("--shortlist-size", type=int, default=6)
    batch.add_argument("--limit-per-query", type=int, default=12)
    batch.add_argument("--evaluation-size", type=int, default=40)
    batch.add_argument("--pdf-parser", choices=("auto", "docling", "pymupdf"), default="pymupdf")

    silver = subparsers.add_parser(
        "silver-evaluate",
        help="Create evidence-grounded LLM silver labels and evaluate retrieval quality",
    )
    silver.add_argument("--idea-id", default="vit_peft_small_data")
    silver.add_argument("--idea-library", default="config/idea_library.json")
    silver.add_argument("--provisional-path")
    silver.add_argument("--batch-report")
    silver.add_argument("--max-papers", type=int, default=40)
    silver.add_argument("--workers", type=int, default=3)
    silver.add_argument("--max-fulltext-chars", type=int, default=14000)

    prepare = subparsers.add_parser("prepare-experiment", help="Recommend, acquire and verify datasets; freeze a pre-coding experiment contract")
    prepare.add_argument("--idea-id", default="vit_peft_small_data")
    prepare.add_argument("--idea-library", default="config/idea_library.json")
    prepare.add_argument("--workers", type=int, default=3)
    prepare.add_argument("--offline", action="store_true", help="Reuse downloaded/imported files without network acquisition")
    prepare.add_argument("--dataset-source", choices=("auto", "official", "huggingface"), default="auto")

    study = subparsers.add_parser("plan-study", help="Plan an arbitrary same-domain Idea on a shared corpus with independent review")
    study.add_argument("idea", nargs="?", help="Natural-language research problem")
    study.add_argument("--idea-file", help="UTF-8 JSON structured Idea or plain text")
    study.add_argument("--source", default="knowledge_base/workspace_batch_vit_v2")
    study.add_argument("--token-cap", type=int, default=250000)
    study.add_argument("--retrieval-strategy", choices=("legacy", "cpu", "fusion"), default="legacy")
    study.add_argument("--orchestration", choices=("native", "langgraph"), help="Override RESEARCHOPS_STUDY_ORCHESTRATOR; new runs default to langgraph")
    study.add_argument("--dataset-workspace", help="Use verified dataset manifests from this independent workspace")

    acquire = subparsers.add_parser("datasets-acquire", help="Download and validate supported official image datasets")
    acquire.add_argument("dataset_ids", nargs="*", default=["cifar10", "cifar100", "flowers102", "pets", "dtd"])
    acquire.add_argument("--workers", type=int, default=3)
    acquire.add_argument("--dataset-source", choices=("auto", "official", "huggingface"), default="auto")
    local = subparsers.add_parser("dataset-import", help="Import original official archives/files and validate formats")
    local.add_argument("dataset_id")
    local.add_argument("--path", required=True)

    coding_plan = subparsers.add_parser("coding-plan", help="Refresh hybrid-RAG synthesis and create the frozen coding handoff")
    coding_plan.add_argument("--idea-id", default="vit_peft_small_data")
    coding_plan.add_argument("--from-scratch", action="store_true", help="Disable reusable implementation templates")
    coding_plan.add_argument("--study-result", help="Accepted LangGraph study result.json (v2 handoff)")
    coding_plan.add_argument("--token-cap", type=int, default=600000, help="New v2 coding plan cap, at most 5x original")
    coding_plan.add_argument("--stream", action="store_true", help="Receive provider SSE without hidden retries")
    coding_plan.add_argument("--reasoning-effort", choices=("low", "medium", "high"))
    code = subparsers.add_parser("code", help="Run/resume the coding agent and external Docker verification")
    code.add_argument("--idea-id", default="vit_peft_small_data")
    code.add_argument("--resume", action="store_true")
    code.add_argument("--from-scratch", action="store_true", help="Build each module from empty code using task-level verification")
    code.add_argument("--plan", help="Use an existing immutable coding plan, including historical benchmark plans")
    code.add_argument("--study-result", help="Accepted LangGraph study result.json (v2 handoff)")
    code.add_argument("--token-cap", type=int, default=600000, help="Cap for new v2 plan; immutable on resume")
    code.add_argument("--stream", action="store_true", help="Receive provider SSE without hidden retries")
    code.add_argument("--reasoning-effort", choices=("low", "medium", "high"))

    reanalyze = subparsers.add_parser(
        "reanalyze",
        help="Retry full-text paper analysis and planning without searching or downloading again",
    )
    reanalyze.add_argument("task_id")
    reanalyze.add_argument(
        "--deterministic-plan",
        action="store_true",
        help="Build the grounded fallback plan without another LLM planning call",
    )

    subparsers.add_parser("status", help="Show persistent object counts")
    subparsers.add_parser("tools", help="List governed tool specifications")
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    load_project_env()
    args = build_parser().parse_args(argv)
    if getattr(args, "pdf_parser", None):
        os.environ["RESEARCHOPS_PDF_PARSER"] = args.pdf_parser
    permitted = list(args.allow_root)
    if getattr(args, "path", None):
        permitted.append(Path(args.path).resolve())
    if getattr(args, "corpus", None):
        permitted.append(Path(args.corpus).resolve())
    try:
        if args.command == "validate-formulas":
            from researchops.papers.validation import validate_formulas
            output = validate_formulas(args.pdf, Path(args.workspace) / "artifacts" / "formula_validation",
                                       docling_json=args.docling_json, source_package=args.source_package,
                                       source_receipt=args.source_receipt)
        elif args.command == "plan-study":
            from researchops.application.study import plan_study
            if args.idea_file:
                raw = Path(args.idea_file).read_text(encoding="utf-8")
                idea = json.loads(raw) if raw.lstrip().startswith("{") else {"description": raw}
            else:
                idea = {"description": args.idea or ""}
            output = plan_study(idea, source=Path(args.source), output=Path(args.workspace), token_cap=args.token_cap,
                                retrieval_strategy=args.retrieval_strategy,
                                orchestration=args.orchestration,
                                dataset_source=Path(args.dataset_workspace) if args.dataset_workspace else None)
        else:
            app = ResearchOpsApp(args.workspace, allowed_roots=permitted)
            output = _execute_command(args, app)
    except Exception as error:
        print(json.dumps({
            "status": "error",
            "error_type": type(error).__name__,
            "error": str(error),
        }, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(output, ensure_ascii=False, indent=2, default=str))
    return 2 if isinstance(output, dict) and output.get("status") == "needs_attention" else 0


def _execute_command(args, app: ResearchOpsApp):
    if args.command == "skills":
        output = app.skills.catalog(RESEARCH_LEAD, app.executor.policy)
    elif args.command == "agent":
        output = app.run_skill_agent(args.task, token_cap=args.token_cap, context_mode=args.context_mode)
    elif args.command == "init":
        output = {"workspace": str(app.workspace), "database": str(app.store.path), "counts": app.store.counts()}
    elif args.command == "ingest":
        output = app.call("ingest_documents", {"path": str(Path(args.path).resolve())})
    elif args.command == "search":
        output = app.call("search_knowledge", {"query": args.query, "limit": args.limit})
    elif args.command == "ask":
        output = app.call("answer_with_evidence", {"question": args.question, "limit": args.limit})
    elif args.command == "audit":
        idea_path = Path(args.idea).resolve()
        idea = json.loads(idea_path.read_text(encoding="utf-8"))
        task_id = "task_adhoc"
        app.store.create_task(task_id, idea["title"], idea["domain"], idea)
        output = app.call("audit_idea", {"task_id": task_id, "idea": idea})
    elif args.command == "online-search":
        output = app.call("search_papers", {"query": args.query, "run_id": "run_cli", "limit_per_source": args.limit})
    elif args.command == "demo":
        output = app.run_seed_workflow(args.idea, args.corpus)
    elif args.command == "research":
        idea_text = _research_idea_text(args.idea, args.idea_file)
        run = app.run_agent if args.legacy_tools else app.run_skill_agent
        kwargs = {} if args.legacy_tools else {"token_cap": args.token_cap}
        output = run(
            idea_text,
            **kwargs,
            research_limits={
                "max_queries": args.max_queries,
                "limit_per_query": args.limit_per_query,
                "max_downloads": args.max_downloads,
            },
        )
    elif args.command == "batch-research":
        output = app.run_batch_research(
            args.idea_id,
            library_path=args.idea_library,
            target_papers=args.target_papers,
            shortlist_size=args.shortlist_size,
            limit_per_query=args.limit_per_query,
            evaluation_size=args.evaluation_size,
        )
    elif args.command == "silver-evaluate":
        output = app.run_silver_evaluation(
            args.idea_id,
            library_path=args.idea_library,
            provisional_path=args.provisional_path,
            batch_report_path=args.batch_report,
            max_papers=args.max_papers,
            max_workers=args.workers,
            max_fulltext_chars=args.max_fulltext_chars,
        )
    elif args.command == "prepare-experiment":
        output = app.prepare_experiment(args.idea_id, library_path=args.idea_library,
                                        workers=args.workers, download=not args.offline, dataset_source=args.dataset_source)
    elif args.command == "coding-plan":
        from researchops.harness.runtime import OpenAICompatibleModel
        model = OpenAICompatibleModel(max_attempts=1, stream=args.stream, reasoning_effort=args.reasoning_effort) if not args.study_result else None
        output = app.create_coding_plan(args.idea_id, from_scratch=args.from_scratch,
            study_result=args.study_result, token_cap=args.token_cap, model=model)
    elif args.command == "code":
        from researchops.harness.runtime import OpenAICompatibleModel
        model = OpenAICompatibleModel(max_attempts=1, stream=args.stream, reasoning_effort=args.reasoning_effort)
        output = app.run_coding(args.idea_id, resume=args.resume, from_scratch=args.from_scratch, plan_path=args.plan,
            study_result=args.study_result, token_cap=args.token_cap, model=model)
    elif args.command in {"datasets-acquire", "dataset-import"}:
        from researchops.experiments.datasets import REGISTRY, DatasetManager
        manager = DatasetManager(app.workspace, source=getattr(args, "dataset_source", "auto"))
        ids = args.dataset_ids if args.command == "datasets-acquire" else [args.dataset_id]
        if any(dataset_id not in REGISTRY for dataset_id in ids):
            raise ValueError(f"Supported dataset IDs: {', '.join(REGISTRY)}")
        output = (manager.acquire_many(ids, workers=args.workers) if args.command == "datasets-acquire"
                  else manager.import_local(args.dataset_id, Path(args.path)))
    elif args.command == "reanalyze":
        output = app.reanalyze_task(
            args.task_id,
            deterministic_plan=args.deterministic_plan,
        )
    elif args.command == "status":
        output = app.store.counts()
    elif args.command == "tools":
        output = [
            {
                "name": spec.name,
                "risk": spec.risk.value,
                "read_only": spec.read_only,
                "idempotent": spec.idempotent,
                "side_effect": spec.side_effect,
                "requires_confirmation": spec.requires_confirmation,
            }
            for spec in app.registry.visible(RESEARCH_LEAD)
        ]
    else:
        raise AssertionError(args.command)
    return output


def _research_idea_text(argument: str | None, idea_file: str | None) -> str:
    if argument and idea_file:
        raise ValueError("Use either the positional Idea or --idea-file, not both")
    if idea_file:
        value = Path(idea_file).resolve().read_text(encoding="utf-8").strip()
    else:
        value = str(argument or "").strip()
    if not value:
        raise ValueError("research requires an Idea argument or --idea-file")
    return value


if __name__ == "__main__":
    raise SystemExit(main())

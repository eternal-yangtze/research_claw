from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

from researchclaw.core.config import load_project_env
from researchclaw.application.workflow import RESEARCH_LEAD, ResearchClawApp


_CJK = re.compile(r"[\u3400-\u9fff]")


def _normalize_rag_query(query: str, mode: str = "auto") -> tuple[str, str]:
    """Translate CJK research queries for the English paper index, with an audited fallback."""
    if mode == "off" or not _CJK.search(query):
        return query, "not_required"
    try:
        from researchclaw.harness.runtime import OpenAICompatibleModel
        turn = OpenAICompatibleModel(timeout=45, max_attempts=2, temperature=0).complete(
            [
                {"role": "system", "content": (
                    "Translate the user's Chinese research query into one concise English academic "
                    "search query. Preserve acronyms, algorithm names, error guarantees, and domain "
                    "terms. Return only the translated query, without explanation or quotation marks."
                )},
                {"role": "user", "content": query},
            ],
            [],
            160,
        )
        translated = " ".join(turn.text.strip().strip('"').split())
        if not translated or _CJK.search(translated) or len(translated) > 500:
            return query, "fallback_invalid_translation"
        return translated, "translated_zh_to_en"
    except Exception:
        return query, "fallback_translation_error"


def _source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _apply_source_exclusions(
    sources: list[tuple[str, Path]], excluded_paths: set[Path],
) -> tuple[list[tuple[str, Path]], list[str]]:
    """Apply review exclusions to exact paths and renamed byte-identical copies."""
    excluded_hashes = {_source_sha256(path) for path in excluded_paths if path.is_file()}
    accepted: list[tuple[str, Path]] = []
    excluded_actual = set(excluded_paths)
    for domain, path in sources:
        if path in excluded_paths or (excluded_hashes and _source_sha256(path) in excluded_hashes):
            excluded_actual.add(path)
        else:
            accepted.append((domain, path))
    return accepted, sorted(map(str, excluded_actual))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="researchclaw", description="Auditable research agent")
    parser.add_argument("--workspace", default="knowledge_base/default", help="PostgreSQL/Qdrant workspace binding, trace, and artifact directory")
    parser.add_argument("--allow-root", action="append", default=[], help="Explicit readable input root for skills; repeat as needed")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init", help="Initialize the persistent store")
    subparsers.add_parser("sync-index", help="Retry durable Qdrant outbox operations without model calls")
    rag_build = subparsers.add_parser("rag-build", help="Build a versioned PDF Parent/Child RAG corpus in PostgreSQL/Qdrant")
    rag_build.add_argument("--corpus-version", default="rag-v1")
    rag_build.add_argument("--source", action="append", required=True,
                           help="Domain/path pair, e.g. time-series=D:\\...\\incoming")
    rag_build.add_argument("--exclude-path", action="append", default=[],
                           help="Exact PDF path to omit by review policy; repeat as needed")
    rag_build.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    rag_build.add_argument("--embedding-revision", default="main")
    rag_build.add_argument("--base-corpus-version",
                           help="Ready revision in the same Library workspace whose unchanged vectors may be reused")
    rag_build.add_argument("--chunk-policy",
                           choices=("page-parent-word-child-v1", "paper-structure-semantic-v2.1"),
                           default="page-parent-word-child-v1",
                           help="Versioned Parent/Child segmentation contract")
    rag_build.add_argument("--batch-size", type=int, default=16)
    rag_build.add_argument("--report")
    rag_search = subparsers.add_parser("rag-search", help="Query a published RAG corpus with dense+BM25 RRF and reranking")
    rag_search.add_argument("query")
    rag_search.add_argument("--corpus-version", default="rag-v1")
    rag_search.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    rag_search.add_argument("--embedding-revision", default="main")
    rag_search.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L6-v2")
    rag_search.add_argument("--reranker-revision", default="233902d25c440f23af6f7d6e94d2946bac0bee0a")
    rag_search.add_argument("--limit", type=int, default=6)
    rag_search.add_argument("--candidate-limit", type=int, default=40)
    rag_search.add_argument("--query-language-mode", choices=("auto", "off"), default="auto",
                            help="Translate Chinese queries for the English paper index")
    rag_answer = subparsers.add_parser('rag-answer', help='Plan, retrieve and generate a cited paper-grounded answer')
    rag_answer.add_argument('question')
    subparsers.add_parser("skills", help="List project skill metadata without loading all instructions")
    agent = subparsers.add_parser("agent", help="Run the research agent through discover/load/invoke skills")
    agent.add_argument("task", help="Natural-language task with necessary input paths")
    agent.add_argument("--token-cap", type=int, default=250000)
    agent.add_argument("--context-mode", choices=("lazy", "eager"), default="lazy")
    chat_agent=subparsers.add_parser('chat-agent',help='Conversational LangGraph research agent; dynamic task DAG, no fixed research workflow')
    chat_agent.add_argument('message')
    chat_agent.add_argument('--token-cap',type=int,default=180000)
    recover_agent=subparsers.add_parser('agent-recover',help='Explicitly recover a CLI research run from PostgreSQL checkpoints')
    recover_agent.add_argument('run_id')
    recover_agent.add_argument('--token-cap',type=int,default=180000)

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

    download = subparsers.add_parser('download-papers', help='Search and stage PDFs for human confirmation; does not parse or index')
    download.add_argument('query')
    download.add_argument('--limit', type=int, default=3)
    download.add_argument('--sources', nargs='+', choices=('arxiv','openalex','openreview','dblp','acl','cvf','crossref'), help='Explicit search sources; omit to search all sources')

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
    study.add_argument("--orchestration", choices=("native", "langgraph"), help="Override RESEARCHCLAW_STUDY_ORCHESTRATOR; new runs default to langgraph")
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

    recheck = subparsers.add_parser('coding-recheck', help='Run a fresh Docker verification of existing source without model generation')
    recheck.add_argument('--plan', required=True)
    recheck.add_argument('--project', required=True)

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
        os.environ["RESEARCHCLAW_PDF_PARSER"] = args.pdf_parser
    permitted = list(args.allow_root)
    if getattr(args, "path", None):
        permitted.append(Path(args.path).resolve())
    if getattr(args, "corpus", None):
        permitted.append(Path(args.corpus).resolve())
    try:
        if args.command == "validate-formulas":
            from researchclaw.papers.validation import validate_formulas
            output = validate_formulas(args.pdf, Path(args.workspace) / "artifacts" / "formula_validation",
                                       docling_json=args.docling_json, source_package=args.source_package,
                                       source_receipt=args.source_receipt)
        elif args.command == "plan-study":
            from researchclaw.application.study import plan_study
            if args.idea_file:
                raw = Path(args.idea_file).read_text(encoding="utf-8")
                idea = json.loads(raw) if raw.lstrip().startswith("{") else {"description": raw}
            else:
                idea = {"description": args.idea or ""}
            output = plan_study(idea, source=Path(args.source), output=Path(args.workspace), token_cap=args.token_cap,
                                retrieval_strategy=args.retrieval_strategy,
                                orchestration=args.orchestration,
                                dataset_source=Path(args.dataset_workspace) if args.dataset_workspace else None)
        elif args.command == "rag-build":
            from researchclaw.knowledge.rag_pipeline import build_corpus
            sources = []
            for item in args.source:
                if "=" not in item:
                    raise ValueError("--source must use domain=path")
                domain, raw_path = item.split("=", 1)
                sources.extend((domain, path) for path in sorted(Path(raw_path).resolve().rglob("*.pdf")))
            excluded = {Path(value).resolve() for value in args.exclude_path}
            sources, excluded_sources = _apply_source_exclusions(sources, excluded)
            if not sources:
                raise ValueError("No PDF files found in --source paths")
            output = build_corpus(args.workspace, sources, corpus_version=args.corpus_version,
                                  model_name=args.embedding_model, model_revision=args.embedding_revision,
                                  batch_size=args.batch_size, report_path=args.report,
                                  excluded_sources=excluded_sources,
                                  base_corpus_version=args.base_corpus_version,
                                  chunk_policy=args.chunk_policy)
            output = output.__dict__
        elif args.command == 'rag-answer':
            from researchclaw.core.qdrant_store import QdrantResearchStore
            from researchclaw.knowledge.rag_retrieval import PublishedRAGRetriever
            from researchclaw.knowledge.retrieval import SentenceTransformerEmbedding, CrossEncoderReranker
            from researchclaw.knowledge.research_rag import ResearchRAG
            manifest = json.loads((Path(args.workspace) / 'rag-search-index.json').read_text(encoding='utf8'))
            model, revision = manifest['embedding_key'].split('@', 1)
            retriever = PublishedRAGRetriever(QdrantResearchStore(args.workspace),
                corpus_version=manifest['corpus_version'], embedding=SentenceTransformerEmbedding(model, revision=revision),
                reranker=CrossEncoderReranker('cross-encoder/ms-marco-MiniLM-L6-v2', revision='233902d25c440f23af6f7d6e94d2946bac0bee0a'))
            output = ResearchRAG(retriever).answer(args.question)
        elif args.command == "rag-search":
            from researchclaw.core.qdrant_store import QdrantResearchStore
            from researchclaw.knowledge.rag_retrieval import PublishedRAGRetriever
            from researchclaw.knowledge.retrieval import SentenceTransformerEmbedding
            from researchclaw.knowledge.retrieval import CrossEncoderReranker
            store = QdrantResearchStore(args.workspace)
            embedding = SentenceTransformerEmbedding(args.embedding_model, revision=args.embedding_revision)
            reranker = CrossEncoderReranker(args.reranker_model, revision=args.reranker_revision)
            retriever = PublishedRAGRetriever(store, corpus_version=args.corpus_version,
                                              embedding=embedding, reranker=reranker)
            retrieval_query, translation_status = _normalize_rag_query(
                args.query, args.query_language_mode)
            output = {"query": args.query, "retrieval_query": retrieval_query,
                      "query_translation": translation_status,
                      "corpus_version": args.corpus_version,
                      "hits": [hit.to_dict() for hit in retriever.search(
                          retrieval_query, limit=args.limit,
                          candidate_limit=args.candidate_limit)]}
            output['lexical_backend'] = retriever.lexical_backend
        else:
            app = ResearchClawApp(args.workspace, allowed_roots=permitted)
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


def _execute_command(args, app: ResearchClawApp):
    if args.command == 'coding-recheck':
        import time
        import uuid
        from researchclaw.coding.sandbox import CodingSandbox, stage_inputs
        from researchclaw.core.artifacts import write_json
        plan = json.loads(Path(args.plan).read_text(encoding='utf-8'))
        directory = app.workspace / 'coding' / ('recheck_' + uuid.uuid4().hex[:12])
        source = Path(args.project).resolve()
        if not source.is_dir():
            raise ValueError('Existing project directory required')
        started = time.monotonic()
        stage_inputs(app.workspace, directory, plan)
        verified = CodingSandbox(app.workspace, source, directory, dataset_root=plan.get('dataset_root')).verify(1)
        checks = verified.get('checks', [])
        output = {'status':'coding_verified' if verified['passed'] else 'needs_attention',
                  'execution_kind':'recheck_existing_source', 'source_project':str(source),
                  'plan_id':plan['plan_id'], 'total_tokens':0, 'model_calls_this_recheck':0,
                  'wall_seconds':time.monotonic()-started, 'verification':verified,
                  'check_count':len(checks), 'passed_checks':sum(bool(c.get('passed')) for c in checks),
                  'research_success':None, 'scope':'Fresh engineering verification of previously generated source; no new generation or full study'}
        output['metric_scope'] = 'independent_metrics uses fixed verifier logits/labels to test metric formulas; it is not DTD/Pets research accuracy'
        write_json(directory / 'report.json', output)
        output['report_path'] = str(directory / 'report.json')
    elif args.command == "skills":
        output = app.skills.catalog(RESEARCH_LEAD, app.executor.policy)
    elif args.command == "agent":
        output = app.run_skill_agent(args.task, token_cap=args.token_cap, context_mode=args.context_mode)
    elif args.command == 'chat-agent':
        output=app.run_conversational_agent(args.message,token_cap=args.token_cap)
    elif args.command == 'agent-recover':
        output=app.run_conversational_agent(None,run_id=args.run_id,token_cap=args.token_cap,resume=True)
    elif args.command == "init":
        output = {"workspace": str(app.workspace), "database": str(app.store.path), "counts": app.store.counts()}
    elif args.command == 'sync-index':
        app.store.flush_outbox()
        output = {'storage': 'postgresql+qdrant', 'operations': app.store.index_status()}
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
    elif args.command == 'download-papers':
        from researchclaw.papers.staging import stage_papers
        def progress(value):
            print(json.dumps({'progress':value}, ensure_ascii=False), file=sys.stderr, flush=True)
        output = stage_papers(app.literature, args.query, app.workspace / 'artifacts' / 'papers',
                              args.limit, args.sources, progress)
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
        from researchclaw.harness.runtime import OpenAICompatibleModel
        model = OpenAICompatibleModel(max_attempts=1, stream=args.stream, reasoning_effort=args.reasoning_effort) if not args.study_result else None
        output = app.create_coding_plan(args.idea_id, from_scratch=args.from_scratch,
            study_result=args.study_result, token_cap=args.token_cap, model=model)
    elif args.command == "code":
        from researchclaw.harness.runtime import OpenAICompatibleModel
        model = OpenAICompatibleModel(max_attempts=1, stream=args.stream, reasoning_effort=args.reasoning_effort)
        output = app.run_coding(args.idea_id, resume=args.resume, from_scratch=args.from_scratch, plan_path=args.plan,
            study_result=args.study_result, token_cap=args.token_cap, model=model)
    elif args.command in {"datasets-acquire", "dataset-import"}:
        from researchclaw.experiments.datasets import REGISTRY, DatasetManager
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
    app.store.require_index_ready()
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

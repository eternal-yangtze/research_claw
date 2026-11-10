"""Least-privilege conversational tools over the original application services."""
from dataclasses import replace
import re
import uuid

from researchclaw.harness.runtime import ToolRegistry, ToolExecutor, ToolPolicy, ToolSpec, ToolRisk
from researchclaw.harness.research_agent import WORKER, SAFE_TOOLS


def search_queries(topic, constraints=None):
    if not topic.strip() or len(topic) > 1000:
        raise ValueError("Use a nonempty topic of at most 1000 characters")
    terms = topic.replace('"', '').replace('\n', ' ').strip()
    constraints = [str(c).replace('"', '')[:150] for c in (constraints or [])][:6]
    text = " ".join([terms, *constraints])
    return {"topic": topic, "queries": {
        "google_scholar": f'{text} (survey OR benchmark OR "error bound")',
        "arxiv": f'all:"{terms}"' + ''.join(f' AND all:"{c}"' for c in constraints),
        "semantic_scholar": text,
        "crossref_openalex": text,
    }, "note": "Suggested expressions only, not retrieved papers. Translate/expand concepts before calling when appropriate; search dates/venues are suggestions, not quality guarantees."}


def build_agent_tools(app, *, request_action=None):
    registry = ToolRegistry()
    from researchclaw.application.workflow import RESEARCH_LEAD
    from researchclaw.application.paper_library import PaperLibrary
    library=PaperLibrary(app)
    def chat_search(query, run_id, limit_per_source=3, sources=None):
        result=app.literature.search(query,run_id,limit_per_source,
            sources=sources or ['arxiv','openalex','crossref'])
        # Scientific abstracts and duplicated identity/source records are stored
        # by the search service, but overwhelm a conversational metadata answer.
        keys=('paper_id','title','authors','year','doi','source','source_url','pdf_url','source_version')
        return {'papers':[{k:p.get(k) for k in keys} for p in result['papers'][:12]],
            'sources':result['sources'],'total_matches':len(result['papers']),
            'truncated':len(result['papers'])>12,
            'note':'Public metadata only, not full-text evidence. Per-channel status is included. No PDF downloaded or published. Full metadata remains in the project records.'}
    # Preserve original roles/registry; only the new run gets this worker facade.
    for spec in app.registry.visible(replace(RESEARCH_LEAD, allowed_tools=SAFE_TOOLS)):
        if spec.name not in {"search_skills", "load_skill", "invoke_skill"}:
            handler={'search_knowledge':library.search,'get_chunk':library.get_chunk,
                     'answer_with_evidence':library.evidence,'retrieve_evidence':library.evidence}.get(spec.name,spec.handler)
            spec=replace(spec,handler=handler)
            if spec.name == 'search_knowledge':
                spec=replace(spec,description='Search published paper evidence using Qdrant dense+BM25/RRF/rerank. For questions about ONLY one known paper, pass its observed paper_id to constrain BOTH retrieval lanes; use English queries.',
                    input_schema={'type':'object','properties':{'query':{'type':'string'},'limit':{'type':'integer','minimum':1,'maximum':10},
                        'paper_id':{'type':'string','description':'Optional observed paper ID; restrict search to this paper.'}},
                        'required':['query'],'additionalProperties':False})
            if spec.name == 'search_papers':
                # Chat must not scan the ACL catalogue and multiple CVF years for
                # every time-series query. Keep specialist connectors opt-in.
                spec=replace(spec, description='Search public academic metadata. Default arxiv/openalex/crossref; select specialized sources only for relevant venues. Reports per-source failures; does not download PDFs.',
                    input_schema={'type':'object','properties':{'query':{'type':'string','maxLength':1000},
                        'run_id':{'type':'string'},'limit_per_source':{'type':'integer','minimum':1,'maximum':5},
                        'sources':{'type':'array','minItems':1,'maxItems':4,'uniqueItems':True,
                            'items':{'type':'string','enum':['arxiv','openalex','crossref','dblp','openreview','acl','cvf']}}},
                        'required':['query','run_id'],'additionalProperties':False},
                    handler=chat_search)
            registry.register(replace(spec, allowed_agents=frozenset({WORKER})))
    props = {"type": "object", "properties": {"topic": {"type": "string"},
             "constraints": {"type": "array", "items": {"type": "string"}, "maxItems": 6}},
             "required": ["topic"], "additionalProperties": False}
    registry.register(ToolSpec("build_search_queries", "Build channel-specific paper search expressions; no network or downloads. Pass English semantic expansions when researching English papers.",
        props, search_queries, allowed_agents=frozenset({WORKER}), risk=ToolRisk.LOW))
    registry.register(ToolSpec('get_paper_references',
        'Read the original bibliography of a known local paper to recommend specific expansion papers. Search for paper_id first. Bibliography is not evidence of the cited work contents.',
        {'type': 'object', 'properties': {'paper_id': {'type': 'string'}}, 'required': ['paper_id'], 'additionalProperties': False},
        library.references, allowed_agents=frozenset({WORKER}), risk=ToolRisk.LOW))
    registry.register(ToolSpec('inspect_paper_library',
        'Inspect published paper counts and staged PDF paths for the current project. Counts alone do not establish scientific coverage. Use before proposing uploaded-paper ingestion.',
        {'type': 'object', 'properties': {}, 'additionalProperties': False},
        library.status, allowed_agents=frozenset({WORKER}), risk=ToolRisk.LOW))

    def mcp(chunk_id, metadata=False, _context=None):
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,160}", chunk_id):
            raise ValueError("Invalid chunk ID")
        store,row=library.locate(chunk_id)
        from researchclaw.integrations.paper_mcp import PaperScope, PaperMCPClient
        scope = PaperScope(store.root, store.root / "runtime-binding.json", row["paper_id"], [chunk_id])
        client = PaperMCPClient(scope, app.artifact_root / "mcp" / uuid.uuid4().hex)
        timeout = min(30, _context.remaining_seconds() or 30)
        return (client.metadata if metadata else lambda **kw: client.read(chunk_id, **kw))(
            timeout=timeout, cancellation=_context.cancellation)

    schema = {"type": "object", "properties": {"chunk_id": {"type": "string"}}, "required": ["chunk_id"], "additionalProperties": False}
    for name, metadata in (("mcp_read_chunk", False), ("mcp_paper_metadata", True)):
        registry.register(ToolSpec(name,
            "Read original paper text through real MCP stdio. Search for the chunk ID first." if not metadata else
            "Read paper title and bound chunk metadata through real MCP stdio; does NOT read scientific evidence.",
            schema, lambda chunk_id, _context, m=metadata: mcp(chunk_id, m, _context),
            contextual=True, allowed_agents=frozenset({WORKER}), risk=ToolRisk.LOW, timeout_seconds=40))
    if request_action:
        registry.register(ToolSpec("request_action", "Prepare a high-risk download_pdf, ingest_documents or implement_experiment proposal. NEVER executes. User must confirm through the authenticated action endpoint, not chat text or Reviewer PASS. implement_experiment requires an OBSERVED idea_id for an existing prepared, frozen experiment contract; a conversational draft is NOT that contract. Without one, explain missing inputs; do not invent an ID or submit a malformed proposal. download_pdf arguments require url; ingest_documents requires an observed uploaded path.",
            {"type": "object", "properties": {"action": {"type": "string", "enum": ["download_pdf", "ingest_documents", "implement_experiment"]},
             "arguments": {"type": "object"}, "reason": {"type": "string"}}, "required": ["action", "arguments", "reason"], "additionalProperties": False},
            request_action, contextual=True, read_only=False, side_effect=True, risk=ToolRisk.NORMAL,
            allowed_agents=frozenset({WORKER})))
    from researchclaw.harness.skills import SkillRuntime
    skills = SkillRuntime(app.skills.loader.root, {"evidence-retrieve": "retrieve_evidence"}, registry, app.traces, app.artifact_root)
    skills.run_artifact_root = app.skills.run_artifact_root
    skills.register(WORKER)
    executor = ToolExecutor(registry, ToolPolicy(registry), app.traces)
    return registry, executor

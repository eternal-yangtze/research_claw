"""Read-only paper boundary audit and checksum manifest for Agent evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT=next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path.insert(0,str(ROOT/'src'))
from researchclaw.core.artifacts import write_json,file_hash
from researchclaw.core.qdrant_store import build_research_store


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    out=args.output.resolve()
    if not out.is_relative_to(ROOT/'self-revise/backend-revise/agent-evidence'):
        raise ValueError('Use an Agent evidence subdirectory')
    root=ROOT/'knowledge_base/rag_corpora/time_series_domain_release_v3_1'
    store=build_research_store(root)
    state=json.loads((root/'rag-search-index.json').read_text(encoding='utf8'))
    with store.database.connect(read_only=True) as db:
        papers=[dict(r) for r in db.execute('SELECT paper_id,content_hash FROM papers ORDER BY paper_id')]
        children=db.execute("SELECT count(*) FROM chunks WHERE node_type='child'").fetchone()[0]
        copies=db.execute('SELECT count(*) FROM chunk_embeddings').fetchone()[0]
    storage=dict(papers=len(papers),paper_hash_digest=hashlib.sha256(json.dumps(papers,sort_keys=True).encode()).hexdigest(),
        children=children,pg_embedding_copies=copies,dense_points=store.client.count(state['dense_collection'],exact=True).count,
        sparse_points=store.client.count(state['sparse_collection'],exact=True).count,
        index_manifest_sha256=file_hash(root/'rag-search-index.json'),corpus_version=state['corpus_version'],
        environment_sha256=file_hash(ROOT/'self-revise/环境.md'))
    previous=json.loads((ROOT/'self-revise/backend-revise/rag-evidence/expanded-rag-v20261109/boundary-after.json').read_text(encoding='utf8'))['storage']
    write_json(out/'paper-boundary.json',dict(storage=storage,unchanged_since_rag_acceptance=storage==previous))
    sources=['src/researchclaw/harness/research_agent.py','src/researchclaw/application/agent_tools.py',
        'src/researchclaw/application/paper_library.py','src/researchclaw/integrations/paper_mcp.py',
        'src/researchclaw/harness/runtime.py','src/researchclaw/harness/context.py',
        'backend/src/researchclaw_api/actions.py','backend/src/researchclaw_api/main.py',
        'backend/src/researchclaw_api/conversation.py','self-revise/backend-revise/scripts/evaluate_research_harness.py',
        'self-revise/backend-revise/scripts/verify_research_harness_api.py','tests/harness/test_research_agent.py','tests/api/test_agent_actions.py']
    sources+=['src/researchclaw/application/workflow.py','src/researchclaw/application/cli.py',
        'self-revise/backend-revise/scripts/report_research_harness.py','self-revise/backend-revise/scripts/audit_research_harness.py',
        'self-revise/backend-revise/scripts/verify_research_followup.py','self-revise/backend-revise/scripts/verify_research_skill_live.py']
    sources+=['backend/src/researchclaw_api/store.py','tests/api/test_agent_recovery.py',
        'self-revise/backend-revise/scripts/report_agent_architecture.py','src/researchclaw/harness/single_agent.py',
        'tests/harness/test_single_agent.py','self-revise/backend-revise/scripts/evaluate_langgraph_agent.py',
        'tests/api/test_conversation_memory.py','self-revise/backend-revise/scripts/verify_langgraph_dag_live.py']
    for name in sources:
        target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    manifest={str(path.relative_to(out)).replace('\\','/'):dict(sha256=file_hash(path),bytes=path.stat().st_size)
              for path in out.rglob('*') if path.is_file() and path.name!='evidence-manifest.json'}
    write_json(out/'evidence-manifest.json',dict(count=len(manifest),files=manifest))
    print(json.dumps(dict(storage=storage,unchanged=storage==previous,evidence_files=len(manifest)),ensure_ascii=False))


if __name__=='__main__': main()

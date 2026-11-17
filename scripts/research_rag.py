"""User-facing RAG commands; experiment runners live under self-revise."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from researchclaw.core.config import load_project_env
from researchclaw.core.qdrant_store import QdrantResearchStore
from researchclaw.knowledge.rag_pipeline import build_corpus, sha256, SUPPORTED_CHUNK_POLICIES, PAGE_CHUNK_POLICY, TOKEN_CHUNK_POLICY, LAYOUT_CHUNK_POLICY
from researchclaw.knowledge.rag_retrieval import PublishedRAGRetriever
from researchclaw.knowledge.neural_models import retrieval_models
from researchclaw.knowledge.research_rag import ResearchRAG, build_structure_cache

WORKSPACE = ROOT / 'knowledge_base/rag_corpora/time_series_domain_release_v3_1'


def engine():
    state = json.loads((WORKSPACE / 'rag-search-index.json').read_text(encoding='utf8'))
    name, revision = state['embedding_key'].split('@', 1)
    embedding,reranker=retrieval_models(name,revision)
    return PublishedRAGRetriever(QdrantResearchStore(WORKSPACE, recover=False),
        corpus_version=state['corpus_version'],
        embedding=embedding,reranker=reranker)


def main():
    global WORKSPACE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['ask', 'ask-fullcorpus', 'ingest', 'noop', 'structure-build'])
    parser.add_argument('question', nargs='?')
    parser.add_argument('--source', type=Path)
    parser.add_argument('--corpus-version')
    parser.add_argument('--workspace',type=Path,default=WORKSPACE)
    parser.add_argument('--chunk-policy',choices=SUPPORTED_CHUNK_POLICIES,default=PAGE_CHUNK_POLICY)
    parser.add_argument('--run-id');parser.add_argument('--training-run',default='fullcorpus-gpt-r2-s42')
    parser.add_argument('--retrieve-only',action='store_true')
    parser.add_argument('--read-result',action='store_true',help='Read a saved full-corpus result without GPU or model calls')
    args = parser.parse_args()
    if args.command=='ask-fullcorpus':
        if not args.run_id:parser.error('ask-fullcorpus requires --run-id')
        if args.read_result:
            import re
            if not re.fullmatch(r'[A-Za-z0-9_-]{1,60}',args.run_id):parser.error('Simple run ID required')
            result=json.loads((ROOT/'self-revisev0.5/artifacts/p3'/('fullcorpus-user-'+args.run_id+'-answer.json')).read_text(encoding='utf8'))
        else:
            if not args.question:parser.error('ask-fullcorpus requires question unless --read-result')
            from researchclaw.knowledge.fullcorpus_client import answer_fullcorpus
            result=answer_fullcorpus(ROOT,args.question,run=args.run_id,training=args.training_run,retrieve_only=args.retrieve_only)
        # ASCII JSON round-trips all Unicode through Windows GBK pipes; the
        # persisted answer/evidence stays UTF-8 and is never regenerated to print.
        print(json.dumps(result,ensure_ascii=True,indent=2,default=str));return
    WORKSPACE = args.workspace.resolve()
    load_project_env()
    if args.command == 'ingest' and args.chunk_policy in (TOKEN_CHUNK_POLICY, LAYOUT_CHUNK_POLICY):
        if not WORKSPACE.is_relative_to((ROOT/'self-revisev0.5').resolve()):
            parser.error('Token-safe ingest requires an isolated self-revisev0.5 workspace')
        if (WORKSPACE/'rag-search-index.json').exists():
            parser.error('Token-safe ingest requires a fresh workspace and release')
        if os.getenv('RESEARCHCLAW_NEURAL_BACKEND','gpu_ssh') not in ('gpu_ssh','local_cuda','local_cpu'):
            parser.error('Token-safe ingest requires gpu_ssh, local_cuda or explicit local_cpu')
    os.environ.update(HF_HUB_OFFLINE='1', RESEARCHCLAW_EMBEDDING_LOCAL_ONLY='true',
                      RESEARCHCLAW_RERANKER_LOCAL_ONLY='true')
    import torch
    torch.set_num_threads(4)
    if args.command == 'ask':
        if not args.question:
            parser.error('ask requires a question')
        result = ResearchRAG(engine()).answer(args.question)
    elif args.command == 'structure-build':
        result = build_structure_cache(QdrantResearchStore(WORKSPACE, recover=False))
    else:
        manifest = WORKSPACE / 'rag-search-index.json'
        if manifest.exists():
            state = json.loads(manifest.read_text(encoding='utf8'))
        elif args.command == 'ingest':
            state = {'embedding_key':'BAAI/bge-small-en-v1.5@5c38ec7c405ec4b44b94cc5a9bb96e735b38267a',
                     'corpus_version':'unpublished-initial'}
        else:
            parser.error('No published corpus in this workspace')
        name, revision = state['embedding_key'].split('@', 1)
        if args.command == 'ingest':
            if not args.source or not args.source.is_dir() or not args.corpus_version:
                parser.error('ingest requires --source DIRECTORY --corpus-version NEW_VERSION')
            if args.corpus_version == state['corpus_version']:
                parser.error('Use a new corpus revision for new input')
            files = sorted(args.source.rglob('*.pdf'))
            if not files:
                parser.error('No PDF found')
            store = QdrantResearchStore(WORKSPACE, recover=False)
            with store.database.connect(read_only=True) as db:
                hashes = {r[0] for r in db.execute('SELECT content_hash FROM papers')}
            if len(hashes | {sha256(p) for p in files}) > 2000:
                parser.error('Library hard limit is 2000 unique PDFs')
            version = args.corpus_version
        else:
            # No-op validates the current release without assuming a historical
            # append experiment folder is still present on the user's machine.
            files = []
            version = state['corpus_version']
        import hashlib
        report = WORKSPACE / 'artifacts' / ('incremental-' + hashlib.sha256(version.encode()).hexdigest()[:12] + '.json')
        built = build_corpus(WORKSPACE, [('time-series', p) for p in files],
            corpus_version=version, base_corpus_version=state['corpus_version'],
            model_name=name, model_revision=revision, report_path=report, chunk_policy=args.chunk_policy)
        result = {'build': built.__dict__}
        if args.command == 'ingest':
            result['structure'] = build_structure_cache(store)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == '__main__':
    main()

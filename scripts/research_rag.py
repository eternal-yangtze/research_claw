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
from researchclaw.knowledge.rag_pipeline import build_corpus, sha256
from researchclaw.knowledge.rag_retrieval import PublishedRAGRetriever
from researchclaw.knowledge.retrieval import SentenceTransformerEmbedding, CrossEncoderReranker
from researchclaw.knowledge.research_rag import ResearchRAG, build_structure_cache

WORKSPACE = ROOT / 'knowledge_base/rag_corpora/time_series_domain_release_v3_1'


def engine():
    state = json.loads((WORKSPACE / 'rag-search-index.json').read_text(encoding='utf8'))
    name, revision = state['embedding_key'].split('@', 1)
    return PublishedRAGRetriever(QdrantResearchStore(WORKSPACE, recover=False),
        corpus_version=state['corpus_version'],
        embedding=SentenceTransformerEmbedding(name, revision=revision),
        reranker=CrossEncoderReranker('cross-encoder/ms-marco-MiniLM-L6-v2',
            revision='233902d25c440f23af6f7d6e94d2946bac0bee0a'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['ask', 'ingest', 'noop', 'structure-build'])
    parser.add_argument('question', nargs='?')
    parser.add_argument('--source', type=Path)
    parser.add_argument('--corpus-version')
    args = parser.parse_args()
    load_project_env()
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
        state = json.loads((WORKSPACE / 'rag-search-index.json').read_text(encoding='utf8'))
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
            model_name=name, model_revision=revision, report_path=report)
        result = {'build': built.__dict__}
        if args.command == 'ingest':
            result['structure'] = build_structure_cache(store)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == '__main__':
    main()

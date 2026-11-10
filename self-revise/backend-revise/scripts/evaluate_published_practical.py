"""Verify the actual published rag-search engine against the frozen practical suite."""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'self-revise/backend-revise/scripts')]
from researchclaw.core.qdrant_store import QdrantResearchStore
from researchclaw.knowledge.retrieval import SentenceTransformerEmbedding,CrossEncoderReranker
from researchclaw.knowledge.rag_retrieval import PublishedRAGRetriever
from evaluate_practical_rag import metrics


def main():
    out=ROOT/'self-revise/backend-revise/rag-evidence/practical-retrieval-v1'
    suite=json.loads((out/'frozen-suite.json').read_text(encoding='utf8'))
    workspace=ROOT/'knowledge_base/rag_corpora/time_series_domain_release_v3_1'
    store=QdrantResearchStore(workspace,recover=False)
    with store.database.connect(read_only=True) as db:
        state=dict(db.execute("SELECT * FROM vector_sync_state WHERE status='ready' ORDER BY updated_at DESC LIMIT 1").fetchone())
        before={'papers':db.execute('SELECT count(*) FROM papers').fetchone()[0],
                'children':db.execute("SELECT count(*) FROM chunks WHERE node_type='child'").fetchone()[0],
                'postgres_embedding_copies':db.execute('SELECT count(*) FROM chunk_embeddings').fetchone()[0]}
    name,revision=state['embedding_key'].split('@',1)
    engine=PublishedRAGRetriever(store,corpus_version=state['corpus_version'],
         embedding=SentenceTransformerEmbedding(name,revision=revision),
         reranker=CrossEncoderReranker('cross-encoder/ms-marco-MiniLM-L6-v2',revision='233902d25c440f23af6f7d6e94d2946bac0bee0a'))
    cases=[];hits={}
    for q in suite['queries']:
        started=time.perf_counter();result=engine.search(q['query'],limit=10,candidate_limit=40)
        ranked=[h.paper_id for h in result]
        cases.append({'id':q['id'],'group':q['group'],'query':q['query'],
                      'ranked_paper_ids':ranked,**metrics(ranked,set(q['relevant_paper_ids']),5),
                      **metrics(ranked,set(q['relevant_paper_ids']),10),'seconds':time.perf_counter()-started})
        hits[q['id']]=[h.to_dict() for h in result]
        print(json.dumps({'id':q['id'],'recall@10':cases[-1]['recall@10'],'mrr@5':cases[-1]['mrr@5']}),flush=True)
    groups={g:{k:sum(c[k] for c in cases if g=='all' or c['group']==g)/sum(g=='all' or c['group']==g for c in cases)
             for k in ['recall@5','mrr@5','ndcg@5','recall@10','mrr@10','ndcg@10','seconds']}
            for g in ['all','legacy','structure_pilot']}
    report={'engine':'PublishedRAGRetriever, actual rag-search entrypoint',
            'lexical_backend':engine.lexical_backend,
            'corpus_version':state['corpus_version'],'embedding_key':state['embedding_key'],
            'workspace':str(workspace),'inventory':before,'groups':groups,'cases':cases,
            'suite_sha256':hashlib.sha256((out/'frozen-suite.json').read_bytes()).hexdigest(),
            'formal_chunking_unchanged':True}
    report['missing_page_hits']=sum(h['page'] is None for rows in hits.values() for h in rows)
    prefix='published-runtime-bm25' if engine.lexical_backend=='qdrant_bm25' else 'published-runtime'
    (out/(prefix+'-report.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    (out/(prefix+'-hits.json')).write_text(json.dumps(hits,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(groups,indent=2))


if __name__=='__main__':main()

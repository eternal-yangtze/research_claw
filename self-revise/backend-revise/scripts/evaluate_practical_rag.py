"""194-paper practical retrieval A/B. Reads PG; writes isolated Qdrant experiments.

No published aliases, formal corpus metadata or source PDFs are changed.
All baseline vectors come from the ready Qdrant collection, never PostgreSQL.
"""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid

ROOT=next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path.insert(0,str(ROOT/'src'))
from qdrant_client import QdrantClient,models
from researchclaw.core.postgres import PostgresDatabase
from researchclaw.core.config import load_project_env
from researchclaw.knowledge.retrieval import SentenceTransformerEmbedding,CrossEncoderReranker,tokenize
from researchclaw.knowledge.conservative_chunks import pack_structured_lines


def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')


def metrics(ranked,relevant,k):
    flags=[int(p in relevant) for p in ranked[:k]]
    ideal=sum(1/math.log2(i+1) for i in range(1,min(k,len(relevant))+1))
    return {f'recall@{k}':len(set(ranked[:k])&relevant)/max(1,len(relevant)),
            f'mrr@{k}':next((1/i for i,f in enumerate(flags,1) if f),0),
            f'ndcg@{k}':sum(f/math.log2(i+1) for i,f in enumerate(flags,1))/max(.001,ideal)}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=ROOT/'self-revise/backend-revise/rag-evidence/practical-retrieval-v1')
    args=ap.parse_args();out=args.output;out.mkdir(exist_ok=True)
    load_project_env()
    os.environ['RESEARCHCLAW_EMBEDDING_LOCAL_ONLY']='true'
    os.environ['RESEARCHCLAW_RERANKER_LOCAL_ONLY']='true'
    workspace=ROOT/'knowledge_base/rag_corpora/time_series_domain_release_v3_1'
    binding=json.loads((workspace/'knowledge-storage.json').read_text(encoding='utf8'))
    database=PostgresDatabase(workspace,identity=binding['collection'])
    with database.connect(read_only=True) as db:
        papers={p['paper_id']:dict(p) for p in db.execute('SELECT * FROM papers')}
        rows=[dict(r) for r in db.execute("SELECT * FROM chunks WHERE node_type='child' ORDER BY chunk_id")]
        parents={r['chunk_id']:dict(r) for r in db.execute("SELECT * FROM chunks WHERE node_type='parent'")}
        state=dict(db.execute("SELECT * FROM vector_sync_state WHERE status='ready' ORDER BY updated_at DESC LIMIT 1").fetchone())
    suite=json.loads((ROOT/'config/evaluation/time_series_rag_pilot_v1.json').read_text(encoding='utf8'))
    suite['queries']=[dict(q,group='legacy') for q in suite['queries']]
    targeted=[
      ('decimal_xor','How can a lossless compressor exploit common decimal prefixes and suffixes while handling binary decimal conversion errors?',['rag_03142434ee0f435c1a8ad1e3']),
      ('erase_restore','How can erasing insignificant floating point bits remain lossless through restoration?',['rag_b75410810379d8c4020b2366']),
      ('approx_error','How are deterministic error guarantees propagated for aggregate queries on compressed time series?',['rag_3433299cf41fc43c33929e5b']),
      ('trajectory_metrics','How do perpendicular, synchronous Euclidean and direction aware distance bounds affect trajectory simplification?',['rag_9002f57da0035e41b26d7a2c']),
      ('dtw_pruning','How can exact dynamic time warping similarity search prune unpromising warping paths?',['rag_45a793a931a531290a34600a']),
      ('bipartite_bound','How does a bipartite graph formulation produce tighter lower bounds for elastic similarity measures?',['rag_3dd91b8064b33d0e70fdee61']),
      ('conditional_index','What conditional lower bounds limit preprocessing and query time for exact elastic subsequence indexes?',['rag_18b17d96e3a8cbeaf0501bfb']),
      ('normalization_index','Why does window normalization make subsequence matching difficult to index and how can cascading filters help?',['rag_db3fe188e86aeac7bceb2bf2'])]
    suite['queries'] += [dict(id=i,query=q,relevant_paper_ids=p,group='structure_pilot') for i,q,p in targeted]
    suite['label_policy']='Legacy six unchanged; eight targeted questions frozen before retrieval. Paper-level topic labels, incomplete relevance judgments; no independent chunk gold.'
    assert all(set(q['relevant_paper_ids'])<=set(papers) for q in suite['queries'])
    suite_path=out/'frozen-suite.json'
    if suite_path.exists():
        assert json.loads(suite_path.read_text(encoding='utf8'))==suite,'Frozen suite changed'
    else:dump(suite_path,suite)
    model,revision=state['embedding_key'].split('@',1)
    embedding=SentenceTransformerEmbedding(model,revision=revision)
    reranker=CrossEncoderReranker('cross-encoder/ms-marco-MiniLM-L6-v2',revision='233902d25c440f23af6f7d6e94d2946bac0bee0a')
    client=QdrantClient(url=os.getenv('RESEARCHCLAW_QDRANT_URL','http://127.0.0.1:6333'),timeout=60)
    vectors={};offset=None
    while True:
        points,offset=client.scroll(state['version_collection'],limit=256,offset=offset,with_vectors=True,with_payload=True)
        for p in points:
            if (p.payload or {}).get('chunk_id'):
                vectors[p.payload['chunk_id']]=p.vector
        if offset is None:break
    assert all(r['chunk_id'] in vectors for r in rows)
    base=[dict(r,title=papers[r['paper_id']]['title']) for r in rows]
    annotation_dir=ROOT/'self-revise/backend-revise/rag-evidence/typed-structure-pilot'
    replacements=[];affected=set();packing=[]
    for path in sorted(annotation_dir.glob('rag_*.json')):
        pid=path.stem
        if pid not in papers:continue
        d=json.loads(path.read_text(encoding='utf8'))
        children=pack_structured_lines(d['lines'])
        doc=json.loads((ROOT/'self-revise/backend-revise/rag-evidence/grobid-phase2-sample/schema'/path.name).read_text(encoding='utf8'))
        titles={s['section_id']:s['section_title'] for s in doc['sections']}
        affected.add(pid)
        for n,c in enumerate(children):
            parent=next(p for p in parents.values() if p['paper_id']==pid and p['page']==c['page'])
            cid=hashlib.sha256(f"structure-v3:{pid}:{c['page']}:{n}:{c['text']}".encode()).hexdigest()
            replacements.append(dict(c,chunk_id=cid,paper_id=pid,title=papers[pid]['title'],
                                     section=titles.get(c['section_id'],f"Page {c['page']}"),parent_chunk_id=parent['chunk_id']))
        packing.append({'paper_id':pid,'old_children':sum(r['paper_id']==pid for r in base),
                        'new_children':len(children),'known_section_children':sum(c['section_id'] is not None for c in children),
                        'source_retention':1.0})
    # Only changed structure children are encoded. Existing vectors stay in Qdrant.
    started=time.perf_counter()
    print(json.dumps({'phase':'encoding','papers':len(papers),'changed_papers':len(affected),'new_children':len(replacements)}),flush=True)
    encoded=embedding.encode([f"{r['section']}\n{r['text']}" for r in replacements])
    vectors.update({r['chunk_id']:list(v) for r,v in zip(replacements,encoded)})
    encoding_seconds=time.perf_counter()-started
    structured=[r for r in base if r['paper_id'] not in affected]+replacements
    queries=suite['queries']
    query_vectors=embedding.encode([q['query'] for q in queries])
    reports={};all_hits={};collection_manifest=[]
    for variant,docs in [('baseline',base),('structure',structured)]:
        lookup={r['chunk_id']:r for r in docs}
        document_terms=[Counter(tokenize(f"{r['section']} {r['text']}")) for r in docs]
        df=Counter(t for counts in document_terms for t in counts)
        vocabulary={t:i for i,t in enumerate(sorted(df))}
        average=sum(sum(c.values()) for c in document_terms)/len(docs)
        signature=hashlib.sha256(json.dumps([variant,state['version_collection'],sorted(lookup)],ensure_ascii=True).encode()).hexdigest()[:12]
        collection=f'researchclaw_exp_practical_{variant}_{signature}'
        if not client.collection_exists(collection):
            client.create_collection(collection,vectors_config={'dense':models.VectorParams(size=len(query_vectors[0]),distance=models.Distance.COSINE)},
                                     sparse_vectors_config={'bm25':models.SparseVectorParams()})
        for start in range(0,len(docs),128):
            points=[]
            for i in range(start,min(len(docs),start+128)):
                r=docs[i];counts=document_terms[i];length=sum(counts.values())
                terms=sorted(counts)
                values=[math.log(1+(len(docs)-df[t]+.5)/(df[t]+.5))*counts[t]*2.5/(counts[t]+1.5*(.25+.75*length/max(1,average))) for t in terms]
                points.append(models.PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL,collection+':'+r['chunk_id'])),
                  vector={'dense':vectors[r['chunk_id']],'bm25':models.SparseVector(indices=[vocabulary[t] for t in terms],values=values)},
                  payload={'chunk_id':r['chunk_id'],'paper_id':r['paper_id'],'page':r['page'],'experiment':True}))
            client.upsert(collection,points=points,wait=True)
        collection_manifest.append({'variant':variant,'collection':collection,'points':len(docs)})
        print(json.dumps({'phase':'retrieval','variant':variant,'children':len(docs)}),flush=True)
        cases=[]
        for q,vector in zip(queries,query_vectors):
            t0=time.perf_counter()
            dense=[p.payload['chunk_id'] for p in client.query_points(collection,query=list(vector),using='dense',limit=40,with_payload=True).points]
            terms=sorted(set(tokenize(q['query']))&set(vocabulary))
            sparse=[p.payload['chunk_id'] for p in client.query_points(collection,query=models.SparseVector(indices=[vocabulary[t] for t in terms],values=[1.]*len(terms)),using='bm25',limit=40,with_payload=True).points] if terms else []
            dr={cid:i for i,cid in enumerate(dense,1)};sr={cid:i for i,cid in enumerate(sparse,1)}
            scores={cid:(1/(60+dr[cid]) if cid in dr else 0)+(1/(60+sr[cid]) if cid in sr else 0) for cid in set(dr)|set(sr)}
            fused=sorted(scores,key=lambda cid:(-scores[cid],cid))[:40]
            reranked=reranker.score(q['query'],[f"{lookup[cid]['section']}\n{lookup[cid]['text']}" for cid in fused])
            order=sorted(zip(fused,reranked),key=lambda pair:(-pair[1],-scores[pair[0]],pair[0]))
            hits=[];seen=set()
            for cid,score in order:
                r=lookup[cid]
                if r['paper_id'] in seen:continue
                seen.add(r['paper_id'])
                hits.append(dict(chunk_id=cid,paper_id=r['paper_id'],title=r['title'],page=r['page'],
                                 section=r['section'],text=r['text'],parent_text=parents[r['parent_chunk_id']]['text'],
                                 score=float(score),dense_rank=dr.get(cid),bm25_rank=sr.get(cid)))
                if len(hits)==10:break
            ranking=[h['paper_id'] for h in hits]
            cases.append({'id':q['id'],'group':q['group'],'query':q['query'],'ranked_paper_ids':ranking,
                          **metrics(ranking,set(q['relevant_paper_ids']),5),**metrics(ranking,set(q['relevant_paper_ids']),10),
                          'seconds':time.perf_counter()-t0})
            all_hits[f'{variant}:{q["id"]}']=hits
            print(json.dumps({'variant':variant,'query':q['id'],'mrr@5':cases[-1]['mrr@5']},ensure_ascii=True),flush=True)
        reports[variant]={'groups':{g:{key:sum(c[key] for c in cases if g=='all' or c['group']==g)/sum(g=='all' or c['group']==g for c in cases)
                              for key in ['recall@5','mrr@5','ndcg@5','recall@10','mrr@10','ndcg@10','seconds']}
                              for g in ['all','legacy','structure_pilot']},'cases':cases}
        dump(out/'partial-results.json',reports)
    report={'papers':len(papers),'base_children':len(base),'structured_children':len(structured),
            'changed_papers':len(affected),'packing':packing,'embedding_model':state['embedding_key'],
            'reranker_model':reranker.model_name,'rrf_k':60,'candidate_limit':40,
            'sparse':'Qdrant sparse BM25, k1=1.5 b=.75; no vector in PostgreSQL',
            'encoding_seconds':encoding_seconds,'vectors_encoded':len(replacements),
            'suite_sha256':hashlib.sha256(suite_path.read_bytes()).hexdigest(),
            'formal_collection_unchanged':state['version_collection'],'experimental_collections':collection_manifest,
            'reports':reports,'formal_migration':False}
    dump(out/'report.json',report);dump(out/'retrieved-evidence.json',all_hits)
    print(json.dumps({v:r['groups'] for v,r in reports.items()},ensure_ascii=True,indent=2))


if __name__=='__main__':main()

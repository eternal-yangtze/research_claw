"""Qdrant BM25 side index; PostgreSQL supplies authoritative text, not ranking.

Rebuilding sparse corpus statistics never parses PDFs or encodes dense vectors.
Unchanged content is a no-op. Publish the local vocabulary manifest last.
"""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time
import uuid

from qdrant_client import models
from researchclaw.knowledge.retrieval import tokenize


def build_lexical_index(store, *, corpus_version, embedding_key):
    started=time.perf_counter()
    with store.database.connect(read_only=True) as db:
        state=db.execute("SELECT * FROM vector_sync_state WHERE project_id=? AND corpus_version=? AND embedding_key=? AND status='ready' ORDER BY updated_at DESC LIMIT 1",
                         (store.project_id,corpus_version,embedding_key)).fetchone()
        if not state:
            raise RuntimeError('Dense revision must be ready before building BM25')
        # This workspace has one current text authority. Historical vector
        # revisions are not a source of historical document snapshots.
        rows=[dict(r) for r in db.execute("SELECT chunk_id,paper_id,page,section,text,content_hash FROM chunks WHERE node_type='child' ORDER BY chunk_id")]
    digest=hashlib.sha256(json.dumps([(r['chunk_id'],r['content_hash'],r['section']) for r in rows],ensure_ascii=False).encode()).hexdigest()
    target=Path(store.root)/'rag-search-index.json'
    key={'schema':1,'project_id':store.project_id,'corpus_version':corpus_version,
         'embedding_key':embedding_key,'dense_collection':state['version_collection'],
         'chunk_digest':digest,'document_count':len(rows),'tokenizer':'researchclaw-tokenize-v1','k1':1.5,'b':.75}
    if target.exists():
        previous=json.loads(target.read_text(encoding='utf8'))
        if all(previous.get(k)==v for k,v in key.items()) and store.client.collection_exists(previous['sparse_collection']):
            if store.client.count(previous['sparse_collection'],exact=True).count==len(rows):
                return {'status':'reused','documents':len(rows),'sparse_points_written':0,
                        'dense_vectors_encoded':0,'seconds':time.perf_counter()-started,
                        'collection':previous['sparse_collection']}
    if not rows:
        raise ValueError('Cannot publish an empty BM25 index')
    documents=[Counter(tokenize(r['section']+' '+r['text'])) for r in rows]
    df=Counter(t for counts in documents for t in counts)
    terms=sorted(df);vocab={t:i for i,t in enumerate(terms)}
    average=sum(sum(c.values()) for c in documents)/len(rows)
    collection='researchclaw_bm25_'+hashlib.sha256(json.dumps(key,sort_keys=True).encode()).hexdigest()[:24]
    if not store.client.collection_exists(collection):
        store.client.create_collection(collection,vectors_config={},sparse_vectors_config={'bm25':models.SparseVectorParams()})
    for start in range(0,len(rows),256):
        points=[]
        for r,counts in zip(rows[start:start+256],documents[start:start+256]):
            length=sum(counts.values());words=sorted(counts)
            values=[math.log(1+(len(rows)-df[t]+.5)/(df[t]+.5))*counts[t]*2.5/(counts[t]+1.5*(.25+.75*length/max(1,average))) for t in words]
            points.append(models.PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL,collection+':'+r['chunk_id'])),
                         vector={'bm25':models.SparseVector(indices=[vocab[t] for t in words],values=values)},
                         payload={'chunk_id':r['chunk_id'],'paper_id':r['paper_id'],'page':r['page'],'project_id':store.project_id}))
        store.client.upsert(collection,points=points,wait=True)
    assert store.client.count(collection,exact=True).count==len(rows)
    manifest={**key,'sparse_collection':collection,'terms':terms,'average_document_length':average}
    temporary=target.with_suffix('.pending.json')
    temporary.write_text(json.dumps(manifest,ensure_ascii=False),encoding='utf8')
    temporary.replace(target)
    return {'status':'built','documents':len(rows),'sparse_points_written':len(rows),
            'dense_vectors_encoded':0,'seconds':time.perf_counter()-started,'collection':collection}


def lexical_candidates(store, state, query, limit, paper_ids=None):
    """None means an old workspace without a BM25 manifest; [] is a real miss."""
    root=getattr(store,'root',None)
    if root is None or not (Path(root)/'rag-search-index.json').exists():
        return None
    data=json.loads((Path(root)/'rag-search-index.json').read_text(encoding='utf8'))
    expected={'project_id':store.project_id,'corpus_version':state['corpus_version'],
              'embedding_key':state['embedding_key'],'dense_collection':state['version_collection']}
    if any(data.get(k)!=v for k,v in expected.items()):
        raise RuntimeError('BM25 index is for a different revision; rebuild the offline search index before querying')
    vocabulary={t:i for i,t in enumerate(data['terms'])}
    indices=sorted(vocabulary[t] for t in set(tokenize(query)) if t in vocabulary)
    if not indices:
        return []
    response=store.client.query_points(data['sparse_collection'],
        query=models.SparseVector(indices=indices,values=[1.]*len(indices)),using='bm25',
        query_filter=models.Filter(must=[models.FieldCondition(key='paper_id',match=models.MatchAny(any=list(paper_ids)))]) if paper_ids else None,
        limit=limit,with_payload=True)
    return [{'chunk_id':p.payload['chunk_id']} for p in response.points]

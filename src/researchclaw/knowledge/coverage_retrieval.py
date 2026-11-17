"""Opt-in complex-question evidence planning over the published retriever."""
import re
from .evidence_coverage import focused_queries,answer_candidate,paragraph_documents,bibliography_request
from .evidence_planner import planner_candidates,choose_evidence


def retrieve_planned(rag,plan,*,limit=5):
    question=plan.get('original_query') or plan['rewritten_query']
    queries=focused_queries(question,rewritten=plan['rewritten_query'])
    if len(queries)==1:
        return rag._retrieve_anchored(plan,limit=limit)
    if not hasattr(rag,'_paper_catalog'):
        with rag.retriever.store.database.connect(read_only=True) as db:
            rag._paper_catalog=[dict(r) for r in db.execute('SELECT paper_id,title FROM papers')]
    explicit=[]
    for paper in rag._paper_catalog:
        title=paper['title'];alias=title.split(':',1)[0].strip() if ':' in title else title
        if len(alias)>=3 and re.search(r'(?<!\w)'+re.escape(alias)+r'(?!\w)',question,re.I):explicit.append(paper['paper_id'])
    # Never broaden an explicit selection because it exceeds the safe scope.
    if len(explicit)>4:raise ValueError('Select at most four papers for a detailed evidence question')
    orders=[];byid={};allrecords={}
    for query in queries:
        for mode in ('dense','bm25'):
            hits=rag.retriever.search(query,limit=20,candidate_limit=60,mode=mode,
                max_hits_per_paper=20,paper_ids=explicit or None,independent_units=True,
                exclude_references=not bibliography_request(question))
            order=[]
            for hit in hits:
                record=rag.retriever.store.get_chunk(hit.chunk_id)
                if not record:continue
                record=dict(record)
                if not answer_candidate(record,question,title=hit.title):continue
                if explicit and record['paper_id'] not in explicit:raise RuntimeError('Paper-scoped retrieval returned another paper')
                paper=rag.retriever.store.get_paper(record['paper_id'])
                if not paper or paper.get('content_hash')!=record['source_hash']:raise RuntimeError('Stale evidence source')
                byid[hit.chunk_id]=(hit,record);order.append(hit.chunk_id)
                if record['paper_id'] not in allrecords:
                    allrecords[record['paper_id']]=[dict(r) for r in rag.retriever.store.paper_chunks(record['paper_id'])]
            orders.append(order)
    pool=[]
    for rank in range(20):
        for order in orders:
            if rank<len(order) and order[rank] not in pool:pool.append(order[rank])
    pool=pool[:120]
    if not pool:return []
    records=[byid[key][1] for key in pool];sources=[r for group in allrecords.values() for r in group]
    documents=paragraph_documents(records,sources)
    scores=[rag.retriever.reranker.score(query,documents) for query in queries]
    packet=planner_candidates(records,scores,sources,question=question,limit=limit)
    selection=choose_evidence(rag.model,packet,limit=limit)
    rag.last_evidence_plan=selection
    rag.last_retrieval_audit=dict(policy='coverage_v05_soft',paper_ids=explicit,query_count=len(queries),
        candidate_count=len(pool),planner_candidates=len(packet['candidates']),planner_tokens=packet['tokens'],
        planner_omitted=packet['omitted'],selected=[s['id'] for s in selection['selected']])
    return [byid[s['id']][0] for s in selection['selected']]

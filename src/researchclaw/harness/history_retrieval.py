"""Query-focused, source-linked history retrieval, independent of paper RAG.

No inferred fact is promoted to confirmed user memory. Assistant statements keep
their role and must not override user corrections. This is lexical BM25, not an
embedding model or an automatic truth/conflict resolver.
"""
from collections import Counter
import math
import re


STOP = set('a an the is are was were be been i me my we our you your it its this that these those do does did have has had of to for from in on at by with and or but as what which who when where how please tell about can could would should'.split())
STOP.update('help helpful tip tips advice recommend recommends recommendation recommendations earlier previous chat discussed discussing looking back thinking trying feel feeling need extra anxious remind provided confirm'.split())


def terms(text):
    words = re.findall(r'[a-z0-9]+|[\u4e00-\u9fff]{2,}', str(text).lower())
    result = []
    for word in words:
        if re.search(r'[\u4e00-\u9fff]', word):
            result.extend(word[i:i+2] for i in range(len(word)-1))
        elif word not in STOP:
            # Conservative plural normalization; retain precise names/numbers.
            result.append(word[:-1] if len(word)>4 and word.endswith('s') and not word.endswith(('ss','us','is')) else word)
    return result


def retrieve_history(records, query, limit=20, excerpt_chars=900):
    """Rank bounded passages, then return chronological source-addressable data.

    Chunking here is for dialogue recall, unrelated to structured paper chunking.
    Overlap prevents a fact at an arbitrary character boundary from disappearing.
    Candidate scoring sees the entire message, not just its first 1200 chars.
    """
    if not query.strip() or limit<=0:
        return []
    candidates=[]
    anchors=[]
    previous_user={}
    for ordinal, record in enumerate(records):
        if record.get('role') not in {'user','assistant'}:
            continue
        text=record.get('content')
        if not isinstance(text,str) or not text.strip():
            continue
        session=record.get('source_session')
        if record['role']=='user':
            previous_user[session]=dict(source_message=record.get('id',f'ordinal:{ordinal}'),content=text[:1400])
        anchor=previous_user.get(session) if record['role']=='assistant' else None
        for offset in range(0,len(text),excerpt_chars-200):
            content=text[offset:offset+excerpt_chars]
            candidates.append(dict(source_message=record.get('id',f'ordinal:{ordinal}'),
                ordinal=record.get('ordinal',ordinal), role=record['role'], offset=offset,
                source_session=record.get('source_session'), date=record.get('created_at',record.get('date')),
                content=content, incomplete=offset>0 or offset+excerpt_chars<len(text)))
            if anchor:
                candidates[-1]['preceding_user']=dict(source_message=anchor['source_message'],role='user',
                    content=anchor['content'][:500],incomplete=len(anchor['content'])>500)
            anchors.append(anchor['content'] if anchor else '')
    if not candidates:
        return []
    # An assistant list often says "7. Transcriptionist" without repeating
    # "work from home jobs for seniors". Carry the preceding user turn as a
    # retrieval anchor, while preserving separate source text and speaker roles.
    counts=[Counter(terms(c['content'])+terms(a)) for c,a in zip(candidates,anchors)]
    lengths=[sum(c.values()) for c in counts]
    average=max(1,sum(lengths)/len(lengths))
    df=Counter(t for c in counts for t in c)
    query_terms=set(terms(query))
    ranked=[]
    for i,(candidate,count,length) in enumerate(zip(candidates,counts,lengths)):
        score=0.0
        for term in query_terms & count.keys():
            frequency=count[term]
            idf=math.log(1+(len(counts)-df[term]+0.5)/(df[term]+0.5))
            score+=idf*frequency*2.2/(frequency+1.2*(0.25+0.75*length/average))
        if score:
            ranked.append((score,i,candidate))
    ranked.sort(key=lambda x:(-x[0],-x[1]))
    selected=[]
    seen=set()
    for score,_,candidate in ranked:
        signature=(candidate['role'],candidate['content'])
        if signature in seen:
            continue
        seen.add(signature)
        selected.append({**candidate,'relevance':round(score,5),'kind':'historical-statement'})
        if len(selected)>=limit:
            break
    # Keep relevance rank for budget packing; the caller restores chronology
    # only AFTER fitting the budget so earliest facts are not evicted first.
    return selected

"""Bounded intent coverage over independent source paragraphs, without gold inputs."""
import json
import re
import unicodedata

INTENTS = (
    ('definition', r'defin|norm|metric|symbol|定义|符号|范数|误差',
     'definition of error metric norm variables normalization and pointwise guarantee'),
    ('conditions', r'assum|condition|proof|theorem|scope|假设|条件|证明|适用',
     'necessary assumptions conditions theorem proof scope and limitations'),
    ('implementation', r'experiment|empiric|implement|precision|error|实验|实现|精度|误差',
     'experimental error threshold numerical precision implementation limitations'),
)


def intent_queries(question):
    return [question] + [query for _, pattern, query in INTENTS if re.search(pattern, question, re.I)]


def bibliography_request(question):
    return bool(re.search(r'bibliograph|reference list|cited works|参考文献列表|书目|列出参考文献', question, re.I))


def provenance(record):
    return record.get('provenance') or json.loads(record.get('provenance_json') or '{}')


def eligible(record, question):
    p = provenance(record)
    return bibliography_request(question) or p.get('structure', {}).get('region') != 'references'


def select_coverage(records, scores, *, limit=5, max_per_paper=5):
    """One best distinct original unit per query lane, then broad-query fill.

    scores[0] is the original question. Further lanes reserve one position each;
    scores are never compared across queries, and packed parent IDs do not
    collapse distinct paragraphs. Caller must filter regions before scoring.
    """
    if limit < 1:return []
    if not records:return []
    if not scores or any(len(row) != len(records) for row in scores):
        raise ValueError('Each score lane must cover every candidate')
    orders=[sorted(range(len(records)), key=lambda i:(-float(row[i]),records[i]['chunk_id'])) for row in scores]
    selected=[];used=set();counts={}
    def add(i):
        r=records[i];p=provenance(r)
        key=(r['paper_id'],p.get('structure',{}).get('node_id') or r.get('parent_chunk_id') or r['chunk_id'])
        if key in used or counts.get(r['paper_id'],0)>=max_per_paper:return False
        selected.append(i);used.add(key);counts[r['paper_id']]=counts.get(r['paper_id'],0)+1
        return True
    for order in orders:
        for i in order:
            if add(i):break
        if len(selected)>=limit:return selected
    for i in orders[0]:
        add(i)
        if len(selected)>=limit:break
    return selected


def focused_queries(question, *, topic='', rewritten=''):
    """Keep the original task/topic in every facet; simple questions stay simple."""
    categories=[bool(re.search(p,question,re.I)) for p in
        (r'defin|norm|metric|symbol|定义|符号|范数',
         r'assum|condition|假设|条件',r'proof|theorem|证明|定理',
         r'experiment|implement|实验|实现')]
    if sum(categories)<2:return [rewritten or question]
    task=rewritten or ('Explain the claim, required assumptions, error metric, scope and proof of the pointwise error guarantee.'
                      if re.search(r'误差',question) else question)
    # Paper scope belongs in the store filter. Repeating the exact title in a
    # relevance query incorrectly rewards title/running-header matches.
    if re.search(r'error|误差',task+' '+question,re.I):
        return [task,
            'What is the error measure and how is relative or absolute approximation error defined?',
            'What assumptions are made about the data and the optimal solution, and what settings are distinguished?',
            'What error bound does the paper claim, and what component or proof is necessary to enforce it?',
            'What error thresholds are used in the experiments and how are these thresholds expressed?',
            'Which implementations violate error bounds, and why can numerical precision cause violations?']
    return [task,task+' What definitions are needed?',task+' What assumptions limit the claim?',task+' Where is the proof?']


def answer_candidate(record, question, *, title=''):
    if not eligible(record,question):return False
    if bibliography_request(question):return True
    kind=provenance(record).get('structure',{}).get('kind')
    if kind=='heading':return False
    if kind=='figure_caption' and len(record['text'].split())<25:return False
    words=re.findall(r'[a-z]+',record['text'].lower())
    titlewords=set(re.findall(r'[a-z]+',title.lower()))
    if titlewords and 2<=len(words)<=22 and sum(w in titlewords for w in words)/len(words)>=.85:
        return False
    return True


def record_unit(record):
    return provenance(record).get('structure',{}).get('node_id') or record.get('parent_chunk_id') or record['chunk_id']


def paragraph_documents(records, all_records, *, neighbor_words=80):
    """Bounded adjacent original paragraphs for scoring only; no new embeddings."""
    parents=[r for r in all_records if r.get('node_type')=='parent' and not provenance(r).get('context_only')]
    by_key={(r['paper_id'],r['source_hash'],r['chunk_id']):i for i,r in enumerate(parents)}
    documents=[]
    for record in records:
        index=by_key.get((record['paper_id'],record['source_hash'],record_unit(record)))
        text=record['section']+'\nMatched paragraph:\n'+record['text']
        if index is not None and len(record['text'].split())<60:
            own=parents[index]
            for offset in (-1,1):
                j=index+offset
                if not 0<=j<len(parents):continue
                other=parents[j]
                if (other['paper_id']==record['paper_id'] and other['source_hash']==record['source_hash']
                    and other['section']==own['section'] and eligible(other,'')
                    and abs((other.get('page') or 0)-(own.get('page') or 0))<=1):
                    words=other['text'].split()
                    excerpt=' '.join(words[-neighbor_words:] if offset<0 else words[:neighbor_words])
                    text+='\nAdjacent source paragraph (context only):\n'+excerpt
        # Normalize the scorer view only; exact source strings/spans and vectors
        # remain unchanged. Repair PDF ligatures and English line-end hyphens.
        text=unicodedata.normalize('NFKC',text)
        text=re.sub(r'(?<=[A-Za-z])-\s*\n\s*(?=[a-z])','',text)
        documents.append(text)
    return documents


def select_soft_coverage(records,scores,*,limit=5,max_per_paper=5):
    """Greedy relevance plus diminishing facet coverage, no hard lane quotas.

    Qwen logit margins are normalized within each facet. Single-fact queries
    use exact broad ranking; facets never promote a source solely by its region.
    """
    if not records or limit<1:return []
    if not scores or any(len(row)!=len(records) for row in scores):raise ValueError('Score shape mismatch')
    if len(scores)==1:return select_coverage(records,scores,limit=limit,max_per_paper=max_per_paper)
    import math
    if any(not math.isfinite(float(s)) for row in scores for s in row):raise ValueError('Non-finite score')
    # Remote Qwen exposes logit(yes)-logit(no), not a probability. Preserve
    # negative valid margins and use stable relative affinities per query.
    scaled=[[math.exp((float(s)-max(map(float,row)))/2.) for s in row] for row in scores]
    covered=[0.]*len(scores);selected=[];used=set();counts={}
    while len(selected)<limit:
        candidates=[]
        for i,r in enumerate(records):
            key=(r['paper_id'],record_unit(r))
            if key in used or counts.get(r['paper_id'],0)>=max_per_paper:continue
            # Actual relevance gate, not a reserved position for every intent.
            if max(float(row[i]) for row in scores)<-4.595:continue
            gain=sum(max(0.,row[i]-covered[k]) for k,row in enumerate(scaled))/len(scores)
            # A tiny caption/formula fragment may inherit a neighbor's good
            # score without containing its explanation. Prefer complete source
            # paragraphs, while retaining concise formulas as usable evidence.
            words=len(r.get('text','').split())
            completeness=min(1.,max(.15,words/60.)) if 'text' in r else 1.
            utility=(.20*scaled[0][i]+.80*gain)*completeness
            candidates.append((utility,float(scores[0][i]),r['chunk_id'],i))
        if not candidates:break
        i=sorted(candidates,key=lambda t:(-t[0],-t[1],t[2]))[0][-1]
        selected.append(i);r=records[i];used.add((r['paper_id'],record_unit(r)))
        counts[r['paper_id']]=counts.get(r['paper_id'],0)+1
        covered=[max(v,scaled[k][i]) for k,v in enumerate(covered)]
    return selected

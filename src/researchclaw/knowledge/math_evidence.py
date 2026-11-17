"""Source-linked mathematical evidence candidates, never theorem verification."""
import hashlib
import re
import json

VERSION = 'math-evidence-v05-rendered-v2'
MARKER = re.compile(r'\b(?P<kind>Theorem|Lemma|Proposition|Corollary|Definition|Assumption|Proof)\b'
                    r'(?:\s+(?P<number>\d+(?:\.\d+)*))?', re.I)
REFERENCE = re.compile(r'\b(Theorem|Lemma|Proposition|Corollary|Definition|Assumption)\s+(\d+(?:\.\d+)*)', re.I)
SYMBOL = re.compile(r'\b(?:where|let|denote|defined as|subject to|provided that|assum(?:e|ing)|if)\b', re.I)


def build_math_graph(segments, *, paper_id, source_hash):
    nodes, edges, last_claim = [], [], None
    for ordinal, segment in enumerate(segments):
        text = segment['text']
        matches = list(MARKER.finditer(text))
        # Only apparent declarations at sentence/paragraph starts become nodes.
        matches = [m for m in matches if m.start() == 0 or text[max(0,m.start()-2):m.start()].strip() in {'', '.', ':'}]
        if not matches and SYMBOL.search(text):
            matches = [None]
        for index, match in enumerate(matches):
            start = match.start() if match else 0
            end = matches[index+1].start() if index+1 < len(matches) and matches[index+1] else len(text)
            quote = text[start:end]
            kind = match.group('kind').lower() if match else 'condition_or_symbol'
            number = match.group('number') if match else None
            nid = hashlib.sha256(f'{source_hash}:{ordinal}:{start}:{quote}'.encode()).hexdigest()[:24]
            node = {'id': nid, 'paper_id': paper_id, 'source_hash': source_hash,
                    'page': segment.get('page'), 'section': segment.get('section', 'unknown'),
                    'segment_ordinal': ordinal, 'char_start': start, 'char_end': end,
                    'kind': kind, 'number': number, 'text': quote,
                    'text_hash': hashlib.sha256(quote.encode()).hexdigest(),
                    'verification': 'source_span_only_semantics_unverified'}
            nodes.append(node)
            if kind == 'proof' and last_claim and last_claim['section'] == node['section']:
                if ordinal - last_claim['segment_ordinal'] <= 1:
                    edges.append({'source': nid, 'target': last_claim['id'],
                                  'relation': 'adjacent_proof_candidate', 'verified': False})
            if kind in {'theorem','lemma','proposition','corollary'}:
                last_claim = node
    lookup = {}
    # Preserve visible mathematical syntax as source spans; no LaTeX repair or
    # claim that extracted PDF characters form a correct mathematical formula.
    formula = re.compile(r'\$\$?[\s\S]+?\$\$?|[^\n]{0,80}[≤≥∥∑∫][^\n]{0,160}')
    for ordinal, segment in enumerate(segments):
        for match in formula.finditer(segment['text']):
            quote = match.group()
            nid = hashlib.sha256(f'{source_hash}:{ordinal}:formula:{match.start()}:{quote}'.encode()).hexdigest()[:24]
            nodes.append({'id':nid,'paper_id':paper_id,'source_hash':source_hash,
                          'page':segment.get('page'),'section':segment.get('section','unknown'),
                          'segment_ordinal':ordinal,'char_start':match.start(),'char_end':match.end(),
                          'kind':'formula','number':None,'text':quote,
                          'text_hash':hashlib.sha256(quote.encode()).hexdigest(),
                          'verification':'source_span_only_transcription_unverified'})
            for other in nodes[:-1]:
                if other['segment_ordinal']==ordinal and other['kind']!='formula':
                    edges.append({'source':nid,'target':other['id'],
                                  'relation':'same_source_segment','verified':False})
    for node in nodes:
        if node['number']:
            lookup.setdefault((node['kind'],node['number']), []).append(node['id'])
    for node in nodes:
        for match in REFERENCE.finditer(node['text']):
            ids = lookup.get((match[1].lower(),match[2]), [])
            if len(ids) == 1 and ids[0] != node['id']:
                edges.append({'source': node['id'], 'target': ids[0],
                              'relation': 'explicit_named_reference', 'verified': False})
    return {'version': VERSION, 'paper_id': paper_id, 'source_hash': source_hash,
            'nodes': nodes, 'edges': edges,
            'warning': 'Textual relationships are candidates, not verified mathematical entailment.'}


def evidence_packet(graph, *, page, child_text, token_counter, budget=1200, max_nodes=8, exclude_ids=()):
    """Budget the complete model projection; retain omission details off-prompt."""
    by_id = {n['id']: n for n in graph['nodes']}
    seeds = [n['id'] for n in graph['nodes'] if n['page'] == page and
             (n['text'].strip() in child_text or child_text.strip() in n['text'])]
    selected, queue, used, omitted = [], list(seeds), 0, []
    visited = set(exclude_ids)
    def project(nodes):
        ids = {n['id'] for n in nodes}
        return {'source':{'paper_id':graph['paper_id'],'sha256':graph['source_hash']},
                'nodes':[{k:n[k] for k in ('id','kind','page','section','segment_ordinal','char_start','char_end','text')} for n in nodes],
                'edges':[e for e in graph['edges'] if e['source'] in ids and e['target'] in ids],
                'warning':'Unverified source links; necessary conditions may be missing. Check original source.'}
    while queue and len(selected) < max_nodes:
        nid = queue.pop(0)
        if nid in visited:
            continue
        visited.add(nid)
        node = by_id[nid]
        cost = token_counter(json.dumps(project(selected+[node]),ensure_ascii=False))
        if cost > budget:
            omitted.append(nid)
            continue
        selected.append(node)
        used = cost
        # One-hop expansion only from original evidence; no unbounded graph walk.
        if nid in seeds:
            for edge in graph['edges']:
                if edge['source'] == nid:
                    queue.append(edge['target'])
                elif edge['target'] == nid:
                    queue.append(edge['source'])
    omitted.extend(n for n in queue if n not in visited)
    kinds = {n['kind'] for n in selected}
    warnings = ['Necessary conditions and formula transcription require source verification.']
    if not kinds & {'assumption','condition_or_symbol'}:
        warnings.append('No explicit assumption/symbol node was recovered; do not assume unconditional validity.')
    if omitted:
        warnings.append('Evidence expansion incomplete because of the token/node budget.')
    return {'version': VERSION, 'nodes': selected, 'tokens': used, 'budget': budget,
            'budget_scope':'complete serialized model_context, including metadata and warning',
            'model_context':project(selected) if selected else None,
            'omitted_ids': sorted(set(omitted)), 'warnings': warnings,
            'edges': [e for e in graph['edges'] if e['source'] in {n['id'] for n in selected}
                      and e['target'] in {n['id'] for n in selected}]}

"""Conservative native PDF structure with auditable line-level provenance.

Layout heuristics are candidates, never mathematical entailment. No neural
inference is performed here; encoding remains the CUDA corpus builder's job.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import re
import statistics

from researchclaw.core.models import Chunk
from researchclaw.knowledge.token_safety import source_windows, token_count

POLICY = 'layout-paragraph-token-v05'
PARAGRAPH_POLICY = 'layout-semantic-paragraph-token-v05.3'
PARSER = 'pymupdf-layout-v05.2'


def _can_join(previous, line, kind, body_size, width):
    if not previous:return False
    last=previous['source_spans'][-1]
    semantic={'proof','definition','assumption','theorem','lemma','table_caption','figure_caption'}
    if kind in semantic:return False
    if previous['kind'] in {'table_caption','figure_caption'} and last['block']!=line['block']:return False
    terminal=bool(re.search(r'[.!?]\s*$',previous['text']))
    if last['page']+1==line['page']:
        return not terminal and line['text'][:1].islower()
    if last['page']!=line['page']:return False
    # A PDF block ID alone does not imply a paragraph, nor does a formula glyph imply a new paragraph.
    if (last['bbox'][0]>width*.45)!=(line['bbox'][0]>width*.45):return False
    gap=line['bbox'][1]-last['bbox'][3]
    if not -body_size*1.5<=gap<=body_size*.85:return False
    same_block=last['block']==line['block']
    if same_block:return True
    aligned=abs(last['bbox'][0]-line['bbox'][0])<body_size*1.5
    font_close=abs(last['size']-line['size'])<body_size*.25
    continuation=not terminal or bool(re.match(r'where\b|and\b|such that\b|with\b',line['text'],re.I))
    return aligned and font_close and continuation


@dataclass
class Structure:
    paper_id: str
    source_hash: str
    nodes: list
    pages: list
    deletions: list
    sections: list

    def to_dict(self):
        return asdict(self)


def _kind(text):
    for kind, pattern in [('definition', r'definition\b'), ('assumption', r'assumption\b'),
                          ('theorem', r'(?:theorem|proposition|corollary)\b'),
                          ('lemma', r'lemma\b'), ('proof', r'proof\b'),
                          ('table_caption', r'table\s+\w+'), ('figure_caption', r'fig(?:ure)?\.?\s+\w+')]:
        if re.match(pattern, text, re.I):
            return kind
    if re.search(r'[=∑∫≤≥]', text) and len(text.split()) < 35:
        return 'formula'
    return 'paragraph'


def parse_paper(path, paper_id=None):
    import fitz
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    paper_id = paper_id or 'rag_' + digest[:24]
    pages, all_lines = [], []
    with fitz.open(path) as doc:
        for number, page in enumerate(doc, 1):
            lines = []
            for bi, block in enumerate(page.get_text('dict')['blocks']):
                for li, line in enumerate(block.get('lines', [])):
                    spans = line.get('spans', [])
                    text = ''.join(s['text'] for s in spans)
                    if text.strip():
                        lines.append(dict(text=text, page=number, bbox=list(line['bbox']), block=bi,
                                          line=li, size=max(s['size'] for s in spans),
                                          bold=any(s['flags'] & 16 for s in spans)))
            pages.append(dict(page=number, width=page.rect.width, height=page.rect.height,
                              raw_lines=lines, status='native' if lines else 'empty_or_scanned_requires_ocr'))
            all_lines.extend(lines)
    body_size = statistics.median([l['size'] for l in all_lines]) if all_lines else 10
    edge_counts = {}
    for page in pages:
        for key in {l['text'].strip() for l in page['raw_lines']
                    if l['bbox'][3] < page['height'] * .055 or l['bbox'][1] > page['height'] * .945}:
            edge_counts[key] = edge_counts.get(key, 0) + 1
    deletions, nodes, sections, heading_stack = [], [], [], []
    region = 'body'
    section_stack = []
    previous = None
    for page in pages:
        retained = []
        for line in page['raw_lines']:
            text = line['text'].strip()
            edge = line['bbox'][3] < page['height'] * .055 or line['bbox'][1] > page['height'] * .945
            # Equations and numbered scientific statements are never frequency-deleted.
            scientific = bool(re.search(r'[=∑∫≤≥<>≠∥∀]|^(?:Assumption|Theorem|Lemma|Definition)\b', text, re.I))
            if edge and not scientific and ((text.isdigit() and text==str(page['page'])) or (not text.isdigit() and edge_counts.get(text, 0) >= max(3, len(pages)*.3))):
                deletions.append(dict(**line, reason='repeated_page_edge' if not text.isdigit() else 'edge_page_number'))
            else:
                retained.append(line)
        width = page['width']
        left = [l for l in retained if l['bbox'][0] < width*.45 and l['bbox'][2] < width*.57]
        right = [l for l in retained if l['bbox'][0] > width*.45]
        double = len(left) >= 5 and len(right) >= 5
        page['layout'] = 'two_column_candidate' if double else 'single_column_or_fallback'
        # Full-width blocks partition the vertical axis, so titles are not read last.
        wide = sorted([l for l in retained if double and l['bbox'][0] < width*.45 and l['bbox'][2] >= width*.57], key=lambda l:l['bbox'][1])
        ordered, rest = [], list(retained)
        for separator in wide:
            above = [l for l in rest if l is not separator and l['bbox'][3] <= separator['bbox'][1]]
            ordered.extend(sorted(above, key=lambda l:(int(l['bbox'][0] > width*.45) if double else 0, l['bbox'][1], l['bbox'][0])))
            ordered.append(separator)
            rest = [l for l in rest if l is not separator and l not in above]
        ordered.extend(sorted(rest, key=lambda l:(int(l['bbox'][0] > width*.45) if double else 0, l['bbox'][1], l['bbox'][0])))
        for line in ordered:
            text = line['text'].strip()
            numbered = re.match(r'^(\d+(?:\.\d+)*|[IVX]+|[A-Z])[.)]?\s+[A-Z][A-Za-z ,:–-]{2,90}$', text)
            known = re.fullmatch(r'abstract|references|bibliography|appendix(?:\s+[A-Z])?|introduction|conclusions?', text, re.I)
            heading = len(text) < 110 and (known or (numbered and (line['bold'] or line['size'] > body_size*1.06 or text.isupper() or re.fullmatch(r'[A-Z]\.\s+[A-Z][A-Za-z ]{2,65}',text))))
            if heading:
                level = (2 if re.fullmatch(r'[A-Z]',numbered[1]) and numbered[1] not in ('I','V','X') else numbered[1].count('.') + 1) if numbered else 1
                heading_stack = heading_stack[:level-1] + [text]
                sid = f'{paper_id}:s{len(sections)}'
                section_stack = section_stack[:level-1]
                sections.append(dict(node_id=sid, parent_id=section_stack[-1] if section_stack else None,
                                     text=text, heading_path=list(heading_stack), source_spans=[line], candidate=True))
                section_stack.append(sid)
                if re.match(r'references|bibliography', text, re.I): region = 'references'
                elif re.match(r'appendix', text, re.I): region = 'appendix'
                elif region != 'references': region = 'body'
                previous = None
                continue
            kind = _kind(text)
            can_join = _can_join(previous,line,kind,body_size,width)
            if can_join:
                # Preserve literal hyphens; PDF does not tell lexical from line-break hyphens.
                previous['text'] += '\n' + line['text']
                previous['source_spans'].append(line)
                previous['normalization'] = 'literal_lines_joined_with_newline'
                previous.setdefault('inline_kinds',[]).append(kind)
                if previous['source_spans'][0]['page']!=line['page']:previous['continuation_candidate']=True
            else:
                previous = dict(node_id=f'{paper_id}:u{len(nodes)}', parent_id=sections[-1]['node_id'] if sections else None,
                                kind=kind, text=line['text'], heading_path=list(heading_stack), region=region,
                                source_spans=[line], parser=PARSER, confidence='heuristic_candidate',
                                fallback_reason=None if double else 'column_structure_not_established')
                nodes.append(previous)
    # Headings themselves remain retrievable evidence with exact locations.
    for section in sections:
        nodes.append(dict(node_id=section['node_id']+':heading', parent_id=section['parent_id'], kind='heading',
                          text=section['source_spans'][0]['text'], heading_path=section['heading_path'],
                          region='references' if re.match(r'references|bibliography',section['text'],re.I) else 'body',
                          source_spans=section['source_spans'], parser=PARSER, confidence='heuristic_candidate', fallback_reason=None))
    for node in nodes:
        offset = 0
        for span in node['source_spans']:
            span = span.copy()
            # Store offsets on copies: raw page records remain untouched.
            span['char_start'], span['char_end'] = offset, offset + len(span['text'])
            node.setdefault('mapped_spans', []).append(span)
            offset = span['char_end'] + 1
        node['page_start'] = min(s['page'] for s in node['source_spans'])
        node['page_end'] = max(s['page'] for s in node['source_spans'])
        node['candidate_links'] = []
        refs=set(re.findall(r'\b(?:Theorem|Lemma|Definition|Assumption|Proposition|Equation|Eq\.)\s*\(?\d+(?:\.\d+)*\)?',node['text'],re.I))
        for ref in sorted(refs):
            for target in nodes:
                if target is not node and re.match(re.escape(ref)+r'\b',target['text'],re.I):
                    node['candidate_links'].append(dict(target_id=target['node_id'],relation='explicit_reference_candidate',verified=False))
    return Structure(paper_id, digest, nodes, pages, deletions, sections)


def _packed_nodes(structure,tokenizer,maximum):
    """Pack adjacent small paragraphs within a section; preserve each original unit and exact spans."""
    packed=[]
    for source in structure.nodes:
        node={**source,'mapped_spans':[dict(s) for s in source['mapped_spans']],
              'original_units':[{'node_id':source['node_id'],'kind':source['kind'],'char_start':0,'char_end':len(source['text'])}]}
        previous=packed[-1] if packed else None
        compatible=previous and previous['kind'] in {'paragraph','formula','paragraph_group'} and node['kind'] in {'paragraph','formula'}
        if compatible:
            compatible=(previous['heading_path']==node['heading_path'] and previous['region']==node['region']
                and previous['page_end']==node['page_start']
                and previous['source_spans'][-1]['bbox'][0]//(structure.pages[node['page_start']-1]['width']*.5)
                    ==node['source_spans'][0]['bbox'][0]//(structure.pages[node['page_start']-1]['width']*.5)
                and token_count(tokenizer,' / '.join(node['heading_path'])+'\n'+previous['text']+'\n\n'+node['text'])<=int(maximum*.78))
        if compatible:
            offset=len(previous['text'])+2
            previous['text']+='\n\n'+node['text']
            previous['mapped_spans'].extend({**s,'char_start':s['char_start']+offset,'char_end':s['char_end']+offset} for s in node['mapped_spans'])
            previous['source_spans']=[*previous['source_spans'],*node['source_spans']]
            previous['original_units'].extend({**u,'char_start':u['char_start']+offset,'char_end':u['char_end']+offset} for u in node['original_units'])
            previous['candidate_links']=[*previous['candidate_links'],*node['candidate_links']]
            previous['kind']='paragraph_group'
            previous['packing']='adjacent_same_section_units; boundaries preserved; no entailment implied'
        else:packed.append(node)
    aliases={u['node_id']:n['node_id'] for n in packed for u in n['original_units']}
    for node in packed:
        node['candidate_links']=[{**link,'target_id':aliases.get(link['target_id'],link['target_id'])} for link in node['candidate_links']]
    return packed


def structure_chunks(structure, tokenizer, *, maximum=512, grade='B', domain='unknown',
                     policy=PARAGRAPH_POLICY):
    if policy not in (POLICY, PARAGRAPH_POLICY):
        raise ValueError('Unknown layout chunk policy')
    chunks = []
    packed = _packed_nodes(structure,tokenizer,maximum)
    groups = {}
    if policy == PARAGRAPH_POLICY:
        for group in packed:
            if len(group['original_units']) > 1:
                group = {**group, 'node_id':group['node_id']+':context'}
                for unit in group['original_units']:
                    groups[unit['node_id']] = group
    for node in (packed if policy == POLICY else structure.nodes):
        # Bound heading prefix itself, explicitly, without dropping source content.
        heading = ' / '.join(node['heading_path'])
        while token_count(tokenizer, heading + '\n') >= maximum//2 and heading:
            heading = heading[:len(heading)//2]
        base = dict(paper_id=structure.paper_id, section=heading, page=node['page_start'], source_hash=structure.source_hash)
        provenance = dict(chunk_policy=policy, parser=PARSER, structure=node, quality_grade=grade, domain=domain,
                          heading_path=node['heading_path'],page_start=node['page_start'],page_end=node['page_end'],
                          fallback_used=bool(node['fallback_reason']))
        chunks.append(Chunk(chunk_id=node['node_id'], text=node['text'], node_type='parent', provenance=provenance, **base))
        windows = source_windows(node['text'], tokenizer, prefix=heading+'\n', maximum=maximum)
        for i, window in enumerate(windows):
            spans = [dict(s, overlap_start=max(window['char_start'],s['char_start']),
                          overlap_end=min(window['char_end'],s['char_end'])) for s in node['mapped_spans']
                     if s['char_end'] > window['char_start'] and s['char_start'] < window['char_end']]
            chunks.append(Chunk(chunk_id=f'{node["node_id"]}:c{i}', text=window['text'], node_type='child',
                                parent_chunk_id=groups.get(node['node_id'],node)['node_id'], content_hash=hashlib.sha256(window['text'].encode()).hexdigest(),
                                provenance={**provenance, 'source_spans':spans, 'input_tokens':window['tokens'],
                                            'source_char_start':window['char_start'], 'source_char_end':window['char_end'],
                                            'continuation_index':i, 'continuation_count':len(windows),
                                            'previous_child_id':f'{node["node_id"]}:c{i-1}' if i else None},
                                **{**base, 'page':spans[0]['page'] if spans else node['page_start']}))
    for group in {g['node_id']:g for g in groups.values()}.values():
        chunks.append(Chunk(chunk_id=group['node_id'], paper_id=structure.paper_id,
                            section=' / '.join(group['heading_path']), text=group['text'],
                            page=group['page_start'], source_hash=structure.source_hash, node_type='parent',
                            provenance=dict(chunk_policy=policy, parser=PARSER, structure=group,
                                            context_only=True, quality_grade=grade, domain=domain)))
    return chunks


def restore_context(hits, chunks, tokenizer, *, maximum=2048, regions=('body', 'appendix'), question='', adjacent=False):
    """Keep hits first; restore parents only when they fit the complete budget."""
    by_id = {c.chunk_id:c for c in chunks}
    selected, omitted = [], []
    used = set()
    def add(chunk):
        if chunk.chunk_id in used: return
        if any(chunk.paper_id != h.paper_id or chunk.source_hash != h.source_hash
               for h in (by_id[h] if isinstance(h,str) else h for h in hits)):
            omitted.append(dict(chunk_id=chunk.chunk_id, reason='source_identity')); return
        if chunk.provenance.get('structure', {}).get('region', 'body') not in regions:
            omitted.append(dict(chunk_id=chunk.chunk_id, reason='region_filter')); return
        from researchclaw.knowledge.evidence_projection import _spans
        item = dict(chunk_id=chunk.chunk_id, text=chunk.text, page=chunk.page, source_hash=chunk.source_hash,
                    source_spans=_spans(chunk.provenance.get('source_spans', chunk.provenance.get('structure',{}).get('mapped_spans',[]))))
        import json
        if token_count(tokenizer, json.dumps(selected+[item], ensure_ascii=False)) > maximum:
            omitted.append(dict(chunk_id=chunk.chunk_id, reason='context_budget')); return
        selected.append(item); used.add(chunk.chunk_id)
    for hit in hits: add(by_id[hit] if isinstance(hit,str) else hit)
    for hit in hits:
        chunk = by_id[hit] if isinstance(hit,str) else hit
        if chunk.parent_chunk_id in by_id and chunk.text != by_id[chunk.parent_chunk_id].text:
            add(by_id[chunk.parent_chunk_id])
        # A long original paragraph may not fit as a parent. Recover at most
        # one neighboring window on each side, without embedding merged text.
        # This is adjacency, not a verified mathematical dependency.
        if chunk.parent_chunk_id not in used:
            unit=chunk.provenance.get('structure',{}).get('node_id')
            index=chunk.provenance.get('continuation_index')
            if unit is not None and index is not None:
                for neighbor_index in (index+1,index-1):
                    neighbor=by_id.get(f'{unit}:c{neighbor_index}')
                    if neighbor is not None:add(neighbor)
        structure=chunk.provenance.get('structure',{})
        if adjacent:
            match=re.fullmatch(r'(.*:u)(\d+)',structure.get('node_id',''))
            if match:
                for offset in (-1,-2,1,2):
                    neighbor=by_id.get(match[1]+str(int(match[2])+offset))
                    if neighbor is None:continue
                    s=neighbor.provenance.get('structure',{})
                    if (s.get('heading_path')==structure.get('heading_path')
                            and abs((neighbor.page or 0)-(chunk.page or 0))<=1):add(neighbor)
        for link in structure.get('candidate_links',[]):
            if link['target_id'] in by_id:add(by_id[link['target_id']])
        if re.search(r'assum|defin|condition|symbol|假设|定义|条件|符号',question,re.I):
            for candidate in chunks:
                s=candidate.provenance.get('structure',{})
                if (candidate.node_type=='parent' and candidate.paper_id==chunk.paper_id
                        and s.get('kind') in ('definition','assumption')
                        and s.get('heading_path')==structure.get('heading_path')):
                    add(candidate)
    import json
    # Only this serialized projection enters a model prompt. Audit omissions stay
    # outside it; their size must not silently invalidate the context budget.
    model_context=json.dumps(selected,ensure_ascii=False)
    return dict(items=selected, model_context=model_context, tokens=token_count(tokenizer,model_context),
                omitted=omitted, maximum_tokens=maximum)

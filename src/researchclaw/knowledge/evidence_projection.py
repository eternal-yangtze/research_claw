"""Budget actual serialized scientific evidence, keeping every matched child."""
import json
from researchclaw.harness.context import ContextOverflow, TokenCounter


def _spans(spans):
    compact=[{k:s[k] for k in ('page','bbox','char_start','char_end','overlap_start','overlap_end','aggregation','original_span_count') if k in s} for s in spans]
    if len(compact)<=8:return compact
    # Full per-line locations remain in the immutable audit contexts. The model
    # needs page attribution and a coarse locator, not hundreds of coordinates.
    # Label envelopes explicitly so they never masquerade as exact line boxes.
    pages={}
    for span in compact:pages.setdefault(span.get('page'),[]).append(span)
    result=[]
    for page,items in pages.items():
        item=dict(page=page,aggregation='page_envelope',original_span_count=len(items))
        boxes=[s['bbox'] for s in items if s.get('bbox')]
        if boxes:item['bbox']=[min(b[0] for b in boxes),min(b[1] for b in boxes),max(b[2] for b in boxes),max(b[3] for b in boxes)]
        for start,end in (('char_start','char_end'),('overlap_start','overlap_end')):
            if all(start in s and end in s for s in items):
                item[start]=min(s[start] for s in items);item[end]=max(s[end] for s in items)
        result.append(item)
    return result


def project_evidence(question, contexts, *, maximum=6000, counter=None, wire_cost=None):
    counter=counter or TokenCounter()
    evidence=[];omitted=[]
    for c in contexts:
        item=dict(id=c['citation'],title=c['title'],pdf_page=c['page'],section=c['section'],text=c['child_text'])
        if c.get('source_hash'):item['source_hash']=c['source_hash']
        if c.get('source_spans'):
            item['source_spans']=_spans(c['source_spans'])
        evidence.append(item)
    encode=lambda rows:json.dumps({'question':question,'evidence':rows},ensure_ascii=False)
    measure=wire_cost or counter.text
    if measure(encode(evidence))>maximum:
        raise ContextOverflow('Matched evidence and source metadata exceed request budget; nothing silently discarded')
    seen={(c.get('source_hash'),c['child_text']) for c in contexts}
    for i,c in enumerate(contexts):
        parent=c.get('parent_text','')
        candidates=[]
        if parent and parent!=c['child_text']:
            candidates.append(('parent_context',dict(text=parent,source_hash=c.get('source_hash'),source_spans=_spans(c.get('parent_source_spans',[])))))
        if c.get('layout_packet'):
            candidates.append(('structure_context',c['layout_packet']['model_context']))
        if c.get('math_evidence',{}).get('model_context'):
            candidates.append(('math_evidence',c['math_evidence']['model_context']))
        for field,value in candidates:
            keys=[]
            if field=='parent_context':
                keys=[(value.get('source_hash'),value['text'])]
                if keys[0] in seen:
                    omitted.append(dict(citation=c['citation'],field=field,reason='duplicate_source_text'));continue
            if field=='structure_context':
                try:
                    packet=json.loads(value)
                except (ValueError,TypeError):packet=None
                if isinstance(packet,list):
                    compact=[]
                    for item in packet:
                        key=(item.get('source_hash'),item['text'])
                        if key in seen or key in keys:continue
                        keys.append(key)
                        compact.append({**{k:item[k] for k in ('chunk_id','text','page','source_hash') if k in item},
                                        'source_spans':_spans(item.get('source_spans',[]))})
                    if not compact:
                        omitted.append(dict(citation=c['citation'],field=field,reason='duplicate_source_text'));continue
                    value=json.dumps(compact,ensure_ascii=False)
            evidence[i][field]=value
            if measure(encode(evidence))>maximum:
                del evidence[i][field];omitted.append(dict(citation=c['citation'],field=field,reason='serialized_evidence_budget'))
            else:seen.update(keys)
    value=encode(evidence)
    return dict(model_context=value,tokens=counter.text(value),maximum_tokens=maximum,evidence=evidence,omitted=omitted)

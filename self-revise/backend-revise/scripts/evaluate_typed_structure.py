"""Import six explicit appended gold labels; evaluate frozen v3 and typed nodes."""
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from collections import Counter
from xml.etree import ElementTree as ET

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path[:0] = [str(ROOT/'src'),str(ROOT/'self-revise/backend-revise/scripts')]
from evaluate_coordinate_alignment import score_gold, coverage
from researchclaw.knowledge.typed_structure import extract_typed_nodes, annotate_typed_nodes


def main():
    base = ROOT/'self-revise/backend-revise/rag-evidence'
    review = base/'alignment-v3-review/REVIEW.md'
    out = base/'typed-structure-pilot'
    out.mkdir(exist_ok=True)
    cases = {c['case_id']:c for c in json.loads((review.parent/'cases.json').read_text(encoding='utf8'))}
    labels = []
    # Generated empty ## blocks are ignored: only appended ### human blocks.
    for match in re.finditer(r'^### (V3-\d{2})[^\n]*\n(.*?)(?=^### V3-|\Z)',review.read_text(encoding='utf8'),re.M|re.S):
        cid, block = match.groups()
        def field(name):
            found = re.search(r'^- '+name+r'：\s*([^\n]+)',block,re.M)
            if not found:
                raise ValueError((cid,name))
            return found[1].strip().strip('`')
        verdict=field('可对齐')
        assert verdict in ('true','false')
        c=cases[cid]
        p,s=field('正确 paragraph_id'),field('正确 section_id')
        labels.append({**{k:c[k] for k in ['case_id','paper_id','page','line_ordinal','text']},
                       'is_alignable':verdict=='true','expected_paragraph_id':None if p=='null' else p,
                       'expected_section_id':None if s=='null' else s,'reviewer':'user_v3_review',
                       'review_notes':block,'source':str(review.relative_to(ROOT))})
    assert len(labels)==len(cases)==len({g['case_id'] for g in labels})==6
    earlier=json.loads((base/'alignment-gold1-refinement/gold1.json').read_text(encoding='utf8'))+json.loads((base/'alignment-gold2-confirmed/gold2.json').read_text(encoding='utf8'))
    frozen,typed={},{}
    papers=[]
    start=time.perf_counter()
    for path in sorted((base/'alignment-gold2-confirmed').glob('rag_*.json')):
        d=json.loads(path.read_text(encoding='utf8'))
        doc=json.loads((base/'grobid-phase2-sample/schema'/path.name).read_text(encoding='utf8'))
        tei_paths=list((base/'grobid-phase2-sample/tei').glob(f'{path.stem}-*.tei.xml'))
        assert len(tei_paths)==1, 'TEI cache must be unique'
        tei=ET.parse(tei_paths[0]).getroot()
        nodes=extract_typed_nodes(doc,tei)
        # Human IDs must exist in the cached document, with consistent section.
        for g in [g for g in earlier+labels if g['paper_id']==path.stem and g['expected_paragraph_id']]:
            p=next(p for p in doc['paragraphs'] if p['paragraph_id']==g['expected_paragraph_id'])
            assert p['section_id']==g['expected_section_id']
        rows=annotate_typed_nodes(d['lines'],nodes,d['page_geometry'])
        assert [(r['page'],r['line_ordinal'],r['text'],r['paragraph_id']) for r in rows]==[(r['page'],r['line_ordinal'],r['text'],r['paragraph_id']) for r in d['lines']]
        for a,z in zip(d['lines'],rows):
            key=(path.stem,a['page'],a['line_ordinal'])
            frozen[key],typed[key]=a,z
        # Use the exact original eligibility tokenizer, including repair rules.
        from researchclaw.knowledge.hybrid_structure import tokens
        eligible=[r for r in rows if len(tokens(r['text']))>=4]
        diagnostic=sum(len(r['text']) for r in eligible if r['paragraph_id'] or r['typed_node_id'])/max(1,sum(len(r['text']) for r in eligible))
        papers.append({'paper_id':path.stem,'title':doc['paper']['title'],
                       'body_coverage_unchanged':coverage(rows,True),
                       'body_or_typed_node_coverage_diagnostic':diagnostic,
                       'typed_rows':dict(Counter(r['typed_content_type'] for r in rows if r['typed_node_id'])),
                       'unverified_container_examples':[{'node_id':n['node_id'],'text_excerpt':n['text'][:350],
                            'pages':sorted({c['page'] for c in n['coordinates']})}
                            for n in nodes if n['content_type'].endswith('_unverified') and
                            any(word in n['text'].casefold() for word in ['theorem','proof','corollary','algorithm'])][:3],
                       'new_typed_rows_without_body':sum(bool(r['typed_node_id']) and not r['paragraph_id'] for r in rows)})
        (out/path.name).write_text(json.dumps({'lines':rows,'typed_nodes':nodes,'page_geometry':d['page_geometry']},ensure_ascii=False),encoding='utf8')
    report={'gold_source_sha256':hashlib.sha256(review.read_bytes()).hexdigest(),
            'new_gold_cases':len(labels),'frozen_v3_new_gold':score_gold(labels,frozen),
            'frozen_v3_all54':score_gold(earlier+labels,frozen),
            'typed_development_all54':score_gold(earlier+labels,typed),
            'scope':'Six targeted positive cases independently reviewed after v3 froze; same papers, not random. Typed annotations are a new development experiment.',
            'papers':papers,'elapsed_seconds':time.perf_counter()-start,
            'body_gate_pass':all(p['body_coverage_unchanged']>=.8 for p in papers),
            'formal_migration':False,
            'cases':[dict(g,frozen_paragraph=frozen[(g['paper_id'],g['page'],g['line_ordinal'])]['paragraph_id'],
                          typed_section=typed[(g['paper_id'],g['page'],g['line_ordinal'])]['section_id']) for g in labels]}
    (out/'gold3.json').write_text(json.dumps(labels,ensure_ascii=False,indent=2),encoding='utf8')
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k!='cases'},ensure_ascii=True,indent=2))


if __name__=='__main__':
    main()

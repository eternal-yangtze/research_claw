"""Freeze gold2 evaluation before comparing a new, explicitly developmental v2."""
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from collections import defaultdict

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'self-revise/backend-revise/scripts')]
import pymupdf
from evaluate_coordinate_alignment import score_gold, coverage
from researchclaw.knowledge.coordinate_alignment import normalized
from researchclaw.knowledge.alignment_refinement import refine_spatial_consensus, refine_instance_evidence


def main():
    base = ROOT / 'self-revise/backend-revise/rag-evidence'
    out = base / 'alignment-gold2-confirmed'
    out.mkdir(exist_ok=True)
    source = base / '金标2.md'
    cases = {c['case_id']: c for c in json.loads((base / 'alignment-human-review2/cases.json').read_text(encoding='utf8'))}
    labels = []
    for m in re.finditer(r'^### (H\d{2}-\d)[^\n]*\n(.*?)(?=^### H|\Z)', source.read_text(encoding='utf8'), re.M | re.S):
        cid, block = m.groups()
        def field(name):
            match = re.search(r'^- ' + name + r'：\s*(.+)$', block, re.M)
            if not match:
                raise ValueError((cid, name))
            return match[1].strip().strip('`')
        verdict = field('可对齐')
        assert verdict in ('true', 'false') or verdict.startswith('null')
        c = cases[cid]
        p, s = field('正确 paragraph_id'), field('正确 section_id')
        labels.append({**{k: c[k] for k in ('case_id','paper_id','page','line_ordinal','text')},
                       'is_alignable': {'true': True, 'false': False}.get(verdict),
                       'expected_paragraph_id': None if p == 'null' else p,
                       'expected_section_id': None if s == 'null' else s,
                       'reviewer': 'user_gold2', 'review_notes': block})
    assert len(labels) == len(cases) == len({g['case_id'] for g in labels}) == 24
    followup = base / 'alignment-gold2-refinement/H08-2-followup.md'
    reviewed = followup.read_text(encoding='utf8')
    # Only the explicit human-filled structured section can supersede pending.
    block = re.search(r'^### H08-2[^\n]*\n(.*)', reviewed, re.M | re.S)
    if not block:
        raise ValueError('Missing structured follow-up gold')
    def follow_field(name):
        found = re.search(r'^- '+name+r'：\s*([^\n]+)',block[1],re.M)
        if not found:
            raise ValueError(name)
        return found[1].strip().strip('`')
    assert follow_field('可对齐') == 'true'
    item = next(g for g in labels if g['case_id']=='H08-2')
    item.update(is_alignable=True, expected_paragraph_id=follow_field('正确 paragraph_id'),
                expected_section_id=follow_field('正确 section_id'), followup_source=str(followup.relative_to(ROOT)),
                review_notes=block[1], mapping_extent='Theorem 4 instance only; repeated text is not globally unique')
    manifest = {m['paper_id']: m for m in json.loads((base/'grobid-phase2-sample/sample-manifest.json').read_text(encoding='utf8'))}
    predictions = {name: {} for name in ('coordinate','frozen_v1','frozen_v2','development_v3')}
    papers = []
    started = time.perf_counter()
    for path in sorted((base/'coordinate-phase3-1-rawline').glob('rag_*.json')):
        original = json.loads(path.read_text(encoding='utf8'))
        frozen = json.loads((base/'alignment-gold1-refinement'/path.name).read_text(encoding='utf8'))
        doc = json.loads((base/'grobid-phase2-sample/schema'/path.name).read_text(encoding='utf8'))
        raw = defaultdict(list)
        detailed = defaultdict(list)
        with pymupdf.open(manifest[path.stem]['path']) as pdf:
            for page in pdf:
                for block in page.get_text('rawdict')['blocks']:
                    for line in block.get('lines', []):
                        text = ''.join(c['c'] for span in line['spans'] for c in span['chars'])
                        raw[(page.number+1, normalized(text))].append(list(line['bbox']))
                        chars = [c for span in line['spans'] for c in span['chars'] if not c['c'].isspace()]
                        detailed[(page.number+1, normalized(text))].append({'bbox':list(line['bbox']),
                            'words':[[*c['bbox'],c['c']] for c in chars]})
        refined = refine_spatial_consensus(original['lines'], doc, original['page_geometry'], raw)
        frozen_v2 = json.loads((base/'alignment-gold2-refinement'/path.name).read_text(encoding='utf8'))['lines']
        assert [r['paragraph_id'] for r in refined] == [r['paragraph_id'] for r in frozen_v2]
        revised = refine_instance_evidence(original['lines'], doc, original['page_geometry'], detailed)
        identity = lambda rows: [(r['page'],r['line_ordinal'],r['text']) for r in rows]
        assert identity(original['lines']) == identity(refined)
        assert identity(original['lines']) == identity(revised)
        for name, rows in [('coordinate',original['lines']),('frozen_v1',frozen['lines']),('frozen_v2',frozen_v2),('development_v3',revised)]:
            for row in rows:
                predictions[name][(path.stem,row['page'],row['line_ordinal'])] = row
        papers.append({'paper_id':path.stem,'coordinate_coverage':coverage(original['lines'],True),
                       'v1_coverage':coverage(frozen['lines'],True),'v2_coverage':coverage(refined,True),
                       'v3_coverage':coverage(revised,True), 'v3_new_accepts':sum(a['paragraph_id'] is None and b['paragraph_id'] is not None for a,b in zip(refined,revised))})
        (out/path.name).write_text(json.dumps({'lines':revised,'page_geometry':original['page_geometry']},ensure_ascii=False),encoding='utf8')
    gold1 = json.loads((base/'alignment-gold1-refinement/gold1.json').read_text(encoding='utf8'))
    report = {'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'followup_sha256':hashlib.sha256(followup.read_bytes()).hexdigest(),
              'label_counts':{'positive':sum(g['is_alignable'] is True for g in labels),
                              'negative':sum(g['is_alignable'] is False for g in labels),
                              'pending':sum(g['is_alignable'] is None for g in labels)},
              'gold2':{n:score_gold(labels,p) for n,p in predictions.items()},
              'gold1_regression':{n:score_gold(gold1,p) for n,p in predictions.items()},
              'scope':'Gold2 plus explicit H08-2 human follow-up; v2 and v3 use gold feedback and are development only',
              'elapsed_seconds':time.perf_counter()-started,'papers':papers,
              'cases':[{**g,'predictions':{n:p[(g['paper_id'],g['page'],g['line_ordinal'])]['paragraph_id'] for n,p in predictions.items()}} for g in labels],
              'formal_migration':False}
    (out/'gold2.json').write_text(json.dumps(labels,ensure_ascii=False,indent=2),encoding='utf8')
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('cases',)},ensure_ascii=True,indent=2))


if __name__ == '__main__':
    main()

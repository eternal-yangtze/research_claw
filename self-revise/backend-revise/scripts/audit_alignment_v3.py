"""Offline residual accounting and six new-rule review positions; no store writes."""
import json
from pathlib import Path
import sys
import re
from collections import Counter

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path[:0] = [str(ROOT/'src')]
import pymupdf
from researchclaw.knowledge.alignment_refinement import sequence
from researchclaw.knowledge.hybrid_structure import tokens
from researchclaw.knowledge.coordinate_alignment import normalized


def main():
    base = ROOT/'self-revise/backend-revise/rag-evidence'
    source = base/'alignment-gold2-confirmed'
    out = base/'alignment-v3-review'
    if (out/'REVIEW.md').exists() and re.search(r'^### V3-\d{2}', (out/'REVIEW.md').read_text(encoding='utf8'),re.M):
        raise RuntimeError('Human labels are appended to REVIEW.md; refusing to regenerate or overwrite this review pack')
    out.mkdir(exist_ok=True)
    (out/'images').mkdir(exist_ok=True)
    manifest = {m['paper_id']:m for m in json.loads((base/'grobid-phase2-sample/sample-manifest.json').read_text(encoding='utf8'))}
    gold = json.loads((source/'gold2.json').read_text(encoding='utf8')) + json.loads((base/'alignment-gold1-refinement/gold1.json').read_text(encoding='utf8'))
    known = {(g['paper_id'],g['page'],g['line_ordinal']) for g in gold}
    audits, cases, unreviewed = [], [], 0
    for path in sorted(source.glob('rag_*.json')):
        rows = json.loads(path.read_text(encoding='utf8'))['lines']
        old = json.loads((base/'alignment-gold2-refinement'/path.name).read_text(encoding='utf8'))['lines']
        doc = json.loads((base/'grobid-phase2-sample/schema'/path.name).read_text(encoding='utf8'))
        paragraphs = doc['paragraphs']
        abstract = sequence(doc['paper'].get('abstract') or '')
        title = sequence(doc['paper']['title'])
        reasons, all_unknown = Counter(), Counter()
        new = []
        for a, row in zip(old,rows):
            if a['paragraph_id'] is None and row['paragraph_id'] and (path.stem,row['page'],row['line_ordinal']) not in known:
                new.append(row)
            if row['paragraph_id']:
                continue
            val = sequence(row['text'])
            if row['page']==1 and len(val)>=12 and val in abstract:
                reason='exact_normalized_abstract_text'
            elif row['page']==1 and len(val)>=12 and val in title:
                reason='exact_normalized_title_text'
            elif row['coordinate_reason']=='no_unique_raw_line':
                reason='missing_or_repeated_raw_line'
            elif not any(len(val)>=6 and val in sequence(p['text']) for p in paragraphs if p['page_start']<=row['page']<=p['page_end']):
                reason='no_continuous_body_text_candidate'
            else:
                reason='body_text_exists_but_geometry_or_ambiguity_gate'
            all_unknown[reason] += 1
            if len(tokens(row['text']))>=4:
                reasons[reason] += len(row['text'])
        unreviewed += len(new)
        audits.append({'paper_id':path.stem,'title':doc['paper']['title'],
                       'unmatched_rows_by_diagnostic':dict(all_unknown),
                       'eligible_unmatched_characters_by_diagnostic':dict(reasons),
                       'unreviewed_new_accepts':len(new)})
        # One new position per paper/rule, excluding every previous human case.
        selected = {}
        for row in new:
            selected.setdefault(row['refinement'],row)
        with pymupdf.open(manifest[path.stem]['path']) as pdf:
            for row in selected.values():
                cid = f'V3-{len(cases)+1:02d}'
                page = pdf[row['page']-1]
                boxes = [list(line['bbox']) for block in page.get_text('rawdict')['blocks'] for line in block.get('lines',[])
                         if normalized(''.join(c['c'] for span in line['spans'] for c in span['chars']))==normalized(row['text'])]
                annotations=[]
                for box in boxes:
                    annot=page.add_rect_annot(pymupdf.Rect(box));annot.set_colors(stroke=(1,.5,0));annot.update();annotations.append(annot)
                page.get_pixmap(matrix=pymupdf.Matrix(1.3,1.3)).save(str(out/'images'/f'{cid}.png'))
                for annot in annotations:
                    page.delete_annot(annot)
                p=next(p for p in paragraphs if p['paragraph_id']==row['paragraph_id'])
                samepage=[p for p in paragraphs if p['page_start']<=row['page']<=p['page_end']]
                alternatives=sorted(samepage,key=lambda p:-len(tokens(row['text'])&tokens(p['text'])))[:3]
                candidates={q['paragraph_id']:q for q in [p,*alternatives]}
                cases.append({'case_id':cid,'paper_id':path.stem,'title':doc['paper']['title'],
                              'pdf_path':manifest[path.stem]['path'],'page':row['page'],
                              'line_ordinal':row['line_ordinal'],'text':row['text'],
                              'context':[r['text'] for r in rows if r['page']==row['page'] and abs(r['line_ordinal']-row['line_ordinal'])<=2],
                              'raw_boxes':boxes,'algorithm_evidence':row,
                              'paragraph_candidates':list(candidates.values()),
                              'silver':{'source':'local_text_and_geometry_rule_not_independent_judge',
                                        'is_alignable':True,'expected_paragraph_id':p['paragraph_id'],
                                        'expected_section_id':p['section_id'],
                                        'reason':row['refinement'],'human_confirmed':False}})
    audit={'scope':'Diagnostic categories, not human content labels; original coverage denominator unchanged',
           'unreviewed_new_accepts':unreviewed,'papers':audits}
    (source/'residual-audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8')
    (out/'cases.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2),encoding='utf8')
    lines=['# v3新增规则复核：6处新位置','',
           '前两批48处已全部导入金标。这里只检查新规则接收、但不在此前金标中的位置。',
           '银标为本地文本与坐标规则建议，**不是独立LLM评审，也不是人工金标**。橙框为全部同文源位置，选中的框和锚点见证据。',
           '请每处填写可对齐、正确paragraph_id、正确section_id与理由；若只认可主归属或部分跨度请注明。页码是PDF物理页。','',
           f'新规则共新增接收48行，其中{unreviewed}行未人工标注；以下按论文/规则选取{len(cases)}处，属于针对性审计，不能估计总体准确率。','']
    for c in cases:
        s=c['silver']
        lines += [f"## {c['case_id']} · {c['title']}",'',f"PDF第{c['page']}页，清洗行号{c['line_ordinal']}（从0开始）；原论文：`{c['pdf_path']}`",'',
                  f"> {c['text']}",'','附近原文：','',*[f'> {t}' for t in c['context']],'',
                  f"![全部同文源位置](images/{c['case_id']}.png)",'',
                  '本地银标：可对齐=true；paragraph_id=`'+s['expected_paragraph_id']+'`；section_id=`'+str(s['expected_section_id'])+'`。',
                  '证据规则：`'+s['reason']+'`。选中源框：`'+json.dumps(c['algorithm_evidence'].get('source_instance_boxes',[]))+'`。',
                  '源锚点：`'+json.dumps(c['algorithm_evidence'].get('source_anchors',[]))+'`。','', '候选段落（含同页替代项）：','']
        for p in c['paragraph_candidates']:
            lines += [f"- `{p['paragraph_id']}` / `{p['section_id']}`",'',p['text'],'']
        lines += ['人工填写：','', '- 可对齐：','- 正确 paragraph_id：','- 正确 section_id：','- 理由：','']
    (out/'REVIEW.md').write_text('\n'.join(lines),encoding='utf8')
    print(json.dumps({'unreviewed_new_accepts':unreviewed,'review_cases':len(cases),
                      'cases':[{k:c[k] for k in ['case_id','title','page','line_ordinal','text']} for c in cases],
                      'eligible_unknown_chars':dict(sum((Counter(p['eligible_unmatched_characters_by_diagnostic']) for p in audits),Counter()))},ensure_ascii=True,indent=2))


if __name__=='__main__':
    main()

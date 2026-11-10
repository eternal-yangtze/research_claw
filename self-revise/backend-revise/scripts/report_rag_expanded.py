"""Summarize frozen experiments without selecting favorable questions or hiding gaps."""
import hashlib
import json
from pathlib import Path
import statistics
import random
import difflib

ROOT=next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
OUT=ROOT/'self-revise/backend-revise/rag-evidence/expanded-rag-v20261109'


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(name,value):(OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def mean(values):
    values=[v for v in values if v is not None]
    return statistics.mean(values) if values else None


def main():
    suite=read(OUT/'suite.json');freeze=read(OUT/'freeze.json')
    assert hashlib.sha256((OUT/'suite.json').read_bytes()).hexdigest()==freeze['sha256']
    development=[q for q in suite['cases'] if q['split']=='development']
    heldout=[q for q in suite['cases'] if q['split']=='heldout']
    near=[]
    for a in development:
        for b in heldout:
            similarity=difflib.SequenceMatcher(None,a['question'].lower(),b['question'].lower()).ratio()
            if similarity>.85:near.append({'development':a['id'],'heldout':b['id'],'similarity':similarity,
                'exact':a['question'].strip().lower()==b['question'].strip().lower()})
    write('split-quality-audit.json',{'unique_questions':len({q['question'].strip().lower() for q in suite['cases']}),
        'total_union_questions':len(suite['cases']),'near_duplicate_threshold':.85,'cross_split_near_duplicates':near,
        'note':'post-evaluation disclosure; no questions removed or labels changed; source-family overlap remains'})
    reports={};failures=[]
    selection=read(OUT/'selection.json') if (OUT/'selection.json').exists() else {}
    candidate_mode=selection.get('candidate','coverage')
    for split in ['development','heldout']:
        judges=[read(p) for p in sorted((OUT/'judges'/split).glob('*.json'))]
        expected=34 if split=='development' else 12
        if len(judges)!=expected:failures.append(f'{split}: {len(judges)}/{expected} judges')
        refined={p.stem:read(p) for p in (OUT/'judges-refined'/split).glob('*.json')}
        if candidate_mode=='anchored':
            if len(refined)!=expected:failures.append(f'{split}: {len(refined)}/{expected} refined judges')
            for j in judges:
                if j['id'] in refined:
                    j['scores']['anchored']=refined[j['id']]['scores']['anchored']
                    j['scores']['legacy_pair']=refined[j['id']]['scores']['legacy']
        reports[split]={}
        for mode in ['dense','legacy','coverage']+(['legacy_pair','anchored'] if candidate_mode=='anchored' else []):
            answerable=[j for j in judges if j['category']!='unanswerable']
            keys=['faithfulness','answer_relevance','reference_fact_recall','context_precision','quote_coverage']
            reports[split][mode]={'cases':len(judges),'answerable_cases':len(answerable),
                'mean':{k:mean([j['scores'][mode].get(k) for j in answerable]) for k in keys},
                'valid_denominators':{k:sum(j['scores'][mode].get(k) is not None for j in answerable) for k in keys},
                'unanswerable_refused':sum(bool(j['scores'][mode].get('refused')) for j in judges if j['category']=='unanswerable'),
                'unanswerable_cases':sum(j['category']=='unanswerable' for j in judges),
                'citation_ids_valid':sum(bool(j['scores'][mode].get('citation_ids_valid')) for j in judges),
                'categories':{category:{k:mean([j['scores'][mode].get(k) for j in judges if j['category']==category]) for k in keys}
                              for category in sorted({j['category'] for j in judges})}}
    write('generation-summary.json',{'label':'same-model blinded LLM silver; quote-validated claims, not RAGAS','splits':reports})
    ragas=[read(p) for p in sorted((OUT/'ragas').glob('*.json'))]
    ragas_report={mode:{metric:{'mean':mean([r.get('score') for r in ragas if r['mode']==mode and r['metric']==metric and r['status']=='scored']),
        'scored':sum(r['mode']==mode and r['metric']==metric and r['status']=='scored' for r in ragas),'expected':12}
        for metric in ['faithfulness','answer_relevancy','context_precision','context_recall']} for mode in ['dense',candidate_mode]}
    write('ragas-summary.json',{'means':ragas_report,'expected':96,'completed':len(ragas),
        'failed':[r for r in ragas if r['status']!='scored'],
        'reference':'source-authored before retrieval, same frozen reference both arms, not baseline-evidence-derived'})
    # Stage attribution using expected sources, never pretend incomplete qrels are exhaustive relevance.
    diagnosis=[]
    for q in suite['cases']:
        if q['category']=='unanswerable':continue
        for mode in ['dense','legacy','coverage']+(['anchored'] if candidate_mode=='anchored' else []):
            path=OUT/'generation'/q['split']/mode/(q['id']+'.json')
            if not path.exists():continue
            record=read(path);contexts=record['contexts']
            papers={c['paper_id'] for c in contexts}
            reason=('expected_paper_absent_from_context' if not set(q['relevant_paper_ids'])<=papers
                    else 'expected_quote_not_preserved_or_different_passage' if record['quote_coverage']<1
                    else 'expected_quotes_present')
            diagnosis.append({'id':q['id'],'split':q['split'],'mode':mode,'stage':reason,
                'quote_coverage':record['quote_coverage'],
                'missing_child_count':sum(' '.join(c['child_text'].split()) not in ' '.join(c['parent_text'].split()) for c in contexts)})
    old=[]
    oldbase=ROOT/'self-revise/backend-revise/rag-evidence/final-rag-v20261109'
    for p in sorted(oldbase.glob('generation-*-full.json')):
        record=read(p)
        old.append({'id':record['id'],'missing_child_citations':[c['citation'] for c in record['contexts']
            if ' '.join(c['child_text'].split()) not in ' '.join(c['parent_text'].split())]})
    write('failure-diagnosis.json',{'old_context_audit':old,'cases':diagnosis,
        'counts':{split:{mode:{stage:sum(r['split']==split and r['mode']==mode and r['stage']==stage for r in diagnosis)
            for stage in ['expected_paper_absent_from_context','expected_quote_not_preserved_or_different_passage','expected_quotes_present']}
            for mode in ['dense','legacy','coverage']+(['anchored'] if candidate_mode=='anchored' else [])} for split in ['development','heldout']},
        'warning':'quote missing can be equivalent evidence, not proof of a wrong answer; source-conditioned silver is incomplete'})
    hold=reports['heldout'];dev=reports['development']
    def gate(split):
        a=split['legacy_pair' if candidate_mode=='anchored' else 'legacy']['mean'];b=split[candidate_mode]['mean']
        return (b['reference_fact_recall'] is not None and b['reference_fact_recall']>=a['reference_fact_recall']
                and b['faithfulness']>=a['faithfulness']-.02 and b['answer_relevance']>=a['answer_relevance']-.02)
    # The original selection uses the same answerable-only denominator below.
    promote=bool(selection.get('promote_if_holdout_passes')) and not failures and gate(hold) and gate(dev)
    decision={'promote':promote,'candidate':candidate_mode,'development_gate':gate(dev),'heldout_gate':gate(hold),
              'selection':selection,'missing_evaluation':failures,'holdout_used_for_retuning':False,
              'scope':'named-paper Qdrant scope, windowed rerank and query/context assembly; no PDF/vector migration or default CLI paper-ranking change'}
    write('decision.json',decision)
    def paired_stats(deltas):
        rng=random.Random(20261109)
        samples=sorted(statistics.mean(rng.choices(deltas,k=len(deltas))) for _ in range(2000))
        return {'n':len(deltas),'mean_delta':mean(deltas),'bootstrap_95_percentile':[samples[49],samples[1949]],
                'improved':sum(x>1e-9 for x in deltas),'unchanged':sum(abs(x)<=1e-9 for x in deltas),'regressed':sum(x< -1e-9 for x in deltas)}
    paired={}
    if candidate_mode=='anchored':
        for split in ['development','heldout']:
            rows=[read(p) for p in (OUT/'judges-refined'/split).glob('*.json')]
            rows=[r for r in rows if r['category']!='unanswerable']
            paired[split]={key:paired_stats([r['scores']['anchored'][key]-r['scores']['legacy'][key] for r in rows
                if r['scores']['anchored'].get(key) is not None and r['scores']['legacy'].get(key) is not None])
                for key in ['faithfulness','reference_fact_recall','answer_relevance','quote_coverage']}
    paired['ragas_heldout']={}
    for metric in ['faithfulness','answer_relevancy','context_precision','context_recall']:
        dense={r['id']:r['score'] for r in ragas if r['mode']=='dense' and r['metric']==metric and r['status']=='scored'}
        candidate={r['id']:r['score'] for r in ragas if r['mode']==candidate_mode and r['metric']==metric and r['status']=='scored'}
        common=sorted(set(dense)&set(candidate))
        if common:paired['ragas_heldout'][metric]=paired_stats([candidate[i]-dense[i] for i in common])
    write('paired-comparison.json',paired)
    source_jobs={j['id']:j for j in read(OUT/'source-selection.json')['jobs']}
    paper_paths={}
    for manifest in [OUT.parent/'time-series-domain-release-v3.1-build.json',OUT.parent/'final-rag-v20261109/incremental-build.json']:
        if manifest.exists():
            paper_paths.update({p['paper_id']:p['path'] for p in read(manifest).get('papers',[]) if 'paper_id' in p and 'path' in p})
    review=['# 后续人工抽查窗口（当前均为银标）','',
            '此窗口不阻塞个人工具试用。按原文接受、修改或拒绝银标；若只否定而未给正确答案，仍为待更正。页码为PDF物理页。盲测完成后才生成本窗口；后续如据此调参，需另备新held-out。','']
    ranked=[]
    for q in suite['cases']:
        path=OUT/('judges-refined' if candidate_mode=='anchored' else 'judges')/q['split']/(q['id']+'.json')
        if path.exists() and q['category']!='unanswerable':
            s=read(path)['scores'][candidate_mode]
            ranked.append((s['reference_fact_recall'],q))
    for score,q in sorted(ranked,key=lambda pair:(pair[0],pair[1]['id']))[:10]:
        sources={s['chunk_id']:s for s in source_jobs[q['id']]['sources']}
        review += [f"## {q['id']} / {q['category']} / {q['split']}",'',q['question'],'',
                   '银标参考：'+q['reference'],'',f'候选参考事实覆盖银标：{score}。','']
        for e in q['evidence']:
            s=sources[e['chunk_id']]
            review += [f"- {s['title']}；PDF第{s['page']}页；paper_id `{s['paper_id']}`；Child `{s['chunk_id']}`；银标原文：{e['quote']}",
                       f"  PDF路径：`{paper_paths.get(s['paper_id'],'按paper_id查询PG权威元数据')}`"]
        review += ['','人工结论：待填写（参考是否正确/是否另有更直接证据/候选是否漏答/修正内容与理由）。','']
    # Once handed to the user, never overwrite appended human adjudications.
    if not (OUT/'人工抽查窗口.md').exists():
        (OUT/'人工抽查窗口.md').write_text('\n'.join(review),encoding='utf-8')
    lines=['# 扩大评测与RAG针对性调试结果','',
        '本轮基于原204份PDF库，不更换embedding、不重建知识库。检索48开发+12盲测；生成34开发+12盲测，两侧共享题目。源文生成银标与精确quote在检索前冻结。held-out在候选选择后运行，未据此回头调参。开发中先试coverage，再据q01/q03等失败记录修订为anchored，见development-amendment.json；不是只试一次就宣称成功。','',
        '## 检索四路基线与保守重排候选','']
    for split in ['development','heldout']:
        lines += [f'### {split}','','|方案|题数|Recall@5|MRR@5|nDCG@5|Recall@10|指定Parent命中@10|','|---|---:|---:|---:|---:|---:|---:|']
        if not (OUT/f'retrieval-{split}.json').exists():continue
        for mode,row in read(OUT/f'retrieval-{split}.json').items():
            m=row['mean'];lines.append(f"|{mode}|{row['n']}|{m['recall@5']:.4f}|{m['mrr@5']:.4f}|{m['ndcg@5']:.4f}|{m['recall@10']:.4f}|{m['expected_parent_hit@10']:.4f}|")
    lines+=['','## 生成盲化银标（自定义，不是RAGAS）','',
        '可答题均值与不可答拒答单独统计；无事实声明时忠实度为null，不填1。均为5组Context，比较不混用旧6题/3组Context结果。legacy_pair是与anchored同次盲化双臂评审的原链路，采用决定比较这两个配对分数；legacy/coverage/dense三臂评审另保留。不同Judge上下文可能产生评分差异，不能混作同一次评审。','',
        '|划分|方案|可答题数|忠实度|答案相关性|参考事实覆盖|精确quote覆盖|不可答拒答|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for split,modes in reports.items():
        for mode,r in modes.items():
            m=r['mean'];lines.append(f"|{split}|{mode}|{r['answerable_cases']}|{m['faithfulness']:.4f}|{m['answer_relevance']:.4f}|{m['reference_fact_recall']:.4f}|{m['quote_coverage']:.4f}|{r['unanswerable_refused']}/{r['unanswerable_cases']}|")
    lines+=['','## 真实RAGAS 0.4.3：12题held-out','',
        '|方案|Faithfulness|Answer relevancy|Context precision|Context recall|','|---|---:|---:|---:|---:|']
    for mode,values in ragas_report.items():
        cells=[f"{r['mean']:.4f} ({r['scored']}/12)" if r['mean'] is not None else f"未出分 ({r['scored']}/12)" for r in values.values()]
        lines.append('|'+mode+'|'+'|'.join(cells)+'|')
    lines+=['',f"已保存{len(ragas)}/96项；失败项单列于ragas-summary.json，未补分。",'',
        '## 根因与改动','',
        '- 旧6题中6条full证据最终Context未完整保留命中Child，见failure-diagnosis.json。候选检查后补回原Child。',
        '- 原每篇仅1条片段，论文发现的多样性约束会影响同篇多页条件覆盖；候选最多2个不同Parent。',
        '- 原多查询先各路rerank、合并再rerank；候选各路RRF后统一一次cross-encoder，并以75%检索排名+25%重排排名保守融合。',
        '- 原original_query仅记录未召回，候选保留原问题；标题加入rerank输入，帮助消歧算法名称。',
        '- 原Context Recall参考答案来自dense证据，本轮从原文独立冻结，降低该项对照偏差；不能据此断言全部下降都只是指标偏差。',
        '- 开发中的具名论文范围漂移，增加精确论文名的元数据限定Qdrant通道，保留全库反证通道；限定内保留BM25命中，局部token分窗评分避免长片段尾部信息被主题稀释。并非所有失败都来自512-token截断，Plato示例Child仅269 tokens，也暴露混合主题干扰。',
        '- Judge返回[S1]与S1的格式差异曾触发错误的0分；从原始响应按统一规则重算，见citation-parser-audit.json，未改动原始判断、quote或参考答案。',
        '- 结构题为页码/结论定位，不能等同完整章节树/附录测试；参考文献后附录缺失仍未修复。','',
        '## 采用决定','',
        (f'候选{candidate_mode}通过预声明开发/held-out门，可以采用该组装策略；具体默认启用状态见deployment.json。' if promote else
         '候选未同时满足开发与held-out门，保留原正式默认策略；改动保留为可复现候选，不通过继续修改同一held-out来制造胜出。'),'',
        '## 证据与限制','',
        'suite.json与freeze.json为冻结输入；retrieval/和generation/为逐题原文、排名与回答；judges/为盲化银标，author-ledger为出题/评审请求响应与usage；ragas/及ragas-ledger为真实RAGAS。类别细分、失败、分母见JSON。','',
        '人工抽查窗口.md列出10个低覆盖样本的论文、PDF物理页、Child和银标原文，方便后续核对。token-length-audit.json记录150/10955个Child单独超过512 token的风险；局部分窗不等于已重做全部embedding。','',
        '配对变化和固定种子2000次bootstrap区间见paired-comparison.json；12题盲测很小，不应把均值改善写成已证明统计显著。原legacy相同回答在三臂/双臂银标中的相关性评分也存在明显差异，说明Judge依赖评审上下文；主结论同时参考原文quote覆盖、真实RAGAS和失败样本，不只挑一张银标表。','',
        '这些是源文条件化、同模型银标，不是人工金标或独立公共基准；qrels只列源证据，不完备。开发与held-out共享语料及部分方法家族，只保证题目未用于候选调参，不声称论文家族隔离。版本6题来自3组同题记录，不能当6种独立版本难例。不可答5题全为开发集的私有/未来事实，不证明所有拒答泛化。','',
        '题目并集65道，全部字符串唯一；事后文本相似度审计发现q15/q50跨划分相似度约0.935，保留且披露，没有删除难题或重算成更有利分数。因此held-out是流程上未参与选候选的题目留出，不是完全独立、无相似问题的外部测试。详见split-quality-audit.json。','',
        '对个人科研辅助工具：扩大评测使局限更清楚，不以取得全满分为目标。实际使用仍需核对原文；后续人工优先复核低分条件题、跨篇比较和附录缺口。']
    (OUT/'实验报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'generation':reports,'decision':{k:v for k,v in decision.items() if k!='selection'},'ragas':ragas_report},ensure_ascii=False))


if __name__=='__main__':main()

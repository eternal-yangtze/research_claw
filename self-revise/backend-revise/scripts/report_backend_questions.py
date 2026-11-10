"""Render question-by-question interview evidence from frozen local artifacts.

No inference, rescoring, dataset modification or database access. Recorded judge
reasons are quoted as silver judgments, not invented as causal proof.
"""
from __future__ import annotations

import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import xml.etree.ElementTree as ET

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
BASE = ROOT / 'self-revise'
OUT = BASE / 'question-analysis-v20261110'
INPUTS = {}
COUNTS = {}
LINKS = []


def read(p):
    p = Path(p)
    raw = p.read_bytes()
    INPUTS[str(p.relative_to(ROOT)).replace('\\', '/')] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw.decode('utf-8-sig'))


def text(value):
    if value is None:
        return '未记录 / 不适用'
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False)
    return value.replace('\x00', '').strip()


def cell(value):
    return text(value).replace('|', '\\|').replace('\n', '<br>')


def num(value):
    if isinstance(value, bool):
        return '通过' if value else '未通过'
    if value is None:
        return 'NA'
    return f'{value:.4f}' if isinstance(value, (float, int)) else cell(value)


def link(p, label='原始证据'):
    p = Path(p)
    assert p.exists(), p
    LINKS.append(p)
    target = os.path.relpath(p, OUT).replace('\\', '/')
    return f'[{label}]({target})'


def block(value):
    return '\n'.join('> ' + s for s in text(value).splitlines())


def table(headers, rows):
    return '\n'.join(['|' + '|'.join(headers) + '|', '|' + '|'.join(['---'] * len(headers)) + '|'] +
                     ['|' + '|'.join(cell(v) for v in row) + '|' for row in rows])


def write(name, lines):
    p = OUT / name
    p.write_text('\n\n'.join(lines) + '\n', encoding='utf-8')


def rag():
    base = BASE / 'rag-evidence/expanded-rag-v20261109'
    suite = read(base / 'suite.json')['cases']
    diagnosis = {(r['id'], r['mode']): r for r in read(base / 'failure-diagnosis.json')['cases']}
    titles = {}
    for p in (base / 'retrieval').rglob('*.json'):
        for hit in read(p).get('hits', []):
            titles[hit['paper_id']] = hit.get('title', '')
    lines = ['# 01 · 最终 RAG：全部问题、逐题对照与解释',
             '来源：本项目 204 份 PDF 的源论文条件化 LLM 银标题，**不是 RPC-Bench、不是公共论文问答排行榜题**。'
             '检索 60 题（48 开发 + 12 held-out），生成 46 题（34 开发 + 12 held-out），共享 41 题，共 65 个不同问题。'
             '原始英文问题逐字列出，不以重新翻译后的题替代实际测评输入。',
             '检索基线：dense、BM25、RRF、full=RRF+rerank、balanced 候选。表内 R5/M5/P10 分别为 Recall@5、MRR@5、ExpectedParentHit@10。'
             '生成配对基线为 legacy/anchored；F/R/C/Q 分别为逐事实忠实性、答案相关性、参考事实召回、精确引文覆盖，均为自定义银标，不是 RAGAS。'
             'RAGAS 另列四项，只有 held-out 12 题，dense/anchored 两组。NA 不补零。',
             '“为什么”分三层：实际排名或上下文审计是可观察定位；judge reason 是模型评审意见；机制归因仍需控制变量实验。'
             '尤其不能从单题升降就断言 query rewrite、embedding 或 reranker 是唯一原因。']
    retrieval_n = generation_n = 0
    for q in suite:
        ident, split = q['id'], q['split']
        lines += [f'## {ident} · {split} · {q["category"]}', '**实际问题**\n\n' + block(q['question']),
                  '**来源与参考**：' + link(base / 'suite.json', '冻结题集') + '。相关论文：' +
                  '; '.join(f'{titles.get(pid) or "题集标注论文（标题未从召回证据解析）"} `{pid}`' for pid in q.get('relevant_paper_ids', [])),
                  '**参考答案**\n\n' + block(q.get('reference'))]
        if q.get('caveat'):
            lines += ['**参考限定**：' + text(q['caveat'])]
        if q.get('evidence'):
            lines += ['**源文位置与关键原句**\n\n' + '\n\n'.join(
                f'Child `{e.get("chunk_id")}`：\n\n{block(e.get("quote"))}' for e in q['evidence'])]
        rows, results = [], {}
        for mode in ['dense', 'bm25', 'rrf', 'full', 'balanced']:
            p = base / f'retrieval/{split}/{mode}/{ident}.json'
            if p.exists():
                r = read(p); results[mode] = r
                rows.append([mode, num(r.get('recall@5')), num(r.get('mrr@5')),
                             num(r.get('expected_parent_hit@10')), link(p)])
        if rows:
            retrieval_n += 1
            lines += ['**检索逐题结果**\n\n' + table(['方案', 'R5', 'M5', 'P10', '证据'], rows)]
            r = results.get('full', next(iter(results.values())))
            expected = set(q.get('relevant_paper_ids', []))
            hits = r.get('hits', [])
            found = [(i + 1, h) for i, h in enumerate(hits) if h.get('paper_id') in expected]
            observed = '; '.join(f'第 {i} 位：{h.get("title")}，PDF p.{h.get("page")}，Parent `{h.get("parent_id")}`' for i, h in found[:4])
            if r.get('recall@5', 0) < 1:
                diagnosis_text = 'full 前 5 项未覆盖全部已标相关论文，观察到的是候选覆盖/排序缺口；不能仅凭它判断正文解析失败。'
            elif not r.get('expected_parent_hit@10'):
                diagnosis_text = '论文级覆盖通过，但标定证据 Parent 未命中；“找到论文”和“找到能回答的段落”在本题分离。'
            else:
                diagnosis_text = 'full 既覆盖标注论文又命中目标 Parent，支持当前题的定位能力，仍不直接证明生成答案完整。'
            lines += ['**检索为何如此、说明什么**：' + diagnosis_text + (' 目标论文实际命中：' + observed if observed else ' 当前 full 返回内未观察到标注论文命中。')]
            a, b = results.get('rrf', {}), results.get('full', {})
            if a and b:
                if b.get('recall@5', 0) < a.get('recall@5', 0):
                    lines += ['**本题消融观察**：加入 rerank 后 R5 下降，说明重排会挤出部分已标相关论文，不能宣传重排逐题必胜。']
                elif b.get('expected_parent_hit@10', 0) > a.get('expected_parent_hit@10', 0):
                    lines += ['**本题消融观察**：加入 rerank 后目标 Parent 命中改善，符合局部证据优先的用途；单题不能证明总体显著性。']
        judge_path = base / f'judges-refined/{split}/{ident}.json'
        if judge_path.exists():
            generation_n += 1
            scores = read(judge_path)['scores']
            rows = [[m] + [num(s.get(k)) for k in ['faithfulness', 'answer_relevance', 'reference_fact_recall', 'quote_coverage']] + [str(s.get('refused'))]
                    for m, s in scores.items()]
            lines += ['**生成逐题结果**\n\n' + table(['方案', 'F', 'R', 'C', 'Q', '拒答'], rows)]
            for mode, score in scores.items():
                p = base / f'generation/{split}/{mode}/{ident}.json'
                lines += [f'**{mode} 评审给出的具体理由（原文，银标意见）**\n\n' + block(score.get('reason'))]
                d = diagnosis.get((ident, mode))
                if d:
                    label = {'expected_paper_absent_from_context': '目标论文未进入生成上下文',
                             'expected_quote_not_preserved_or_different_passage': '目标原句未保留或选到了该论文其他片段',
                             'expected_quotes_present': '目标原句已存在于上下文'}.get(d['stage'], d['stage'])
                    lines += [f'**{mode} 上下文审计定位**：{label}；目标引文覆盖 {num(d.get("quote_coverage"))}，丢失 Child 数 {d.get("missing_child_count")}。这是文本位置审计，不是单一因果证明。']
                if p.exists():
                    lines += [f'{mode} 完整回答/上下文：' + link(p) + '；配对评审：' + link(judge_path)]
            s = scores.get('anchored', {})
            if q['category'] == 'unanswerable' or q.get('unanswerable'):
                meaning = '本题应拒答或明确证据不足；正确拒答验证这一具体不可答问题的边界控制，不外推通用拒答率。'
            elif s.get('refused'):
                meaning = '当前方案没有给出实质答案；需区分证据不足时安全停止与本可作答但漏检，不能把拒答本身记为科研成功。'
            elif s.get('reference_fact_recall', 0) < .999:
                meaning = '当前回答仍有参考事实遗漏；忠实性再高也不等于完整。科研使用中应追问缺失条件，并核对参考原句。'
            else:
                meaning = '当前评审认为参考事实已覆盖；仅支持本题、给定参考和给定证据，不证明全库其他问题或所有数学细节正确。'
            lines += ['**本题结果的实际含义**：' + meaning]
        ragas_rows = []
        for mode in ['dense', 'anchored']:
            values = []
            for metric in ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']:
                p = base / f'ragas/{ident}-{mode}-{metric}.json'
                if p.exists():
                    values.append(num(read(p).get('score')))
            if values:
                assert len(values) == 4
                ragas_rows.append([mode] + values)
        if ragas_rows:
            lines += ['**本题真实 RAGAS**\n\n' + table(['方案', 'Faithfulness', 'Answer relevancy', 'Context precision', 'Context recall'], ragas_rows),
                      'RAGAS 标量没有逐题根因说明，解释应结合上面的原文、上下文审计和配对 reason；不能从一个数反推唯一故障组件。逐指标 JSON 位于 ' + link(base / 'ragas', 'ragas 目录') + '。']
    COUNTS['rag_final'] = {'unique_questions': len(suite), 'retrieval': retrieval_n, 'generation': generation_n}
    assert (len(suite), retrieval_n, generation_n) == (65, 60, 46)
    write('01_RAG最终65题逐题分析.md', lines)


def memory(directory, name, public):
    base = BASE / 'memory-evidence' / directory
    freeze = read(base / 'freeze.json')
    data = read(ROOT / 'self-revise/backend-revise/internal_experiments/assets' / freeze['dataset'])
    references = {q['question_id']: q for q in data}
    lines = [f'# {name}',
             '来源：公开 [LongMemEval](https://huggingface.co/datasets/xiaowu0162/longmemeval) 数据。问题保留原始英文；参考答案来自公开记录；'
             '本项目使用本地同模型语义银标，不是官方评分脚本或人工金标。',
             ('当前目标：单会话 oracle。oracle 已给出正确目标会话，因此不测跨会话选择。156 个公共问题中 132 个前轮未出现，不能将二者相加。'
              '另外 36 个为 12 个问题加入背景的派生案例，不是 36 个独立公共问题。' if public else
              '历史跨会话方向：LongMemEval-S 500 题中固定选 48 题，24 开发/24 保留。这一方向已移出当前默认承诺；不能当作当前产品跨会话记忆成绩。'),
             '基线 legacy=统一提示下旧历史检索/压缩，query=问题驱动检索+对话锚点。两者窗口上限相同，但实际输入 token 不同。'
             '每题“为什么”的首要依据是保存的 judge_reason；它不等于已独立验证的算法因果归因。']
    tally = Counter()
    for entry in freeze['cases']:
        ident = entry['id']
        track = entry.get('track', entry.get('split'))
        original = ident if ident in references else re.sub(r'-(20|50|100)-(early|middle|late)$', '', ident)
        ref = references[original]
        a, b = [read(base / f'calls/{ident}-{mode}.json') for mode in ['legacy', 'query']]
        tally['cases'] += 1; tally['legacy'] += bool(a['correct']); tally['query'] += bool(b['correct'])
        lines += [f'## {ident} · {track} · {entry["category"]}', '**实际问题**\n\n' + block(ref['question']),
                  '**公开参考答案**\n\n' + block(ref['answer']),
                  f'**原始来源身份**：`question_id={original}`，数据文件 `{freeze["dataset"]}`，SHA256 `{freeze["dataset_sha256"]}`。']
        if entry.get('rounds'):
            lines += [f'**派生方式**：在该原问题历史之外加入 {entry["rounds"]} 轮通用背景，原会话位置 `{entry["position"]}`。原问题与参考答案不变；轮数和位置同时变化，不能当作严格位置消融。']
        if public:
            lines += ['**曝光状态**：' + ('该公共原题已在前轮出现。' if a.get('previously_exposed') else '该公共原题未在前轮 48 题中出现；属于 132 题子集。')]
        rows = []
        for mode, row in [('legacy', a), ('query', b)]:
            rows.append([mode, num(row['correct']), row.get('usage', {}).get('prompt_tokens', 'NA'), num(row.get('seconds')),
                         link(base / f'calls/{ident}-{mode}.json')])
        lines += [table(['方案', '正确性银标', '实际 prompt tokens', '生成秒数', '完整请求/回答/评审'], rows)]
        for mode, row in [('legacy', a), ('query', b)]:
            lines += [f'**{mode} 实际答案**\n\n' + block(row['answer']),
                      f'**{mode} 为什么获得该判定（保存的银标理由）**\n\n' + block(row.get('judge_reason'))]
        if a['correct'] and b['correct']:
            meaning = '两组均覆盖了本题评分所需事实；此题不能证明新策略优于旧策略。应结合 token 成本看保留收益。'
        elif not a['correct'] and b['correct']:
            meaning = '新策略在本题补足了旧答案的事实/偏好缺口，具体差异见两段回答和评审理由；说明上下文选择改善了本例，但实际装入信息更多，不能将增益全部归因于算法。'
        elif a['correct'] and not b['correct']:
            meaning = '本题是新策略回退，保留失败，不重新判分。说明问题驱动召回/锚点也会遗漏信息或改变解释，不能宣称逐题支配旧策略。'
        else:
            meaning = '两组都没有满足本题参考要求，说明现有上下文选择和回答仍有缺口；需对照请求判断是事实未进入上下文还是已进入但未使用，单凭错误标签不能确定。'
        lines += ['**本题说明什么**：' + meaning]
        if 'preference' in entry['category']:
            lines += ['**科研使用映射**：这是隐含偏好/建议适配题；对应科研中的选题偏好和设备约束，不是论文知识检索正确率。重要约束应显式确认进 State。']
        else:
            lines += ['**科研使用映射**：对应回忆前文设备、误差阈值、助手已给方案或历史修正；不测试论文 RAG，也不能证明长对话任意细节无损。']
    COUNTS['memory_' + directory] = dict(tally)
    if public:
        assert tally['cases'] == 192 and tally['query'] == 165 and tally['legacy'] == 81
    else:
        assert tally['cases'] == 48 and tally['query'] == 32 and tally['legacy'] == 5
    write(('03_单会话156题及36派生案例.md' if public else '04_历史LongMemEval48题.md'), lines)


def agents():
    base = BASE / 'agent-evidence/v20261110-architecture-live'
    freeze = read(base / 'freeze.json')
    reports = [('旧单 ReAct · 复测', read(base / 'task-summary.json'), 'single', base / 'task-summary.json'),
               ('三角色 · 复测', read(base / 'task-summary.json'), 'multi', base / 'task-summary.json'),
               ('新 LangGraph', read(base / 'langgraph-validation-v2/task-summary.json'), 'langgraph', base / 'langgraph-validation-v2/task-summary.json'),
               ('旧单 · 首轮', read(BASE / 'agent-evidence/v20261110-final/task-summary.json'), 'single', BASE / 'agent-evidence/v20261110-final/task-summary.json'),
               ('三角色 · 首轮', read(BASE / 'agent-evidence/v20261110-final/task-summary.json'), 'multi', BASE / 'agent-evidence/v20261110-final/task-summary.json'),
               ('三角色 · 最初非紧凑', read(BASE / 'agent-evidence/v20261110-final/task-summary-initial.json'), 'multi', BASE / 'agent-evidence/v20261110-final/task-summary-initial.json'),
               ('三角色 · 定向三题复验', read(BASE / 'agent-evidence/v20261110-final/multi-candidate-summary.json'), 'multi', BASE / 'agent-evidence/v20261110-final/multi-candidate-summary.json')]
    lines = ['# 05 · Agent 24 个实际科研任务与 32 个路由问题',
             '来源：本项目自建科研助手任务集，不是 SWE-bench、GAIA 或其他公开 Agent 基准。24 题覆盖局部事实、跨篇比较、idea、基线草案、检索式、会话修正、安全和 MCP。'
             '同一批题已曝光，评审为本地 LLM 银标；首轮、复测、新引擎为不同批次，不能混合挑最好结果。',
             'success=是否满足该任务验收要求；supported=证据支持银标。正确澄清/拒绝危险操作也可能 success=true；仅拿到引文或内部状态 complete 不等于任务成功。'
             '表内 tokens 读取任务汇总的 tokens 字段（Agent 消耗），不误用 usage 字段里的独立 judge 消耗。定向三题只在相关题显示，不替换任何整组结果。']
    for ident, category, prompt in freeze['cases']:
        lines += [f'## {ident} · {category}', '**实际用户问题**\n\n' + block(prompt),
                  '**来源/要求**：' + link(base / 'freeze.json', '冻结任务定义') + '；此处问题中的 cite/local evidence/no execution 等限制都是验收条件。']
        matched = []
        for title, report, arm, path in reports:
            case = next((x for x in report.get('cases', []) if x.get('id') == ident and x.get('arm') == arm), None)
            if case is not None:
                matched.append((title, case, path))
        lines += [table(['批次/方案', '成功', '证据支持', '秒数', 'tokens', '实际工具序列', '证据'],
                        [[title, num(c.get('success')), num(c.get('supported')), num(c.get('seconds')),
                          c.get('tokens', 'NA'), ', '.join(c.get('tools', [])), link(p)] for title, c, p in matched])]
        for title, c, _ in matched:
            lines += [f'**{title}：为什么得到这个结果（原始评审理由）**\n\n' + block(c.get('reason', c.get('error')))]
        current = next(c for title, c, _ in matched if title == '新 LangGraph')
        meanings = {'fact': '检验在库内取证和限定结论；不证明全局学术事实或数学保证。',
                    'comparison': '检验跨来源对齐假设、误差含义与适用范围；不能把重构误差自动推广为下游任务保证。',
                    'idea': '检验相对于库内文献的重叠检查与可检验假设，不是证明全球新颖性。',
                    'baseline': '检验能否提出公平控制、指标与小规模验证草案；没有执行或验证实验收益。',
                    'safety': '检验是否把危险/有副作用请求交给宿主确认；正确停止是成功，不是“不会做任务”。',
                    'query': '检验能否按用户科研目标产生检索式，不证明外部检索召回或下载成功。',
                    'context': '检验同会话目标/阈值修正与约束保留，不代表跨会话自动记忆。',
                    'mcp': '检验经实际协议读取指定对象，不代表多个远程 MCP 服务或完整研究任务。'}
        lines += ['**本题说明什么**：' + meanings.get(category, '本题只支持问题中明确限定的能力；不能外推为全链路科研成功。') +
                  (' 当前组通过，只限该本地银标要求。' if current.get('success') else ' 当前组未通过，不以工具调用成功或部分有据替代任务完成。')]
        if ident == 'fact-sz':
            lines += ['**人工分析**：主干预测/量化/编码有据，PSNR 模式和 2.1 版本细节却由只含 “be extended” 的片段支撑。'
                      '这是引用存在但语义蕴含不足；应精准回读或删除无依据细节，不能由宿主 ID 检查兜底。保持此题失败，没有重跑抬分。']
    routepath = BASE / 'agent-evidence/v20261110-final/routing-summary.json'
    routes = read(routepath)['cases']
    lines += ['# 工具路由 32 题', '来源：16 类本地工具意图各两个问法，不是 32 个独立科研任务。允许答案是工具集合；只检查选工具，不能等同执行后的科研成功。']
    for r in routes:
        lines += [f'## 路由 {r["id"]} / variant {r["variant"]}', block(r['prompt']),
                  f'预期工具：`{r.get("expected")}`；实际：`{r.get("selected")}`；判定：{num(r["correct"])}。',
                  '**为何、说明什么**：' + ('实际选中工具满足该题允许集合，支持这一问法的工具语义匹配。' if r['correct'] else '实际工具未满足允许集合，属于路由错误。') +
                  ' 这一步没有验证外部数据是否充分、返回证据是否支持结论，也不证明所有改写都可正确选择。' + link(routepath)]
    COUNTS['agent'] = {'tasks': len(freeze['cases']), 'routing_prompts': len(routes)}
    assert len(freeze['cases']) == 24 and len(routes) == 32
    write('05_Agent24题与32路由问题.md', lines)


def legacy_memory():
    base = BASE / 'memory-evidence/v20261109'
    suite = read(base / 'suite.json')
    lines = ['# 06 · 早期 30 个记忆案例与四组基线',
             '来源：本项目模拟 20/50/100 轮科研对话，非公开基准。四臂 recent-window、summary-window、llm-summary-window、state-source-memory。'
             'state-source-memory 预写了答案对应已确认记忆，这一设置只验读取、替换、日期与拒答，不能证明从普通聊天自主发现事实。']
    for q in suite:
        lines += [f'## {q["id"]} · {q["category"]} · {q["rounds"]} 轮', '**实际问题**\n\n' + block(q['query']),
                  '**参考答案**：' + text(q['expected']), '**历史关键事件**\n\n' + '\n\n'.join(block(e['text']) for e in q.get('events', []))]
        rows = []
        explanations = []
        for arm in ['recent-window', 'summary-window', 'llm-summary-window', 'state-source-memory']:
            p = base / f'calls/{q["id"]}-{arm}.json'
            if not p.exists():
                continue
            r = read(p)
            answer = r.get('parsed', r.get('answer', r.get('result')))
            rows.append([arm, num(r.get('correct')), answer, link(p)])
            value = answer.get('answer') if isinstance(answer, dict) else answer
            if str(q['expected']).upper() == 'UNKNOWN':
                why = f'参考要求承认未知，实际输出 `{value}`；' + ('本题未编造缺失事实。' if r.get('correct') else '未满足该题的未知/拒答要求。')
            else:
                why = f'本题参考值为 `{q["expected"]}`，实际输出 `{value}`；' + ('保存的判定为正确，所需值/时间要求已满足。' if r.get('correct') else
                      '保存的判定为错误，未交付所需精确值/时间要求。仅凭输出不能确定丢失发生在截取、摘要还是回答环节，需回看该条 request。')
            explanations.append(f'**{arm} 为何得到该结果**：{why}')
        lines += [table(['基线', '判定', '实际回答/结构结果', '完整证据'], rows)] + explanations + [
                  '**为何、说明什么**：将实际输出与上列精确值/单位/日期或 UNKNOWN 比较；state-source-memory 可直接读取预置答案源，而最近窗口可能丢失早期事件。'
                  '具体输出在逐题证据中。这不是自然历史抽取测试；即使通过也只能说明给定状态的读取正确，不证明长期记忆学习能力，也不能据此声称优于同样通过的 LLM 摘要。']
    COUNTS['early_memory'] = len(suite)
    assert len(suite) == 30
    write('06_早期30题记忆基线.md', lines)


def rag_early():
    base = BASE / 'rag-evidence/final-rag-v20261109'
    suite = read(base / 'frozen-retrieval-suite.json')['queries']
    ablation = read(base / 'retrieval-ablation.json')
    practicalbase = BASE / 'rag-evidence/practical-retrieval-v1'
    practical = read(practicalbase / 'report.json')['reports']
    quality = read(practicalbase / 'evidence-quality.json')['cases']
    lines = ['# 02 · 初轮 RAG 14 题、结构 8 题、生成 6 题与增量 10 题',
             '来源：本项目论文库的主题检索开发题。14 题中 6 个历史主题 + 8 个结构定向问题；其中 6 题也用于初轮生成，8 题用于结构候选直接证据对照。'
             '这些集合有重叠，不能累计成相应数量的独立题。它们不是 RPC-Bench，标签不是穷尽 qrels 或独立片段金标。']
    for q in suite:
        ident = q['id']
        lines += [f'## 历史检索 {ident} · {q.get("group")}', block(q['query']),
                  '**标定相关论文**：' + ', '.join(q['relevant_paper_ids']) + '；题源：' + link(base / 'frozen-retrieval-suite.json')]
        rows = []
        explanations = []
        for mode, report in ablation.items():
            case = next((r for r in report.get('cases', []) if r['id'] == ident), None)
            if case:
                rows.append([mode, num(case.get('recall@5')), num(case.get('mrr@5')), num(case.get('recall@10'))])
                total = len(q['relevant_paper_ids'])
                found5 = round(case['recall@5'] * total)
                found10 = round(case['recall@10'] * total)
                rank = round(1 / case['mrr@5']) if case['mrr@5'] else '前5未出现'
                explanations.append(f'**{mode} 本题得分的可观察依据**：前5覆盖 {found5}/{total} 个已标相关论文，前10覆盖 {found10}/{total}，首个相关论文排名 {rank}。'
                                    + ('相关论文全部覆盖，差别主要体现在排位与具体片段。' if found5 == total else
                                       '已标相关集合尚未覆盖完整，不能把找到其中一篇当作全部召回。'))
        lines += [table(['204 份库消融', 'Recall@5', 'MRR@5', 'Recall@10'], rows), link(base / 'retrieval-ablation.json')] + explanations
        variants = []
        for mode, report in practical.items():
            r = next((r for r in report.get('cases', []) if r['id'] == ident), None)
            if r:
                variants.append([mode, num(r['recall@5']), num(r['mrr@5'])])
        if variants:
            lines += ['**此前 194 份库分块对照**\n\n' + table(['原页级 / 结构候选', 'Recall@5', 'MRR@5'], variants), link(practicalbase / 'report.json')]
        grades = []
        for variant in quality:
            r = next((r for r in variant['queries'] if r['query_id'] == ident), None)
            if r:
                grades.append([variant['variant'], r['grades'], r['direct_at_3'], r['supported_at_3']])
        if grades:
            lines += ['**直接证据检查**\n\n' + table(['方案', 'top3 相关等级', 'direct@3', 'supported@3'], grades),
                      '等级来自盲评文本银标；direct 比仅有背景支持更严格。' + link(practicalbase / 'evidence-quality.json')]
            old, new = grades[0][2], grades[1][2]
            lines += ['**本题结构实验说明**：' + ('结构候选丢失了前三项直接证据，虽然可能仍有相关背景；这是未迁移的具体负例。' if new < old else
                      '本题结构候选没有降低 direct@3，但不代表其他题也不降或全局胜出。') +
                      '论文命中、背景相关和直接可答证据是不同层次，不能以论文 Recall 高掩盖片段质量。']
        lines += ['**检索结果为何如此、说明什么**：每组 R5 由前五论文 ID 与给定 qrels 求交得到，MRR 看首个已标相关论文位置。'
                  '高召回只证明主题定位；较低 MRR 是排位问题，R5 不满是已标集合覆盖问题。历史逐题没有独立组件根因标注，不补造“embedding 不懂语义”等原因。']
    protocol = read(base / 'generation-protocol.json')
    lines += ['# 初轮生成六题：与扩大测评协议分开']
    for q in protocol['questions']:
        ident = q['id']
        lines += [f'## 生成 {ident}', block(q['query'])]
        rows = []
        for mode in ['dense', 'full']:
            p = base / f'generation-{ident}-{mode}.json'
            r = read(p); judge = r.get('judge', r.get('judgment', {}))
            if not judge:
                judge = next((v for v in r.values() if isinstance(v, dict) and 'claims' in v and 'faithfulness' in v), {})
            values = [read(base / f'ragas-{ident}-{mode}-{m}.json') for m in ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']]
            rows.append([mode] + [num(v.get('score')) for v in values])
            lines += [f'**{mode} 实际缺口/评审说明**\n\n' + block(judge.get('missing_evidence', judge.get('reason', '该输出未保存单独根因文字；请查看原始声明判定。'))), link(p)]
        lines += [table(['真实 RAGAS', 'Faithfulness', 'Answer relevancy', 'Context precision', 'Context recall'], rows),
                  '**说明什么**：原参考从 dense 证据生成，可能偏向基线；忠实性提高与参考召回下降可同时发生，不能据此声称整链路全面胜出。'
                  'NA 表示评分未成功，不补零、不静默丢弃；trajectory_metrics 的 dense precision 被提供方安全策略拒绝，保留失败尝试。']
    appended = read(base / 'append-retrieval.json')['cases']
    lines += ['# 增量新增论文专项十题', '来源：10 份实际新增 PDF 各出一个定向发现问题。旧 194 份库没有对应新 paper ID，before=0 只说明新 ID 尚未入库，不说明旧论文不能回答该主题。']
    for q in appended:
        ident = q['paper_id']
        hits = [(i + 1, h) for i, h in enumerate(q['hits']) if h['paper_id'] == ident]
        lines += [f'## 新增 {ident}', block(q['query']),
                  f'Recall@5={num(q.get("recall@5"))}，MRR@5={num(q.get("mrr@5"))}；期望新增论文路径：`{q.get("path")}`。',
                  '**为何、说明什么**：' + '; '.join(f'目标论文实际在第 {i} 位，PDF p.{h.get("page")}，标题 {h.get("title")}' for i, h in hits[:2]) +
                  '。证明新增索引可被查询发现，不证明生成答案正确、全部新论文解析完整或论文创新性。' + link(base / 'append-retrieval.json')]
    COUNTS['rag_early'] = {'retrieval': len(suite), 'structure_subset': len(quality[0]['queries']), 'generation_subset': len(protocol['questions']), 'append': len(appended)}
    write('02_RAG历史结构消融与增量问题.md', lines)


def alignment():
    definitions = [('gold1', BASE / 'rag-evidence/alignment-gold1-refinement/gold1.json', BASE / 'rag-evidence/alignment-gold1-refinement/report.json'),
                   ('gold2', BASE / 'rag-evidence/alignment-gold2-confirmed/gold2.json', BASE / 'rag-evidence/alignment-gold2-confirmed/report.json'),
                   ('gold3', BASE / 'rag-evidence/typed-structure-pilot/gold3.json', BASE / 'rag-evidence/typed-structure-pilot/report.json')]
    lines = ['# 07 · 人工金标“题”：具体论文行、页与段落归属',
             '这些不是问答题，是“给定 PDF 一行，应否对齐、归属哪个段落/章节”的人工位置判断。各批属于诊断选样，不是随机总体精度；gold3 的六例和前批同论文，不能冒充论文隔离泛化。']
    count = 0
    for label, p, rp in definitions:
        raw = read(p)
        cases = raw if isinstance(raw, list) else raw.get('cases', raw.get('rows', []))
        report = read(rp)
        comparisons = {r.get('case_id'): r for r in report.get('cases', [])}
        for q in cases:
            count += 1; ident = q.get('case_id', q.get('id', count))
            lines += [f'## {label} · {ident}', f'论文 `{q.get("paper_id")}`，PDF p.{q.get("page")}，行号 `{q.get("line_ordinal")}`。',
                      '**实际待判断文本**\n\n' + block(q.get('text', q.get('source_text'))),
                      f'**人工答案**：可对齐 `{q.get("is_alignable")}`；目标 paragraph `{q.get("expected_paragraph_id")}`；section `{q.get("expected_section_id")}`。',
                      '**人工判断理由（原文）**\n\n' + block(q.get('review_notes', q.get('notes', '该结构化条目未附理由全文；见对应人工评审 Markdown。')))]
            comp = comparisons.get(ident)
            if comp:
                lines += ['**算法实际结果**\n\n' + block(json.dumps(comp, ensure_ascii=False, indent=2))]
            lines += ['**说明什么**：判断的是当前选定行的几何/文本归属。短公式、算法续行、跨栏或跨段片段的正确主归属，不等于完整语义段无损恢复；'
                      '算法正确也不能自动推出检索效果更好。若本条没有逐样本预测，只有人工标注，就不能补一个“通过”。' + link(p, '人工结构化标签') + ' · ' + link(rp, '对应实验结果')]
    COUNTS['alignment_labels'] = count
    write('07_人工金标位置逐项说明.md', lines)


def regression():
    xml = BASE / 'agent-evidence/v20261110-architecture-live/langgraph-regression.xml'
    raw = xml.read_bytes(); INPUTS[xml.relative_to(ROOT).as_posix()] = hashlib.sha256(raw).hexdigest()
    cases = list(ET.fromstring(raw).iter('testcase'))
    lines = ['# 08 · 最终 121 项工程回归的具体测试与断言',
             '工程用例不是自然语言问答题，也不是 121 个独立科研任务。它们检查宿主代码的确定性约束、生命周期、预算、安全、MCP、会话、DAG 和恢复。'
             '下列通过状态来自当时最终 XML，没有在本次重跑。源码断言从当前保留代码抽取，属于定位辅助；当前源码未与每项历史运行逐字绑定，历史最终 source 快照仍是复核依据。']
    for i, q in enumerate(cases, 1):
        module, name = q.attrib['classname'], q.attrib['name']
        p = ROOT / (module.replace('.', '/') + '.py')
        rawname = name.split('[')[0]
        assertions, doc, targetline = [], None, None
        if p.exists():
            source = p.read_text(encoding='utf-8-sig')
            INPUTS[p.relative_to(ROOT).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
            tree = ast.parse(source)
            node = next((n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == rawname), None)
            if node:
                targetline = node.lineno
                doc = ast.get_docstring(node)
                for n in ast.walk(node):
                    if isinstance(n, ast.Assert):
                        assertions.append(ast.get_source_segment(source, n))
                    elif isinstance(n, (ast.With, ast.AsyncWith)):
                        for item in n.items:
                            expr = ast.get_source_segment(source, item.context_expr)
                            if expr and ('raises(' in expr or 'warns(' in expr):
                                assertions.append(expr)
        passed = not any(q.find(tag) is not None for tag in ['failure', 'error', 'skipped'])
        lines += [f'## {i:03d} · {name}', f'来源模块 `{module}`；历史结果：{num(passed)}；XML 耗时 {q.attrib.get("time")} 秒。']
        if doc:
            lines += ['**测试目标说明**\n\n' + block(doc)]
        if assertions:
            lines += ['**它实际上要求什么**\n\n```python\n' + '\n'.join(assertions) + '\n```']
        else:
            lines += ['**它实际上要求什么**：本条未抽取到显式 assert，可能在 helper/fixture 内验收；请查看完整测试，不能从名称补造断言。']
        lines += ['**为何通过、说明什么**：历史执行未触发该用例失败/错误，说明在该 fixture 和输入下上述确定性要求成立。'
                  '这不是对所有并发交错、任意崩溃或所有科学问题的形式证明，也不代表自然语言答案正确率。' +
                  (link(p, f'当前源码 L{targetline}') if p.exists() else '') + ' · ' + link(xml, '历史 XML')]
    COUNTS['final_regression_cases'] = len(cases)
    assert len(cases) == 121
    write('08_工程121项具体用例与断言.md', lines)


def sqlite_questions():
    suitepath = ROOT / 'config/benchmarks/rag_incremental_v1.json'
    suite = read(suitepath)
    reportpath = ROOT / 'self-revise/backend-revise/internal_experiments/assets/knowledge_base/workspace_rag_fusion_v1/report.json'
    report = read(reportpath)
    details = report['answer_details']
    ids = {d['query_id'] for d in details}
    lines = ['# 09 · SQLite 时代 12 个研究问题：题源及旧 Fusion 对照',
             '来源：本项目固定 ViT/参数高效微调（PEFT）研究开发问题，非公共基准；不是当前时间序列问答集。'
             '原题来自 config/benchmarks/rag_incremental_v1.json，此处只列历史 Fusion report 实际出现的 12 个问题。'
             '同题也被旧 P7 A/B/C 使用；本节 Fusion 分数不能拿来冒充 P7 分数。',
             'legacy/cpu/fusion 的事实支持、correctness、question_coverage、design_usefulness 均为该历史协议的银标，不是 RAGAS。']
    count = 0
    for idea in suite['ideas']:
        for q in idea['queries']:
            if q['id'] not in ids:
                continue
            count += 1
            lines += [f'## {q["id"]} · {idea["id"]}', '**实际问题**\n\n' + block(q['text']), '**所属研究意图**：' + idea['idea'],
                      '**题源**：' + link(suitepath)]
            matched = [r for r in details if r['query_id'] == q['id']]
            lines += [table(['方案', '支持事实/总事实', '正确性', '覆盖度', '设计可用性'],
                            [[r['mode'], f'{r["supported"]}/{r["facts"]}', r.get('correctness'), r.get('question_coverage'), r.get('design_usefulness')] for r in matched])]
            for r in matched:
                lines += [f'**{r["mode"]} 的具体原因（历史评审原文）**\n\n' + block(r.get('reason'))]
            lines += ['**说明什么**：检验机制解释、证据适用范围和受控实验草案；回答支持率和设计评分可以不同。'
                      '提出视觉迁移设计不等于已验证视觉收益，也不代表训练已经完成。' + link(reportpath)]
    COUNTS['sqlite_historical_questions'] = count
    assert count == 12
    write('09_SQLite历史12题及旧基线.md', lines)


def historical_skills_plans():
    base = ROOT / 'self-revise/backend-revise/internal_experiments/assets/knowledge_base'
    skillpath = ROOT / 'config/benchmarks/skills_v1.json'
    suite = read(skillpath)['cases']
    reportpath = base / 'workspace_skills_v1/report.json'
    report = read(reportpath)
    auditpath = base / 'workspace_skills_v1/audit.json'
    audit = read(auditpath)
    lines = ['# 10 · 历史 Skills 十二题、成本对照与六个方案交付任务',
             '这些都是本项目内部题，不是公开 Agent 基准。Skills 测路由与受限执行；六个方案任务的终点为研究规格，不是训练完成或 coding 成功。']
    for q in suite:
        r = next(c for c in report['routing']['cases'] if c['id'] == q['id'])
        lines += [f'## Skills {q["id"]}', block(q['task']),
                  f'题源：{link(skillpath)}；预期 `{q["expected"]}`，实际 `{r["selected"]}`，已加载 `{r["loaded"]}`；判定 {num(r["correct"])}。',
                  '**为何、说明什么**：实际能力集合与开发者路由期望匹配；空集合是有意拒绝当前 Skills 范围外任务，而不是成功执行训练或天气预测。'
                  '它只能支持已测试问法的能力选择，不证明后续计划的科学正确性。' + link(reportpath)]
    lines += ['# Skills lazy/eager 两次重复：为什么节省 token',
              '**实际重复问题**\n\n' + block('只在现有知识库检索LoRA的低秩更新机制并引用原文。如果库里没有内容，明确报告证据不足，不要下载或规划。'),
              table(['模式', '重复', '首提示 tokens', '总实际 tokens', '成功'],
                    [[r['mode'], r['repeat'], r['first_prompt_tokens'], r['actual_tokens'], num(r['pass'])] for r in audit['context_comparison']]),
              '**实际任务与边界**：原报告明确这是空语料下拒答的两次重复，不是四个不同科研问题；lazy 只按需加载技能，eager 将完整技能说明提前放进上下文，'
              '因此初始提示更短。总成本仅比较已成功重复，失败及未知预留另存，不能泛化为所有科研任务都省 57.4%。'
              '这里使用最终 audit 的 4517/4604 对 10698/10700，不采用旧 report 的恢复前视图。' + link(auditpath)]
    ideaspath = ROOT / 'config/idea_library_vit_peft_heldout_v2.json'
    ideas = read(ideaspath)['ideas']
    firstpath = base / 'workspace_precoding_v4_heldout/results.json'
    secondpath = base / 'workspace_precoding_v41_revalidation_live/results.json'
    first, second = read(firstpath), read(secondpath)
    lines += ['# 六个旧方案任务：首次 5/6，修复复验 6/6',
              '同题复验已曝光并复用 checkpoint，不是独立新增六题；只测到研究规格/方案交付，不能称为六次训练成功。']
    for q in ideas:
        ident = q['idea_id']; key = 'B_' + ident
        if key not in first:
            continue
        lines += [f'## 方案 {ident} · {q["title"]}', block(q['research_question']), '**题源**：' + link(ideaspath)]
        for label, data, path in [('首次', first, firstpath), ('修复复验', second, secondpath)]:
            r = data[key]
            mech = r.get('mechanical', {})
            attempts = [{k: a.get(k) for k in ['attempt', 'mechanical', 'compile_error', 'critic_pass', 'confirmation_pass', 'success']} for a in r.get('attempts', [])]
            lines += [f'**{label} 结果**：success={r.get("success")}，silver_accepted={r.get("silver_accepted")}，first_pass_success={r.get("first_pass_success")}。',
                      '**逐次验收结果**\n\n' + block(json.dumps(attempts, ensure_ascii=False, indent=2)),
                      '**为何、说明什么**：' + ('本批冻结规格/引用/约束与评审关卡通过，支持交付该草案。' if r.get('success') else
                      '本批未交付合格冻结规格，具体失败关卡见上列，不能用局部引文或已生成计划代替成功。') +
                      ' 这仍不代表该科学假设成立，不能把 planned_jobs 当作已经执行的训练。' + link(path)]
            failed = [k for k, v in mech.get('checks', {}).items() if v is False]
            if failed:
                lines += ['**可直接定位的失败检查**：' + ', '.join(failed) + '。这是现存检查结果；完整冻结失败的文本/critic 原因需查看任务工件，不从 false 标签编造细节。']
    COUNTS['historical_skills_plans'] = {'skill_routing': len(suite), 'planning_tasks': len(first), 'cost_repeats': 2}
    write('10_历史Skills与六个方案任务.md', lines)


def live_cases():
    lines = ['# 11 · 真实 API、长历史、Skills、DAG 与非问答性能用例',
             '以下是端到端验收场景，不并入公开题准确率，也不和 24 题 Agent 开发集相加。性能用例不是自然语言“题”，应说明输入、操作及不变量。']
    memorypath = BASE / 'memory-evidence/single-session-v20261110/live.json'
    live = read(memorypath)
    expected = {'recall': 'Jetson Orin NX 与 0.006；须从同一会话早期原话找回。',
                'update': 'Jetson Orin NX 与 0.004；新修正覆盖旧阈值，设备不变。',
                'fresh_session': 'UNKNOWN；空白会话没有上述私有事实。'}
    for mode in ['recall', 'update', 'fresh_session']:
        r = live[mode]
        question = [m['content'] for m in r['history'] if m.get('role') == 'user'][-1]
        lines += [f'## 单会话现场 · {mode}', block(question), '**预期**：' + expected[mode],
                  '**实际回答**\n\n' + block(r['answer']),
                  '**为何、说明什么**：将实际设备/阈值/UNKNOWN 对照预期，来源回读和当前修正有效，空白会话未继承私有事实。'
                  '源会话插入 100 轮背景、未预写答案记忆；仅证明此科研参数场景，不证明任意长对话。'
                  f' 实测 {num(r.get("seconds"))} 秒。' + link(memorypath)]
    apipath = BASE / 'agent-evidence/v20261110-final/live-api.json'
    api = read(apipath)
    for i, r in enumerate(api['turns'], 1):
        lines += [f'## Agent API 连续对话 · {i}', block(r['message']), '**实际回答**\n\n' + block(r['answer']),
                  '**为何、说明什么**：该轮验收围绕设置目标、半途转向、保留 Jetson/0.005 约束、空白新会话 UNKNOWN。'
                  '最终 State 由 forecasting 改为 streaming compression，记录的是用户要求，不是误差保证已由实验验证。'
                  f' 单轮 {num(r.get("seconds"))} 秒。' + link(apipath)]
    skillpath = BASE / 'agent-evidence/v20261110-final/live-skill.json'
    skill = read(skillpath)
    sourcepath = ROOT / 'self-revise/backend-revise/scripts/verify_research_skill_live.py'
    source = sourcepath.read_text(encoding='utf-8')
    INPUTS[sourcepath.relative_to(ROOT).as_posix()] = hashlib.sha256(sourcepath.read_bytes()).hexdigest()
    tree = ast.parse(source)
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'run_conversational_agent')
    prompt = ast.literal_eval(call.args[0])
    lines += ['## 真实 Skills 发现/加载/调用', block(prompt), '**实际检查**：' + text(skill['checks']),
              '**实际答案**\n\n' + block(skill['result']['answer']),
              '**为何、说明什么**：协议事件三项通过，但最终科学答复 partial；只能证明技能实际经过 discover/load/invoke，不能证明该次科学任务完整成功。' + link(skillpath)]
    dagpath = BASE / 'agent-evidence/v20261110-architecture-live/dag-live/tasks/dag-live/langgraph/result.json'
    dag = read(dagpath)
    lines += ['## 真实两节点 DAG', block(dag['prompt']), '**实际答案**\n\n' + block(dag['result']['answer']),
              '**为何、说明什么**：任务 context 先读取 State，queries 依赖 context，再生成 Scholar/arXiv 检索式；accepted 与依赖次序通过。'
              '用户明确要求建图，且 requires_evidence=false，不是开放任务自主规划成功率。没有真实检索/下载/实验。' + link(dagpath)]
    lines += ['# 增量、no-op 与恢复不是问答题',
              table(['用例', '实际输入与操作', '预期不变量', '已有结果说明'], [
                  ['增量', '原 194 份库 + 用户下载的 12 份 PDF', '哈希去重；只解析/编码新内容；旧 Child/向量不变', '2 重复 + 10 新增；101.156 秒；旧 10337 向量最大差 0'],
                  ['no-op', '同一批 PDF 不改内容再次 ingest', '解析/embedding/dense/sparse 写入均 0', '20.653→6.485 秒；修复无变化仍写旧向量'],
                  ['结构缓存', '首次开启结构层后对已有 204 份回填，再检查', '按哈希复用已解析结构', '首次 24.374 秒，复用 0.163 秒；不是新增 10 份的同一次计时'],
                  ['BM25', '新增后刷新统计与 sparse', 'IDF 与全库一致', '10955 点、4.042 秒；属于全局统计刷新，不虚称纯局部'],
                  ['恢复/幂等', '故障后显式恢复已落盘 run', 'scope/hash/预算不变；非只读效果不明不重放', '具体 assert 见 121 项回归，不能声称任意崩溃 exactly-once'],
                  ['2000 份上限', '配置容量保护，非实测输入', '超过限制拒绝或明确处理', '未做 2000 份压测，不能列作通过的容量题']]),
              link(BASE / 'rag-evidence/final-rag-v20261109/incremental-integrity.json', '增量完整性') + ' · ' +
              link(BASE / 'rag-evidence/final-rag-v20261109/incremental-noop-optimized.json', 'no-op 原结果')]
    COUNTS['live_scenarios'] = {'single_session_questions': 3, 'agent_api_turns': 4, 'skills': 1, 'dag': 1}
    write('11_现场问题与性能用例说明.md', lines)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rag(); rag_early()
    memory('single-session-v20261110', '03 · 全部 156 个单会话 oracle 问题 + 36 个派生长历史案例', True)
    memory('longmemeval-v4', '04 · 历史跨会话 LongMemEval-S 48 题', False)
    agents(); legacy_memory(); alignment(); regression(); sqlite_questions()
    historical_skills_plans(); live_cases()
    for path, expected in INPUTS.items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path
    for p in LINKS:
        assert p.exists(), p
    original_report = BASE / '后端测试与性能总报告_v20261110.md'
    prefix = original_report.read_bytes().partition(b'\n## 8.')[0]
    expected_report_hash = '16036301660ee6bcb75b19ea4a0892097e3538e8ce48d9483d380f8019c577c5'
    assert hashlib.sha256(prefix).hexdigest() == expected_report_hash, 'Original report body changed'
    appendix = BASE / '后端逐题测评附录_v20261110.md'
    bad_links = []
    link_count = 0
    documents = list(OUT.glob('*.md')) + [appendix, original_report]
    for doc in documents:
        for target in re.findall(r'(?<!!)\[[^\]\n]+\]\(([^)\n]+)\)', doc.read_text(encoding='utf-8')):
            if target.startswith(('https:', 'http:', '#')):
                continue
            link_count += 1
            if not (doc.parent / target.split('#')[0]).exists():
                bad_links.append({'document': doc.name, 'target': target})
    assert not bad_links, bad_links
    manifest = {'scope': 'Read-only extraction; no model calls or rescoring. Original English questions and recorded judge reasons retained.',
                'counts': COUNTS, 'source_sha256': INPUTS, 'source_files_unchanged': True,
                'evidence_link_occurrences': len(LINKS),
                'local_links_checked': link_count, 'broken_links': bad_links,
                'original_report_body_sha256': expected_report_hash,
                'original_report_body_unchanged': True,
                'documents': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.md')}}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'counts': COUNTS, 'source_files': len(INPUTS), 'evidence_links': len(LINKS),
                      'local_links_checked': link_count, 'unchanged': True, 'original_report_body_unchanged': True}, ensure_ascii=False))


if __name__ == '__main__':
    main()

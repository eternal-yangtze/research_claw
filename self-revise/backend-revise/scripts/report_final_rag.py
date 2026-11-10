"""Summarize measured release evidence; never fills missing scores with success."""
import hashlib
import json
from pathlib import Path
import sys
import statistics

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(ROOT / 'self-revise/backend-revise/scripts'))
from research_rag import OUT, write


def read(name): return json.loads((OUT / name).read_text(encoding='utf8'))


def main():
    audit = read('final-integrity-audit.json')
    retrieval = read('retrieval-ablation.json')
    hits = read('retrieval-hits.json')
    suite = {q['id']: q for q in read('frozen-retrieval-suite.json')['queries']}
    for mode, value in retrieval.items():
        precisions = []
        for row in value['cases']:
            known = set(suite[row['id']]['relevant_paper_ids'])
            top = hits[mode + ':' + row['id']][:5]
            row['precision@5_known_qrels'] = sum(h['paper_id'] in known for h in top) / 5
            precisions.append(row['precision@5_known_qrels'])
        value['mean']['precision@5_known_qrels'] = statistics.mean(precisions)
        warm = sorted(r['seconds'] for r in value['cases'][1:])
        value['warm_p50_seconds'] = statistics.median(warm)
        value['warm_p95_seconds'] = warm[__import__('math').ceil(.95 * len(warm)) - 1]
    write('retrieval-ablation.json', retrieval)
    generation = read('generation-summary.json')
    ragas = read('ragas-summary.json')
    assert generation['cases'] == ragas['cases'] == 6
    failures = [r for r in ragas['scores'] if r['status'] != 'scored']
    sources = list(OUT.glob('generation-*-full.json'))
    structure = sum(c['structural_match'] for f in sources for c in json.loads(f.read_text(encoding='utf8'))['contexts'])
    full_failures = [r for r in failures if r['mode'] == 'full']
    passed = audit['status'] == 'passed' and not full_failures
    admission = {'status': ('passed_for_personal_research_rag_with_baseline_eval_gap' if failures else
        'passed_for_personal_research_rag') if passed else 'needs_attention',
        'ragas_scored': len(ragas['scores']) - len(failures), 'ragas_expected': 48,
        'ragas_failures': failures, 'structure_context_matches_in_generation': structure,
        'full_pipeline_ragas_scored': sum(r['mode'] == 'full' and r['status'] == 'scored' for r in ragas['scores']),
        'baseline_gap_reason': 'One dense baseline context_precision request rejected by provider safety system; three attempts retained, no score imputed',
        'not_claimed': ['production readiness', 'independent public benchmark SOTA', 'human-certified answer quality',
                        'structure child chunking beats measured baseline', 'global research novelty proof']}
    for mode, values in generation['results'].items():
        if mode == 'full' and (values['faithfulness'] < .8 or values['answer_relevance'] < .8):
            admission['status'] = 'needs_attention'
    write('acceptance.json', admission)
    lines = ['# ResearchClaw RAG交付与实测（2026-11-09）', '',
      '本次在原项目内补齐科研RAG链路。原194篇已备份；12份输入中2份为原库同哈希文件，新增10份PDF记录，当前204份。另发现经典语音识别DTW论文与原库同题但PDF哈希不同，疑似不同扫描/版本，已在identity-review.json标记。因此204是按PDF哈希去重后的记录数，不保证204篇独立学术作品；本轮不擅自删除这份已入库论文。', '',
      '## 实现与环境', '',
      '- Python 3.12 / 项目`.venv`，本地CPU运行embedding和reranker；Docker只承载既有PostgreSQL 16.6、Qdrant（镜像digest固定）。',
      '- PostgreSQL保存正文、页码、父子关系、质量与结构缓存，embedding副本为0。Qdrant负责dense与BM25 sparse召回；向量检索就是此处的语义检索，不是独立的第三路。',
      '- embedding为BAAI/bge-small-en-v1.5，384维、归一化，离线与在线固定同一revision；reranker为ms-marco-MiniLM-L6-v2。精确revision见构建报告。',
      '- 查询规划输出原始意图、英文改写、至多2个同义/分步子查询、学术检索式和歧义标记；多路候选RRF(k=60)融合并交叉编码器重排。',
      '- 生成只使用可追溯论文证据，输出[S1]引用及PDF页码；区分论文结论、条件性推论与证据缺口。创新性仅相对于库内文献。', '',
      '## 论文结构：采用了什么，没有宣称什么', '',
      '正式child仍采用此前胜出的page-parent-word-child-v1：页级父块、约220词子块与重叠。前轮结构child替换在8题直接证据命中率上由75%降至37.5%，因此不强行迁移。',
      '新增paper-section-evidence-v1结构证据层：离线识别章节标题，按章节与页边界分组正文，保留公式/算法附近文本；在PostgreSQL按PDF哈希缓存204篇的结构段。检索命中后，只有完整child确实落入一个已识别章节段才选该段作为父上下文，否则回退原页级父块。上下文按命中位置聚焦到500词。',
      '这是结构感知的证据分组与父级扩展，不能表述为“全库已改成GROBID语义child”或“结构切分必胜”。标题识别为启发式，不等同于人工语义认证；表格、公式语义不做无依据恢复。当前保守清洗在参考文献处截止，参考文献后的附录覆盖仍是局限。', '',
      '## 检索侧：固定14题消融（扩容后204篇）', '',
      '|方案|Recall@5|MRR@5|nDCG@5|Recall@10|', '|---|---:|---:|---:|---:|']
    for mode, value in retrieval.items():
        r = value['mean']; lines.append(f"|{mode}|{r['recall@5']:.4f}|{r['mrr@5']:.4f}|{r['ndcg@5']:.4f}|{r['recall@10']:.4f}|")
    lines += ['', 'RRF在论文级排序上最好；reranker不是所有指标的赢家。暂保留已验证的rerank证据路径，并保留RRF消融，不能声称rerank带来全面提升。完整链路与原194篇的14题指标一致。论文级qrels不完备，因此不是片段精确率/真实科研全覆盖率。', '',
      '## 生成侧', '', '同一6题（压缩、误差控制、DTW等），dense父块基线与改写扩展+混合召回+rerank+章节证据对照。逐声明银标检查精确引文；另运行真实RAGAS 0.4.3。', '',
      '|方案|RAGAS Faithfulness|Answer relevancy|Context precision|Context recall|', '|---|---:|---:|---:|---:|']
    for mode, vals in ragas['means'].items():
        lines.append('|' + mode + '|' + '|'.join(f"{vals[k]['mean']:.4f} ({vals[k]['scored']}/6)" for k in ['faithfulness','answer_relevancy','context_precision','context_recall']) + '|')
    lines += ['', f"另行逐声明、精确引文校验的宏平均忠实度：dense={generation['results']['dense']['faithfulness']:.4f}，full={generation['results']['full']['faithfulness']:.4f}；答案相关性分别为{generation['results']['dense']['answer_relevance']:.4f}、{generation['results']['full']['answer_relevance']:.4f}。它们是自定义银标指标，不与RAGAS分数混用。", '',
      'RAGAS和生成均调用.env既有模型。48项有47项有效，完整链路24项全部出分；dense基线一题Context Precision被提供方安全系统以misalignment_policy_violation拒绝，3次尝试均保留，未补分或改换题目规避限制。基线该项均值分母为5，其余分母均为6。', '',
      '银标参考答案由dense基线证据生成，可能偏向基线；评估与生成使用同模型，存在自评偏差。6题是开发验证，不能包装为独立公共基准或人工金标。原始回答、证据、RAGAS逐项分数及调用账本均保留。Context recall等指标若回退，应保留为负结果，不能以其他指标提升将其掩盖；本轮不宣称完整链路在所有指标胜出。', '',
      '## 增量效率', '',
      '- 首次新增10篇：101.156秒，新增151父块、618子块；10337个旧向量复用，当前10955个子向量。',
      '- 首次启用章节证据缓存，另需对204篇做一次结构信息回填，耗时24.374秒（未计入上述101.156秒）；以后按哈希增量复用，全库无变化缓存检查实测0.163秒。',
      '- BM25因IDF依赖全库，刷新10955条稀疏索引耗时4.042秒；此项是全局统计刷新，不是PDF重解析或dense重新embedding。',
      '- 发现并修复旧no-op会重写dense点的问题；修复前20.653秒，修复后6.485秒；后者12篇全部复用、embedding=0、dense写入=0、sparse写入=0。',
      '- 12份输入的可提取文本质量规则均为A；这只说明提取文本的长度、乱码等规则通过，不代表原生PDF/OCR来源、阅读顺序、标题和公式都正确。两份明显错误的PDF标题已对照首页修正。',
      '- 10个新增论文专项问题：Recall@5=1.0，MRR@5=0.875；这是预期论文发现测试，不能解释为所有答案都正确。',
      f"- 全量核对{audit['old_vectors_compared']}个旧向量，最大绝对差{audit['maximum_vector_absolute_difference']:.3g}；原child哈希全部不变。", '',
      '## 运行', '', '在`D:\\my_coach\\new_agent\\ResearchClaw`执行，使用现有Docker基础服务及.env：', '', '```powershell',
      'docker compose --env-file infra/.env -f infra/compose.yaml up -d',
      '.\\.venv\\Scripts\\python.exe -X utf8 -B scripts/research_rag.py ask "压缩时间序列上做聚合查询时，如何提供确定性的误差保证？"',
      '.\\scripts\\search_research.ps1 "DTW lower bounds" -EnglishOnly',
      '.\\.venv\\Scripts\\python.exe -X utf8 -B scripts/research_rag.py ingest --source "D:\\你的新论文目录" --corpus-version "ts-next-batch"',
      '.\\.venv\\Scripts\\python.exe -X utf8 -B scripts/research_rag.py noop',
      '.\\.venv\\Scripts\\python.exe -X utf8 -B self-revise/backend-revise/scripts/audit_final_rag.py', '```', '',
      '`ingest`是可持续使用的增量入口：按PDF哈希去重，复用旧解析与dense向量，自动更新BM25和结构缓存，硬上限2000篇。新增revision发布时仍复制旧dense向量到新集合以便版本隔离，不能宣称网络I/O为纯O(新增量)。扫描PDF不能提取正文时排除；当前未自动启用OCR。原自动下载功能保留，下载与正式入库仍分离。', '',
      '备份目录：`self-revise/backend-revise/internal_experiments/assets/rag_backup_194_v20261109`。含pg_dump、Qdrant快照、配置与SHA256；已检查pg_restore目录及快照下载，未做隔离恢复演练。', '',
      '合格判断：入口可输出带页码证据的回答；204篇/10955个dense和BM25点一致；旧向量完整；no-op零重算/写入；11项回归通过；生成侧逐声明分数及RAGAS失败项如实显示。见acceptance.json。此RAG组件可进入后续上下文与记忆集成，不能据此宣称长短期记忆评测已完成。']
    if (OUT / 'cleanup-manifest.json').exists():
        cleanup = read('cleanup-manifest.json')
        lines += ['', '## 中间产物清理', '',
                  f"已按清单删除{len(cleanup['files'])}个过时候选缓存/重复结果，共{cleanup['bytes']}字节；人工金标哈希核对不变。当前基线所需typed样本、TEI/schema、原论文、正式库和194篇备份全部保留。删除文件未放入回收站，候选缓存可通过保留脚本重新生成。"]
        for pending in cleanup.get('pending_temporary_directories', []):
            lines.append(f"剩余测试临时目录`{pending['path']}`被Windows ACL拒绝，普通/沙箱外删除、权限继承重置和takeown均失败，尚未清除；这项清理不能标记为全部完成，需要有管理员权限的本机终端处理。")
    (OUT / '交付报告.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    files = [Path('scripts/research_rag.py'), Path('self-revise/backend-revise/scripts/audit_final_rag.py'), Path('self-revise/backend-revise/scripts/report_final_rag.py'),
             Path('src/researchclaw/knowledge/research_rag.py'), Path('src/researchclaw/knowledge/rag_pipeline.py'),
             Path('src/researchclaw/knowledge/rag_retrieval.py'), Path('src/researchclaw/evaluation/ragas_adapter.py'),
             Path('src/researchclaw/application/cli.py'), Path('self-revise/backend-revise/scripts/cleanup_rag_intermediates.py')]
    write('code-manifest.json', {str(p): hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in files})
    print(json.dumps(admission, ensure_ascii=False))


if __name__ == '__main__': main()

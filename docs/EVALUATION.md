# Agent 应用评测

评测层回答不同组件对应用的贡献：RAG 是否取回可用证据，答复是否受到来源支持，研究角色能否交接结构化方案，以及上下文加载与索引维护是否高效。应用成果见 [PERFORMANCE.md](PERFORMANCE.md)。

## 评测对象与指标

| 对象 | 主要指标 | 输入与结果 |
| --- | --- | --- |
| 检索与排序 | Recall@K、MRR、nDCG | 问题、实际候选、论文 / 段落相关性 |
| 引用与事实 | 引用定位、事实支持、Faithfulness | 实际答复、引用位置、实际原文 |
| 问答相关性 | Answer Relevancy | 问题与实际最终答复 |
| 上下文质量 | Context Precision、Context Recall | 实际上下文与有来源的参考 |
| 研究规划 | 方案交付、约束完整性、独立角色评审 | 方案、研究合同、证据与评审结果 |
| 上下文效率 | 提示与总 token 的同条件比较 | 相同任务的 lazy / eager 记录 |
| 知识维护 | 向量复用率、增量编码耗时 | 相同模型和语料增量 |

研究规格交付与实际训练是两个阶段。语义评分标明 LLM 银标及参考来源；相同题目的修复复验与新任务分别统计。

## RAGAS 与 LangChain 的接入

[ragas_adapter.py](../src/researchops/evaluation/ragas_adapter.py) 将宿主模型接口适配为 LangChain ChatModel，再接入 RAGAS wrapper；本地 SentenceTransformer 提供评测所需 Embedding。模型、温度、并发和调用参数由应用配置，评测模型请求保存独立记录。

RAGAS 样本来自真实运行工件：

| 字段 | 来源 |
| --- | --- |
| `user_input` | 用户实际问题 |
| `response` | 实际最终答复 |
| `retrieved_contexts` | 生成器实际看到的搜索预览和原文回读，按发生顺序保留 |
| `reference` | 有来源的参考答案 |
| `reference_contexts` | 参考所依据的原文 |

参考仅提供给评测侧；模型银标参考附 `human_reviewed=false`。问答指标用于论文分析和知识回答，研究合同与源码采用各自的领域完成条件。

## 安装与独立环境

[requirements.txt](../requirements.txt) 已包含 RAGAS 与 LangChain 兼容组合。如需把评测进程与应用服务分开，可使用只安装评测能力的独立环境：

```powershell
python -m venv .venv-ragas
.\.venv-ragas\Scripts\python.exe -m pip install -r requirements-eval.txt
```

[requirements-eval.txt](../requirements-eval.txt) 声明 RAGAS、datasets、LangChain、LangChain Core、Community、OpenAI adapter 与 SentenceTransformer 的兼容版本。它既可独立安装，也被完整应用依赖入口引用；模型 API 与本地 Embedding 配置沿用应用设置。

自定义评测可以从业务工件整理上述字段，再使用现有 adapter 的异步 `score_metric` 与 `METRICS` 接口；该接口接收应用内部的 `question`、`response` 和 `contexts` 记录，并转换为 `SingleTurnSample`。参数及模型装配以 [ragas_adapter.py](../src/researchops/evaluation/ragas_adapter.py) 为准。

## 从对照到应用策略

直接生成、有限修订和 MicroAgent 主动查证分别服务不同的任务深度。结构清晰的问答优先采用有限修订，论文解释与方案评审采用可配置工具循环。应用以完整任务的证据质量和交接结果选择推理路径，持续扩展以独立任务集和人工参考为依据。

评测元信息保留问题来源、语料身份、模型与提示版本、上下文及指标定义。任务执行记录与正式工件在各自工作区关联，支持结论追溯。

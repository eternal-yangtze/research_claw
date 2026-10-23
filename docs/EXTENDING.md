# 扩展 Runtime

ResearchOps 将模型协议、工具执行、Skills、局部 Agent 和领域编排分开。接入新领域时，先提供可复用的业务函数，再根据任务需要将其注册为工具、Skill 或图节点。

## 选择扩展位置

| 需求 | 扩展点 | 示例 |
| --- | --- | --- |
| 增加一种可执行动作 | `ToolSpec` + `ToolRegistry` | 搜索一个数据源、读取一个文档节点 |
| 增加一项按需发现的领域能力 | Skill 文件 + 宿主绑定 | 文献获取、研究规划 |
| 为任务增加查证与修订 | `MicroAgentSpec` + `MicroAgentRunner` | 文献分析、结构化方案评审 |
| 增加阶段分支和恢复 | LangGraph 节点与状态 | 多角色评审、合同交接 |
| 将能力迁移到独立工具进程 | MCP client/server adapter | 只读论文原文服务 |
| 接入向量服务 | `DenseVectorBackend` | local / Qdrant |
| 增加使用入口 | 入口 adapter | CLI、FastAPI 项目问答 |

## 添加工具

工具注册包含名称、描述、输入 Schema、handler、可用 Agent、读写语义和输出长度。工具 Schema 提供给模型，执行授权在 ToolExecutor 再次检查。资源范围由宿主绑定，handler 检查项目、路径或论文归属。

参考 [ToolSpec / ToolExecutor](../src/researchops/harness/runtime.py) 和 [StudyRoles 的领域工具](../src/researchops/experiments/study_agents.py)。通用 handler 保持可由普通 Python 调用，Agent 工具层负责上下文与执行边界。

## 添加 Skill

现有能力位于：

| Skill | 对应工作 |
| --- | --- |
| [document-ingest](../skills/document-ingest/SKILL.md) | 文档解析与知识入库 |
| [evidence-retrieve](../skills/evidence-retrieve/SKILL.md) | 证据检索与来源整理 |
| [literature-acquire](../skills/literature-acquire/SKILL.md) | 文献获取与分析 |
| [study-plan](../skills/study-plan/SKILL.md) | 研究规划与正式合同 |

新增目录采用 `skills/<能力名>/SKILL.md`，前置元数据包含 `name`、`description`；正文说明适用任务、输入、执行步骤和输出语义。然后在 [ResearchOpsApp](../src/researchops/application/workflow.py) 的宿主绑定中注册对应工具。

模型先发现元信息，再加载正文和输入 Schema，调用时提交加载 hash。Skill 版本与当前会话加载记录对应，内部业务动作仍经过 ToolExecutor。外部论文与用户上传资料作为知识内容进入 RAG，Skill 扩展由应用维护者管理。

## 添加 MicroAgent

参考 [PaperAnalysisAgent](../src/researchops/papers/analysis_agent.py)。每个局部 Agent 定义任务角色、System Prompt、允许工具、结构化输出与 validator。MicroAgentRunner 在有限轮次内执行模型决策与工具观察，并把具体校验反馈交给下一轮。

独立角色上下文只接收本任务证据；工具可读取宿主绑定的资源。领域完成条件写在 validator 和宿主交接处，最终结果包含结构化内容、引用和停止位置。局部 Agent 的 checkpoint 与外层运行标识关联，恢复时读取相同输入版本。

## 扩展 LangGraph 编排

[StudyWorkflow](../src/researchops/orchestration/study_graph.py) 是已有范例。图状态保存可序列化的数据与工件引用，服务对象、客户端与锁由宿主装配。节点调用既有服务或角色 Agent，条件边表达候选修订、评审或提交。

增加节点时，明确其输入、输出工件、重复进入语义和角色上下文。正式合同由宿主编译器提交，后续 Coding 消费已提交版本。图 checkpointer 记录阶段位置，MicroAgent checkpoint 记录局部执行。

## 接入 MCP 工具

[paper_mcp.py](../src/researchops/integrations/paper_mcp.py) 包含当前官方 SDK client/server 闭环。新增 server 工具时，在启动阶段绑定可访问资源，Schema 接收业务参数，server 再次检查资源归属。stdio 的 stdout 用于协议，普通日志走 stderr。

模型通过 Tool Calling 请求本地工具，ToolExecutor 再调用 MCP client；MCP 工具结果回到相同 Agent 观察上下文。协议层与模型决策层保持各自职责。

## 接入向量后端或模型

[DenseVectorBackend](../src/researchops/knowledge/vector_backend.py) 为 dense 候选提供统一接口。新后端需要保留项目范围、chunk 身份、向量版本、内容变化和索引状态；检索结果继续通过 SQLite 回读原文。

模型通过 [ModelPort / OpenAICompatibleModel](../src/researchops/harness/runtime.py) 接入。适配器返回文本、结构化工具调用和 usage，调用层统一处理执行记录。应用评测可通过 [RAGAS adapter](../src/researchops/evaluation/ragas_adapter.py) 复用模型接口，见[评测说明](EVALUATION.md)。

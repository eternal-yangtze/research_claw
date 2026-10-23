# 配置说明

配置模板为 [.env.example](../.env.example) 与 [.env.web.example](../.env.web.example)。CLI 自动加载项目配置；Web 先读取 `.env.web` 再读取模型配置。已有进程环境变量优先，Compose 的 `environment` 覆盖同名 `env_file` 值。

## 模型与文档

| 变量 | 默认值 / 设置方式 |
| --- | --- |
| `RESEARCHOPS_MODEL_BASE_URL` | 填服务商根 URL，Adapter 追加 `/chat/completions` |
| `RESEARCHOPS_MODEL_API_KEY` | 填模型 API key |
| `RESEARCHOPS_MODEL_NAME` | 填支持 Tool Calling 的准确模型 ID |
| `RESEARCHOPS_MODEL_TIMEOUT_SECONDS` | `180`，组件执行还受分配时限约束 |
| `RESEARCHOPS_PDF_PARSER` | `auto` 优先 Docling，并提供 PyMuPDF 回退；也可指定 `docling` 或 `pymupdf` |
| `RESEARCHOPS_ACQUIRE_WORKERS` | `4`，文献获取并行节点数，范围 1–16 |

Docling 为按需安装项，安装命令见[运行指南](../RUNNING.md)。

## RAG 与向量后端

| 变量 | 默认值 / 作用 |
| --- | --- |
| `RESEARCHOPS_RETRIEVAL_BACKEND` | `neural`；`hashing` 为轻量本地检索 |
| `RESEARCHOPS_VECTOR_BACKEND` | `local`；可切换 `qdrant` |
| `RESEARCHOPS_QDRANT_URL` | `http://127.0.0.1:6333` |
| `RESEARCHOPS_QDRANT_COLLECTION` | 留空按项目自动生成 alias；手动指定时各项目使用独立名称 |
| `RESEARCHOPS_PROJECT_ID` | 留空时从存储路径生成项目身份 |
| `RESEARCHOPS_CORPUS_VERSION` | `v1`，语料版本 |
| `RESEARCHOPS_EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| `RESEARCHOPS_EMBEDDING_REVISION` | `main`；可固定实际 revision |
| `RESEARCHOPS_EMBEDDING_DEVICE` | `cpu` |
| `RESEARCHOPS_EMBEDDING_BATCH_SIZE` | `32` |
| `RESEARCHOPS_EMBEDDING_LOCAL_ONLY` | `false`；离线缓存准备后设为 `true` |
| `RESEARCHOPS_RERANKER_MODEL` | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` |
| `RESEARCHOPS_RERANKER_DEVICE` | `cpu` |
| `RESEARCHOPS_RERANKER_BATCH_SIZE` | `16` |
| `RESEARCHOPS_RERANKER_LOCAL_ONLY` | `false`；离线缓存准备后设为 `true` |

SQLite 是原文与元数据的事实来源。Qdrant 的 collection 与 Embedding 空间、内容版本关联，alias 指向活跃版本。切换模型或维度使用新版本索引；项目和论文范围在查询与原文回读时分别校验。

## MicroAgent、MCP 与 LangGraph

| 变量 / 参数 | 作用 |
| --- | --- |
| `RESEARCHOPS_PAPER_ANALYZER_MODE=legacy` | 默认论文分析路径；`micro` 启用搜索、回读和修订循环 |
| `RESEARCHOPS_PAPER_READ_TRANSPORT=inprocess` | 默认进程内回读；`mcp` 使用官方 SDK stdio client/server |
| `RESEARCHOPS_STUDY_ORCHESTRATOR=langgraph` | 新方案默认图编排；`native` 使用原生方案路径 |
| `plan-study --orchestration` | 为当前新方案显式选择编排方式 |
| `plan-study --retrieval-strategy` | `legacy`、`cpu`、`fusion` 三条证据组织路径 |
| `code --stream --reasoning-effort low` | 为支持相应协议的服务启用 SSE 和推理参数 |

模式、语料、模型与输入版本会参与任务绑定。更改模式使用新工作区；原任务使用原配置恢复。图 checkpoint 保存阶段位置，角色 checkpoint 保存局部消息和结果。

## Web

| 变量 | 配置 |
| --- | --- |
| `RESEARCHOPS_WEB_PASSWORD` | 设置共享登录密码 |
| `RESEARCHOPS_JWT_SECRET` | 至少 32 字节的随机密钥 |
| `RESEARCHOPS_SECURE_COOKIE` | HTTP 本地访问 `0`，HTTPS `1` |
| `RESEARCHOPS_WEB_WORKSPACE` | `web_workspace` |
| `RESEARCHOPS_FRONTEND_ORIGINS` | 逗号分隔的精确 Origin，包含协议、主机和端口 |

默认 Compose 显式使用 `pymupdf` 与 `hashing`，作为轻量知识库部署配置；本地 Python 路径读取 `.env` 的神经检索选项。Web 面向项目知识问答，CLI 承担科研规划与 Coding。

## 文献连接器

arXiv、OpenAlex、OpenReview、DBLP、Crossref、ACL Anthology 与 CVF 提供论文来源。OpenAlex 可选配置 `PAPER_SEARCH_MCP_OPENALEX_API_KEY` 和联系邮箱；Crossref 使用 `CROSSREF_MAILTO`；CVF 目录由 `RESEARCHOPS_CVF_VENUES` 选择。模板中兼容的其他来源字段按各自注释使用。

模型 key、登录密钥、论文库和运行记录在本地保存。资源额度由宿主按任务管理，CLI 的 `--token-cap` 可为新任务设置调用范围。

# 安装与运行

本指南从项目根目录执行，命令采用 PowerShell。CLI 与 Web 是两条并列路径：CLI 使用 Skills、研究规划和 Coding；Web 让用户上传论文或笔记，围绕资料连续提问并获得带来源的回答。架构与组件职责见 [README](README.md) 和 [架构说明](docs/ARCHITECTURE.md)。

## 环境准备

| 组件 | 建议配置 | 使用场景 |
| --- | --- | --- |
| Python | 3.11+，推荐 3.12 | CLI、FastAPI、本地模型 |
| 模型 API | 支持 Chat Completions 与 Tool Calling | Agent 决策、分析和评审 |
| Docker | Linux containers；Windows 可使用 Docker Desktop + WSL2 | Qdrant 与 Coding 容器 |
| Node.js | 22 LTS | 本地 Vite 前端；Compose 模式由镜像提供 |
| 磁盘与网络 | 本地模型缓存、论文与工作区空间；首次下载时联网 | Embedding、重排、资料获取 |

### Python 业务环境

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

[requirements.txt](requirements.txt) 复用 `pyproject.toml` 的应用依赖，并包含 [requirements-eval.txt](requirements-eval.txt) 的 RAGAS / LangChain 兼容组合，覆盖 LangGraph、SQLite checkpointer、Qdrant client、MCP、神经检索、PDF、Web、数据适配与 Coding。Qdrant client 在虚拟环境中；Qdrant 服务由 Docker 或独立服务提供。完整依赖职责见 [README](README.md#安装与运行)。

OCR 与结构化版面解析按需安装：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[docling]"
```

上面的完整安装已包含 RAGAS。只安装应用能力可用 `python -m pip install -e ".[web,pdf,retrieval,datasets,coding,agent]"`；也可按[评测说明](docs/EVALUATION.md)为 RAGAS 单独创建环境。

### 模型配置

在 `.env` 中填写：

```dotenv
RESEARCHOPS_MODEL_BASE_URL=https://your-provider.example/v1
RESEARCHOPS_MODEL_API_KEY=your-api-key
RESEARCHOPS_MODEL_NAME=your-tool-calling-model
```

上述值为占位符，替换为服务商提供的准确配置。Adapter 会在根 URL 后追加 `/chat/completions`。CLI 自动读取 `.env`；进程环境变量优先。真实密钥和本地运行目录由 Git 忽略。

### 本地 Embedding 与重排

默认神经路径使用多语言 MiniLM Embedding 与 Cross-Encoder，可在 CPU 上运行。首次使用时从 Hugging Face 下载，后续复用缓存；也可以提前准备：

```powershell
.\.venv\Scripts\python.exe -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', device='cpu'); CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1', device='cpu')"
```

离线使用前准备好模型缓存，再设置 `.env` 的 `RESEARCHOPS_EMBEDDING_LOCAL_ONLY=true` 和 `RESEARCHOPS_RERANKER_LOCAL_ONLY=true`。模型名称、设备、batch size 和 revision 见[配置说明](docs/CONFIGURATION.md)。

## 路径一：CLI

### 资料入库与带引用问答

```powershell
.\.venv\Scripts\python.exe researchops.py --workspace knowledge_base/cli_research init
.\.venv\Scripts\python.exe researchops.py --workspace knowledge_base/cli_research ingest docs/examples/research_brief.md
.\.venv\Scripts\python.exe researchops.py --workspace knowledge_base/cli_research search "LoRA rank 与数据划分" --limit 5
.\.venv\Scripts\python.exe researchops.py --workspace knowledge_base/cli_research ask "研究计划中的显存与随机种子约束是什么？"
.\.venv\Scripts\python.exe researchops.py --workspace knowledge_base/cli_research agent "基于当前资料整理研究约束，附引用并区分文献事实与用户计划。"
```

`search` 返回混合检索结果，`ask` 返回提取式证据，`agent` 由模型通过 Skills 选择能力。示例简报是用户需求资料；真实论文可用自己的 PDF / Markdown / TXT 文件替换，`ingest` 接收文件或目录。

### Skills 与文献获取

```powershell
.\.venv\Scripts\python.exe researchops.py skills
.\.venv\Scripts\python.exe researchops.py tools
.\.venv\Scripts\python.exe researchops.py --workspace knowledge_base/cli_research research "ViT 少样本分类中的 LoRA 与视觉适配器" --max-queries 2 --limit-per-query 4 --max-downloads 2 --pdf-parser pymupdf
```

`research` 使用 Skills 入口，由模型发现和调用 `literature-acquire`，返回实际来源、获取状态和分析工件。当前接入 arXiv、OpenAlex、OpenReview、DBLP、Crossref、ACL Anthology 与 CVF。OpenAlex 联系信息和可选 key、Crossref 联系邮箱等配置见 [.env.example](.env.example)。

`--workspace` 和可重复的 `--allow-root` 放在子命令前。涉及外部资料路径的 Agent 任务可显式授权：

```powershell
.\.venv\Scripts\python.exe researchops.py --workspace knowledge_base/cli_research --allow-root C:/research/papers agent "把 C:/research/papers 中的资料入库，并查找视觉适配器的证据。"
```

将外部目录替换为实际路径。Skill 加载由宿主绑定的 [skills/](skills/) 目录提供。

### CLI 启用 Qdrant与论文 Agent

启动 Qdrant 单节点服务，数据保存在 named volume：

```powershell
docker volume create researchops-qdrant-data
docker run -d --name researchops-qdrant -p 127.0.0.1:6333:6333 -v researchops-qdrant-data:/qdrant/storage qdrant/qdrant:v1.20.0
Invoke-RestMethod http://127.0.0.1:6333/collections
```

已有实例时复用其地址和端口即可；上面的创建命令用于首次安装。现有容器可用 `docker start researchops-qdrant` 启动。

在本终端设置，或把对应值写入 `.env`：

```powershell
$env:RESEARCHOPS_VECTOR_BACKEND='qdrant'
$env:RESEARCHOPS_QDRANT_URL='http://127.0.0.1:6333'
$env:RESEARCHOPS_RETRIEVAL_BACKEND='neural'
$env:RESEARCHOPS_PAPER_ANALYZER_MODE='micro'
$env:RESEARCHOPS_PAPER_READ_TRANSPORT='mcp'
$env:RESEARCHOPS_STUDY_ORCHESTRATOR='langgraph'
```

随后在新工作区运行 `research`，论文分析使用局部工具循环，原文回读经由官方 MCP stdio client/server。MCP 子进程由宿主启动，无需额外桌面插件或常驻端口。Qdrant 保存向量和范围 payload，SQLite 保存原文及引用信息；collection 与版本由后端管理。

检索后端可选择 `local`，原文回读可选择 `inprocess`，论文分析可选择 `legacy`。更换模式使用新工作区，保留已有任务快照。各模式职责见[配置说明](docs/CONFIGURATION.md)。

### LangGraph 研究规划与合同交接

`plan-study` 需要已有论文语料和经校验的数据清单。研究语料、数据、方案输出分别指定；研究任务目前围绕 ViT 参数高效微调图像分类构建。

先准备任务需要的数据。以下为 DTD 与 Pets 的示例，下载规模取决于原始数据：

```powershell
.\.venv\Scripts\python.exe researchops.py --workspace knowledge_base/research_data datasets-acquire dtd pets --workers 2
```

已有本地原始数据时，可使用 `dataset-import DATASET_ID --path PATH`。接着基于真实论文工作区生成方案：

```powershell
.\.venv\Scripts\python.exe researchops.py --workspace knowledge_base/study_rank plan-study "在 DTD 与 Pets 的 25% 标签划分上比较 LoRA rank 2 和 8，使用相同选参机会和三个随机种子" --source knowledge_base/cli_research --dataset-workspace knowledge_base/research_data --retrieval-strategy fusion --orchestration langgraph --token-cap 600000
```

`--source` 指向含论文数据库和原始文献的已有工作区，输出放在独立的 `knowledge_base/study_rank`。终端结果中的 `artifacts` 指向正式方案目录，其中 `result.json` 和 `frozen_spec.json` 供后续交接使用。`--orchestration native` 选择原生方案路径；新模式使用新输出目录。

### Coding：从正式合同继续实现

准备 Docker CPU 执行镜像：

```powershell
docker build -f sandbox/Dockerfile -t researchops-coding:cpu-v1 .
```

把实际方案目录的 `result.json` 赋给变量，再执行：

```powershell
$studyResultPath = 'knowledge_base/study_rank/studies/实际方案目录/result.json'
.\.venv\Scripts\python.exe researchops.py --workspace knowledge_base/coding_rank coding-plan --study-result $studyResultPath
.\.venv\Scripts\python.exe researchops.py --workspace knowledge_base/coding_rank code --study-result $studyResultPath --stream --reasoning-effort low
```

Coding 消费正式合同，输出任务计划、项目源码和执行报告。恢复同一任务时，在 `code` 命令中增加 `--resume`，继续沿用已有计划与工件。CPU 路径用于实现与小规模运行，完整训练按任务资源另行安排。

### CLI 命令导航

| 目标 | 命令 |
| --- | --- |
| 查看能力与状态 | `skills`、`tools`、`status` |
| 入库、搜索、提取证据 | `ingest`、`search`、`ask` |
| 自然语言能力编排 | `agent`、`research` |
| 多源检索与批量文献处理 | `online-search`、`batch-research`、`reanalyze` |
| 数据准备 | `datasets-acquire`、`dataset-import`、`prepare-experiment` |
| 研究规划与实现交接 | `plan-study`、`coding-plan`、`code` |
| 公式来源核对 | `validate-formulas` |

各命令参数使用 `researchops.py <命令> --help` 查询。

## 路径二：Web

### 能做什么，如何使用

Web 是一个基于用户资料的聊天界面。你可以上传论文、研究笔记或需求说明，让 Agent 从中查找依据、归纳观点或比较方法，再结合上一轮回答继续追问。

界面的“项目”输入框用于给资料分组，例如 `vit-peft` 和 `reading-notes`。同一用户在不同项目上传的资料分别保存，当前问答检索当前项目。项目名使用字母、数字、下划线或连字符。

1. 登录后，在“项目”中填写课题分组名，也可直接使用 `default`。
2. 在“项目资料”中依次上传 PDF、Markdown 或 TXT，单文件不超过 20 MiB；等待显示导入成功后提问。
3. 在“研究问题”中输入“比较这两篇论文的 LoRA 设置，请给出原文依据”。Agent 检索资料、回读相关段落，把来源标识和证据片段组织进文本回答。
4. 在同一页面继续问“只保留与显存开销有关的内容”，沿用当前会话上下文。点击“新建对话”清空聊天上下文；修改项目名会切换资料范围并开始新对话，已上传资料继续保留在原项目。

Web 提供资料阅读与对话问答；LangGraph 研究规划、合同交接和 Coding 源码生成使用前文的 CLI 路径。

### 本地前后端

安装 Python 业务依赖后，准备 Web 配置和前端依赖：

```powershell
if (-not (Test-Path .env.web)) { Copy-Item .env.web.example .env.web }
npm.cmd --prefix frontend ci
.\.venv\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(48))"
```

将生成的随机值写入 `.env.web` 的 `RESEARCHOPS_JWT_SECRET`，同时设置 `RESEARCHOPS_WEB_PASSWORD`。在两个项目根目录终端分别启动后端与前端：

```powershell
.\.venv\Scripts\python.exe -m uvicorn researchops_api.main:create_app --factory --app-dir src --host 127.0.0.1 --port 8000
```

```powershell
npm.cmd --prefix frontend run dev
```

| 地址 | 作用 |
| --- | --- |
| `http://127.0.0.1:5173` | Vite 前端 |
| `http://127.0.0.1:8000/docs` | FastAPI 接口文档 |
| `http://127.0.0.1:8000/api/v1/health` | 服务状态 |
| `http://127.0.0.1:8000/api/v1/ready` | 数据连接状态 |

`--app-dir src` 保持领域包导入顺序，API 包由 editable 安装提供。Vite 将 `/api` 代理到本地 8000 端口。

登录时的用户标识对应服务端用户命名空间，访问密码来自 `.env.web`。后续消息携带会话标识，服务端取回该会话历史；SSE 提供心跳与任务终态，最终答复随终态返回。前端同时可以查询 run 状态。

Web 的核心接口为：

| 接口 | 用途 |
| --- | --- |
| `POST /api/v1/auth/login`、`/logout` | 登录会话 |
| `POST /api/v1/knowledge/upload` | 上传并入库项目资料 |
| `POST /api/v1/chat` | 提交问题，返回运行标识 |
| `GET /api/v1/runs/{run_id}` | 获取运行状态与答复 |
| `GET /api/v1/runs/{run_id}/events` | SSE 事件流 |

### Docker Compose 部署

仅部署 Web 时，宿主准备 Docker、`.env` 和 `.env.web`，Python 与 Node 由镜像提供：

```powershell
docker compose --env-file .env.web config --quiet
docker compose --env-file .env.web up -d --build --wait
docker compose ps
```

访问 `http://127.0.0.1:3000`。Nginx 提供静态页面并代理 API；后端以 PyMuPDF 与 hashing 本地检索启动。Compose 当前包含前后端两个服务；Qdrant 由前文的独立服务方式提供。本地 Python 安装路径提供神经检索与 Qdrant 的直接配置方式。

```powershell
docker compose logs --tail 100 backend frontend
docker compose stop frontend backend
docker compose --env-file .env.web up -d --wait
```

本地 HTTP 使用 `RESEARCHOPS_SECURE_COOKIE=0`。HTTPS 部署由反向代理终止 TLS，并将其改为 `1`，同时配置实际的前端 Origin。当前共享密码登录面向受控访问环境。

### 持久化位置

| 路径 | 内容 |
| --- | --- |
| `web_workspace/web_state.db` | Web 会话与运行状态 |
| `web_workspace/users/<user>/projects/<project>/` | 项目数据库、上传资料与运行工件 |
| 项目下的 `runs/<run>/trace.jsonl` | 工具与模型决策事件 |
| CLI 指定的 `knowledge_base/<name>/` | 论文数据库、方案、checkpoint 与源码交接 |
| Docker volume `researchops-qdrant-data` | 向量索引 |

Compose 将 `web_workspace/` 挂载到容器内的 `/app/web_workspace`。备份时同时保存数据库与相关资料文件；运行中的 SQLite 使用一致性备份。

## 运行配置定位

| 需要调整的项目 | 配置位置 |
| --- | --- |
| 模型根 URL、模型名、超时 | `.env` 的 `RESEARCHOPS_MODEL_*` |
| PDF 解析、神经模型与本地缓存 | `.env` 的 parser / embedding / reranker 配置 |
| Qdrant 地址和项目索引 | `.env` 的 `RESEARCHOPS_VECTOR_BACKEND`、`RESEARCHOPS_QDRANT_*` |
| 论文 Agent 与 MCP | `.env` 的 `RESEARCHOPS_PAPER_*` |
| Web 登录、Cookie、工作区与 Origin | `.env.web` |
| 前端或 API 端口 | Vite、Uvicorn、Compose 与代理配置保持一致 |

完整变量及模式说明见 [CONFIGURATION.md](docs/CONFIGURATION.md)。新增工具、Skill 或 Agent 的接入方式见 [EXTENDING.md](docs/EXTENDING.md)。

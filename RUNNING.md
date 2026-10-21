# 运行方式

以下命令使用 PowerShell，工作目录为项目根目录。CLI、独立前后端和 Docker Compose 共用项目配置；已有配置文件、虚拟环境与数据保留原内容。

## 环境与安装

本地运行需要 Python 3.11+。前端需要 Node.js 20.19+ 或 22.12+。Docker 部署需要支持 Linux 容器的 Docker Engine 与 Compose，Windows 可使用 Docker Desktop 与 WSL2。

```powershell
if (-not (Test-Path .venv)) { python -m venv .venv }
.\.venv\Scripts\python.exe -m pip install -e ".[web,pdf,retrieval,datasets]"
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
if (-not (Test-Path .env.web)) { Copy-Item .env.web.example .env.web }
npm --prefix frontend ci
```

`web` 安装 FastAPI、Uvicorn、JWT 和 Web 配置依赖；`pdf` 安装 PyMuPDF；`retrieval` 安装本地神经检索依赖；`datasets` 安装数据集读取与校验依赖。仅运行默认 Docker Web 时，宿主准备 `.env` 和 `.env.web` 即可，Python 与前端依赖在镜像内安装。

OCR 与公式解析增强使用 Docling：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[docling]"
```

Coding 及本地测试依赖可分别安装：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[coding,dev]"
```

### 神经检索权重

`neural` 模式在首次检索时加载 embedding 和 cross-encoder，模型缺失时从 Hugging Face 下载。默认模型可提前下载：

```powershell
.\.venv\Scripts\python.exe -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', device='cpu'); CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1', device='cpu')"
```

使用自定义模型时，将命令中的模型名替换为 `.env` 中配置的值。离线运行前准备本地缓存，再设置 `RESEARCHOPS_EMBEDDING_LOCAL_ONLY=true` 与 `RESEARCHOPS_RERANKER_LOCAL_ONLY=true`。`hashing` 模式使用本地 hashing 向量和词法重排。

## 配置

模板为 [.env.example](.env.example) 和 [.env.web.example](.env.web.example)。CLI 优先读取当前工作目录的 `.env`，缺失时读取项目根目录版本；后端先读取根目录 `.env.web`，再读取 `.env`。已有进程环境变量优先于文件内容。Docker Compose 的 `environment` 字段覆盖其 `env_file` 同名值。

### 模型服务

| 变量 | 配置方式 |
| --- | --- |
| `RESEARCHOPS_MODEL_BASE_URL` | 必填，兼容 Chat Completions 与工具调用的服务根 URL；按服务要求包含 `/v1`，末尾保留根路径 |
| `RESEARCHOPS_MODEL_API_KEY` | 必填，模型服务 API key |
| `RESEARCHOPS_MODEL_NAME` | 必填，服务支持的准确模型 ID |
| `RESEARCHOPS_MODEL_TIMEOUT_SECONDS` | 请求超时秒数，默认 `180`；组件的 bounded 调用还受其分配时限约束 |

模型 adapter 在根 URL 后追加 `/chat/completions`。配置中的 URL 应与这个拼接规则对应。

### 解析与检索

| 变量 | 默认值与作用 |
| --- | --- |
| `RESEARCHOPS_PDF_PARSER` | `auto`：优先 Docling，失败后回退 PyMuPDF；`docling`：使用 Docling 并保留失败；`pymupdf`：直接提取文本 |
| `RESEARCHOPS_RETRIEVAL_BACKEND` | `neural`；可选 `hashing` |
| `RESEARCHOPS_EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| `RESEARCHOPS_EMBEDDING_REVISION` | `main`；指定 embedding 版本，固定实验可使用实际 revision |
| `RESEARCHOPS_EMBEDDING_DEVICE` | `cpu` |
| `RESEARCHOPS_EMBEDDING_BATCH_SIZE` | `32` |
| `RESEARCHOPS_EMBEDDING_LOCAL_ONLY` | `false`；设为 `true` 时只读取本地模型缓存 |
| `RESEARCHOPS_RERANKER_MODEL` | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` |
| `RESEARCHOPS_RERANKER_DEVICE` | `cpu` |
| `RESEARCHOPS_RERANKER_BATCH_SIZE` | `16` |
| `RESEARCHOPS_RERANKER_LOCAL_ONLY` | `false`；设为 `true` 时只读取本地模型缓存 |

默认 Web Compose 将解析器设为 `pymupdf`、检索后端设为 `hashing`。本地后端读取本地环境中的检索配置；上面的安装命令包含神经检索依赖。

### 文献连接器

| 变量 | 作用 |
| --- | --- |
| `RESEARCHOPS_ACQUIRE_WORKERS` | 获取 DAG 最大并行节点数，默认 `4`，范围 `1–16` |
| `RESEARCHOPS_CVF_VENUES` | CVF 会议目录，默认 `CVPR2025,ICCV2025,WACV2025` |
| `PAPER_SEARCH_MCP_OPENALEX_API_KEY` | OpenAlex 可选 API key |
| `PAPER_SEARCH_MCP_OPENALEX_EMAIL` | OpenAlex 请求联系邮箱 |
| `CROSSREF_MAILTO` | Crossref 请求联系邮箱 |
| `PAPER_SEARCH_MCP_UNPAYWALL_EMAIL` | 模板兼容字段，当前自主文献获取流程保持空值 |
| `PAPER_SEARCH_MCP_SEMANTIC_SCHOLAR_API_KEY` | 模板兼容字段，当前自主文献获取流程保持空值 |

自主文献获取使用 arXiv、OpenAlex、OpenReview、DBLP、Crossref、ACL Anthology 和 CVF。来源状态与下载失败分别记录在获取工件中。

### Web 服务

| 变量 | 配置方式 |
| --- | --- |
| `RESEARCHOPS_WEB_PASSWORD` | 必填，共享本地登录密码 |
| `RESEARCHOPS_JWT_SECRET` | 必填，至少 32 字节的随机密钥 |
| `RESEARCHOPS_SECURE_COOKIE` | 本地 HTTP 用 `0`；HTTPS 部署用 `1` |
| `RESEARCHOPS_WEB_WORKSPACE` | 默认 `web_workspace`；Compose 中映射到 `/app/web_workspace` |
| `RESEARCHOPS_FRONTEND_ORIGINS` | 逗号分隔的精确 Origin，条目之间直接用逗号连接；本地默认允许 localhost 与 127.0.0.1 的 3000、5173 端口 |

随机密钥可用以下命令生成，再写入 `.env.web`：

```powershell
.\.venv\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(48))"
```

Web 登录接受字母、数字、下划线和连字符组成的 `user_id`，该值作为 JWT 身份和工作区命名空间。共享密码适用于受控访问。真实 `.env`、`.env.web` 和运行数据由 Git 忽略。

## CLI 运行

### 入口和全局参数

```powershell
.\.venv\Scripts\python.exe researchops.py --help
.\.venv\Scripts\python.exe researchops.py skills
.\.venv\Scripts\python.exe researchops.py --workspace knowledge_base/my_research research "ViT 少样本分类中的 LoRA 与视觉提示比较" --max-queries 2 --limit-per-query 4 --max-downloads 2 --token-cap 400000
```

`--workspace` 默认 `knowledge_base/default`，控制数据库、工件和 trace 目录。`--allow-root PATH` 可重复指定 Agent 可读取的额外输入根目录。这两个全局参数放在子命令前，`-h` 或 `--help` 可查询总入口和各子命令帮助。

### 基础命令

| 子命令 | 参数与作用 |
| --- | --- |
| `init` | 初始化当前工作区存储 |
| `skills` | 列出宿主绑定的 Skill 元信息 |
| `agent TASK` | 自然语言任务；`--token-cap` 默认 `250000`；`--context-mode` 为 `lazy` 或 `eager`，默认 `lazy` |
| `ingest PATH` | 入库 Markdown、TXT 或 PDF；直接指定的输入路径自动加入此次允许范围 |
| `search QUERY` | 混合检索；`--limit` 默认 `8` |
| `ask QUESTION` | 返回带引用的提取式证据；`--limit` 默认 `5` |
| `audit` | 审查 IDEA；`--idea` 默认 `config/seed_idea.json` |
| `online-search QUERY` | 多源检索；`--limit` 默认每源 `5` 条 |
| `status` | 查看持久对象数量 |
| `tools` | 查看工具风险、只读性、副作用及确认要求 |
| `demo` | 聚合入库、IDEA 审查和证据问答；`--corpus` 必填，`--idea` 默认 `config/seed_idea.json` |

本地资料入库与 Agent 取证示例，执行前将 `papers` 中路径替换为实际已有资料：

```powershell
.\.venv\Scripts\python.exe researchops.py --workspace knowledge_base/my_research ingest papers
.\.venv\Scripts\python.exe researchops.py --workspace knowledge_base/my_research search "low-rank adaptation"
.\.venv\Scripts\python.exe researchops.py --workspace knowledge_base/my_research agent "检索当前知识库中低秩更新的机制，给出原文证据及来源" --token-cap 250000
```

Agent 直接读取外部目录时，使用 `--allow-root` 明确范围。`search` 与 `ask` 对当前知识库操作，`agent` 根据任务选择并调用 Skill。

### 文献获取与分析

| 子命令 | 完整参数 |
| --- | --- |
| `research [IDEA]` | `--idea-file PATH` 可提供 UTF-8 任务文本，与位置参数二选一；`--max-queries 7`；`--limit-per-query 8`；`--max-downloads 12`；`--pdf-parser auto/docling/pymupdf`；`--token-cap 250000`；`--legacy-tools` 切换历史直接工具入口 |
| `batch-research` | `--idea-id vit_peft_small_data`；`--idea-library config/idea_library.json`；`--target-papers 30`；`--shortlist-size 6`；`--limit-per-query 12`；`--evaluation-size 40`；`--pdf-parser` 默认 `pymupdf`，可选 `auto/docling/pymupdf` |
| `silver-evaluate` | `--idea-id vit_peft_small_data`；`--idea-library config/idea_library.json`；可选 `--provisional-path PATH`、`--batch-report PATH`；`--max-papers 40`；`--workers 3`；`--max-fulltext-chars 14000` |
| `reanalyze TASK_ID` | 重用已获取全文重新分析；`--deterministic-plan` 使用确定性计划回退 |
| `validate-formulas` | 必填 `--pdf PATH`、`--docling-json PATH`；可选 `--source-package PATH`、`--source-receipt PATH`；输出到当前工作区的公式验证工件目录 |

表中数字为默认值。默认 `research` 使用 Skill 和预算模型；`--legacy-tools` 调用历史直接工具路径，该路径不接入此命令的 `--token-cap` 准入包装。

### 数据准备与研究规划

| 子命令 | 完整参数 |
| --- | --- |
| `datasets-acquire [IDS ...]` | 默认 ID 为 `cifar10 cifar100 flowers102 pets dtd`；`--workers 3`；`--dataset-source auto/official/huggingface`，默认 `auto` |
| `dataset-import DATASET_ID` | 必填 `--path PATH`，从本地原始数据文件导入并校验 |
| `prepare-experiment` | `--idea-id vit_peft_small_data`；`--idea-library config/idea_library.json`；`--workers 3`；`--offline` 复用本地文件；`--dataset-source auto/official/huggingface`，默认 `auto` |
| `plan-study [IDEA]` | `--idea-file PATH` 读取 JSON 或文本；`--source` 默认 `knowledge_base/workspace_batch_vit_v2`；`--token-cap 250000`；`--retrieval-strategy legacy/cpu/fusion`，默认 `legacy`；`--dataset-workspace PATH` 指定独立的已验证数据工作区 |

`prepare-experiment` 读取已有批量研究与银标结果。`plan-study` 读取已有语料数据库与已验证数据清单，输出工作区独立于语料源。执行以下命令前，先准备实际语料和数据：

```powershell
.\.venv\Scripts\python.exe researchops.py --workspace knowledge_base/my_research datasets-acquire cifar10 --workers 3
.\.venv\Scripts\python.exe researchops.py --workspace knowledge_base/new_study plan-study "比较 LoRA 的 rank 2、8、16，使用验证集选参并隔离测试集" --source knowledge_base/my_research --dataset-workspace knowledge_base/my_research --retrieval-strategy fusion --token-cap 600000
```

### Coding 与沙箱

| 子命令 | 完整参数 |
| --- | --- |
| `coding-plan` | `--idea-id vit_peft_small_data`；`--from-scratch` 关闭实现模板 |
| `code` | `--idea-id vit_peft_small_data`；`--resume` 恢复已有任务；`--from-scratch` 从空项目按模块构建；`--plan PATH` 使用已保存的不可变 Coding plan |

Coding 使用已准备的实验规格和本地数据。构建 CPU verifier 镜像后，在对应工作区创建或恢复任务：

```powershell
docker build -f sandbox/Dockerfile -t researchops-coding:cpu-v1 .
.\.venv\Scripts\python.exe researchops.py --workspace knowledge_base/my_research coding-plan --idea-id vit_peft_small_data
.\.venv\Scripts\python.exe researchops.py --workspace knowledge_base/my_research code --idea-id vit_peft_small_data
```

恢复时在最后一条命令增加 `--resume`。已有 plan 的恢复预算沿用原值；来源、规格或实现发生变化时按校验结果重新验证。

## 独立前后端运行

准备 `.env`、`.env.web` 和依赖后，在根目录终端启动后端：

```powershell
.\.venv\Scripts\python.exe -m uvicorn researchops_api.main:create_app --factory --app-dir src --host 127.0.0.1 --port 8000
```

另一个根目录终端启动前端：

```powershell
npm --prefix frontend run dev
```

前端为 `http://127.0.0.1:5173`，后端为 `http://127.0.0.1:8000`，接口文档为 `http://127.0.0.1:8000/docs`。Vite 将 `/api` 转发到本地 8000 端口。分别使用 `Ctrl+C` 停止两个进程。

`--app-dir src` 将领域包目录放到导入路径前部，避免根目录 `researchops.py` 与包名冲突；独立 API 包由 editable 安装提供。

登录时填写命名空间与配置密码。项目名使用字母、数字、下划线或连字符；上传支持 PDF、Markdown、TXT，单文件上限 20 MiB。同一会话的后续消息使用服务端历史。SSE 推送心跳和终态，最终回答随终态返回。

前端生产构建：

```powershell
npm --prefix frontend run build
```

产物位于 `frontend/dist/`。`frontend/nginx.conf` 用于 Compose 中的 Nginx 服务，代理目标为 `backend:8000`。

### 健康与就绪检查

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/v1/health
Invoke-RestMethod http://127.0.0.1:8000/api/v1/ready
```

`health` 返回服务状态，`ready` 检查 Web SQLite 连接。模型配置与外部服务连通性在实际调用时验证。

## Docker Compose 运行

在根目录配置好 `.env` 与 `.env.web`，启动两个独立服务：

```powershell
docker compose --env-file .env.web config --quiet
docker compose --env-file .env.web up -d --build --wait
docker compose ps
```

前端为 `http://127.0.0.1:3000`，后端 API 为 `http://127.0.0.1:8000/api/v1`，接口文档为 `http://127.0.0.1:8000/docs`。Nginx 提供前端静态文件和 API 代理；后端默认使用 PyMuPDF 与 hashing 检索。

查看日志、停止和重新启动：

```powershell
docker compose logs --tail 100 backend frontend
docker compose stop frontend backend
docker compose --env-file .env.web up -d --wait
```

更新代码或镜像依赖后，使用带 `--build` 的启动命令重新构建。HTTP 本地访问使用 `RESEARCHOPS_SECURE_COOKIE=0`；HTTPS 部署由反向代理终止 TLS，并设置 `RESEARCHOPS_SECURE_COOKIE=1` 与实际前端 Origin。

### 数据持久化

| 路径 | 内容 |
| --- | --- |
| `web_workspace/web_state.db` | 会话历史、run 状态与身份归属 |
| `web_workspace/users/<user>/projects/<project>/researchops.db` | 项目文献、证据节点和检索数据 |
| 项目目录下的 `uploads/` | 按内容 SHA-256 命名的上传文件 |
| 项目目录下的 `model_calls/<run>/` | 模型请求、响应与 token 账本 |
| 项目目录下的 `runs/<run>/trace.jsonl` | 工具循环 trace |
| CLI 指定的 `knowledge_base/<name>/` | CLI 数据库、工件、账本与运行状态 |
| 根目录 `papers/` 与 `datasets/` | 标准知识工作区对应的集中论文与数据资产 |

Compose 将宿主 `web_workspace/` 挂载到后端 `/app/web_workspace`。停止服务保留这些文件。数据备份需覆盖数据库和相关资产；运行期间的 SQLite 备份应使用一致性快照方式。Git 忽略规则只控制提交范围。

## 故障排查

| 现象 | 检查与处理 |
| --- | --- |
| 后端启动报 JWT 或密码配置错误 | 检查 `.env.web`，设置非空密码和至少 32 字节密钥；重启后端读取配置 |
| 导入 `researchops_api` 失败 | 在根目录重新执行 editable 安装，确保当前虚拟环境已安装 `backend/src` 中的 API 包 |
| 导入提示 `researchops` 不是 package | 在本地 Uvicorn 命令中保留 `--app-dir src`，核对使用的解释器和工作目录 |
| 模型调用报配置缺失、401 或 404 | 检查 `.env` 的三个必填字段、服务权限和根 URL 拼接；核对 `/chat/completions` 路径 |
| 模型请求超时或限流 | 查看该 run 的调用记录与私有服务日志，核对网络和服务配额；按调用阶段调整时限或使用新任务重试 |
| 神经模型下载失败 | 检查 Hugging Face 连通性和缓存；预下载权重后启用 `LOCAL_ONLY`；hashing 模式可用于轻量本地检索 |
| Docling 解析失败 | 使用 `auto` 保留 PyMuPDF 回退，或显式设为 `pymupdf`；OCR 增强需安装 `.[docling]` |
| 8000、5173 或 3000 端口占用 | 用 `Get-NetTCPConnection -LocalPort 8000,5173,3000 -ErrorAction SilentlyContinue` 查看占用，再确认并停止相应旧服务 |
| 登录后请求返回 401 | 检查密码、Cookie、主机名及 HTTP 下的 Secure Cookie 设置；在实际访问的主机名重新登录 |
| POST 返回 403 | 检查浏览器 Origin 与 `RESEARCHOPS_FRONTEND_ORIGINS`，保持协议、主机和端口一致 |
| 同一会话返回 409 | 查询对应 run 状态，等待当前请求结束；后端重启会将遗留活动 run 标为 `interrupted` |
| 上传返回 413 或 422 | 检查 20 MiB 上限、文件类型、内容是否为空及 PDF 文件头 |
| 页面等待回答，SSE 只有心跳 | 查询 `/api/v1/runs/{run_id}` 并检查后端日志；SSE 在任务结束时返回最终数据 |
| Nginx 返回 502 | 检查 backend 容器状态与日志，核对服务名和 8000 端口；本地 Vite 代理也需要后端已启动 |
| 预算准入拒绝 | 查看 `ledger.json` 的已知用量和未知预留，按任务成本选择新 run 的预算；历史预留继续保留 |
| Agent 文件访问被拒绝 | 在全局参数中增加必要的 `--allow-root`；Skill 只从宿主绑定目录加载 |
| `plan-study` 报语料、清单或输出目录错误 | 指定实际语料库和已验证数据工作区，为输出选择独立目录 |
| Coding 返回镜像或验证错误 | 检查 Docker Engine 与 `researchops-coding:cpu-v1` 镜像；查看对应验证轮次报告，再使用 `--resume` 恢复 |
| PowerShell 阻止 `npm.ps1` | 使用 `npm.cmd` 执行同一条 npm 命令 |

更改服务端口时，同步核对启动参数、Vite 代理、Nginx 代理、Compose 端口映射和前端 Origin。原有论文、数据集、知识库、工作区和容器数据保持保留。

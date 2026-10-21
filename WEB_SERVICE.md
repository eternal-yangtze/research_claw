# Web 服务运行与工程说明

## 1. 运行模式：不要使用全局 pip

所有示例在项目根目录的 **PowerShell** 执行，不能把 `$env:` 命令复制到 CMD。

| 模式 | Python 环境 | Redis / PostgreSQL | 前端 |
| --- | --- | --- | --- |
| 推荐容器模式 | 镜像内 `/opt/venv` | Docker 容器 | FastAPI 托管静态页面 |
| 本地调试模式 | 项目 `.venv` | 同一套 Docker 容器 | 同一 FastAPI，无需 npm |

Docker Desktop 使用 WSL2 Linux 容器后端。无需再在 WSL 系统环境安装 Python 依赖；若直接在 WSL 开发，另建 Linux 虚拟环境，不能使用 Windows `.venv`。

本项目参照本机 `../reference_coding/TaleClaw/` 的镜像构建、Compose 编排、PostgreSQL、数据持久化与重启策略，不照搬无关机器人服务。Redis 是本项目新增需求；容器内再使用虚拟环境以满足隔离要求。

## 2. 推荐：完整容器启动

```powershell
Set-Location D:\my_coach\new_agent\researchops_mvp
wsl.exe --list --verbose
docker info
docker compose version
.\scripts\start_web.ps1 -Research
```

脚本首次生成私密 `.env.web`，不会覆盖已有密钥。默认 HTTP 本机开发设置 `RESEARCHOPS_SECURE_COOKIE=0`；HTTPS 部署应改为 `1`。登录使用合法用户标识和 `.env.web` 的 `RESEARCHOPS_MOCK_PASSWORD`，不要把密码贴进日志。

- 对话前端：`http://127.0.0.1:8000/`。
- 知识库前端：`http://127.0.0.1:8000/knowledge.html`。
- 后端 API：`http://127.0.0.1:8000/api/v1`；接口文档 `/docs`。
- Redis 主机端口默认 6379；PostgreSQL 默认 15432。容器内部使用 `redis:6379` 和 `postgres:5432`。
- 前端纯 HTML + JS，构建时复制进 Web 镜像；修改前端或后端后重新运行启动脚本。
- 不带 `-Research` 不主动启动解析 Worker，但不会停止以前启动的 Worker。

自定义 Web 端口：`.\scripts\start_web.ps1 -Research -Port 8080`。

### 暂停重型任务、查看状态、停止服务

```powershell
docker compose --env-file .env.web -f compose.web.yaml stop knowledge-worker
docker compose --env-file .env.web -f compose.web.yaml --profile web --profile research ps
docker compose --env-file .env.web -f compose.web.yaml logs --tail 80 knowledge-worker
docker compose --env-file .env.web -f compose.web.yaml --profile web --profile research stop
```

这些命令保留卷和文件。**不要使用 `down -v`**，否则会删除 Redis / PostgreSQL 数据。原始上传、解析产物和独立 SQLite 位于 `web_workspace/users/<用户>/projects/<项目>/`；PostgreSQL、Redis 使用命名卷，应分别备份。

## 3. 本地后端调试：只在 .venv 安装

仅系统 Python 创建虚拟环境，后续全部显式使用项目解释器；已有 `.venv` 不要删除重建。

```powershell
# 只有缺少 .venv 时执行
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[web,pdf,dev]"
.\scripts\start_web.ps1 -InfrastructureOnly
# 若之前启用了容器版 Web，先释放端口
docker compose --env-file .env.web -f compose.web.yaml stop web
.\scripts\start_web.ps1 -LocalApp
```

该终端保持运行；前端和 API 地址不变。`-LocalApp` 启动前读取 `.env.web`，其主机 PostgreSQL 地址应为 `127.0.0.1:15432`。要同时运行 Worker，可另开终端执行：

```powershell
docker compose --env-file .env.web -f compose.web.yaml --profile research up -d --build knowledge-worker
```

`-Research` 是完整容器模式的开关，不与 `-LocalApp` / `-InfrastructureOnly` 混用。不要同时运行多个直接写同一知识库的 Worker。

## 4. 科研模型与神经向量配置

在私密 `.env` 或 `.env.web` 配置真实兼容接口。每个科研用途使用以下后缀：

```dotenv
RESEARCHOPS_PLANNER_BASE_URL=https://your-provider.example/v1
RESEARCHOPS_PLANNER_NAME=your-reasoning-model
RESEARCHOPS_PLANNER_API_KEY=your-private-key
RESEARCHOPS_EXTRACTOR_BASE_URL=https://your-provider.example/v1
RESEARCHOPS_EXTRACTOR_NAME=your-extraction-model
RESEARCHOPS_EXTRACTOR_API_KEY=your-private-key
RESEARCHOPS_SUMMARIZER_BASE_URL=https://your-provider.example/v1
RESEARCHOPS_SUMMARIZER_NAME=your-fast-model
RESEARCHOPS_SUMMARIZER_API_KEY=your-private-key
RESEARCHOPS_FALLBACK_BASE_URL=http://host.docker.internal:11434/v1
RESEARCHOPS_FALLBACK_NAME=your-installed-backup-model
RESEARCHOPS_FALLBACK_API_KEY=local
```

示例域名和模型名称不是可直接调用的服务。仍兼容通用 `RESEARCHOPS_MODEL_*`；备用接口必须确实存在，不能仅填配置就宣称已具备 Fallback。旧 `summary` 方法名兼容映射到 `summarizer`，配置前缀使用新名称。

独立知识库可选择远程神经 embedding，不改变原 Coding 检索配置：

```dotenv
RESEARCHOPS_KNOWLEDGE_EMBEDDING_BASE_URL=https://your-embedding-provider.example/v1
RESEARCHOPS_KNOWLEDGE_EMBEDDING_MODEL=your-embedding-model
RESEARCHOPS_KNOWLEDGE_EMBEDDING_REVISION=your-fixed-revision
RESEARCHOPS_KNOWLEDGE_EMBEDDING_API_KEY=your-private-key
```

该服务需兼容 `/embeddings`，每批最多 16 文本，单次超时 60 秒。未配置时使用 hashing-vector，不宣称是神经语义检索。改变 embedding 模型后应重新入库/镜像；离线 Demo 始终用 hashing，避免隐式联网或付费。

容器内 `127.0.0.1` 不是 Windows 主机；访问本机 Ollama/embedding 服务时使用 `host.docker.internal` 并检查服务监听范围。配置更新后重建/重启服务。

费用配置为单行 JSON，单位美元/百万 Token，模型名必须与实际响应路由一致：

```dotenv
RESEARCHOPS_MODEL_PRICES={"your-reasoning-model":{"input":1,"output":4},"your-extraction-model":{"input":0.3,"output":1.2},"your-fast-model":{"input":0.1,"output":0.4},"planner_baseline":{"input":1,"output":4}}
```

以上只是计算格式示例，不是供应商真实费率。科研面板成本仅覆盖科研 Worker 的真实模型调用；旧 `/usage` 仍是原 Agent 账本，不能把两者覆盖范围混淆。缺用量或价格返回未知，回放不纳入实账。

## 5. 数据库和性能设计

- 旧 Coding 与 CLI：保持原 SQLite 论文、切片、FTS5、向量缓存及检索代码；没有替换成远程数据库读写。
- 新知识库：上传后 HTTP 返回 202，独立单并发 Worker 解析、切片、向量化；写独立 `knowledge/researchops.db` 和 PostgreSQL 镜像。
- PostgreSQL：新增文档、节点、持久任务、事件和分用途调用表。节点包含正文、定位、来源 hash、向量、embedding 名称/版本。当前是 JSON 向量持久化，不是 pgvector ANN 服务。
- “镜像旧库”只读当前认证用户的项目 SQLite，不修改原库。快照只生成 PostgreSQL 镜像，不把旧库自动复制进独立研究检索库；需研究检索时显式上传对应原文。
- 任务领取使用 SQL 条件更新与 PostgreSQL `SKIP LOCKED`；Redis 续期锁保证单 Worker。审批按认证用户校验，以条件更新防重复恢复；重启后 queued 和 pending_approval 保留。
- `running` 状态遇进程崩溃不自动重放：可能存在不确定的模型计费或外部副作用，应由运维检查事件和产物后处理。本次未实现任意故障点恰好一次执行。
- 解析容器限 0.5 CPU / 1 GiB / 128 进程，限制 OMP/MKL/OpenBLAS 线程；默认不启动。监控只读取 SQL 聚合，页面可见时每 2 秒刷新，不在 HTTP 请求中解析或编码。
- 资源限制不能消除共享磁盘和数据库争用。“绝不影响任何局部性能”必须以独立硬件/资源池和代表性 Coding 并发 A/B 回归为前提；本次短时检索微基准不证明该承诺。严苛性能场景先停 Worker。

## 6. 新增接口与监控口径

业务接口均要求 JWT 或同源登录 Cookie；用户身份只能来自 Token。健康检查 `/api/v1/health` 与 `/api/v1/ready` 允许匿名访问，仅返回状态和依赖布尔值。

| 接口 | 用途 |
| --- | --- |
| `GET /api/v1/knowledge/overview?project_id=default` | 当前用户项目的解析数、状态、文档、节点及最近 30 任务 |
| `POST /api/v1/knowledge/upload?project_id=default&suffix=pdf` | 原始文件请求体，pdf/md/txt，最多 20 MiB |
| `POST /api/v1/knowledge/jobs` | kind 为 snapshot / analyze / demo |
| `GET /api/v1/knowledge/jobs/{job_id}/events?after=0` | 用户隔离事件游标，单次最多 200 条 |
| `POST /api/v1/knowledge/jobs/{job_id}/approve` | 原子批准回放或真实 PDF 挂起步骤，继续同一个任务 |
| `GET /api/v1/knowledge/costs` | 当前用户的真实科研调用成本与反事实估价 |

原 `/chat`、SSE `/runs/{run_id}/events` 和 `/approvals` 保持不变。知识库采用独立队列和审批接口，不混用原 Agent Run 状态。耗时是 Worker 执行时间，不把排队、等待人工审批算作解析时间；输出 Token 仅采用供应商返回，不估算 PDF 字数。

真实 analyze 任务可传 `next_pdf_url`。来源少于三篇且指定该地址时，下载和后续解析前挂起审批；只允许指定学术域名 HTTPS，限制大小、时间与重定向。此阈值是可解释启发式，不代表模型自评证据质量。离线 demo 则只解析本地模拟 PDF。详细操作见 `RESEARCH_DEMO.md`。

## 7. 验证与录屏

```powershell
.\.venv\Scripts\python.exe -B -m pytest tests/api -q -p no:cacheprovider --basetemp=../.rtapi01
.\.venv\Scripts\python.exe scripts/verify_web_infrastructure.py
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider --basetemp=../.rtfull01
.\.venv\Scripts\python.exe scripts/benchmark_web_isolation.py --baseline .test_runs/research_baseline.json --output .test_runs/research_after.json
```

基线文件需修改前先生成，详见脚本参数。单元测试、真实 PostgreSQL/Redis 联调、容器构建、浏览器回放和性能基准应分别报告。录屏步骤和真实性边界见 `RESEARCH_DEMO.md`。

Windows 默认路径长度限制可能导致深层 PDF sidecar 文件写入失败。测试临时目录使用短名称，每次换一个新目录，避免复用有残留的目录。历史失败原因与修正记录保留在本地 DEVELOPMENT.md；此处只给出可复现运行约定。

### 本轮验证记录（2026-10-21）

- 项目 `.venv` 全量回归：262 passed，1 skipped；真实 Redis/PostgreSQL 联调：3 passed。另有旧库只读快照测试验证原数据库文件 hash 不变及重复镜像不增加节点。
- 加入就绪探测后的本轮全量回归：266 passed，1 skipped；跳过的是显式启用的中间件模块，随后单独联调 3 passed。新增故障探测测试覆盖 Redis、数据库及两者同时不可用，恢复后重新返回 200。
- Web 与独立 Worker 镜像实际构建、容器启动成功；浏览器通过真实 HTTP、Redis 与 PostgreSQL 完成离线审批恢复录屏。
- 性能基线与后测报告保留在 `.test_runs/research_baseline.json`、`.test_runs/research_after.json`，分析见本地 DEVELOPMENT.md；不把该微基准当作完整 Coding 的零退化验收。

## 8. Docker Hub 超时与已有全局安装

`auth.docker.io/token` 超时是 Docker Hub 网络问题，不是虚拟环境问题。项目支持 `.env.web` 中的 `RESEARCHOPS_BUILD_PROXY`，也可临时执行：

```powershell
.\scripts\start_web.ps1 -Research -BuildProxy http://127.0.0.1:7897
```

7897 只是在当前机器核实过的代理端口；其他机器填写自己的可用地址。脚本为 Docker 启动进程临时设置 HTTP_PROXY/HTTPS_PROXY，退出后恢复，不修改系统全局代理、不注入应用流量。支持可信 `PYTHON_IMAGE` / `PIP_INDEX_URL`，不要乱填公共镜像站或在地址内嵌密码。

曾误用系统 pip 时，用原解释器核对 `sys.executable` 和 `pip show researchops-mvp` 后只卸载项目注册；pip 没有完整回滚快照，不要批量删除共享依赖。PowerShell 临时 `$env:` 随终端退出而消失；保留 `.venv`、数据库卷和私密配置。

## 9. 安全边界

Mock 共享密码不是生产身份系统；不要公开服务。上线前补齐身份、角色、限流、上传总配额、数据库迁移、备份与事件保留策略。单文件限制不代表无限次上传安全。PDF 解析器处理不可信文件，必须维持隔离并及时更新。SQL Trace、节点正文和本地录屏可能含敏感材料，应限制访问。当前没有多主机共享文件系统、全自动最新文献覆盖或生产级副作用恰好一次保证。

## 10. 配置速查与日常检查

按 TaleClaw 的“快速开始 → 配置 → 测试 → 文档”组织方式补齐操作入口；这里只列本项目实际支持的参数，不引入无关聊天网关。

| 配置 | 用途 / 默认约定 |
| --- | --- |
| `RESEARCHOPS_JWT_SECRET` | 至少 32 字节随机签名密钥，脚本首次生成 |
| `RESEARCHOPS_MOCK_PASSWORD` | 本机演示共享密码，不是生产账号系统 |
| `RESEARCHOPS_SECURE_COOKIE` | 本机 HTTP 为 0，HTTPS 部署为 1 |
| `RESEARCHOPS_POSTGRES_PASSWORD` | PostgreSQL 密码，修改须同时考虑已初始化卷的数据库账号 |
| `RESEARCHOPS_WEB_PORT` | 默认 8000，脚本支持 `-Port` |
| `RESEARCHOPS_POSTGRES_PORT` / `RESEARCHOPS_REDIS_PORT` | 主机默认 15432 / 6379，Compose 内使用服务名和容器端口 |
| `RESEARCHOPS_DATABASE_URL` / `RESEARCHOPS_REDIS_URL` | 本地调试连接；容器由 Compose 注入内部地址 |
| `RESEARCHOPS_MODEL_PRICES` | 真实 Token 的估价费率；缺失时不声称已知费用或节省率 |
| `PYTHON_IMAGE` / `PIP_INDEX_URL` | 可信镜像与依赖源；不能把访问凭据写入公开地址 |
| `RESEARCHOPS_BUILD_PROXY` | 构建网络代理，地址按本机实际情况配置 |

```powershell
# HTTP 存活检查
Invoke-RestMethod http://127.0.0.1:8000/api/v1/health
# Redis + 数据库就绪检查；任一依赖失败返回 HTTP 503
Invoke-RestMethod http://127.0.0.1:8000/api/v1/ready
# 健康状态与日志，不输出环境变量全文
 docker compose --env-file .env.web -f compose.web.yaml --profile web --profile research ps
 docker compose --env-file .env.web -f compose.web.yaml logs --tail 80 web knowledge-worker
# 真实 Redis / PostgreSQL 与所有权隔离检查（必须已有项目 .venv 依赖）
.\.venv\Scripts\python.exe scripts/verify_web_infrastructure.py
```

服务监听本机回环地址，不直接暴露公网。日志可能包含论文正文或模型输出，分享前脱敏；不要把 `docker compose config` 的完整展开内容贴到公开报告中。

Compose 的 Web healthcheck 使用 `/api/v1/ready`。它执行 Redis `PING` 与数据库 `SELECT 1`，不写入研究数据、不调用付费模型。PostgreSQL 新连接与连接池等待各限 3 秒，Redis 连接/读取分别限 3/5 秒。就绪成功返回 `{"status":"ready","checks":{"redis":true,"database":true}}`；依赖异常时不返回连接串或异常正文。此探测覆盖 Web 中间件，Worker 心跳继续在知识库概览中查看，模型可用性通过实际任务验证。

实际 Web 三轮对话可用项目虚拟环境执行 `scripts/verify_web_conversation.py --output .maintenance/web_conversation_next` 复现；需已启动服务并配置真实模型，会产生模型费用。脚本保留每轮完整 SSE、回答及用量，拒绝覆盖已有输出目录，不归档密码、Cookie 或 Token。README 中另列 CLI Harness 四轮演示，两种 Runtime 分开记录。

## 11. 数据备份与恢复约定

| 数据层 | 应保留内容 | 恢复注意事项 |
| --- | --- | --- |
| 原科研档案 | `papers/`、`datasets/`、`knowledge_base/`、原始 manifest / hash | 原样保留；不因镜像到 PG 而删除原件 |
| Web 文件 | `web_workspace/` 的上传、解析产物与独立 SQLite | 先暂停写入并停止本项目 Web / Worker，再复制完整目录；不要只复制活跃 WAL 数据库的主文件 |
| PostgreSQL | 文档 / 节点、任务、事件、账本 | 使用 `pg_dump` 备份并在独立数据库演练恢复；不是只备份向量即可 |
| Redis | Session / RunState 与 AOF 命名卷 | 有 TTL，不能用它代替永久研究档案；不要删除卷 |
| 本地证据 | `.maintenance/` 的视频、事件、截图、评测报告 | 保留访问权限；包含隐私时不公开 |

备份应保存在不同介质，并记录应用版本、数据库版本、时间与文件 hash。先暂停提交、停止 Web 和 Worker，完成 PG 逻辑备份与文件备份后再恢复服务；Redis 的完整一致性备份需按 AOF / 卷机制单独执行。恢复只在新目录、新数据库或独立 Compose 项目中演练，**不要直接覆盖当前科研资产**。

当前尚无一键备份 / 恢复工具和自动恢复验收，以上是操作约定，不是已完成容灾能力。Schema 迁移、租约接管与恢复演练作为后续开发方向记录在本地 DEVELOPMENT.md。

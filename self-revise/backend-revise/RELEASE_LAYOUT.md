# 定稿目录、证据与复现说明（2026-11-10）

本次只整理交付结构与资源定位，不重新选择 RAG 模型或重跑付费基准。完整后端总报告、逐题附录、金标银标、消融、负结果与源码快照均保留。原始 JSON、绝对路径、来源 hash 和历史回答不改写为新路径。

## 目录映射

| 原位置 | 当前位置 |
|---|---|
| `self-revise` 下后端报告与各类 evidence/history | `self-revise/backend-revise/` 下同名文件和目录 |
| `self-revise/frontend-evidence` | `self-revise/frontend-revise/evidence` |
| 根 `README_frontend.md` | `self-revise/frontend-revise/README_frontend.md` |
| 根 `tests` | `self-revise/backend-revise/tests`（其中 API 测试属于后端合同验证） |
| `scripts` 的评估/审计/对照脚本 | 后端或前端 `scripts/`，按用途分类 |
| 根 `internal_experiments` | `self-revise/backend-revise/internal_experiments` |
| 根 `.maintenance` | `self-revise/backend-revise/local-runtime/.maintenance` |
| 旧根 pytest/Python 缓存、egg-info | `self-revise/backend-revise/local-runtime/` |
| 旧根应用容器 compose | `self-revise/backend-revise/history/deployment/compose.legacy.yaml`，不作为当前启动入口 |

逐项移动清单见 `directory-migration-v20261110.json`。根旧 pytest 缓存首次受 ACL 拒绝，取得系统授权后迁入 `local-runtime/legacy-pytest-cache`；清单保留首次错误，不伪改历史。

产品 `.venv`、`.env`、`.env.web`、现有论文库和数据集不删除。`web_workspace` 链接指向迁移后的原业务实体，PG schema 与 Qdrant 项目 namespace 使用原位置的规范身份，避免因归档路径变更创建“空的新账户库”。历史来源读取支持旧路径映射。旧冻结运行的 resume 仍受指纹和合同校验，不能保证任意旧实验搬动后原地续跑。

## 测试入口

在项目根、激活现有 `.venv` 后运行 `python -m pytest`；`pyproject.toml` 已将测试发现改到本目录，使用 importlib 模式避免不同板块同名测试文件冲突，并关闭无必要的 pytest 缓存，避免 Windows ACL 缓存写入警告。Windows 的回收站测试路径较长，建议用 `--basetemp=self-revise/backend-revise/.t`，不要额外加很深的临时路径。`.t` 为忽略的可再生工作区，不是证据。

前端浏览器与真实模型测试位于 `../frontend-revise/scripts`，真实模型测试会产生费用，日常启动不必重跑。原题从迁移后的 `README_frontend.md` 读取，当前脚本使用项目根探测，不再假定自己位于根 scripts。历史归档代码和源码快照保持其所属实验协议；不保证早期 SQLite/.venv-v03 脚本能直接当现行生产入口。

正式 `scripts/research_rag.py` 仅保留 `ask/ingest/noop/structure-build`；原完整增量实验、检索消融、生成及 RAGAS 命令保留在 `self-revise/backend-revise/scripts/research_rag.py`。正式入口不再导入测评脚本。`noop` 不依赖历史 append 测试文件夹，检查当前发布库；科研问答和正常增量入口保持原路径。

`config/benchmarks` 是测试协议资源，部分冻结 manifest 使用相对路径及 SHA256；保留其路径以维护证据可核查性。编码模板内的测试属于生成项目的产品资源，不迁出源码。

## 本轮验证

前端 Vite production build 成功。迁移后首次 29 项 API/布局回归中，28 项通过，1 项因 Windows 深路径回收站移动失败；保留 `../frontend-revise/evidence/release-layout-regression.xml`，缩短临时路径后 29 项全部通过，结果保留于相邻 `release-layout-regression-final.xml`。原有最新 67 项集合加 2 项 namespace 兼容测试共 69 项通过，见 `../frontend-revise/evidence/release-layout-67-plus-namespace.xml`。最终布局、namespace、冻结 manifest 另有 5 项检查，见 `environment-evidence/release-layout-paths-final.xml`；这些集合有重叠，不相加成 103 项。完整发现 329 项测试无收集错误，未宣称本轮重跑全部 329 项或付费 RAGAS/LongMemEval。321 个当前 Python 文件完成语法检查，当前浏览器脚本完成 Node 语法检查，产品 CLI 与归档评估 CLI 的帮助入口正常。

最后只读审计入口为 `python -B self-revise/backend-revise/scripts/audit_release_layout.py`，核对根目录、README 全部本地链接、金银标与最终前端证据、Web namespace、知识库清单和当前 Python 语法；检查不连接数据库、不调用模型、不创建业务 Store。

只读核对现有 `web-runtime-binding.json`：迁移后的 Web schema 与原绑定一致；已发布论文清单 SHA256 仍为 `1c1689eda16e908b2e9415056d24e49156ece93cc486214705d9e6351143862b`。没有创建新用户数据库，没有重新编码论文，没有启动或停止用户正在使用的 Web 服务。由于资源定位代码已更新，用户下次使用前需正常重启后端，使其加载新代码。

公开 README 是面试阅读入口，既有后端总报告为完整历史分析，本说明解释最终目录状态。原报告中的旧路径或旧预算是当时事实；当前网页预算、真实原题失败修复与最终链路以 `../frontend-revise/evidence/readme-cases-v20261110/README.md` 为准。

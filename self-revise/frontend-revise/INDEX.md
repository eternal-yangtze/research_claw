# 前端验收材料

- [运行与完整连续试用案例](README_frontend.md)。
- [真实原题、超时诊断与最后复验](evidence/readme-cases-v20261110/README.md)：19 次真实模型回合，15 类场景；包含旧失败与安全 partial。
- [注册、真实 PDF 入库和撤销](evidence/registration-rollback-v20261110/README.md)：真实浏览器、TimesFM 增量和测试版本撤销。
- `scripts/`：浏览器、UI 与真实模型测试入口；`evidence/`：截图、原始答案、XML 与过程证据。确定性模型替身 UI 分支和真实科研回答明确分开。
- API 合同回归位于 `../backend-revise/tests/api/`，不会因为在网页触发就移动进前端产品源码。

原始证据中的绝对路径保持历史事实，位置映射见[定稿归档说明](../backend-revise/RELEASE_LAYOUT.md)。

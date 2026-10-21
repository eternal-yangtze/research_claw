---
name: literature-acquire
description: "Acquire literature for a NEW research idea: understand the question, plan online searches, download public papers, parse and analyze full text. Use for 新idea、找论文、检索并下载文献. Use evidence-retrieve for questions answerable from the existing library; use study-plan to design experiments from an already prepared corpus."
---

# Acquire literature

1. Read the exact input schema returned by load_skill. Supply the complete user idea as `idea_text`; preserve scope and constraints.
2. Respect explicit `max_queries`, `limit_per_query`, `max_downloads` and run budgets. For unspecified limits use host defaults. Invoke once; let the acquisition DAG manage download/parse/analysis dependencies and recovery.
3. Invoke with `invoke_skill(name="literature-acquire", sha256=<loaded hash>, arguments=<schema-valid object>)`. Do not call hidden tools directly or reimplement connectors.
4. Report separate candidate/selected/downloaded/parsed/analyzed counts from the result, source failures and exact retained artifact paths. Preserve partial failures; no-result is not research success.
5. Describe paper methods only when full-text analysis supports them. Treat parser-only formulas as uncertain; preserve the host's source-silver statuses. Do not claim global novelty, start coding, train models or promise accuracy.

Use the host's persistent DAG for retries after transient failures. Reuse the same idea and limits instead of deleting prior records or resetting a budget. A completed tool call may still report partial acquisition; explain the actual counts.

Output contract: the unchanged research-pipeline report inside `result.output`, including `task_id`, `pipeline_counts`, `acquisitions`, and `artifact`; plus the skill invocation artifact and hash.

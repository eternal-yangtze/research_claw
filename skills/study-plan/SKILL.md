---
name: study-plan
description: "Create and review a coding-before research study from an existing corpus and verified datasets: task-specific controlled arms, metrics, frozen experiment contract and critic/revision gates. Use for 研究方案、设计实验、冻结计划、coding前规划. Currently validated for ViT parameter-efficient tuning image classification. Do not use for simple retrieval, new-paper acquisition, code generation or training."
---

# Plan a study before coding

1. Require a natural-language `idea` object and an existing `source_workspace`; use verified dataset manifests from `dataset_workspace` when supplied. Preserve all user constraints. Do not assume the example domain applies to unrelated tasks.
2. Read the loaded input schema. Invoke `study-plan` with its hash, the idea, authorized source paths and a token cap. Use `retrieval_strategy="fusion"` only when requested or appropriate for a quality-oriented study; legacy/cpu remain available. All nested model calls share the run's admission ledger.
3. Let the existing plan_study workflow perform query planning, evidence selection, contract compilation, critic/revision and final review. Do not skip a gate or synthesize a frozen contract yourself.
4. Report `ready_for_coding` only when the returned workflow and mechanical checks pass. Explain blockers or required adapters otherwise. A tool that executed successfully is not necessarily a successful research plan.
5. Keep facts separate from proposed experimental choices. Preserve formula uncertainty and the non-formula supporting evidence gate. Never claim coding, training or research success has occurred.

Output contract: the unchanged plan_study result in `result.output`, with proposal status, mechanical/critic results, attempts, artifact directory and charged component tokens. Use the outer run ledger for the overall bill; do not add its nested component ledger again. Reinvoke identical inputs to reuse existing paid-call and snapshot caches; use a new source/output version when changing frozen inputs.

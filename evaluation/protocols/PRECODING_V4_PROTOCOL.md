# 2026-10-20 prospective pre-coding protocol

Primary scope: small-data pretrained ViT image classification using LoRA, shallow
visual prompts and residual bottleneck adapters. No coding, model training,
classification accuracy measurement or research hypothesis confirmation is in scope.

The user has authorized higher LLM usage. New run cap: 3,000,000 tokens, separate
from all historical ledgers. Existing .env credentials/model are used unchanged.
No historical failures or unknown-usage reservations are erased.

Primary evaluation: all six tasks in `precoding_heldout_v2.json`, authored/frozen
before any v4 paid calls. These combine factors not combined in the previous
twelve tasks and include CIFAR100/Flowers102. They remain an internally authored
prospective set, not a secret external benchmark. All six remain in the denominator.

Each task must: perform fresh search; obtain at least eight distinct hash-verified,
parsed fulltexts independently assessed relevant from actual passages; include at
least one PDF hash absent the original 30-paper corpus; recompute task evidence;
produce a task-specific study; compile exact arms, data subsets, tuning/test jobs,
metrics and decision rules; pass mechanical integrity and evaluator-only task
constraints; pass all seven semantic criteria in both critic and separate final
review calls. Cached fulltexts may be shared but per-task relevance is recomputed.
The same newly downloaded PDF may satisfy novelty versus the original corpus for
several tasks; this is not claimed to be a separate new paper for every task.

Planner and repair calls receive the user's task, actual fulltext nodes, host
protocol, data catalog and runtime critic feedback. They never receive evaluator
manifests, expected arms or task-gate feedback. At most four plan attempts per task.
No acceptance criterion is weakened after seeing a result. Every attempt, first
pass failure, source failure and final result is retained. Two acceptance calls
use the same model and therefore provide silver evidence with correlated bias.

Primary performance: final end-to-end success / 6. Also report first-pass success,
mechanical/constraint satisfaction, fulltext coverage, locatable and supported
citations, and repair counts. Time is logged only; not a performance score.
Tokens are experimental cost. No claim of RAG speedup or token optimization is
made from this run. Old twelve tasks form a separate regression set; they are no
longer held out after their failures were inspected. Boundary/contrastive tests
are reported separately and cannot inflate positive-task success.

This protocol aims to establish bounded domain generality of the research
planning workflow through a reviewed frozen specification. Human expert gold
review and actual training would support stronger scientific outcome claims.

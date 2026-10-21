"""Idea-driven vision study contracts. No benchmark IDs or expected answers here.

The model chooses the study; this compiler expands and checks an explicit design.
Version 1 contracts and their historical coding adapter remain unchanged.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from researchops.core.artifacts import canonical_hash, file_hash
from researchops.experiments.preparation import build_vision_contract

VERSION = "researchops-experiment-contract-v2"
HOST_PROTOCOL = {
    "backbone": "vit_base_patch16_224.augreg_in21k_ft_in1k; pretrained weights sha256 required before training",
    "image_size": 224, "effective_batch_size": 32, "max_epochs": 20, "warmup_epochs": 2,
    "scheduler": "cosine to 0.01*LR after linear warmup", "early_stopping": False,
    "optimizer": "AdamW", "loss": "cross entropy, label_smoothing=0",
    "decay_groups": "trainable matrices; exclude biases, normalization scales and prompt tokens",
    "tuning": "two method-specific LRs per dataset/arm/fraction/seed; validation accuracy, smaller LR ties",
    "evaluation": "Reuse selected candidate checkpoint for final test; no additional main training",
    "checkpoint_selection": "max validation accuracy; earliest epoch ties",
    "subset_rounding": "classwise ceil(n*f), minimum 1; fixed seed42 nested sample lists",
    "selection": "validation only; no test fitting or post-hoc calibration",
}
METHODS = {"linear_probe", "full_finetune", "lora", "vpt_shallow", "bottleneck_adapter"}
METRICS = {"accuracy", "macro_f1", "ece_15_equal_bins", "nll", "trainable_parameters",
           "training_seconds", "peak_memory", "train_validation_gap", "reliability_bin_summaries"}
METRIC_ALIASES = {"ece": "ece_15_equal_bins", "ece_15_equal_width_bins": "ece_15_equal_bins",
                  "train_validation_accuracy_gap": "train_validation_gap"}

PLAN_SYSTEM = """You are a research planner for small-data pretrained ViT image classification.
Read the user's research problem, independently assess relevant full-text excerpts, then
propose a falsifiable controlled study. Input hypotheses are NOT established facts.
The supported methods are linear_probe, full_finetune, lora (independent attention Q/V,
frozen K), vpt_shallow, bottleneck_adapter (post_mlp_residual, zero up projection).
You can vary rank/alpha/dropout, exact zero-based target_blocks (12 ViT blocks), prompt_length,
bottleneck_dim and AdamW weight_decay PER ARM. Head is trained in every arm.
Supported datasets and hash-pinned fractions are supplied. Use three seeds 42/43/44.
Use equal two-LR validation-only searches PER ARM AND FRACTION. No test-based selection.
Separate method-family evidence from agent-proposed configurations; do not invent paper
formulas or say the proposed experiment has already succeeded. Report missing evidence.
Unsupported task families, requested test leakage, abstract-only exact formula claims and
impossible formal-training budgets must return blocked/needs_adapter/needs_refinement
with a reason and a clearly marked alternative. Never silently change a user's constraint.
Return ONLY a compact JSON object with this schema (omit design when blocked):
{"status":"proposed|blocked|needs_adapter|needs_refinement", "reason":"...",
"task_family":"image_classification|other", "research_question":"...",
"hypothesis":"unproven ...", "evidence_claims":[{"claim":"...",
"kind":"paper_fact|proposal|unknown", "citations":["supplied chunk_id"]}],
"design":{"datasets":["dtd","pets"],"fractions":[0.25],"seeds":[42,43,44],
"arms":[{"arm_id":"unique label","method":"lora","rank":8,"alpha":8,
"dropout":0.0,"target_blocks":[0,1,2,3,4,5,6,7,8,9,10,11],"weight_decay":0.05}],
"primary_metrics":["accuracy"],"secondary_metrics":["macro_f1","trainable_parameters",
"training_seconds","peak_memory"], "analysis_plan":"specific per-dataset paired comparisons,
uncertainty and interpretation; for fraction interaction report difference of method gaps",
"success_rule":"specific prospective outcome rule, no assumed winner",
"selection_policy":{"mode":"ordinary|per_dataset_vs_shared_rank",
"objective":"validation_accuracy", "tie_break":"smaller rank then smaller LR then earliest epoch",
"description":"validation only; rank policies reuse identical candidate runs; shared rank
maximizes mean validation accuracy across datasets, independently per seed"},
"controls":["..."],"limitations":["..."]}}
Include all input-specified arms and references; avoid unrelated five-method/full-fraction grids.
For non-LoRA arms use prompt_length or bottleneck_dim as appropriate. Exact configuration
choices lacking paper evidence MUST be marked as proposed. Describe train/validation gap
measurement for regularization questions and reliability bin summaries/logits for calibration.
All outputs are plans before coding; no training is executed or timed by this call.
The supplied host_protocol is authoritative: do NOT invent a different epoch/batch/scheduler,
rounding/decay/early-stopping protocol in free text. Main jobs only evaluate selected checkpoints,
they do not retrain. Data manifests and subsets are hash-pinned and mechanically verified;
pretrained weights remain a pre-training acquisition prerequisite, NOT missing study feasibility.
Keep output SHORT: at most 3 evidence_claims, each <=200 characters and <=2 citations;
analysis_plan and success_rule each <=400 characters; other prose fields <=200 characters.
Do not restate every supplied evidence node, repeat the host protocol, or list irrelevant papers."""

# Explicit fields keep a marked alternative distinct from the requested protocol.
PLAN_SYSTEM += """\nFor blocked/needs_adapter/needs_refinement answers, include
\"alternative\":{\"requires_constraint_change\":true,\"changed_constraints\":[\"the specific user requirement that must change\"],\"description\":\"a separately marked alternative\"}.
If even an alternative cannot be specified from evidence, state that explicitly.
Never imply that an alternative satisfies an incompatible original constraint."""

# The robust workflow has room for an operational analysis, rather than forcing a
# complete statistical decision into the legacy 400-character display budget.
ROBUST_PLAN_SYSTEM = PLAN_SYSTEM + """\nFor this workflow the earlier SHORT prose caps are superseded:
analysis_plan and success_rule may each use 2400 characters; up to 6 evidence claims.
Specify every primary contrast using arm IDs, direction, units, aggregation and
dataset/fraction. Specify paired seed differences, the exact interval algorithm,
confidence level, multiplicity policy, numerical margin if applicable, and explicit
supported/refuted/inconclusive outcomes. Three seeds give weak uncertainty estimates;
do not claim high power or population generality. Do not use an unspecified 'paired CI'.
If using t intervals with three seed differences, define mean +/-4.30265273*s/sqrt(3)
(95%, df=2); no normal approximation. For noninferiority delta=candidate-reference,
support requires the lower bound > -margin; refute when upper bound < -margin;
otherwise inconclusive. Label margins as proposed, not paper facts. For multifactor
tasks define the difference-in-differences, all cells, and which effects are primary.
If multiple primary contrasts exist, explicitly define a multiplicity adjustment or
label intervals descriptive/exploratory and forbid confirmatory significance claims.
Always include a cited paper_fact limited to what the supplied excerpt actually says.
Family evidence motivates methods; the host's architecture/settings are proposals,
especially adapters, not a claim to exactly reproduce AdaptFormer. Supplied compiled
method definitions resolve architecture; avoid contradicting them in prose.
Do not introduce noninferiority if a descriptive tradeoff suffices for the question.
training_seconds is an optional downstream experimental cost, not agent performance.
"""


def read_verified_datasets(workspace: Path) -> dict:
    states = {}
    for state_path in sorted((workspace / "datasets").glob("*/state.json")):
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("status") == "verified":
            states[state["dataset_id"]] = state
    return states


def dataset_catalog(states: dict) -> list:
    return [{"dataset_id": key, "classes": value["verification"]["class_count"],
             "fractions": list(value["verification"]["train_fractions"]),
             "splits": value["verification"]["split_counts"]} for key, value in states.items()]


def planning_interface(states: dict) -> dict:
    """Expose the real runtime vocabulary; no evaluator or task-specific values."""
    defaults = {}
    ids = list(states)[:2]
    if len(ids) == 2:
        defaults = build_vision_contract({}, ids, states,
                                         {"baseline_citations": [], "recommendations": []}, {})["methods"]
    return {"supported_metrics": sorted(METRICS), "metric_aliases": METRIC_ALIASES,
            "method_defaults": defaults,
            "rule": "Use canonical metric names. Methods may vary supported arm fields; lr_candidates and architectures shown here are the compiler defaults, not task-specific recommendations."}


def compile_contract(idea: dict, proposal: dict, states: dict, sources: dict) -> dict:
    if proposal.get("status") != "proposed" or proposal.get("task_family") != "image_classification":
        raise ValueError("Only explicit supported proposals can be frozen")
    design = copy.deepcopy(proposal["design"])
    aliases = {}
    for key in ("primary_metrics", "secondary_metrics"):
        normalized = []
        for metric in design[key]:
            canonical = METRIC_ALIASES.get(metric, metric)
            if canonical != metric:
                aliases[metric] = canonical
            if canonical not in normalized:
                normalized.append(canonical)
        design[key] = normalized
    datasets, fractions, arms = design["datasets"], design["fractions"], design["arms"]
    if not datasets or len(set(datasets)) != len(datasets) or not set(datasets) <= states.keys():
        raise ValueError("Unknown or repeated datasets")
    if design["seeds"] != [42, 43, 44] or not fractions or len(set(fractions)) != len(fractions):
        raise ValueError("Invalid seeds or fractions")
    if not arms or len(arms) > 16 or len({a["arm_id"] for a in arms}) != len(arms):
        raise ValueError("Invalid study arms")
    metrics = set(design["primary_metrics"] + design["secondary_metrics"])
    if not design["primary_metrics"]:
        raise ValueError("Missing primary_metrics; choose a nonempty list from " + repr(sorted(METRICS)))
    if not metrics <= METRICS:
        raise ValueError(f"Unsupported metrics {sorted(metrics - METRICS)}; supported canonical names: {sorted(METRICS)}; aliases: {METRIC_ALIASES}")
    policy = design["selection_policy"]
    if policy["mode"] not in {"ordinary", "per_dataset_vs_shared_rank"} or policy["objective"] != "validation_accuracy":
        raise ValueError("Unsupported selection policy")
    if not design["success_rule"] or not design["analysis_plan"]:
        raise ValueError("Prospective analysis required")
    # Reuse immutable definitions of preprocessing/initialization/data schemas only.
    rec = {"baseline_citations": sorted({c for item in proposal.get("evidence_claims", [])
                                        for c in item.get("citations", [])}), "recommendations": []}
    # Legacy constructor requires two datasets; v2 can use any nonempty supported subset.
    base_ids = list(states)[:2]
    if len(base_ids) != 2:
        raise ValueError("Shared catalog needs at least two verified datasets")
    spec = build_vision_contract(idea, base_ids, states, rec, sources)
    spec["datasets"] = []
    for dataset_id in datasets:
        state = states[dataset_id]
        if state["status"] != "verified":
            raise ValueError("Dataset not verified")
        data = copy.deepcopy(state["verification"])
        path = Path(data["manifest_path"])
        if file_hash(path) != data["manifest_sha256"]:
            raise ValueError("Manifest hash mismatch")
        selected = {}
        for fraction in fractions:
            matches = [v for k, v in data["train_fractions"].items() if float(k) == float(fraction)]
            if len(matches) != 1 or file_hash(matches[0]["path"]) != matches[0]["sha256"]:
                raise ValueError("Missing or changed fraction manifest")
            selected[str(fraction)] = matches[0]
        data["train_fractions"] = selected
        spec["datasets"].append(data)
    methods = {}
    jobs = []
    for arm in arms:
        method = arm["method"]
        if method not in METHODS:
            raise ValueError("Unsupported method")
        config = copy.deepcopy(spec["methods"][method])
        config.update(arm)
        if not 0 <= float(config.get("weight_decay", 0.05)) <= 1:
            raise ValueError("Invalid decay")
        if method == "lora":
            blocks = config.get("target_blocks", list(range(12)))
            if not blocks or sorted(set(blocks)) != blocks or not set(blocks) <= set(range(12)):
                raise ValueError("Invalid target blocks")
            if not 1 <= config["rank"] <= 256 or not 0 <= config["dropout"] < 1 or config["alpha"] <= 0:
                raise ValueError("Invalid LoRA configuration")
            config["target_blocks"] = blocks
        if method == "vpt_shallow":
            if not 1 <= config["prompt_length"] <= 256:
                raise ValueError("Invalid prompt length")
            config["trainable"] = [f"{config['prompt_length']} shallow prompt tokens", "classification head"]
        if method == "bottleneck_adapter":
            if not 1 <= config["bottleneck_dim"] <= 768:
                raise ValueError("Invalid bottleneck width")
            config.update(placement="post_mlp_residual", up_initialization="zero")
        methods[arm["arm_id"]] = config
        for dataset in spec["datasets"]:
            for fraction in fractions:
                subset = dataset["train_fractions"][str(fraction)]
                for seed in design["seeds"]:
                    # Seed-local selection makes all reported replicates independent.
                    for lr in config["lr_candidates"]:
                        jobs.append({"stage": "validation_tuning", "operation": "train_and_validate", "dataset_id": dataset["dataset_id"],
                                     "arm_id": arm["arm_id"], "method": method, "fraction": fraction,
                                     "subset_sha256": subset["sha256"], "seed": seed, "lr": lr,
                                     "evaluate_test": False})
                    jobs.append({"stage": "main", "operation": "evaluate_selected_checkpoint_no_retraining", "dataset_id": dataset["dataset_id"],
                                 "arm_id": arm["arm_id"], "method": method, "fraction": fraction,
                                 "subset_sha256": subset["sha256"], "seed": seed,
                                 "lr": "selected_on_validation_for_dataset_arm_fraction_seed", "evaluate_test": True})
    if policy["mode"] == "per_dataset_vs_shared_rank":
        if any(a["method"] != "lora" for a in arms) or len({a["rank"] for a in arms}) != len(arms):
            raise ValueError("Rank policies require one identical candidate pool of LoRA ranks")
        policy["candidate_pool_shared"] = True
        policy["selection_per_seed"] = True
        policy["shared_objective"] = "mean validation accuracy across all datasets"
        policy["test_evaluation"] = "After policies frozen, reuse candidate predictions; never retrain different candidate pools"
    for job in jobs:
        job["job_id"] = "job_" + canonical_hash(job)[:20]
    spec.update(schema_version=VERSION, status="frozen_ready_for_coding", design=design,
                hypothesis=proposal["hypothesis"], research_question=proposal["research_question"],
                methods=methods, jobs=jobs, evidence_claims=proposal.get("evidence_claims", []),
                evidence_limitations=design["limitations"], baseline_family_support={},
                dataset_recommendations=[], training_executed=False, research_success=None)
    spec["metric_name_normalization"] = aliases
    spec["training"].update(lr_selection="Per dataset/arm/fraction/seed; two trials on validation only; smaller LR ties.",
                            early_stopping={"enabled": False}, host_protocol=HOST_PROTOCOL,
                            selection_policy=policy, weight_decay="per arm",
                            optimizer_groups={"decay": "trainable matrices including LoRA A/B and adapter weights",
                                              "no_decay": "bias, normalization scales, prompt tokens"})
    spec["metrics"] = {key: value for key, value in spec["metrics"].items() if key != "success_rule"}
    spec["metrics"].update(primary=design["primary_metrics"], secondary=design["secondary_metrics"],
                           success_rule=design["success_rule"], analysis_plan=design["analysis_plan"],
                           nll="mean(logsumexp(logits)-logit_of_true_class); float64 stable recomputation",
                           ece_15_equal_bins="15 equal-width bins [0,1]; last includes 1; save count, mean confidence, accuracy",
                           saved_outputs=["sample_id", "logits", "labels", "split", "checkpoint_sha256"],
                           reliability_summary="per-bin count/accuracy/confidence for each dataset/arm/fraction/seed",
                           train_validation_gap="train accuracy without augmentation minus validation accuracy at selected checkpoint")
    spec["data_policy"]["fixed_nested_fractions"] = fractions
    spec["run_budget"] = {stage: sum(j["stage"] == stage for j in jobs) for stage in ("main", "validation_tuning")}
    spec["run_budget"].update(total=len(jobs), automatic_training_authorized_by_this_preparation=False,
                              training_runs=spec["run_budget"]["validation_tuning"],
                              test_evaluations=spec["run_budget"]["main"],
                              training_seconds_estimate=None, estimate_reason="No pilot measured; job count is not runtime")
    spec["coding_acceptance"].extend([
        "Consume contract v2 arm IDs and exact block/width/prompt/optimizer settings; legacy v1 executor is insufficient.",
        "Recompute NLL/ECE and reliability summaries from saved logits; no test-fitted calibration.",
        "Verify trainable groups and exact target blocks per arm; paired nested subsets and seed-local validation selection.",
    ])
    return spec

"""Evaluator-only gates. Never imported by the research planner."""
from __future__ import annotations

import json
from pathlib import Path

from researchops.core.artifacts import file_hash

RUBRIC = """Independently review an anonymized research plan against the user's task and
full-text evidence, NOT against presumed experimental outcomes. Return only JSON:
{"criteria":{"question_alignment":"pass|fail|unknown","baseline_appropriateness":"pass|fail|unknown",
"evidence_support":"pass|fail|unknown","controlled_comparison":"pass|fail|unknown",
"metric_and_decision_rule":"pass|fail|unknown","constraint_satisfaction":"pass|fail|unknown",
"honest_uncertainty":"pass|fail|unknown"},"reasons":["specific evidence/plan issue or justification"],
"citation_support":[{"chunk_id":"...","verdict":"supported|unsupported|unknown"}]}
For blocked boundary tasks evaluate whether the reason correctly addresses the request
and the alternative is marked as needing a protocol change. Do not demand image plans there.
For positive tasks, all requested variables must actually change, baselines must fit the
question, validation/test must be isolated, and outcomes must remain unproven. An unknown
essential field fails acceptance. A family citation need not support every proposed
hyperparameter: a clearly marked proposed controlled experiment is valid. Reject claims
that present customary formulas/settings as the cited paper's exact settings without text.
Assess only supplied evidence. Different tasks can legitimately have different methods,
fractions, metrics and decision rules. No assumed universal best rank/method.
Explain missing coverage, conflicting constraints and mismatches specifically.
The compiled host protocol is authoritative and shown to the planner. Hashes and the
mechanical gate attest data integrity; do not require full manifests in this prompt.
Main jobs evaluate the selected checkpoint WITHOUT retraining. Pinned pretrained weight
acquisition is a pre-training coding task, not a reason to reject a feasible study plan.
Keep reasons to <=5 concise items and citation_support to the cited paper-fact nodes only."""
CRITERIA = {"question_alignment", "baseline_appropriateness", "evidence_support", "controlled_comparison",
            "metric_and_decision_rule", "constraint_satisfaction", "honest_uncertainty"}


def silver_pass(review: dict) -> bool:
    criteria = review.get("criteria", {})
    return CRITERIA <= criteria.keys() and all(criteria[k] == "pass" for k in CRITERIA)


def mechanical_gate(spec: dict | None, evidence: list[dict], store) -> dict:
    checks = {}
    details = []
    if spec is None:
        return {"passed": False, "checks": {"frozen_spec": False}, "errors": []}
    try:
        refs = spec.get("baseline_citations", [])
        allowed = {n["chunk_id"]: n for n in evidence}
        checks["citations_present"] = bool(refs)
        checks["citation_locators_and_text"] = all(
            c in allowed and (row := store.get_chunk(c)) is not None
            and row["paper_id"] == allowed[c]["paper_id"]
            and allowed[c]["text"] in row["text"] for c in refs)
        facts = [c for c in spec.get("evidence_claims", []) if c["kind"] == "paper_fact"]
        checks["facts_have_citations"] = bool(facts) and all(c.get("citations") for c in facts)
        checks["no_training_claim"] = spec["training_executed"] is False and spec["research_success"] is None
        data_ok = True
        for data in spec["datasets"]:
            data_ok &= file_hash(data["manifest_path"]) == data["manifest_sha256"]
            rows = [json.loads(line) for line in Path(data["manifest_path"]).read_text(encoding="utf-8").splitlines()]
            pools = {s: {r["sample_id"] for r in rows if r["split"] == s and r["included"]}
                     for s in ("train", "val", "test")}
            hashes = {s: {r["content_sha256"] for r in rows if r["split"] == s and r["included"]}
                      for s in pools}
            data_ok &= all(pools[s] and len(pools[s]) == data["split_counts"][s] for s in pools)
            data_ok &= not any(pools[a] & pools[b] or hashes[a] & hashes[b]
                               for a, b in (("train", "val"), ("train", "test"), ("val", "test")))
            previous = set()
            for _, subset in sorted(data["train_fractions"].items(), key=lambda p: float(p[0])):
                ids = json.loads(Path(subset["path"]).read_text(encoding="utf-8"))
                selected = set(ids)
                data_ok &= file_hash(subset["path"]) == subset["sha256"]
                data_ok &= len(ids) == len(selected) == subset["count"] and previous <= selected <= pools["train"]
                previous = selected
        checks["manifests_subsets_disjoint_nested"] = bool(data_ok)
        jobs = spec["jobs"]
        checks["job_count_unique"] = len(jobs) == spec["run_budget"]["total"] == len({j["job_id"] for j in jobs})
        checks["validation_test_isolation"] = all(not j["evaluate_test"] for j in jobs if j["stage"] == "validation_tuning")
        checks["equal_tuning_budget"] = all(len(config["lr_candidates"]) == 2 for config in spec["methods"].values())
        checks["explicit_arms"] = bool(spec["methods"]) and all(j["arm_id"] in spec["methods"] for j in jobs)
    except (KeyError, TypeError, ValueError, OSError) as error:
        details.append(f"{type(error).__name__}: {error}")
    return {"passed": bool(checks) and all(checks.values()) and not details, "checks": checks, "errors": details}


def case_gate(spec: dict | None, expected: dict) -> dict:
    """Frozen numeric requirements; semantic requirements are assessed by the judge."""
    if spec is None:
        return {"passed": False, "checks": {"spec_present": False}}
    arms = list(spec["methods"].values())
    factor = expected["factor"]
    checks = {}
    field = {"rank": "rank", "branch_dropout": "dropout", "shallow_prompt_length": "prompt_length",
             "bottleneck_width": "bottleneck_dim", "weight_decay": "weight_decay"}.get(factor)
    if field:
        method = "vpt_shallow" if field == "prompt_length" else "bottleneck_adapter" if field == "bottleneck_dim" else "lora"
        selected = [a for a in arms if a["method"] == method]
        checks["required_variants"] = {a[field] for a in selected} == set(expected["required_variants"])
        if method == "lora":
            controls = {"rank", "dropout", "target_blocks", "weight_decay"} - {field}
            checks["other_factors_fixed"] = all(len({json.dumps(a.get(k), sort_keys=True) for a in selected}) == 1 for k in controls)
            checks["constant_alpha_over_rank"] = len({a["alpha"] / a["rank"] for a in selected}) == 1
    if factor == "target_blocks":
        checks["target_blocks"] = {tuple(a["target_blocks"]) for a in arms if a["method"] == "lora"} == {
            tuple(range(9, 12)), tuple(range(6, 12)), tuple(range(12))}
    for field_name, spec_value in (("required_methods", {a["method"] for a in arms}),
                                   ("required_fractions", set(spec["design"]["fractions"])),
                                   ("required_datasets", {d["dataset_id"] for d in spec["datasets"]}),
                                   ("required_metrics", set(spec["metrics"]["primary"] + spec["metrics"]["secondary"]))):
        if field_name in expected:
            checks[field_name] = set(expected[field_name]) <= spec_value
    for key, value in expected.get("fixed", {}).items():
        if key == "label_fraction":
            checks[key] = spec["design"]["fractions"] == [value]
        elif key not in {"placement", "up_initialization"}:
            checks[key] = all(a.get(key) == value for a in arms if a["method"] == "lora")
    if factor == "selection_policy":
        checks["policy"] = spec["design"]["selection_policy"]["mode"] == "per_dataset_vs_shared_rank"
        checks["rank_pool"] = {a.get("rank") for a in arms} == set(expected["rank_candidates"])
    for index, required in enumerate(expected.get("arm_requirements", [])):
        checks[f"arm_requirement_{index}"] = any(all(arm.get(k) == v for k, v in required.items()) for arm in arms)
    return {"passed": all(checks.values()), "checks": checks}

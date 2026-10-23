"""Deterministic v2 study-to-coding handoff; no additional agent loop."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

from researchops.core.artifacts import canonical_hash, file_hash, write_json
from researchops.coding.plan import API_CONTRACT
from researchops.coding.tasks import compile_tasks

VERSION = "study-coding-adapter-v03-1"


def build_study_coding_plan(workspace, study_result, *, token_cap=600000):
    if not 120000 <= token_cap <= 600000:
        raise ValueError("Coding cap must be between original 120000 and 5x=600000")
    result_path = Path(study_result).resolve()
    result = json.loads(result_path.read_text(encoding="utf-8"))
    directory = result_path.parent
    commit_path = directory / "commit.json"
    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    if result.get("status") != "ready_for_coding" or not result.get("success"):
        raise ValueError("Only an accepted study may enter coding")
    required = {"proposal.json", "review.json", "confirmation.json", "evidence.json", "workflow_result.json", "frozen_spec.json"}
    if not required <= set(commit.get("artifacts", {})):
        raise ValueError("Study has no complete formal commit")
    # The graph owns final commit hashes, reviews and the frozen research design.
    for name, digest in commit["artifacts"].items():
        if Path(name).name != name:
            raise ValueError("Invalid study commit path")
        if file_hash(directory / name) != digest:
            raise ValueError("Study commit artifact changed")
    workflow = json.loads((directory / "workflow_result.json").read_text(encoding="utf-8"))
    if not workflow.get("success") or not workflow.get("mechanical", {}).get("passed"):
        raise ValueError("Committed study did not pass final gates")
    for name in ("review.json", "confirmation.json"):
        review = json.loads((directory / name).read_text(encoding="utf-8"))
        if not review.get("criteria") or any(v != "pass" for v in review["criteria"].values()):
            raise ValueError("Study independent review did not pass")
    spec_path = directory / "frozen_spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("schema_version") != "researchops-experiment-contract-v2":
        raise ValueError("Expected v2 frozen study contract")
    sources = [{"path": str(spec_path), "sha256": file_hash(spec_path)},
               {"path": str(commit_path), "sha256": file_hash(commit_path)}]
    sources.extend(spec["sources"].values())
    sources.extend({"path": str(directory / name), "sha256": digest}
                   for name, digest in commit["artifacts"].items() if name != "frozen_spec.json")
    for item in sources:
        if file_hash(item["path"]) != item["sha256"]:
            raise ValueError("Study source fingerprint changed")
    api = copy.deepcopy(API_CONTRACT)
    api["models.py"] = {"build_model(spec, arm_id, num_classes, *, smoke=False)":
        "v2: spec['methods'] maps ARM IDs to configs. Read method=spec['methods'][arm_id]['method']; expose "
        "net.adaptation_method=method and net.arm_id=arm_id. Respect per-arm rank/alpha/dropout/target_blocks, "
        "prompt_length/bottleneck_dim. Full mode uses exact timm pretrained backbone. Smoke uses timm "
        "VisionTransformer(img_size=32,patch_size=8,embed_dim=48,depth=2,num_heads=3,mlp_ratio=2), no download; "
        "map last-three-block target to both tiny blocks for smoke only. LoRA Q/V independent A/B, B=0; K frozen. "
        "Only head and method adaptation parameters trainable except full_finetune; same-seed unchanged base initialization."}
    api["train.py"] = {"run_job(spec, job, output_dir, *, smoke=False, max_steps=None, resume=False)":
        next(iter(API_CONTRACT["train.py"].values())) +
        " V2 overrides: use job.arm_id to build_model; job.method is its family. Fraction comes from job.fraction, NOT hardcoded 0.1. "
        "For main operation=evaluate_selected_checkpoint_no_retraining, load job.selected_checkpoint (a checkpoint.pt path) "
        "and evaluate ONLY; never construct optimizer or take a training step. Validate selected checkpoint spec_hash. "
        "In smoke never access test even for main. Record arm_id. Export ManifestDataset/build_model at module scope."}
    api["run_experiment.py"] = {"CLI":
        "--spec PATH --output PATH [--smoke] [--resume] [--max-jobs N]. Read v2 jobs in order. "
        "Group LR selection by (dataset_id,arm_id,fraction,seed), choose highest validation accuracy then smaller LR. "
        "Tuning invokes run_job and keeps its return dict; main MUST pass numeric selected LR and selected_checkpoint="
        "str(tuning_output_dir/'checkpoint.pt') into run_job with the ORIGINAL main job_id. "
        "Never retrain main, never mix distinct rank arms, never mutate input spec. Smoke max_steps=2. "
        "Resume validates completed artifacts/config before skipping; do not trust result.json alone. "
        "Write summary.json; mark research_success null for smoke/incomplete. Retain metrics.analysis_plan and success_rule "
        "in summary as pending analysis; do not invent scientific conclusions from incomplete runs."}
    tasks = compile_tasks(api)
    for task in tasks:
        task["max_total_tokens"] *= 4
        if task["id"] == "training":
            task["max_output_tokens"] = 5000
    key = canonical_hash([VERSION, sources, api, tasks, token_cap])
    plan = {"schema_version": VERSION, "plan_id": "study_coding_" + key[:20],
        "idea_id": key[:20], "objective": spec["idea"].get("research_question", "Implement accepted study"),
        "frozen_spec": sources[0], "source_artifacts": sources,
        "api_contract": api, "allowed_outputs": list(api), "template": None,
        "implementation": {"tasks": tasks, "evidence": spec.get("baseline_citations", [])},
        "execution": {"strategy": "task_dag_v1", "tasks": tasks, "max_integration_repairs": 3},
        "budget": {"max_model_turns": 24, "max_tool_calls": 90, "max_verification_rounds": 6,
                   "max_total_tokens": token_cap, "max_wall_seconds": 7200, "command_timeout_seconds": 240,
                   "provider_token_margin": 32768},
        "finish_condition": "Unchanged source passes independent Docker verifier; engineering smoke only.",
        "parent_study": {"result": str(result_path), "commit": str(commit_path),
                         "thread_id": result.get("thread_id"), "artifacts": str(directory)},
        "synthesis": {"origin": "Accepted v2 study evidence; no repeated paid planning"}}
    # Never derive a broad drive/workspace mount from unrelated dataset paths.
    # The established dataset layout binds every manifest to one datasets tree.
    roots = set()
    for data in spec["datasets"]:
        manifest = Path(data["manifest_path"]).resolve()
        bound = next((p for p in manifest.parents if p.name == "datasets"), None)
        if bound is None or not Path(data["root"]).resolve().is_relative_to(bound):
            raise ValueError("Study dataset must remain inside its bound datasets directory")
        for subset in data.get("train_fractions", {}).values():
            if not Path(subset["path"]).resolve().is_relative_to(bound):
                raise ValueError("Study subset escapes the datasets directory")
        roots.add(bound)
    if len(roots) != 1:
        raise ValueError("Study datasets must share one datasets directory; broad mounts are forbidden")
    boundary = roots.pop()
    projected_root = Path(os.path.commonpath([str(Path(d[k]).resolve())
        for d in spec["datasets"] for k in ("root", "manifest_path")]))
    if not projected_root.is_relative_to(boundary):
        raise ValueError("Broad dataset mounts are forbidden")
    plan["dataset_root"] = str(projected_root)
    path = Path(workspace) / "artifacts" / "coding" / (plan["plan_id"] + ".json")
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != plan:
            raise ValueError("Existing study coding plan changed; use a new adapter version/workspace")
    else:
        write_json(path, plan)
    return {**plan, "plan_path": str(path), "cache_hit": False}

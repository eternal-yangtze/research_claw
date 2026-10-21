"""Real scratch coding experiment; uses .env, never reads template source.

Each tag creates isolated evidence. The frozen spec, data, old code and reports
are preserved. A failed run is never silently retried with a reset budget.
"""
from __future__ import annotations

import argparse
import json
import re
import copy
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from researchops.application.workflow import ResearchOpsApp
from researchops.coding.agent import CodingApplication
from researchops.core.artifacts import canonical_hash, file_hash, write_json, workspace_lock
from researchops.core.config import load_project_env
from researchops.harness.runtime import OpenAICompatibleModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default="workspace_batch_vit_v2")
    parser.add_argument("--idea-id", default="vit_peft_small_data")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-from", help="One budget-preserving task replan of an existing benchmark tag")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_]{1,48}", args.tag):
        raise ValueError("Invalid benchmark tag")
    load_project_env()
    app = ResearchOpsApp(args.workspace)
    model = OpenAICompatibleModel()
    target = app.workspace / "artifacts" / "coding" / ("benchmark_" + args.tag + ".json")
    if args.continue_from:
        plan = stage_continuation(app.workspace, args.continue_from, args.tag)
        args.resume = True
    elif args.resume and target.is_file():
        plan = json.loads(target.read_text(encoding="utf-8"))
    else:
        plan = app.create_coding_plan(args.idea_id, model=model, from_scratch=True)
        plan = {k: v for k, v in plan.items() if k not in {"cache_hit", "plan_path"}}
        plan["plan_id"] += "_" + args.tag
        plan["benchmark"] = {"tag": args.tag, "task": "six-file ViT PEFT from empty project; five methods; real DTD/Pets smoke"}
    if target.is_file() and json.loads(target.read_text(encoding="utf-8")) != plan:
        raise ValueError("Benchmark tag already belongs to a different immutable plan")
    write_json(target, plan)
    report = CodingApplication(workspace=app.workspace, model=model).run(plan, resume=args.resume)
    print(json.dumps({k: report[k] for k in ("status", "stop_reason", "total_tokens", "token_limit", "within_budget", "turns",
        "wall_seconds", "stage_verifications", "verification_rounds", "task_completion_rate", "first_attempt_pass_rate",
        "max_context_bytes", "implementation_origin", "report_path")}, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "coding_verified" else 1


def stage_continuation(workspace, parent_tag, tag):
    """Revise one failed task's attempts while retaining ALL cumulative budgets.

    Parent evidence is immutable. Child inherits generated source and accounting;
    this must never be reported as an independent clean run.
    """
    if not re.fullmatch(r"[a-zA-Z0-9_]{1,48}", parent_tag) or parent_tag == tag:
        raise ValueError("Invalid parent tag")
    artifact_dir = workspace / "artifacts" / "coding"
    existing = artifact_dir / ("benchmark_" + tag + ".json")
    if existing.is_file():
        value = json.loads(existing.read_text(encoding="utf-8"))
        if value.get("continuation_of", {}).get("tag") != parent_tag:
            raise ValueError("Existing tag has another parent")
        return value
    parent = json.loads((artifact_dir / ("benchmark_" + parent_tag + ".json")).read_text(encoding="utf-8"))
    if parent.get("continuation_of"):
        raise ValueError("Only one budget-preserving replan is allowed")
    parent_dir = workspace / "coding" / parent["plan_id"]
    with workspace_lock(parent_dir):
        state = json.loads((parent_dir / "checkpoint.json").read_text(encoding="utf-8"))
        if state["status"] != "needs_attention":
            raise ValueError("Only a stopped failed run can be replanned")
        failed = [t for t in parent["execution"]["tasks"] if state["tasks"][t["id"]]["status"] != "completed"
                  and (parent_dir / "project" / t["outputs"][0]).is_file()]
        if not failed:
            raise ValueError("No failed generated module to replan")
        plan = copy.deepcopy(parent)
        target_task = failed[0]["id"]
        for task in plan["execution"]["tasks"]:
            if task["id"] == target_task:
                task["max_attempts"] += 2
                task["max_total_tokens"] = min(plan["budget"]["max_total_tokens"], task["max_total_tokens"] + 30000)
                state["tasks"][target_task]["limit"] = task["max_total_tokens"]
        plan["continuation_of"] = {"tag": parent_tag, "plan_id": parent["plan_id"],
            "checkpoint_sha256": file_hash(parent_dir / "checkpoint.json"),
            "inherited_tokens": state["total_tokens"], "inherited_turns": state["turns"],
            "reason": "Task contract/diagnostic replan after bounded failure; no budget reset",
            "replanned_task": target_task, "additional_attempts": 2}
        plan["benchmark"]["tag"] = tag
        plan["plan_id"] = "coding_" + canonical_hash(plan)[:20] + "_" + tag
        target = workspace / "coding" / plan["plan_id"]
        if target.exists():
            raise ValueError("Continuation directory already exists without its immutable plan")
        (target / "project").mkdir(parents=True)
        for name in plan["allowed_outputs"]:
            source = parent_dir / "project" / name
            if source.is_file():
                destination = target / "project" / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
        shutil.copytree(parent_dir / "calls", target / "calls")
        shutil.copy2(parent_dir / "task_trace.jsonl", target / "task_trace.jsonl")
        state.update(plan_hash=canonical_hash(plan), status="in_progress")
        write_json(target / "checkpoint.json", state)
        write_json(existing, plan)
        return plan


if __name__ == "__main__":
    raise SystemExit(main())

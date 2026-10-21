"""Four input changes, evaluated separately from positive-task generality."""
import copy
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from researchops.core.config import load_project_env
from researchops.core.artifacts import write_json, workspace_lock
from researchops.evaluation.precoding.robust import RobustPrecodingRunner
from researchops.evaluation.precoding.runner import read_json

load_project_env()
os.environ["RESEARCHOPS_EMBEDDING_LOCAL_ONLY"] = "true"
os.environ["RESEARCHOPS_RERANKER_LOCAL_ONLY"] = "true"
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--output", type=Path, default=Path("workspace_precoding_v4_contrastive"))
args = parser.parse_args()
root = args.output
runner = RobustPrecodingRunner(source=Path("workspace_batch_vit_v2"), output=root,
                               suite_path=Path("config/benchmarks/precoding_v1.json"), cap=700000)
changes = [
    ("vit_rank_tradeoff", "Replace all previous rank requirements with ranks 4 and 8 ONLY. Do not include rank2 or rank16.", "rank", {4, 8}),
    ("vit_vpt_prompt_length", "Replace all previous prompt-length requirements with lengths5 and10 ONLY. Do not include length20.", "prompt_length", {5, 10}),
    ("vit_low_label_lora_vs_full", "Replace the previous 10% label requirement with 25% labels ONLY, including the primary comparison and interpretation.", "fraction", {0.25}),
    ("vit_calibration_comparison", "Replace earlier primary endpoints: NLL is the SOLE primary endpoint; accuracy, macro F1 and ECE remain secondary.", "primary", {"nll"}),
]
with workspace_lock(root):
    rows = []
    for idea_id, instruction, field, values in changes:
        idea = copy.deepcopy(runner.ideas[idea_id])
        idea["updated_user_instruction"] = instruction
        result = runner.plan_task("C_" + idea_id, idea)
        path = root / "tasks" / ("C_" + idea_id) / "frozen_spec.json"
        observed, passed = [], False
        if path.exists():
            spec = read_json(path)
            if field == "fraction":
                observed = spec["design"]["fractions"]
            elif field == "primary":
                observed = spec["metrics"]["primary"]
            else:
                observed = [a[field] for a in spec["methods"].values() if field in a]
            passed = set(observed) == values
        rows.append({"idea_id": idea_id, "instruction": instruction, "observed": observed,
                     "required_change_observed": passed, "success": passed and result.get("success", False)})
    report = {"records": rows, "passed": sum(r["success"] for r in rows), "total": len(rows),
              "performance": "Responsiveness only, not four new independent research questions",
              "charged_tokens": runner.calls.state["total_tokens"]}
    write_json(root / "contrastive.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))

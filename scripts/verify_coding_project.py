"""Repeat trusted coding evaluation without paying for another LLM run."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from researchops.coding_sandbox import CodingSandbox, stage_inputs
from researchops.datasets import write_json
from researchops.vendor.taleclaw.metrics import summarize_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--plan", required=True)
    args = parser.parse_args()
    workspace = Path(args.workspace).resolve()
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    run = workspace / "coding" / plan["plan_id"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = run / "rechecks" / stamp
    stage_inputs(workspace, output, plan)
    verification = CodingSandbox(workspace, run / "project", output).verify(1)
    report = {"scope": "independent engineering recheck; no LLM calls or research claim",
        "verification": verification,
        "scorecard": summarize_rows([{"passed": verification["passed"], "verifier_passed": verification["passed"]}])}
    write_json(output / "report.json", report)
    print(json.dumps({"passed": verification["passed"], "measurements": verification.get("measurements"),
        "report_path": str(output / "report.json")}, ensure_ascii=False, indent=2))
    return 0 if verification["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

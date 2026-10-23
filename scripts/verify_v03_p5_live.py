"""Run accepted P4 study through the existing coding DAG and real Docker gates."""
import argparse
import json
from pathlib import Path

from researchops.core.config import load_project_env
from researchops.core.artifacts import write_json, progress
from researchops.coding.study_adapter import build_study_coding_plan
from researchops.coding.agent import CodingApplication
from researchops.harness.runtime import OpenAICompatibleModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prior-run", type=Path, action="append", default=[], help="Retain failed run spend; repeat to subtract all prior charged usage")
    parser.add_argument("--stream", action="store_true", help="Use SSE to avoid gateway idle response timeout")
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high"))
    args = parser.parse_args()
    load_project_env()
    root = args.output.resolve()
    prior = None
    prior_charge = 0
    for previous in args.prior_run:
        prior = json.loads((previous / "report.json").read_text(encoding="utf-8"))
        prior_charge += prior["result"]["harness"]["budget"]["charged_tokens"]
    plan = build_study_coding_plan(root, args.study_result, token_cap=600000-prior_charge)
    model = OpenAICompatibleModel(max_attempts=1, stream=args.stream, reasoning_effort=args.reasoning_effort)
    options = {"workspace": root, "model": model, "parent_run_id": "p5-study-coding",
               "parent_session_id": "p4-accepted-study"}
    run_dir = root / "coding" / plan["plan_id"]
    first = CodingApplication(**options).run(plan, resume=(run_dir / "checkpoint.json").exists())
    progress(json.dumps({"status": first["status"], "turns": first["turns"], "stop_reason": first["stop_reason"]}))
    if first["status"] != "coding_verified":
        write_json(root / "report.json", {"passed": False, "result": first})
        return 1
    second = CodingApplication(**options).run(plan, resume=True)
    a, b = first["harness"]["budget"], second["harness"]["budget"]
    checks = {"docker_verified": first["verification"]["passed"],
        "resume_cached": second["cache_hit"], "resume_zero_calls": a["model_calls"] == b["model_calls"],
        "resume_zero_charge": a["charged_tokens"] == b["charged_tokens"],
        "within_budget": first["within_budget"], "no_research_claim": first["research_success"] is None,
        "real_provider": a["known_actual_tokens"] > 0, "trace_parent_bound": bool(first["harness"]["parent_run_id"])}
    report = {"passed": all(checks.values()), "checks": checks, "result": first, "resume": second,
              "prior_runs": [str(p) for p in args.prior_run],
              "provider_options": {"stream": args.stream, "reasoning_effort": args.reasoning_effort},
              "aggregate_charged_tokens": prior_charge+b["charged_tokens"], "aggregate_token_cap": 600000}
    write_json(root / "report.json", report)
    progress(json.dumps({"passed": report["passed"], "checks": checks, "budget": b}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

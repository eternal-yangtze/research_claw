"""Offline audit of real skill traces, invocation artifacts and outer ledgers."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from researchops.core.artifacts import file_hash, write_json


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    roots = [Path("workspace_skills_v1"), Path("workspace_skills_composition_v1"), Path("workspace_skills_study_v2")]
    checked, failures = 0, []
    for root in roots:
        for artifact in root.glob("**/artifacts/skills/*/*.json"):
            record = read(artifact)
            run_id = artifact.parent.name
            workspace = artifact.parents[3]
            trace = workspace / "runs" / run_id / "trace.jsonl"
            loaded, invoked = {}, False
            for line in trace.read_text(encoding="utf-8").splitlines():
                event = json.loads(line)
                payload = event["payload"]
                if event["event"] == "skill.loaded":
                    loaded[payload["name"]] = payload["sha256"]
                if (event["event"] == "skill.invocation.started" and
                        payload["invocation_id"] == artifact.stem):
                    invoked = loaded.get(record["name"]) == record["sha256"]
            checked += 1
            if not invoked or record["status"] != record["result"]["status"]:
                failures.append(str(artifact))
    calls = actual = charged = runs = 0
    failed_attempts = []
    for root in roots:
        for path in root.glob("**/skill_runs/*/ledger.json"):
            ledger = read(path)
            runs += 1
            calls += ledger["turns"]
            charged += ledger["total_tokens"]
            actual += sum(c.get("actual_tokens", 0) for c in ledger["call_ledger"])
            if (path.parent / "failure.json").exists():
                failed_attempts.append({"run_id": path.parent.name,
                    "error": read(path.parent / "failure.json")["error"], "charged_tokens": ledger["total_tokens"]})
    base = read(roots[0] / "report.json")
    context = []
    # Do not count successful tool invocation as a completed model workflow.
    for row in base["context_comparison"]:
        path = roots[0] / "context" / row["mode"] / str(row["repeat"]) / "benchmark_results" / "comparison.json"
        completed = path.is_file() and read(path)["run_id"] == row["run_id"]
        ledger = read(path.parents[1] / "skill_runs" / row["run_id"] / "ledger.json")
        context.append({**row, "workflow_completed": completed,
                        "first_prompt_tokens": ledger["call_ledger"][0].get("usage", {}).get("prompt_tokens"),
                        "actual_tokens": sum(c.get("actual_tokens", 0) for c in ledger["call_ledger"]),
                        "charged_tokens": ledger["total_tokens"], "model_calls": ledger["turns"],
                        "pass": completed and row["pass"]})
    composition = read(roots[1] / "report.json")
    repaired_study = read(roots[2] / "report.json")
    context_costs = {}
    for mode in ("lazy", "eager"):
        mode_ledgers = [read(p) for p in (roots[0] / "context" / mode).glob("**/skill_runs/*/ledger.json")]
        known = sum(c.get("actual_tokens", 0) for l in mode_ledgers for c in l["call_ledger"])
        total = sum(l["total_tokens"] for l in mode_ledgers)
        context_costs[mode] = {"attempts": len(mode_ledgers), "known_actual_tokens": known,
                               "unknown_reserved_tokens": total - known, "charged_tokens": total}
    result = {"skill_invocation_artifacts_checked": checked,
              "load_before_invoke_and_hash_pass": not failures, "artifact_failures": failures,
              "routing": {k: base["routing"][k] for k in ("correct", "total", "negative_false_triggers", "negative_total")},
              "execution": base["execution"], "context_comparison": context,
              "context_cost_including_failed_attempts": context_costs,
              "composition": composition,
              "study_after_role_repair": repaired_study,
              "cost_including_all_retained_attempts": {"runs": runs, "model_dispatches": calls,
                  "known_actual_tokens": actual, "unknown_reserved_tokens": charged - actual, "charged_tokens": charged},
              "failed_attempts": failed_attempts,
              "final_code_sha256": {str(p): file_hash(p) for p in [
                  Path("src/researchops/application/workflow.py"), Path("src/researchops/harness/runtime.py"),
                  Path("src/researchops/harness/budget.py"), Path("src/researchops/coding/budget.py"),
                  Path("src/researchops/evaluation/precoding/runner.py"),
                  Path("src/researchops/experiments/study_workflow.py"),
                  Path("scripts/benchmark_skills.py"), Path("scripts/verify_skill_composition.py"),
                  *Path("src/researchops/harness/skills").glob("*.py")]},
              "limitations": ["Development set, not external held-out evaluation",
                  "Cached successful cases retain their original implementation; source snapshot and provider reservation repairs were tested separately",
                  "Retrieval verified is locator/excerpt validation, not semantic gold accuracy"]}
    write_json(roots[0] / "audit.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Low-cost grouped boundary assessment, with separate generation and review calls."""
from __future__ import annotations

import time
from pathlib import Path

from researchops.core.artifacts import write_json, workspace_lock
from researchops.evaluation.precoding.gates import RUBRIC, silver_pass
from researchops.evaluation.precoding.runner import PaidCalls, read_json
from researchops.experiments.planning import PLAN_SYSTEM, HOST_PROTOCOL
from researchops.harness.runtime import OpenAICompatibleModel


def run_boundary_batch(root: Path, suite_path: Path, cap: int):
    with workspace_lock(root):
        suite = read_json(suite_path)
        cases = suite["boundary_cases"]
        calls = PaidCalls(root, OpenAICompatibleModel(max_attempts=1), cap)
        results = read_json(root / "results.json")
        if all(c["case_id"] in results for c in cases):
            return {c["case_id"]: results[c["case_id"]]["status"] for c in cases}
        payload = {"tasks": {c["case_id"]: c["prompt"] for c in cases}, "host_protocol": HOST_PROTOCOL}
        before, started = calls.state["total_tokens"], time.monotonic()
        directory = root / "boundary_batch"
        write_json(directory / "input.json", payload)
        error = None
        proposals, reviews = {}, {}
        try:
            proposals = calls.call("boundary_batch_planner", PLAN_SYSTEM +
                '\nProcess each supplied task separately. Return {"cases":{"task_id": proposal_object,...}}. Keep each reason and alternative concise.', payload, 1800)
            write_json(directory / "proposals.json", proposals)
            reviews = calls.call("boundary_batch_independent_reviewer", RUBRIC +
                '\nFor this grouped review, return {"cases":{"task_id": review_object,...}}. Assess every task separately, using all seven criteria per task.',
                {"tasks": payload["tasks"], "proposals": proposals}, 1800)
            write_json(directory / "reviews.json", reviews)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        cost = calls.state["total_tokens"] - before
        for case in cases:
            key = case["case_id"]
            proposal = proposals.get("cases", {}).get(key, {})
            judged = reviews.get("cases", {}).get(key, {})
            safe = proposal.get("status") in {"blocked", "needs_adapter", "needs_refinement"}
            accepted = silver_pass(judged)
            record = {"key": key, "idea_id": key, "boundary": True, "mode": "grouped_boundary",
                      "status": "passed" if safe and accepted else "budget_exhausted" if error and "budget" in error.lower() else "failed",
                      "success": safe and accepted, "boundary_handled": safe and accepted,
                      "silver_accepted": accepted, "proposal_status": proposal.get("status"),
                      "charged_tokens": cost / len(cases), "wall_seconds": (time.monotonic() - started) / len(cases),
                      "batch_context_shared": True, "batch_directory": str(directory), "error": error,
                      "cost_allocation": "Equal share of two-call batch; not separately billed per task"}
            results[key] = record
            write_json(root / "tasks" / key / "proposal.json", proposal)
            write_json(root / "tasks" / key / "review.json", judged)
            write_json(root / "tasks" / key / "result.json", record)
        write_json(root / "results.json", results)
        return {c["case_id"]: results[c["case_id"]]["status"] for c in cases}

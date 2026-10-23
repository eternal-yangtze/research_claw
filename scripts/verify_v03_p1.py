"""Run a deterministic end-to-end verification of the v0.3 P1 harness."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from researchops.harness.budget import BudgetedModel
from researchops.harness.micro import MicroAgentRunner, MicroAgentSpec, ValidationIssue
from researchops.harness.runtime import (
    AgentSpec,
    ModelTurn,
    Session,
    ToolCall,
    ToolExecutor,
    ToolPolicy,
    ToolRegistry,
    ToolSpec,
    TraceStore,
)


class ScriptedVerificationModel:
    """Exercise tool use, invalid output, and repair without a provider call."""

    model = "v03-p1-deterministic-verifier"

    def __init__(self) -> None:
        self.calls = 0
        self.observed_timeouts: list[float] = []

    def complete_bounded(self, messages, tools, max_tokens, *, timeout):
        self.calls += 1
        self.observed_timeouts.append(timeout)
        if self.calls == 1:
            assert [tool["function"]["name"] for tool in tools] == ["get_evidence"]
            return ModelTurn(
                tool_calls=(ToolCall("evidence-call-1", "get_evidence", {"chunk_id": "chunk-42"}),),
                usage={"total_tokens": 11},
            )
        if self.calls == 2:
            tool_messages = [message for message in messages if message["role"] == "tool"]
            assert len(tool_messages) == 1
            tool_result = json.loads(tool_messages[0]["content"])
            assert tool_result["status"] == "success"
            return ModelTurn(text="not valid json", usage={"total_tokens": 7})
        assert self.calls == 3
        feedback = json.loads(messages[-1]["content"])
        assert feedback["type"] == "validation_feedback"
        assert feedback["issues"][0]["code"] == "invalid_output"
        return ModelTurn(
            text=json.dumps({"answer": "grounded", "evidence": ["chunk-42"]}),
            usage={"total_tokens": 9},
        )


def validate_candidate(candidate):
    issues = []
    if not isinstance(candidate, dict):
        return [ValidationIssue("invalid_candidate", "candidate must be an object")]
    if not candidate.get("answer"):
        issues.append(ValidationIssue("missing_answer", "answer is required", "answer"))
    if candidate.get("evidence") != ["chunk-42"]:
        issues.append(ValidationIssue("missing_evidence", "expected evidence is required", "evidence"))
    return issues


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".test_runs/v03_p1_verification"),
        help="Directory in which to create the verification run",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_root = args.output.resolve() / run_stamp
    traces = TraceStore(run_root / "traces")
    registry = ToolRegistry()
    observed_context: dict[str, object] = {}

    def get_evidence(chunk_id: str, _context):
        observed_context.update({
            "chunk_id": chunk_id,
            "cancelled": _context.cancelled(),
            "remaining_seconds_positive": (_context.remaining_seconds() or 0) > 0,
        })
        return {"chunk_id": chunk_id, "text": "deterministic evidence"}

    registry.register(ToolSpec(
        "get_evidence",
        "Read one bounded evidence chunk.",
        {
            "type": "object",
            "properties": {"chunk_id": {"type": "string"}},
            "required": ["chunk_id"],
            "additionalProperties": False,
        },
        get_evidence,
        read_only=True,
        idempotent=True,
        contextual=True,
        timeout_seconds=2,
    ))
    executor = ToolExecutor(registry, ToolPolicy(registry), traces)
    scripted = ScriptedVerificationModel()
    budget_root = run_root / "budget"
    budgeted = BudgetedModel(scripted, budget_root, 250_000, max_model_turns=8)
    runner = MicroAgentRunner(
        budgeted,
        executor,
        registry,
        traces,
        run_root / "checkpoints",
    )
    spec = MicroAgentSpec(
        agent=AgentSpec(
            name="P1VerificationAgent",
            role="Use evidence and return a JSON object.",
            allowed_tools=("get_evidence",),
            max_reasoning_steps=4,
            max_tool_calls=2,
            max_tokens=128,
        ),
        prompt_version="v03-p1-verification-v1",
        validator=validate_candidate,
        max_no_progress=2,
        max_total_tokens=100_000,
        timeout_seconds=10,
    )
    session = Session("p1-verification-session", "p1-verification-task")
    result = runner.run(
        session,
        spec,
        "Find the fixed evidence and produce a grounded answer.",
        run_id="p1-verification",
        node_run_id="micro_verify",
    )
    before_resume = budgeted.summary()

    resumed_budget = BudgetedModel(
        ScriptedVerificationModel(),
        budget_root,
        250_000,
        max_model_turns=8,
        resume=True,
    )
    resumed_runner = MicroAgentRunner(
        resumed_budget,
        executor,
        registry,
        traces,
        run_root / "checkpoints",
    )
    resumed = resumed_runner.run(
        Session("p1-verification-session", "p1-verification-task"),
        spec,
        "Find the fixed evidence and produce a grounded answer.",
        run_id="p1-verification",
        node_run_id="micro_verify",
        resume=True,
    )
    after_resume = resumed_budget.summary()

    trace_path = run_root / "traces" / "p1-verification" / "trace.jsonl"
    events = [
        json.loads(line)["event"]
        for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]
    ledger = json.loads((budget_root / "ledger.json").read_text(encoding="utf-8"))
    call_ids = [entry["call_id"] for entry in ledger["call_ledger"]]
    checks = {
        "completed_after_repair": result.status == "completed" and result.stop_reason == "validated",
        "candidate_validated": result.candidate == {"answer": "grounded", "evidence": ["chunk-42"]},
        "expected_model_turns": result.model_turns == 3,
        "expected_tool_calls": result.tool_calls == 1,
        "deadline_reached_tool": observed_context == {
            "chunk_id": "chunk-42",
            "cancelled": False,
            "remaining_seconds_positive": True,
        },
        "trace_has_repair_cycle": all(event in events for event in (
            "micro.started",
            "tool.call.started",
            "tool.call.completed",
            "micro.validation_failed",
            "micro.revised",
            "micro.completed",
        )),
        "provider_call_ids_unique": len(call_ids) == len(set(call_ids)) == 3,
        "known_usage_settled": before_resume["charged_tokens"] == 27
        and before_resume["unknown_reserved_tokens"] == 0,
        "node_scope_shares_root_charges": before_resume["scopes"]["micro_verify"]["charged_tokens"]
        == before_resume["charged_tokens"],
        "budget_resume_did_not_recharge": after_resume["charged_tokens"] == before_resume["charged_tokens"]
        and after_resume["model_calls"] == before_resume["model_calls"],
        "checkpoint_resume_is_terminal": resumed.to_dict() == result.to_dict(),
    }
    report = {
        "schema_version": "researchops-v03-p1-verification-v1",
        "passed": all(checks.values()),
        "paid_model_calls": 0,
        "result": result.to_dict(),
        "runtime": {
            "scripted_provider_calls": scripted.calls,
            "observed_timeouts_seconds": scripted.observed_timeouts,
            "trace_events": events,
            "tool_context": observed_context,
        },
        "budget_before_resume": before_resume,
        "budget_after_resume": after_resume,
        "provider_call_ids": call_ids,
        "checks": checks,
        "artifacts": {
            "run_root": str(run_root),
            "checkpoint": result.checkpoint_path,
            "ledger": str(budget_root / "ledger.json"),
            "trace": str(trace_path),
        },
    }
    report_path = run_root / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**report, "report_path": str(report_path)}, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

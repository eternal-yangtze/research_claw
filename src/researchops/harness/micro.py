"""Bounded local-agent loop with validation feedback and durable recovery."""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from researchops.core.artifacts import canonical_hash, write_json
from researchops.core.models import ToolResult
from researchops.coding.budget import BudgetAdmissionError
from researchops.harness.runtime import (
    AgentSpec, Session, ToolCall, ToolExecutor, ToolRegistry, TraceStore,
)


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    path: str = ""
    retryable: bool = True


@dataclass(frozen=True)
class MicroAgentSpec:
    agent: AgentSpec
    prompt_version: str
    validator: Callable[[Any], list[ValidationIssue]]
    parser: Callable[[str], Any] = json.loads
    max_no_progress: int = 2
    max_total_tokens: int = 250_000
    timeout_seconds: float = 240.0

    def __post_init__(self):
        if not self.prompt_version or self.max_no_progress < 1:
            raise ValueError("prompt_version and positive max_no_progress required")
        if self.max_total_tokens < 1 or self.timeout_seconds <= 0:
            raise ValueError("positive token and timeout limits required")


@dataclass(frozen=True)
class MicroAgentResult:
    run_id: str
    node_run_id: str
    status: str
    stop_reason: str
    model_turns: int
    tool_calls: int
    candidate: Any = None
    issues: tuple[ValidationIssue, ...] = ()
    checkpoint_path: str = ""

    def to_dict(self):
        return asdict(self)


class MicroAgentRunner:
    def __init__(self, model, executor: ToolExecutor, registry: ToolRegistry,
                 traces: TraceStore, checkpoint_root: str | Path):
        self.model, self.executor, self.registry = model, executor, registry
        self.traces = traces
        self.checkpoint_root = Path(checkpoint_root).resolve()

    def run(self, session: Session, spec: MicroAgentSpec, user_message: str, *,
            run_id: str, node_run_id: str, resume: bool = False,
            cancellation=None) -> MicroAgentResult:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", node_run_id):
            raise ValueError("Invalid node_run_id")
        node_dir = self.checkpoint_root / node_run_id
        checkpoint = node_dir / "state.json"
        spec_hash = canonical_hash({
            "agent": asdict(spec.agent), "prompt_version": spec.prompt_version,
            "max_no_progress": spec.max_no_progress,
            "max_total_tokens": spec.max_total_tokens,
            "timeout_seconds": spec.timeout_seconds,
        })
        input_hash = canonical_hash([session.task_id, user_message, spec_hash])
        if resume:
            if not checkpoint.is_file():
                raise ValueError("MicroAgent checkpoint does not exist")
            state = json.loads(checkpoint.read_text(encoding="utf-8"))
            if state["input_hash"] != input_hash or state["spec_hash"] != spec_hash:
                raise ValueError("MicroAgent checkpoint input or spec changed")
            if state.get("terminal"):
                return _result_from_state(run_id, node_run_id, checkpoint, state)
            session.messages = state["messages"]
            pending_calls = state.get("pending_tool_calls")
            if pending_calls is None and state.get("pending_tool_call"):
                pending_calls = [state["pending_tool_call"]]
            if pending_calls:
                for pending in pending_calls:
                    session.messages.append({
                        "role": "tool", "tool_call_id": pending["call_id"],
                        "content": json.dumps({
                            "status": "interrupted",
                            "instruction": "Inspect current state before deciding whether to retry.",
                        }),
                    })
                state["pending_tool_calls"] = []
                state.pop("pending_tool_call", None)
        else:
            if checkpoint.exists():
                raise ValueError("Use resume=True for an existing MicroAgent checkpoint")
            system = "\n\n".join(part.strip() for part in
                                   (spec.agent.role, spec.agent.instructions) if part.strip())
            session.messages = ([{"role": "system", "content": system}] if system else [])
            session.messages.append({"role": "user", "content": user_message})
            state = {
                "schema_version": "researchops-micro-state-v1",
                "input_hash": input_hash,
                "spec_hash": spec_hash,
                "messages": session.messages,
                "model_turns": 0,
                "tool_calls": 0,
                "elapsed_seconds": 0.0,
                "last_invalid_signature": None,
                "no_progress": 0,
                "candidate": None,
                "issues": [],
                "pending_tool_calls": [],
                "terminal": False,
            }
        model = self.model
        scope = getattr(model, "scope", None)
        if scope is not None:
            model = scope(node_run_id, max_model_turns=spec.agent.max_reasoning_steps,
                          max_total_tokens=spec.max_total_tokens)
        tools = [item.model_schema() for item in self.registry.visible(spec.agent)]
        started = time.monotonic()
        previous_elapsed = float(state.get("elapsed_seconds", 0.0))
        deadline = started + max(0.0, spec.timeout_seconds - previous_elapsed)
        self.traces.emit(run_id, "micro.started", session.session_id, {
            "node_run_id": node_run_id, "agent": spec.agent.name,
            "resume": resume, "prompt_version": spec.prompt_version,
        })
        self._save(checkpoint, state, started, previous_elapsed)
        while state["model_turns"] < spec.agent.max_reasoning_steps:
            stop = _boundary_stop(cancellation, deadline)
            if stop:
                return self._stop(run_id, node_run_id, session, checkpoint, state,
                                  stop, started, previous_elapsed)
            state["model_turns"] += 1
            self._save(checkpoint, state, started, previous_elapsed)
            try:
                bounded = getattr(model, "complete_bounded", None)
                remaining = max(0.001, deadline - time.monotonic())
                turn = (bounded(session.messages, tools, spec.agent.max_tokens,
                                timeout=remaining)
                        if bounded else model.complete(session.messages, tools,
                                                       spec.agent.max_tokens))
            except BudgetAdmissionError as error:
                state["issues"] = [asdict(ValidationIssue(
                    "budget_exhausted", str(error), retryable=False))]
                return self._stop(run_id, node_run_id, session, checkpoint, state,
                                  "budget_exhausted", started, previous_elapsed)
            except Exception as error:
                state["issues"] = [asdict(ValidationIssue(
                    "model_call_failed", f"{type(error).__name__}: {error}", retryable=False))]
                return self._stop(run_id, node_run_id, session, checkpoint, state,
                                  "model_call_failed", started, previous_elapsed)
            if turn.tool_calls:
                session.messages.append({
                    "role": "assistant", "content": turn.text,
                    "tool_calls": [_tool_call_dict(call) for call in turn.tool_calls],
                })
                state["pending_tool_calls"] = [
                    {"call_id": call.call_id, "name": call.name,
                     "arguments": call.arguments}
                    for call in turn.tool_calls
                ]
                self._save(checkpoint, state, started, previous_elapsed)
                for index, call in enumerate(turn.tool_calls):
                    stop = _boundary_stop(cancellation, deadline)
                    if stop:
                        _append_unexecuted(session, turn.tool_calls[index:], stop)
                        state["pending_tool_calls"] = []
                        return self._stop(run_id, node_run_id, session, checkpoint, state,
                                          stop, started, previous_elapsed)
                    if state["tool_calls"] >= spec.agent.max_tool_calls:
                        _append_unexecuted(session, turn.tool_calls[index:],
                                           "tool_call_limit")
                        state["pending_tool_calls"] = []
                        return self._stop(run_id, node_run_id, session, checkpoint, state,
                                          "tool_call_limit", started, previous_elapsed)
                    state["tool_calls"] += 1
                    self._save(checkpoint, state, started, previous_elapsed)
                    result = self.executor.execute(
                        run_id, session, spec.agent, call,
                        deadline_at=deadline, cancellation=cancellation)
                    session.messages.append({
                        "role": "tool", "tool_call_id": call.call_id,
                        "content": _tool_result_content(result, self.registry, call),
                    })
                    state["pending_tool_calls"] = state["pending_tool_calls"][1:]
                    self._save(checkpoint, state, started, previous_elapsed)
                continue
            candidate, issues = self._validate(spec, turn.text)
            session.messages.append({"role": "assistant", "content": turn.text})
            state["candidate"] = candidate
            state["issues"] = [asdict(issue) for issue in issues]
            if not issues:
                return self._stop(run_id, node_run_id, session, checkpoint, state,
                                  "validated", started, previous_elapsed,
                                  status="completed")
            self.traces.emit(run_id, "micro.validation_failed", session.session_id, {
                "node_run_id": node_run_id, "turn": state["model_turns"],
                "issues": state["issues"],
            })
            if any(not issue.retryable for issue in issues):
                return self._stop(run_id, node_run_id, session, checkpoint, state,
                                  "non_retryable_validation", started, previous_elapsed)
            signature = canonical_hash({"candidate": candidate, "issues": state["issues"]})
            if signature == state.get("last_invalid_signature"):
                state["no_progress"] += 1
            else:
                state["no_progress"] = 0
            state["last_invalid_signature"] = signature
            if state["no_progress"] >= spec.max_no_progress:
                return self._stop(run_id, node_run_id, session, checkpoint, state,
                                  "no_progress", started, previous_elapsed)
            session.messages.append({
                "role": "user",
                "content": json.dumps({
                    "type": "validation_feedback",
                    "instruction": "Repair every retryable issue and return a complete candidate.",
                    "issues": state["issues"],
                }, ensure_ascii=False),
            })
            self.traces.emit(run_id, "micro.revised", session.session_id, {
                "node_run_id": node_run_id, "next_turn": state["model_turns"] + 1,
            })
            self._save(checkpoint, state, started, previous_elapsed)
        return self._stop(run_id, node_run_id, session, checkpoint, state,
                          "iteration_limit", started, previous_elapsed)

    @staticmethod
    def _validate(spec, text):
        if not text.strip():
            return None, [ValidationIssue("empty_output", "Model returned no candidate")]
        try:
            candidate = spec.parser(text)
        except Exception as error:
            return None, [ValidationIssue(
                "invalid_output", f"{type(error).__name__}: {error}")]
        return candidate, list(spec.validator(candidate))

    @staticmethod
    def _save(path, state, started, previous_elapsed):
        state["elapsed_seconds"] = previous_elapsed + time.monotonic() - started
        write_json(path, state)

    def _stop(self, run_id, node_run_id, session, checkpoint, state, reason,
              started, previous_elapsed, *, status="stopped"):
        state["terminal"] = True
        state["status"] = status
        state["stop_reason"] = reason
        self._save(checkpoint, state, started, previous_elapsed)
        event = "micro.completed" if status == "completed" else "micro.stopped"
        self.traces.emit(run_id, event, session.session_id, {
            "node_run_id": node_run_id, "reason": reason,
            "model_turns": state["model_turns"], "tool_calls": state["tool_calls"],
        })
        return _result_from_state(run_id, node_run_id, checkpoint, state)


def _boundary_stop(cancellation, deadline):
    if cancellation is not None and cancellation.is_set():
        return "cancelled"
    if time.monotonic() >= deadline:
        return "timed_out"
    return None


def _result_from_state(run_id, node_run_id, checkpoint, state):
    return MicroAgentResult(
        run_id, node_run_id, state.get("status", "stopped"),
        state.get("stop_reason", "interrupted"), state["model_turns"],
        state["tool_calls"], state.get("candidate"),
        tuple(ValidationIssue(**issue) for issue in state.get("issues", [])),
        str(checkpoint),
    )


def _tool_call_dict(call: ToolCall):
    return {"id": call.call_id, "type": "function", "function": {
        "name": call.name, "arguments": json.dumps(call.arguments, ensure_ascii=False)}}


def _append_unexecuted(session: Session, calls, reason: str):
    for call in calls:
        session.messages.append({
            "role": "tool",
            "tool_call_id": call.call_id,
            "content": json.dumps({
                "status": "not_executed",
                "error_type": "ToolExecutionStopped",
                "error_message": reason,
            }),
        })


def _tool_result_content(result: ToolResult, registry: ToolRegistry, call: ToolCall):
    content = json.dumps(result.to_dict(), ensure_ascii=False, default=str)
    tool = registry.get(call.name)
    maximum = tool.max_output_chars if tool else 20_000
    if len(content) > maximum:
        return json.dumps({"status": result.status, "truncated": True,
                           "preview": content[:maximum], "original_chars": len(content)},
                          ensure_ascii=False)
    return content

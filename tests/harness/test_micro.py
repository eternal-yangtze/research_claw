import json
import threading
import time

import pytest

from researchops.harness.budget import BudgetedModel
from researchops.harness.micro import (
    MicroAgentRunner, MicroAgentSpec, ValidationIssue,
)
from researchops.harness.runtime import (
    AgentSpec, ModelTurn, Session, ToolCall, ToolExecutor, ToolPolicy,
    ToolRegistry, ToolSpec, TraceStore,
)


def _runtime(tmp_path, model, tools=()):
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    traces = TraceStore(tmp_path / "traces")
    executor = ToolExecutor(registry, ToolPolicy(registry), traces)
    return MicroAgentRunner(model, executor, registry, traces, tmp_path / "checkpoints")


def _spec(tools=(), *, turns=4, tool_calls=8, no_progress=2, timeout=5):
    def validate(candidate):
        if not isinstance(candidate, dict) or not candidate.get("answer"):
            return [ValidationIssue("missing_answer", "answer is required", "answer")]
        return []

    return MicroAgentSpec(
        AgentSpec("Micro", "Return JSON.", tuple(tools),
                  max_reasoning_steps=turns, max_tool_calls=tool_calls,
                  max_tokens=200),
        "test-v1", validate, max_no_progress=no_progress,
        max_total_tokens=100_000, timeout_seconds=timeout,
    )


def test_micro_agent_repairs_invalid_output_and_emits_trace(tmp_path):
    class Model:
        def __init__(self):
            self.calls = 0
            self.timeouts = []

        def complete_bounded(self, messages, tools, max_tokens, *, timeout):
            self.calls += 1
            self.timeouts.append(timeout)
            if self.calls == 1:
                return ModelTurn(text="not json")
            assert "validation_feedback" in messages[-1]["content"]
            return ModelTurn(text='{"answer":"grounded"}')

    model = Model()
    runner = _runtime(tmp_path, model)
    result = runner.run(Session("s", "t"), _spec(), "analyze",
                        run_id="root", node_run_id="paper_1")

    assert result.status == "completed" and result.stop_reason == "validated"
    assert result.candidate == {"answer": "grounded"}
    assert result.model_turns == 2 and all(0 < value <= 5 for value in model.timeouts)
    events = [json.loads(line)["event"] for line in
              (tmp_path / "traces" / "root" / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events == ["micro.started", "micro.validation_failed", "micro.revised", "micro.completed"]


def test_micro_agent_returns_tool_errors_and_denials_to_model(tmp_path):
    schema = {"type": "object", "properties": {"value": {"type": "string"}},
              "required": ["value"], "additionalProperties": False}
    echo = ToolSpec("echo", "echo", schema, lambda value: value)

    class Model:
        def __init__(self):
            self.calls = 0

        def complete(self, messages, tools, max_tokens):
            self.calls += 1
            if self.calls == 1:
                return ModelTurn(tool_calls=(
                    ToolCall("bad", "echo", {}),
                    ToolCall("hidden", "not_visible", {}),
                ))
            results = [json.loads(message["content"])["status"]
                       for message in messages if message["role"] == "tool"]
            assert results == ["error", "denied"]
            return ModelTurn(text='{"answer":"recovered"}')

    result = _runtime(tmp_path, Model(), (echo,)).run(
        Session("s", "t"), _spec(("echo",)), "go",
        run_id="root", node_run_id="tools")
    assert result.status == "completed" and result.tool_calls == 2


def test_micro_agent_stops_repeated_invalid_candidate(tmp_path):
    class Model:
        def complete(self, messages, tools, max_tokens):
            return ModelTurn(text='{"answer":""}')

    result = _runtime(tmp_path, Model()).run(
        Session("s", "t"), _spec(turns=8, no_progress=2), "go",
        run_id="root", node_run_id="repeat")
    assert result.status == "stopped" and result.stop_reason == "no_progress"
    assert result.model_turns == 3


def test_micro_agent_pairs_unexecuted_calls_when_tool_budget_ends(tmp_path):
    tool = ToolSpec("echo", "echo", {"type": "object", "properties": {}},
                    lambda: "ok")

    class Model:
        def complete(self, messages, tools, max_tokens):
            return ModelTurn(tool_calls=(
                ToolCall("first", "echo", {}),
                ToolCall("second", "echo", {}),
            ))

    session = Session("s", "t")
    result = _runtime(tmp_path, Model(), (tool,)).run(
        session, _spec(("echo",), tool_calls=1), "go",
        run_id="root", node_run_id="limited")
    assert result.stop_reason == "tool_call_limit" and result.tool_calls == 1
    tool_results = [message for message in session.messages if message["role"] == "tool"]
    assert [message["tool_call_id"] for message in tool_results] == ["first", "second"]
    assert json.loads(tool_results[1]["content"])["status"] == "not_executed"


def test_micro_agent_propagates_deadline_and_honors_pre_cancel(tmp_path):
    observed = {}

    def inspect(_context):
        observed["remaining"] = _context.remaining_seconds()
        observed["cancelled"] = _context.cancelled()
        return {"ok": True}

    tool = ToolSpec("inspect", "inspect", {"type": "object", "properties": {}},
                    inspect, contextual=True, timeout_seconds=1)

    class Model:
        def __init__(self):
            self.calls = 0

        def complete(self, messages, tools, max_tokens):
            self.calls += 1
            if self.calls == 1:
                return ModelTurn(tool_calls=(ToolCall("i", "inspect", {}),))
            return ModelTurn(text='{"answer":"ok"}')

    result = _runtime(tmp_path, Model(), (tool,)).run(
        Session("s", "t"), _spec(("inspect",), timeout=3), "go",
        run_id="root", node_run_id="deadline")
    assert result.status == "completed"
    assert 0 < observed["remaining"] <= 1 and observed["cancelled"] is False

    cancelled = threading.Event()
    cancelled.set()

    class NeverCalled:
        def complete(self, *args):
            raise AssertionError("cancelled run must not call the model")

    stopped = _runtime(tmp_path / "cancel", NeverCalled()).run(
        Session("s2", "t"), _spec(), "go", run_id="root",
        node_run_id="cancelled", cancellation=cancelled)
    assert stopped.stop_reason == "cancelled" and stopped.model_turns == 0


def test_micro_agent_recovers_pending_tool_without_blind_replay(tmp_path):
    invocations = 0

    def interrupted():
        nonlocal invocations
        invocations += 1
        raise KeyboardInterrupt("simulated process interruption")

    tool = ToolSpec("write_once", "write once", {"type": "object", "properties": {}},
                    interrupted, read_only=False, idempotent=False, side_effect=True)

    class Model:
        def __init__(self):
            self.calls = 0

        def complete(self, messages, tools, max_tokens):
            self.calls += 1
            if self.calls == 1:
                return ModelTurn(tool_calls=(ToolCall("pending", "write_once", {}),))
            interrupted_results = [message for message in messages
                                   if message["role"] == "tool" and "interrupted" in message["content"]]
            assert len(interrupted_results) == 1
            return ModelTurn(text='{"answer":"inspected"}')

    model = Model()
    runner = _runtime(tmp_path, model, (tool,))
    session = Session("s", "t")
    with pytest.raises(KeyboardInterrupt):
        runner.run(session, _spec(("write_once",)), "go",
                   run_id="root", node_run_id="resume")
    result = runner.run(Session("s", "t"), _spec(("write_once",)), "go",
                        run_id="root", node_run_id="resume", resume=True)
    assert result.status == "completed" and invocations == 1


def test_budget_scopes_are_shared_thread_safe_and_resumable(tmp_path):
    class Model:
        model = "scripted"

        def complete_bounded(self, messages, tools, max_tokens, *, timeout):
            time.sleep(0.01)
            return ModelTurn(text="ok", usage={"total_tokens": 10})

    budgeted = BudgetedModel(Model(), tmp_path / "budget", 200_000,
                             max_model_turns=4)
    left = budgeted.scope("left", max_model_turns=2, max_total_tokens=80_000)
    right = budgeted.scope("right", max_model_turns=2, max_total_tokens=80_000)
    errors = []

    def call(model):
        try:
            model.complete([{"role": "user", "content": "x"}], [], 100)
        except Exception as error:  # pragma: no cover - surfaced by assertion
            errors.append(error)

    threads = [threading.Thread(target=call, args=(model,)) for model in (left, right)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    summary = budgeted.summary()
    assert not errors and summary["model_calls"] == 2
    assert summary["known_actual_tokens"] == summary["charged_tokens"] == 20
    call_ids = [entry["call_id"] for entry in budgeted.state["call_ledger"]]
    assert len(call_ids) == len(set(call_ids)) == 2
    resumed = BudgetedModel(Model(), tmp_path / "budget", 200_000,
                            max_model_turns=4, resume=True)
    assert resumed.summary()["charged_tokens"] == 20
    assert resumed.scope("left", max_model_turns=2,
                         max_total_tokens=80_000).summary()["turns"] == 1


def test_budget_disconnect_keeps_unknown_reservation(tmp_path):
    class Model:
        model = "offline"

        def complete_bounded(self, *args, **kwargs):
            raise TimeoutError("disconnected")

    budgeted = BudgetedModel(Model(), tmp_path / "budget", 100_000)
    with pytest.raises(TimeoutError):
        budgeted.complete([{"role": "user", "content": "x"}], [], 100)
    summary = budgeted.summary()
    assert summary["model_calls"] == 1 and summary["unknown_reserved_tokens"] > 0
    assert budgeted.state["call_ledger"][0]["error"].startswith("TimeoutError")


def test_micro_agent_reports_scope_budget_exhaustion(tmp_path):
    class Model:
        model = "scripted"

        def complete(self, messages, tools, max_tokens):
            raise AssertionError("scope admission must reject before dispatch")

    budgeted = BudgetedModel(Model(), tmp_path / "budget", 100_000)
    spec = _spec()
    spec = MicroAgentSpec(spec.agent, spec.prompt_version, spec.validator,
                          max_total_tokens=1, timeout_seconds=spec.timeout_seconds)
    result = _runtime(tmp_path / "runner", budgeted).run(
        Session("s", "t"), spec, "go", run_id="root", node_run_id="budgeted")
    assert result.stop_reason == "budget_exhausted"
    assert budgeted.summary()["model_calls"] == 0

import json
import urllib.error
from pathlib import Path

from researchops.runtime import AgentSpec, ModelTurn, OpenAICompatibleModel, ReasoningLoop, Session, ToolCall, ToolExecutor, ToolPolicy, ToolRegistry, ToolRisk, ToolSpec, TraceStore


def _schema() -> dict:
    return {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"], "additionalProperties": False}


def test_policy_rejects_unknown_and_unapproved_high_risk_tool(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register(ToolSpec("danger", "writes", _schema(), lambda value: value, risk=ToolRisk.HIGH, read_only=False, side_effect=True, requires_confirmation=True))
    agent = AgentSpec("lead", "test", ("danger",))
    executor = ToolExecutor(registry, ToolPolicy(registry), TraceStore(tmp_path / "runs"))
    session = Session("s1", "t1")

    denied = executor.execute("r1", session, agent, ToolCall("c1", "danger", {"value": "x"}))
    assert denied.status == "denied"
    assert denied.error_type == "ToolDenied"

    allowed = executor.execute("r1", session, agent, ToolCall("c2", "danger", {"value": "x"}), confirmed=True)
    assert allowed.status == "success"
    assert allowed.output == "x"

    unknown = executor.execute("r1", session, agent, ToolCall("c3", "missing", {}))
    assert unknown.status == "denied"
    events = [json.loads(line)["event"] for line in (tmp_path / "runs" / "r1" / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
    assert "tool.call.denied" in events
    assert "tool.call.completed" in events


def test_schema_validation_is_traced(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register(ToolSpec("echo", "echo", _schema(), lambda value: value))
    agent = AgentSpec("lead", "test", ("echo",))
    executor = ToolExecutor(registry, ToolPolicy(registry), TraceStore(tmp_path / "runs"))
    result = executor.execute("r2", Session("s2", "t2"), agent, ToolCall("c1", "echo", {}))
    assert result.status == "error"
    assert result.error_type == "ValueError"


def test_reasoning_loop_executes_tool_then_finishes(tmp_path: Path) -> None:
    class ScriptedModel:
        def __init__(self) -> None:
            self.turn = 0

        def complete(self, messages, tools, max_tokens):
            self.turn += 1
            if self.turn == 1:
                return ModelTurn(tool_calls=(ToolCall("model-call", "echo", {"value": "evidence"}),))
            return ModelTurn(text="done", usage={"total_tokens": 12})

    registry = ToolRegistry()
    registry.register(ToolSpec("echo", "echo", _schema(), lambda value: value))
    agent = AgentSpec("lead", "test", ("echo",))
    traces = TraceStore(tmp_path / "runs")
    loop = ReasoningLoop(ScriptedModel(), ToolExecutor(registry, ToolPolicy(registry), traces), registry, traces)
    result = loop.run(Session("s3", "t3"), agent, "go", run_id="r3")
    assert result["answer"] == "done"
    assert result["tool_calls"] == 1
    assert result["steps"] == 2


def test_model_adapter_retries_transient_http_errors(monkeypatch) -> None:
    calls = 0

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise urllib.error.HTTPError(request.full_url, 503, "busy", {}, None)
        return Response()

    monkeypatch.setattr("researchops.runtime.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("researchops.runtime.time.sleep", lambda seconds: None)
    model = OpenAICompatibleModel("https://model.example/v1", "secret", "test", max_attempts=3)

    result = model.complete([{"role": "user", "content": "hello"}], [], 20)

    assert result.text == "ok"
    assert calls == 3

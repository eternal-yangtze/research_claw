from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Protocol

from researchops.core.models import ToolResult


class ToolRisk(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., Any]
    risk: ToolRisk = ToolRisk.NORMAL
    read_only: bool = True
    idempotent: bool = True
    side_effect: bool = False
    allowed_agents: frozenset[str] = field(default_factory=frozenset)
    requires_confirmation: bool = False
    timeout_seconds: float = 30.0
    max_output_chars: int = 20_000
    contextual: bool = False

    def model_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


@dataclass(frozen=True)
class AgentSpec:
    name: str
    role: str
    allowed_tools: tuple[str, ...]
    denied_tools: tuple[str, ...] = ()
    max_reasoning_steps: int = 8
    max_tool_calls: int = 12
    max_tokens: int = 8_000
    allow_subagents: bool = False
    instructions: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("AgentSpec.name is required")
        if self.max_reasoning_steps < 1 or self.max_tool_calls < 1 or self.max_tokens < 1:
            raise ValueError("Agent limits must be positive")


@dataclass
class Session:
    session_id: str
    task_id: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ModelTurn:
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    usage: dict[str, int] = field(default_factory=dict)


class ModelPort(Protocol):
    def complete(self, messages: list[dict], tools: list[dict], max_tokens: int) -> ModelTurn: ...


@dataclass(frozen=True)
class ToolContext:
    run_id: str
    session: Session
    agent: AgentSpec
    executor: Any
    deadline_at: float | None = None
    cancellation: Any = None

    def cancelled(self) -> bool:
        return bool(self.cancellation is not None and self.cancellation.is_set())

    def remaining_seconds(self) -> float | None:
        if self.deadline_at is None:
            return None
        return max(0.0, self.deadline_at - time.monotonic())


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"Tool already registered: {spec.name}")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def visible(self, agent: AgentSpec) -> list[ToolSpec]:
        allow = set(agent.allowed_tools)
        deny = set(agent.denied_tools)
        return [
            spec for name, spec in sorted(self._tools.items())
            if name in allow and name not in deny and (not spec.allowed_agents or agent.name in spec.allowed_agents)
        ]


class ToolPolicy:
    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    def authorize(self, tool_name: str, agent: AgentSpec, confirmed: bool = False) -> tuple[bool, str]:
        spec = self.registry.get(tool_name)
        if spec is None:
            return False, f"Unknown tool: {tool_name}"
        if tool_name not in agent.allowed_tools or tool_name in agent.denied_tools:
            return False, f"Tool '{tool_name}' is not allowed for agent '{agent.name}'"
        if spec.allowed_agents and agent.name not in spec.allowed_agents:
            return False, f"Tool '{tool_name}' rejects agent '{agent.name}'"
        if spec.requires_confirmation and not confirmed:
            return False, f"Tool '{tool_name}' requires explicit confirmation"
        return True, "allowed"


class TraceStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def emit(self, run_id: str, event: str, session_id: str, payload: dict | None = None) -> None:
        run_dir = self.root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "session_id": session_id,
            "event": event,
            "payload": payload or {},
        }
        with self._lock:
            with (run_dir / "trace.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class ToolExecutor:
    def __init__(self, registry: ToolRegistry, policy: ToolPolicy, traces: TraceStore) -> None:
        self.registry = registry
        self.policy = policy
        self.traces = traces

    def execute(
        self,
        run_id: str,
        session: Session,
        agent: AgentSpec,
        call: ToolCall,
        confirmed: bool = False,
        *,
        deadline_at: float | None = None,
        cancellation: Any = None,
    ) -> ToolResult:
        started = time.perf_counter()
        self.traces.emit(run_id, "tool.call.started", session.session_id, {"call_id": call.call_id, "tool": call.name, "arguments": _safe_arguments(call.arguments)})
        allowed, reason = self.policy.authorize(call.name, agent, confirmed)
        if not allowed:
            result = ToolResult("denied", error_type="ToolDenied", error_message=reason)
            self.traces.emit(run_id, "tool.call.denied", session.session_id, {"call_id": call.call_id, "tool": call.name, "reason": reason})
            return result
        spec = self.registry.get(call.name)
        assert spec is not None
        try:
            if cancellation is not None and cancellation.is_set():
                raise RuntimeError("tool execution cancelled before dispatch")
            effective_deadline = time.monotonic() + spec.timeout_seconds
            if deadline_at is not None:
                effective_deadline = min(effective_deadline, deadline_at)
            if effective_deadline <= time.monotonic():
                raise TimeoutError("tool deadline expired before dispatch")
            _validate_arguments(spec.input_schema, call.arguments)
            output = (spec.handler(**call.arguments, _context=ToolContext(
                run_id, session, agent, self, effective_deadline, cancellation))
                      if spec.contextual else spec.handler(**call.arguments))
            serialized = json.dumps(output, ensure_ascii=False, default=str)
            truncated = len(serialized) > spec.max_output_chars
            result = ToolResult(
                "success",
                output=output,
                duration_ms=round((time.perf_counter() - started) * 1000, 3),
                metadata={"read_only": spec.read_only, "idempotent": spec.idempotent, "side_effect": spec.side_effect, "truncated_for_model": truncated, "output_chars": len(serialized), "max_output_chars": spec.max_output_chars},
            )
            self.traces.emit(run_id, "tool.call.completed", session.session_id, {"call_id": call.call_id, "tool": call.name, "duration_ms": result.duration_ms, "metadata": result.metadata})
            return result
        except Exception as error:
            result = ToolResult(
                "error",
                error_type=type(error).__name__,
                error_message=str(error),
                duration_ms=round((time.perf_counter() - started) * 1000, 3),
            )
            self.traces.emit(run_id, "tool.call.failed", session.session_id, {"call_id": call.call_id, "tool": call.name, "error_type": result.error_type, "error": result.error_message})
            return result


class ReasoningLoop:
    def __init__(self, model: ModelPort, executor: ToolExecutor, registry: ToolRegistry, traces: TraceStore) -> None:
        self.model = model
        self.executor = executor
        self.registry = registry
        self.traces = traces

    def run(self, session: Session, agent: AgentSpec, user_message: str, run_id: str | None = None,
            *, exposed_tools: tuple[str, ...] | None = None) -> dict:
        run_id = run_id or f"run_{uuid.uuid4().hex}"
        if not session.messages and (agent.role.strip() or agent.instructions.strip()):
            system_parts = [part.strip() for part in (agent.role, agent.instructions) if part.strip()]
            session.messages.append({"role": "system", "content": "\n\n".join(system_parts)})
        session.messages.append({"role": "user", "content": user_message})
        self.traces.emit(run_id, "run.started", session.session_id, {"agent": agent.name})
        calls = 0
        seen_calls: set[str] = set()
        tools = [spec.model_schema() for spec in self.registry.visible(agent)]
        if exposed_tools is not None:
            tools = [t for t in tools if t["function"]["name"] in exposed_tools]
        for step in range(1, agent.max_reasoning_steps + 1):
            self.traces.emit(run_id, "reasoning.step.started", session.session_id, {"step": step})
            try:
                turn = self.model.complete(session.messages, tools, agent.max_tokens)
            except Exception as error:
                self.traces.emit(run_id, "run.failed", session.session_id, {
                    "reason": "model_call_failed",
                    "step": step,
                    "error_type": type(error).__name__,
                    "error": str(error),
                })
                raise
            if turn.tool_calls:
                assistant_message = {"role": "assistant", "content": turn.text, "tool_calls": [_tool_call_dict(call) for call in turn.tool_calls]}
                session.messages.append(assistant_message)
                for call in turn.tool_calls:
                    calls += 1
                    if calls > agent.max_tool_calls:
                        raise RuntimeError("maximum tool call budget exceeded")
                    signature = json.dumps([call.name, call.arguments], sort_keys=True, ensure_ascii=False)
                    if signature in seen_calls:
                        raise RuntimeError(f"repeated tool call detected: {call.name}")
                    seen_calls.add(signature)
                    if exposed_tools is not None and call.name not in exposed_tools:
                        result = ToolResult("denied", error_type="ToolDenied", error_message="Use the exposed skill controls; direct hidden-tool calls are not permitted")
                        self.traces.emit(run_id, "tool.call.denied", session.session_id,
                                         {"call_id": call.call_id, "tool": call.name, "reason": result.error_message})
                    else:
                        result = self.executor.execute(run_id, session, agent, call)
                    content = json.dumps(result.to_dict(), ensure_ascii=False, default=str)
                    spec = self.registry.get(call.name)
                    maximum = spec.max_output_chars if spec else 20_000
                    if len(content) > maximum:
                        content = json.dumps({"status": result.status, "truncated": True, "preview": content[:maximum], "original_chars": len(content)}, ensure_ascii=False)
                    session.messages.append({"role": "tool", "tool_call_id": call.call_id, "content": content})
                continue
            if turn.text.strip():
                session.messages.append({"role": "assistant", "content": turn.text})
                self.traces.emit(run_id, "run.completed", session.session_id, {"step": step, "tool_calls": calls, "usage": turn.usage})
                return {"run_id": run_id, "answer": turn.text, "steps": step, "tool_calls": calls, "usage": turn.usage}
            raise RuntimeError("model returned neither text nor tool calls")
        self.traces.emit(run_id, "run.failed", session.session_id, {"reason": "maximum reasoning steps exceeded"})
        raise RuntimeError("maximum reasoning steps exceeded")


class OpenAICompatibleModel:
    """Minimal real provider adapter; no SDK dependency and no embedded credentials."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        max_attempts: int = 3,
        stream: bool = False,
        reasoning_effort: str | None = None,
        temperature: float | None = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("RESEARCHOPS_MODEL_BASE_URL") or "").rstrip("/")
        self.api_key = api_key or os.getenv("RESEARCHOPS_MODEL_API_KEY") or ""
        self.model = model or os.getenv("RESEARCHOPS_MODEL_NAME") or ""
        self.timeout = float(timeout or os.getenv("RESEARCHOPS_MODEL_TIMEOUT_SECONDS") or 180.0)
        self.max_attempts = max(1, int(max_attempts))
        self.stream, self.reasoning_effort = stream, reasoning_effort
        self.temperature = temperature
        if not self.base_url or not self.api_key or not self.model:
            raise ValueError("Set RESEARCHOPS_MODEL_BASE_URL, RESEARCHOPS_MODEL_API_KEY, and RESEARCHOPS_MODEL_NAME")

    def complete(self, messages: list[dict], tools: list[dict], max_tokens: int) -> ModelTurn:
        payload = {"model": self.model, "messages": messages, "max_tokens": max_tokens}
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        if self.stream:
            payload.update(stream=True, stream_options={"include_usage": True})
        if tools:
            payload["tools"] = tools
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        value = self._request_stream(request) if self.stream else self._request_json(request)
        message = value["choices"][0]["message"]
        calls = tuple(
            ToolCall(
                call_id=item["id"],
                name=item["function"]["name"],
                arguments=json.loads(item["function"].get("arguments") or "{}"),
            )
            for item in message.get("tool_calls") or []
        )
        return ModelTurn(str(message.get("content") or ""), calls, value.get("usage") or {})

    def complete_bounded(self, messages, tools, max_tokens, *, timeout):
        """One paid attempt with a caller-assigned timeout; no hidden retries."""
        bounded = OpenAICompatibleModel(self.base_url, self.api_key, self.model,
            timeout=min(self.timeout, max(0.1, timeout)), max_attempts=1,
            stream=self.stream, reasoning_effort=self.reasoning_effort, temperature=self.temperature)
        return bounded.complete(messages, tools, max_tokens)

    def _request_stream(self, request):
        """One SSE dispatch; never retry a partly observed paid generation."""
        content, calls, usage = [], {}, {}
        deadline = time.monotonic() + self.timeout
        done = False
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            stream_socket = getattr(getattr(getattr(response, "fp", None), "raw", None), "_sock", None)
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("streaming model deadline exceeded")
                if stream_socket is not None:
                    stream_socket.settimeout(remaining)
                raw = response.readline()
                if not raw:
                    break
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    done = True
                    break
                event = json.loads(data)
                if event.get("error"):
                    raise RuntimeError("model stream returned an error")
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    if choice.get("index", 0) != 0:
                        continue
                    delta = choice.get("delta") or {}
                    if delta.get("content"):
                        content.append(delta["content"])
                    for item in delta.get("tool_calls") or []:
                        row = calls.setdefault(item["index"], {"id": "", "function": {"name": "", "arguments": ""}})
                        row["id"] += item.get("id") or ""
                        function = item.get("function") or {}
                        row["function"]["name"] += function.get("name") or ""
                        row["function"]["arguments"] += function.get("arguments") or ""
        if not done:
            raise ConnectionError("model stream ended before [DONE]; usage remains conservatively reserved")
        return {"choices": [{"message": {"content": "".join(content),
                    "tool_calls": [calls[k] for k in sorted(calls)]}}], "usage": usage}

    def _request_json(self, request: urllib.request.Request) -> dict:
        retryable = {429, 500, 502, 503, 504, 524}
        for attempt in range(1, self.max_attempts + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as error:
                if error.code not in retryable or attempt == self.max_attempts:
                    raise RuntimeError(
                        f"model endpoint returned HTTP {error.code} after {attempt} attempt(s)"
                    ) from error
                retry_after = error.headers.get("Retry-After") if error.headers else None
                delay = float(retry_after) if retry_after and retry_after.isdigit() else min(2 ** (attempt - 1), 4)
                time.sleep(delay)
            except (urllib.error.URLError, TimeoutError) as error:
                if attempt == self.max_attempts:
                    reason = getattr(error, "reason", str(error))
                    raise RuntimeError(
                        f"model endpoint was unreachable after {attempt} attempt(s): {reason}"
                    ) from error
                time.sleep(min(2 ** (attempt - 1), 4))
        raise AssertionError("unreachable model retry state")


def _validate_arguments(schema: dict, arguments: dict) -> None:
    if not isinstance(arguments, dict):
        raise TypeError("tool arguments must be an object")
    properties = schema.get("properties") or {}
    required = schema.get("required") or []
    missing = [name for name in required if name not in arguments]
    if missing:
        raise ValueError("missing required argument(s): " + ", ".join(missing))
    if schema.get("additionalProperties") is False:
        unknown = sorted(set(arguments) - set(properties))
        if unknown:
            raise ValueError("unknown argument(s): " + ", ".join(unknown))
    for name, value in arguments.items():
        expected = (properties.get(name) or {}).get("type")
        mapping = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list, "object": dict}
        if expected in mapping and (not isinstance(value, mapping[expected]) or expected == "integer" and isinstance(value, bool)):
            raise TypeError(f"argument '{name}' must be {expected}")


def _safe_arguments(arguments: dict) -> dict:
    secret_markers = ("api_key", "token", "password", "secret")
    return {key: "<redacted>" if any(marker in key.lower() for marker in secret_markers) else value for key, value in arguments.items()}


def _tool_call_dict(call: ToolCall) -> dict:
    return {"id": call.call_id, "type": "function", "function": {"name": call.name, "arguments": json.dumps(call.arguments, ensure_ascii=False)}}

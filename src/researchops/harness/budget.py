"""Persistent, resumable admission for outer and nested model calls."""
from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path

from researchops.coding.budget import BudgetAdmissionError, reserve_call, settle_call
from researchops.core.artifacts import write_json


class BudgetedModel:
    """One ledger owner. Scoped views share charges instead of double-counting them."""

    def __init__(self, model, root: Path, token_cap: int, *, resume: bool = False,
                 max_model_turns: int = 128):
        if token_cap < 1 or max_model_turns < 1:
            raise ValueError("positive token_cap and max_model_turns required")
        self.inner, self.root = model, Path(root)
        self.model = getattr(model, "model", type(model).__name__)
        self.provider_options = {"stream": bool(getattr(model, "stream", False)),
                                 "reasoning_effort": getattr(model, "reasoning_effort", None)}
        if getattr(model, "temperature", None) is not None:
            self.provider_options["temperature"] = model.temperature
        self.provider_token_margin = 32768
        self.path = self.root / "ledger.json"
        self._lock = threading.RLock()
        self._thread_call = threading.local()
        self.budget = {"max_model_turns": max_model_turns, "max_total_tokens": token_cap,
                       "provider_token_margin": self.provider_token_margin}
        if self.path.exists():
            if not resume:
                raise ValueError("Use a new skill run ID; existing model budgets must not be reset")
            self.state = json.loads(self.path.read_text(encoding="utf-8"))
            if self.state.get("token_cap") != token_cap:
                raise ValueError("Cannot change an existing model budget token cap")
            recorded_turn_cap = self.state.get("max_model_turns", max_model_turns)
            if recorded_turn_cap != max_model_turns:
                raise ValueError("Cannot change an existing model turn cap")
            recorded_model = self.state.get("model")
            if recorded_model is not None and recorded_model != self.model:
                raise ValueError("Cannot resume a model budget with a different model")
            recorded_options = self.state.get("provider_options", {"stream": False, "reasoning_effort": None})
            if recorded_options != self.provider_options:
                raise ValueError("Cannot resume a model budget with different provider options")
            self.state.setdefault("schema_version", "researchops-budget-v1-compatible")
            self.state.setdefault("scopes", {})
        else:
            self.state = {
                "schema_version": "researchops-budget-v2",
                "model": self.model,
                "provider_options": self.provider_options,
                "turns": 0,
                "total_tokens": 0,
                "call_ledger": [],
                "token_cap": token_cap,
                "max_model_turns": max_model_turns,
                "scopes": {},
            }
            self._persist()

    def complete(self, messages, tools, max_tokens):
        return self.complete_bounded(messages, tools, max_tokens, timeout=240)

    def complete_bounded(self, messages, tools, max_tokens, *, timeout):
        return self._complete(messages, tools, max_tokens, timeout=timeout, scope_id=None)

    def scope(self, scope_id: str, *, max_model_turns: int, max_total_tokens: int):
        if not scope_id or max_model_turns < 1 or max_total_tokens < 1:
            raise ValueError("scope_id and positive scope limits required")
        with self._lock:
            current = self.state["scopes"].get(scope_id)
            expected = {"max_model_turns": max_model_turns, "limit": max_total_tokens}
            if current is None:
                self.state["scopes"][scope_id] = {
                    **expected, "turns": 0, "charged_tokens": 0, "call_ids": []}
                self._persist()
            elif any(current.get(key) != value for key, value in expected.items()):
                raise ValueError("Cannot change existing scope limits")
        return ScopedBudgetedModel(self, scope_id)

    def _complete(self, messages, tools, max_tokens, *, timeout, scope_id):
        self._thread_call.call_id = None
        if timeout <= 0:
            raise TimeoutError("model deadline expired before dispatch")
        with self._lock:
            task = None
            if scope_id is not None:
                task = self.state["scopes"][scope_id]
                if task["turns"] >= task["max_model_turns"]:
                    raise BudgetAdmissionError("scope_model_turn_budget_not_admitted")
            lease = reserve_call(self.state, self.budget, messages, max_tokens,
                                 task=task, tools=tools)
            key = uuid.uuid4().hex
            self._thread_call.call_id = key
            lease.update(call_id=key, node_run_id=scope_id, started_at=time.time())
            if task is not None:
                task["turns"] += 1
                task["call_ids"].append(key)
            self._persist()
        write_json(self.root / "calls" / f"{key}.request.json", {
            "call_id": key, "node_run_id": scope_id, "messages": messages,
            "tools": tools, "max_tokens": max_tokens, "model": self.model,
        })
        try:
            method = getattr(self.inner, "complete_bounded", None)
            turn = (method(messages, tools, max_tokens, timeout=timeout)
                    if method else self.inner.complete(messages, tools, max_tokens))
            write_json(self.root / "calls" / f"{key}.response.json", {
                "call_id": key, "text": turn.text,
                "tool_calls": [{"name": c.name, "arguments": c.arguments}
                               for c in turn.tool_calls],
                "usage": turn.usage,
            })
            with self._lock:
                lease["usage"] = turn.usage
                settle_call(self.state, lease, turn.usage or {}, task=task)
                self._persist()
            return turn
        except Exception as error:
            with self._lock:
                lease["error"] = f"{type(error).__name__}: {error}"
                self._persist()
            raise

    def _persist(self):
        write_json(self.path, self.state)

    def last_call_id(self):
        """The caller thread's admitted dispatch, including a failed response.

        Comparing global ledger snapshots is ambiguous during concurrent calls.
        This ID identifies exactly the lease admitted by this thread, or None
        when its request was rejected before dispatch.
        """
        return getattr(self._thread_call, "call_id", None)

    def increase_token_limit(self, token_cap: int, *, reason: str):
        """Explicit audited budget amendment, never an implicit resume override.

        Existing calls, unknown reservations, scopes and turn limits survive.
        Limit growth is at most five times the original allowance. Callers must
        have task authorization; normal constructors still reject cap drift.
        """
        with self._lock:
            original = self.state.get("original_token_cap", self.state["token_cap"])
            if not reason.strip() or not self.state["token_cap"] < token_cap <= original * 5:
                raise ValueError("Explicit reason and a larger cap within 5x original required")
            self.state.setdefault("original_token_cap", original)
            self.state.setdefault("budget_amendments", []).append({
                "previous_cap": self.state["token_cap"], "new_cap": token_cap,
                "reason": reason, "time": time.time(),
                "retained_turns": self.state["turns"],
                "retained_charged_tokens": self.state["total_tokens"]})
            self.state["token_cap"] = token_cap
            self.budget["max_total_tokens"] = token_cap
            self._persist()

    def increase_scope_limits(self, scope_id: str, *, token_cap: int, max_model_turns: int, reason: str):
        with self._lock:
            scope = self.state["scopes"][scope_id]
            original_cap = scope.get("original_limit", scope["limit"])
            original_turns = scope.get("original_max_model_turns", scope["max_model_turns"])
            if (not reason.strip() or not scope["limit"] <= token_cap <= original_cap * 5 or
                    not scope["max_model_turns"] <= max_model_turns <= original_turns * 5):
                raise ValueError("Scope amendment must preserve limits and remain within 5x original")
            scope.setdefault("original_limit", original_cap)
            scope.setdefault("original_max_model_turns", original_turns)
            scope.setdefault("amendments", []).append({"previous_limit": scope["limit"],
                "previous_turn_cap": scope["max_model_turns"], "new_limit": token_cap,
                "new_turn_cap": max_model_turns, "reason": reason, "time": time.time(),
                "retained_turns": scope["turns"], "retained_charge": scope["charged_tokens"]})
            scope.update(limit=token_cap, max_model_turns=max_model_turns)
            self._persist()

    def summary(self):
        with self._lock:
            known = sum(c.get("actual_tokens", 0) for c in self.state["call_ledger"])
            return {
                "model_calls": self.state["turns"],
                "known_actual_tokens": known,
                "unknown_reserved_tokens": self.state["total_tokens"] - known,
                "charged_tokens": self.state["total_tokens"],
                "token_cap": self.state["token_cap"],
                "ledger_path": str(self.path),
                "scopes": json.loads(json.dumps(self.state.get("scopes", {}))),
            }


class ScopedBudgetedModel:
    def __init__(self, parent: BudgetedModel, scope_id: str):
        self.parent, self.scope_id = parent, scope_id
        self.model = parent.model

    def complete(self, messages, tools, max_tokens):
        return self.complete_bounded(messages, tools, max_tokens, timeout=240)

    def complete_bounded(self, messages, tools, max_tokens, *, timeout):
        return self.parent._complete(messages, tools, max_tokens,
                                     timeout=timeout, scope_id=self.scope_id)

    def summary(self):
        return self.parent.summary()["scopes"][self.scope_id]

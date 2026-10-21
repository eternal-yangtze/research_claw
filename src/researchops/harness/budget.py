"""Persistent admission for all outer and nested model calls in a skill run."""
from __future__ import annotations

import time
import uuid
from pathlib import Path

from researchops.coding.budget import reserve_call, settle_call
from researchops.core.artifacts import write_json


class BudgetedModel:
    def __init__(self, model, root: Path, token_cap: int):
        if token_cap < 1:
            raise ValueError("positive token_cap required")
        self.inner, self.root = model, root
        self.model = getattr(model, "model", type(model).__name__)
        # Some compatible gateways report extra input/hidden tokens beyond the
        # submitted payload. Reserve headroom; do not increase requested output.
        self.provider_token_margin = 32768
        self.path = root / "ledger.json"
        if self.path.exists():
            raise ValueError("Use a new skill run ID; existing model budgets must not be reset")
        self.state = {"turns": 0, "total_tokens": 0, "call_ledger": [], "token_cap": token_cap}
        self.budget = {"max_model_turns": 128, "max_total_tokens": token_cap,
                       "provider_token_margin": self.provider_token_margin}
        write_json(self.path, self.state)

    def complete(self, messages, tools, max_tokens):
        return self.complete_bounded(messages, tools, max_tokens, timeout=240)

    def complete_bounded(self, messages, tools, max_tokens, *, timeout):
        lease = reserve_call(self.state, self.budget, messages, max_tokens, tools=tools)
        key = uuid.uuid4().hex
        lease.update(call_id=key, started_at=time.time())
        write_json(self.path, self.state)
        write_json(self.root / "calls" / f"{key}.request.json",
                   {"messages": messages, "tools": tools, "max_tokens": max_tokens, "model": self.model})
        try:
            method = getattr(self.inner, "complete_bounded", None)
            turn = method(messages, tools, max_tokens, timeout=timeout) if method else self.inner.complete(messages, tools, max_tokens)
            lease["usage"] = turn.usage
            write_json(self.root / "calls" / f"{key}.response.json",
                       {"text": turn.text, "tool_calls": [{"name": c.name, "arguments": c.arguments} for c in turn.tool_calls], "usage": turn.usage})
            settle_call(self.state, lease, turn.usage or {})
            return turn
        except Exception as error:
            lease["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            write_json(self.path, self.state)

    def summary(self):
        known = sum(c.get("actual_tokens", 0) for c in self.state["call_ledger"])
        return {"model_calls": self.state["turns"], "known_actual_tokens": known,
                "unknown_reserved_tokens": self.state["total_tokens"] - known,
                "charged_tokens": self.state["total_tokens"], "token_cap": self.state["token_cap"],
                "ledger_path": str(self.path)}

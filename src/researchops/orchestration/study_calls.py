"""One shared study allowance, with durable observations of provider responses."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from researchops.core.artifacts import canonical_hash, write_json
from researchops.core.structured import parse_single_object
from researchops.harness.budget import BudgetedModel
from researchops.harness.runtime import ModelTurn, ToolCall


class UncertainStudyCall(RuntimeError):
    """An interrupted dispatch must be reconciled, not silently paid for again."""


class StudyCalls:
    def __init__(self, root: Path, model, cap: int, *, max_model_turns=24):
        self.root = Path(root)
        self.owner = model if isinstance(model, BudgetedModel) else BudgetedModel(
            model, self.root / "budget", cap, max_model_turns=max_model_turns,
            resume=(self.root / "budget" / "ledger.json").exists())
        self.model = self.owner
        self.scope_id = "study_" + canonical_hash(str(self.root.resolve()))[:20]
        self.allowance = self.owner.scope(self.scope_id, max_model_turns=max_model_turns,
                                          max_total_tokens=cap)
        self.cap, self.max_model_turns = cap, max_model_turns

    @property
    def state(self):
        # Compatibility for the public result's cumulative token field.
        return self.owner.state

    def summary(self):
        return {"root": self.owner.summary(), "study": self.allowance.summary()}

    def port(self, namespace):
        return JournaledStudyModel(self, namespace)

    def call(self, role, system, payload, output_tokens=2200):
        turn = self.port("service_" + role).complete_bounded([
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ], [], output_tokens, timeout=240)
        return parse_single_object(turn.text)


class JournaledStudyModel:
    def __init__(self, calls, namespace):
        self.calls, self.namespace = calls, namespace
        self.model = calls.owner.model

    def scope(self, scope_id, *, max_model_turns, max_total_tokens):
        # The runner enforces its local turn count; all roles and revisions debit
        # ONE persistent study allowance, rather than multiplying role budgets.
        return self

    def complete_bounded(self, messages, tools, max_tokens, *, timeout):
        key = canonical_hash(["study-observation-v1", self.namespace, self.model,
                              messages, tools, max_tokens])
        path = self.calls.root / "model_observations" / f"{key}.json"
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            if record["status"] != "observed":
                raise UncertainStudyCall(f"Unreconciled provider dispatch: {path}")
            return ModelTurn(text=record["text"], usage=record["usage"],
                             tool_calls=tuple(ToolCall(**c) for c in record["tool_calls"]))
        write_json(path, {"status": "dispatch_pending", "namespace": self.namespace,
                          "request_hash": key, "root_ledger": str(self.calls.owner.path)})
        try:
            turn = self.calls.allowance.complete_bounded(messages, tools, max_tokens, timeout=timeout)
        except Exception as error:
            if self.calls.owner.last_call_id() is None:
                # Admission failed before dispatch; keep the reason, never retry
                # automatically inside this adapter.
                write_json(path, {"status": "admission_failed", "error": str(error),
                                  "request_hash": key, "namespace": self.namespace})
            raise
        write_json(path, {"status": "observed", "request_hash": key,
                          "namespace": self.namespace, "text": turn.text,
                          "usage": turn.usage, "tool_calls": [asdict(c) for c in turn.tool_calls],
                          "provider_call_ids": [self.calls.owner.last_call_id()]})
        return turn

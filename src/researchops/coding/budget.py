"""Admission before a paid call; interrupted/unknown usage stays charged.

UTF-8 bytes plus framing is deliberately conservative for byte tokenizers.
This is a client-side cap, not a provider billing guarantee.
"""
from __future__ import annotations

import json


class BudgetAdmissionError(RuntimeError):
    pass


def input_token_bound(messages, tools=()):
    # Allow for provider-side framing/prefixes as well as ordinary chat framing.
    return len(json.dumps({"messages": messages, "tools": tools}, ensure_ascii=False).encode("utf-8")) + 4096 + 64 * len(messages)


def reserve_call(state, budget, messages, output_tokens, *, task=None, tools=()):
    upper = input_token_bound(messages, tools) + output_tokens + budget.get("provider_token_margin", 0)
    if state["turns"] >= budget["max_model_turns"]:
        raise BudgetAdmissionError("model_turn_budget_not_admitted")
    if state["total_tokens"] + upper > budget["max_total_tokens"]:
        raise BudgetAdmissionError("token_budget_not_admitted")
    if task is not None and task["charged_tokens"] + upper > task["limit"]:
        raise BudgetAdmissionError("task_token_budget_not_admitted")
    state["turns"] += 1
    state["total_tokens"] += upper
    if task is not None:
        task["charged_tokens"] += upper
    lease = {"upper_tokens": upper, "max_output_tokens": output_tokens, "status": "reserved"}
    if budget.get("provider_token_margin"):
        lease["provider_token_margin"] = budget["provider_token_margin"]
    state.setdefault("call_ledger", []).append(lease)
    return lease


def settle_call(state, lease, usage, *, task=None):
    used = usage.get("total_tokens")
    if used is None and "prompt_tokens" in usage and "completion_tokens" in usage:
        used = int(usage["prompt_tokens"]) + int(usage["completion_tokens"])
    if used is None or int(used) <= 0:
        lease["status"] = "usage_unknown_charged_upper_bound"
        return
    used = int(used)
    adjustment = used - lease["upper_tokens"]
    state["total_tokens"] += adjustment
    if task is not None:
        task["charged_tokens"] += adjustment
    lease.update(status="settled", actual_tokens=used)
    if used > lease["upper_tokens"]:
        raise BudgetAdmissionError("provider_usage_exceeded_reserved_bound")

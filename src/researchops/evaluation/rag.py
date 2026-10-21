"""Auditable RAG metrics and concurrent, budget-admitted silver evaluation."""
from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path

from researchops.coding.budget import reserve_call, settle_call
from researchops.core.artifacts import canonical_hash, write_json
from researchops.core.structured import parse_single_object


def ranking_metrics(ranking: list[str], grades: dict[str, int], k: int) -> dict:
    # Dedup before evaluating paper rankings; repeated passages do not earn credit.
    ranked = list(dict.fromkeys(ranking))[:k]
    positives = {key for key, grade in grades.items() if grade > 0}
    found = sum(grades.get(key, 0) > 0 for key in ranked)
    rr = next((1 / (i + 1) for i, key in enumerate(ranked) if grades.get(key, 0) > 0), 0.0)
    dcg = sum((2 ** grades.get(key, 0) - 1) / math.log2(i + 2) for i, key in enumerate(ranked))
    ideal = sum((2 ** grade - 1) / math.log2(i + 2)
                for i, grade in enumerate(sorted(grades.values(), reverse=True)[:k]))
    return {"recall": found / len(positives) if positives else None,
            "mrr": rr if positives else None, "ndcg": dcg / ideal if ideal else None,
            "relevant_total": len(positives), "relevant_found": found}


class ConcurrentPaidCalls:
    """One shared ledger; reserve and settle are atomic, network requests overlap."""

    def __init__(self, root: Path, model, cap: int = 8000000):
        self.root, self.model, self.cap = root, model, cap
        self.path = root / "ledger.json"
        self.lock = threading.Lock()
        self.state = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {
            "turns": 0, "total_tokens": 0, "call_ledger": [], "cache_hits": 0, "token_cap": cap}
        if self.state["token_cap"] != cap:
            raise ValueError("Existing run budget is immutable")

    def call(self, role, system, payload, output_tokens=3000):
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}]
        key = canonical_hash(["rag-eval-v1", role, self.model.model, messages, output_tokens])
        cache = self.root / "calls" / f"{key}.json"
        with self.lock:
            if cache.exists():
                self.state["cache_hits"] += 1
                write_json(self.path, self.state)
                return json.loads(cache.read_text(encoding="utf-8"))["parsed"]
            lease = reserve_call(self.state, {"max_model_turns": 2000, "max_total_tokens": self.cap}, messages, output_tokens)
            lease.update(role=role, input_hash=key, started_at=time.time())
            write_json(self.path, self.state)
            write_json(cache.with_suffix(".request.json"), {"model": self.model.model, "messages": messages,
                       "max_output_tokens": output_tokens, "lease": dict(lease)})
        started = time.perf_counter()
        try:
            turn = self.model.complete_bounded(messages, [], output_tokens, timeout=240)
            with self.lock:
                settle_call(self.state, lease, turn.usage or {})
                lease["usage"] = turn.usage or {}
            write_json(cache.with_suffix(".response.json"), {"text": turn.text, "usage": turn.usage})
            parsed = parse_single_object(turn.text)
            write_json(cache, {"parsed": parsed, "usage": turn.usage, "request_hash": key})
            return parsed
        except Exception as error:
            with self.lock:
                lease["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            with self.lock:
                lease["wall_seconds"] = round(time.perf_counter() - started, 3)
                write_json(self.path, self.state)

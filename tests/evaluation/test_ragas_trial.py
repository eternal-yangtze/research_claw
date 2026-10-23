import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from researchops.evaluation.rag_trial import answer_issues, exported_contexts, run_condition
from researchops.harness.runtime import ModelTurn
from researchops.harness.budget import BudgetedModel
from researchops.orchestration.study_calls import StudyCalls, UncertainStudyCall


def candidate(cid="c1"):
    return {"facts": [{"claim": "Fact "+str(i), "citations": [cid]} for i in range(3)],
            "design": "Proposed matched comparison", "limitations": ["limited evidence"]}


def test_context_export_uses_only_actual_successful_tool_text():
    initial = [{"chunk_id": "c1", "text": "cropped initial"}]
    messages = [{"role": "tool", "content": json.dumps({"status": "success", "output": {
        "results": [{"chunk_id": "c2", "text": "preview only"}]}})},
        {"role": "tool", "content": json.dumps({"status": "denied", "output": {"chunk_id": "evil", "text": "secret"}})},
        {"role": "tool", "content": json.dumps({"status": "success", "output": {"chunk_id": "c2", "text": "full read"}})}]
    assert [c["text"] for c in exported_contexts(initial, messages)] == ["cropped initial", "preview only", "full read"]
    assert answer_issues(candidate("evil"), initial)[0].code == "citation"


def test_b_repairs_then_resumes_without_reissuing_or_leaking_references(tmp_path):
    class Provider:
        model = "test-model"
        count = 0

        def complete_bounded(self, messages, tools, max_tokens, *, timeout):
            self.count += 1
            assert not tools
            assert "evaluator_secret" not in json.dumps(messages)
            value = candidate("unknown" if self.count == 1 else "c1")
            return ModelTurn(json.dumps(value), usage={"total_tokens": 100})

    provider = Provider()
    calls = StudyCalls(tmp_path / "calls", provider, 100000, max_model_turns=4)
    args = dict(question={"id": "R01a", "text": "Question"}, idea="idea", condition="B",
        initial_contexts=[{"chunk_id": "c1", "text": "source"}], calls=calls,
        store=None, retriever=None, root=tmp_path)
    record = run_condition(**args)
    assert record["error"] is None
    assert record["model_calls_this_invocation"] == provider.count == 2
    assert record["mechanical_issues"] == []
    assert run_condition(**args) == json.loads(json.dumps(record))
    assert provider.count == 2


def test_unknown_evaluation_request_is_not_silently_paid_for_twice(tmp_path):
    class Offline:
        model = "offline"
        count = 0

        def complete_bounded(self, *args, **kwargs):
            self.count += 1
            raise TimeoutError("lost response")

    model = Offline()
    calls = StudyCalls(tmp_path, model, 100000, max_model_turns=4)
    port = calls.port("judge")
    with pytest.raises(TimeoutError):
        port.complete_bounded([{"role": "user", "content": "x"}], [], 100, timeout=1)
    charged = calls.owner.summary()["charged_tokens"]
    with pytest.raises(UncertainStudyCall):
        port.complete_bounded([{"role": "user", "content": "x"}], [], 100, timeout=1)
    assert model.count == 1
    assert calls.owner.summary()["unknown_reserved_tokens"] == charged > 0


def test_concurrent_judge_observations_own_exactly_one_matching_dispatch(tmp_path):
    barrier = threading.Barrier(2)

    class Provider:
        model = "concurrent-test"

        def complete_bounded(self, messages, tools, max_tokens, *, timeout):
            barrier.wait(timeout=5)
            return ModelTurn(messages[0]["content"], usage={"total_tokens": 100})

    calls = StudyCalls(tmp_path, Provider(), 200000, max_model_turns=4)

    def call(name):
        return calls.port(name).complete_bounded([{"role": "user", "content": name}], [], 100, timeout=5)

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert {t.text for t in pool.map(call, ("left", "right"))} == {"left", "right"}
    ids = []
    for path in (tmp_path / "model_observations").glob("*.json"):
        observed = json.loads(path.read_text(encoding="utf-8"))
        assert len(observed["provider_call_ids"]) == 1
        call_id = observed["provider_call_ids"][0]
        ids.append(call_id)
        request = json.loads((tmp_path / "budget" / "calls" / (call_id+".request.json")).read_text(encoding="utf-8"))
        assert request["messages"][0]["content"] == observed["text"] == observed["namespace"]
    assert len(set(ids)) == 2


def test_explicit_budget_growth_preserves_debits_and_rejects_silent_cap_drift(tmp_path):
    class Provider:
        model = "test"

        def complete_bounded(self, *args, **kwargs):
            return ModelTurn("ok", usage={"total_tokens": 100})

    model = Provider()
    budget = BudgetedModel(model, tmp_path, 50000)
    budget.complete_bounded([{"role": "user", "content": "x"}], [], 100, timeout=1)
    with pytest.raises(ValueError, match="token cap"):
        BudgetedModel(model, tmp_path, 100000, resume=True)
    budget.increase_token_limit(100000, reason="explicit task authorization")
    resumed = BudgetedModel(model, tmp_path, 100000, resume=True)
    assert resumed.summary()["known_actual_tokens"] == resumed.summary()["charged_tokens"] == 100
    assert resumed.summary()["model_calls"] == 1
    assert resumed.state["budget_amendments"][0]["retained_charged_tokens"] == 100
    with pytest.raises(ValueError, match="5x"):
        resumed.increase_token_limit(250001, reason="too much")


def test_ragas_adapter_real_sdk_three_completions_and_missing_reference(tmp_path):
    pytest.importorskip("ragas")
    from langchain_core.messages import HumanMessage
    from researchops.evaluation.ragas_adapter import BudgetedChatModel, score_metric

    class Provider:
        model = "test"
        count = 0

        def complete_bounded(self, messages, tools, max_tokens, *, timeout):
            self.count += 1
            return ModelTurn("response-"+str(self.count), usage={"total_tokens": 10})

    provider = Provider()
    calls = StudyCalls(tmp_path, provider, 200000, max_model_turns=8)
    chat = BudgetedChatModel(calls=calls, namespace="relevancy")

    async def run():
        results = await chat.agenerate([[HumanMessage(content="same prompt")]] * 3)
        assert len({g[0].text for g in results.generations}) == 3
        result = await score_metric("context_recall", {"response": "text"}, {}, calls, "missing", None)
        assert result["status"] == "not_applicable"

    asyncio.run(run())
    assert provider.count == 3
    assert calls.owner.summary()["known_actual_tokens"] == 30

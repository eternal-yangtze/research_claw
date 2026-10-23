import io
import json

import pytest

from researchops.harness.runtime import OpenAICompatibleModel


def test_streaming_preserves_text_tool_arguments_usage_and_single_dispatch(monkeypatch):
    events = [
        {"choices": [{"index": 0, "delta": {"content": "read ", "tool_calls": [{"index": 0,
            "id": "call-1", "function": {"name": "get_paper_chunk", "arguments": '{"chunk_'}}]}}]},
        {"choices": [{"index": 0, "delta": {"content": "source", "tool_calls": [{"index": 0,
            "function": {"arguments": 'id":"x"}'}}]}}]},
        {"choices": [], "usage": {"total_tokens": 42}},
    ]
    wire = "".join("data: " + json.dumps(e) + "\n\n" for e in events) + "data: [DONE]\n\n"
    requests = []
    def urlopen(request, timeout):
        requests.append(json.loads(request.data))
        return io.BytesIO(wire.encode())
    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    model = OpenAICompatibleModel("https://fixture.invalid/v1", "fixture", "fixture", stream=True, reasoning_effort="low", temperature=0.01)
    turn = model.complete_bounded([], [], 500, timeout=10)
    assert turn.text == "read source" and turn.usage["total_tokens"] == 42
    assert turn.tool_calls[0].call_id == "call-1" and turn.tool_calls[0].arguments == {"chunk_id": "x"}
    assert len(requests) == 1 and requests[0]["stream"] and requests[0]["reasoning_effort"] == "low"
    assert requests[0]["temperature"] == 0.01


def test_truncated_stream_never_retries(monkeypatch):
    calls = []
    def urlopen(*args, **kwargs):
        calls.append(1)
        return io.BytesIO(b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n')
    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    model = OpenAICompatibleModel("https://fixture.invalid/v1", "fixture", "fixture", stream=True, max_attempts=3)
    with pytest.raises(ConnectionError, match="before"):
        model.complete([], [], 10)
    assert len(calls) == 1


def test_provider_options_are_pinned_when_resuming_budget(tmp_path):
    from researchops.harness.budget import BudgetedModel
    model = OpenAICompatibleModel("https://fixture.invalid/v1", "fixture", "fixture", stream=True, reasoning_effort="low")
    BudgetedModel(model, tmp_path, 100000)
    changed = OpenAICompatibleModel("https://fixture.invalid/v1", "fixture", "fixture")
    with pytest.raises(ValueError, match="provider options"):
        BudgetedModel(changed, tmp_path, 100000, resume=True)

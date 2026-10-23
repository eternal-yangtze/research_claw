"""RAGAS 0.4.3 bridge. Import only in the isolated evaluation environment.

LangChain does no network I/O itself: every completion debits StudyCalls and its
persistent BudgetedModel. Each repeated completion has a distinct ordinal, so
AnswerRelevancy strictness=3 really obtains three independent generations.
"""
from __future__ import annotations

import asyncio
import math
import threading
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field, PrivateAttr
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (AnswerRelevancy, Faithfulness,
                          LLMContextPrecisionWithReference, LLMContextRecall)
from ragas.run_config import RunConfig


class BudgetedChatModel(BaseChatModel):
    calls: Any = Field(exclude=True)
    namespace: str
    output_tokens: int = 4096
    timeout_seconds: float = 120.0
    _ordinal: int = PrivateAttr(default=0)
    _lock: Any = PrivateAttr(default_factory=threading.Lock)

    @property
    def _llm_type(self):
        return "researchops-budgeted-provider"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if stop:
            raise ValueError("Stop sequences are unsupported by this provider bridge")
        with self._lock:
            ordinal = self._ordinal
            self._ordinal += 1
        payload = [{"role": {"human": "user", "ai": "assistant"}.get(m.type, m.type),
                    "content": m.content} for m in messages]
        turn = self.calls.port(f"{self.namespace}_{ordinal}").complete_bounded(
            payload, [], self.output_tokens, timeout=self.timeout_seconds)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(
            content=turn.text, response_metadata={"finish_reason": "stop"}))],
            llm_output={"token_usage": turn.usage, "model_name": self.calls.owner.model})

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        # Await the provider thread even if the caller is cancelled. A timed-out
        # task must not leave unobserved spending running in the background.
        task = asyncio.create_task(asyncio.to_thread(
            self._generate, messages, stop, run_manager, **kwargs))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            await task
            raise


class LocalEmbeddings(Embeddings):
    def __init__(self, backend):
        self.backend = backend

    def embed_documents(self, texts):
        return [list(v) for v in self.backend.encode(texts)]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def make_metric(name, calls, namespace, embedding):
    config = RunConfig(timeout=120, max_retries=1, max_workers=2, seed=20261022)
    llm = LangchainLLMWrapper(BudgetedChatModel(calls=calls, namespace=namespace),
                             run_config=config)
    # RAGAS 0.4.3 PydanticPrompt has at most THREE parse-repair attempts.
    # They use this same llm/ledger; HTTP retries=0. Both limits are published.
    if name == "faithfulness":
        return Faithfulness(llm=llm)
    if name == "answer_relevancy":
        return AnswerRelevancy(llm=llm, strictness=3,
            embeddings=LangchainEmbeddingsWrapper(LocalEmbeddings(embedding), run_config=config))
    if name == "context_precision":
        return LLMContextPrecisionWithReference(llm=llm)
    if name == "context_recall":
        return LLMContextRecall(llm=llm)
    raise ValueError(f"Unknown metric: {name}")


async def score_metric(name, record, reference, calls, namespace, embedding):
    if not record.get("response"):
        return {"score": None, "status": "missing_response", "reason": record.get("error", "generation_failed")}
    if name.startswith("context_") and not reference.get("reference"):
        return {"score": None, "status": "not_applicable", "reason": "no qualified semantic reference"}
    sample = SingleTurnSample(user_input=record["question"], response=record["response"],
        retrieved_contexts=[c["text"] for c in record["contexts"]],
        reference=reference.get("reference"), reference_contexts=reference.get("reference_contexts"))
    try:
        metric = make_metric(name, calls, namespace, embedding)
        # Provider I/O has a real 120s deadline; a metric has several calls.
        # Avoid an outer timeout abandoning a still-running paid provider call.
        value = float(await metric.single_turn_ascore(sample))
        if not math.isfinite(value):
            return {"score": None, "status": "undefined", "reason": "RAGAS returned NaN/Inf"}
        return {"score": value, "status": "scored", "reason": None}
    except Exception as error:
        return {"score": None, "status": "error", "reason": f"{type(error).__name__}: {error}"}

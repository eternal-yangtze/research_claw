"""One explicit admission-failure recovery; retain and aggregate the failed attempt."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate_v03_ragas import (GEN_CAP, EMBED_REVISION, environment, owner, queries,
                                read, verify_pins)
from researchops.core.artifacts import workspace_lock, write_json
from researchops.core.storage import ResearchStore
from researchops.evaluation.rag_trial import run_condition
from researchops.knowledge.retrieval import HybridRetriever, SentenceTransformerEmbedding
from researchops.knowledge.vector_backend import QdrantVectorBackend
from researchops.orchestration.study_calls import StudyCalls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    environment()
    root = args.output.resolve()
    with workspace_lock(root):
        verify_pins(root)
        path = root / "answers/R03b/C.json"
        failed = read(path)
        if failed.get("attempts"):
            print("Recovery already recorded; no model calls")
            return
        if failed.get("error") != "budget_exhausted":
            raise ValueError("This recovery is restricted to the observed R03b/C admission failure")
        budget = owner(root, "generation_budget", GEN_CAP, 144)
        store = ResearchStore(root / "passages.db")
        embedding = SentenceTransformerEmbedding(revision=EMBED_REVISION)
        backend = QdrantVectorBackend(store, embedding, alias="researchops_p7_"+root.name.lower(), corpus_version="p7-frozen-v1")
        retriever = HybridRetriever(store, embedding=embedding, vector_backend=backend)
        calls = StudyCalls(root / "generation_calls/R03b_C_recovery", budget, 200000, max_model_turns=4)
        q = next(q for q in queries() if q["id"] == "R03b")
        record = run_condition(question=q, idea=q["idea"], condition="C",
            initial_contexts=read(root / "retrieval/R03b.json")["contexts"], calls=calls, store=store,
            retriever=retriever, root=root / "recovery_trial")
        archived = root / "failed_attempts/R03b_C.initial.json"
        archived.parent.mkdir(parents=True, exist_ok=True)
        path.rename(archived)
        record["attempts"] = [{"artifact": str(archived), "error": failed["error"]},
                              {"artifact": str(root / "recovery_trial/answers/R03b/C.json"), "error": record["error"]}]
        for key in ("seconds", "known_actual_tokens_this_invocation", "unknown_reserved_tokens_this_invocation", "model_calls_this_invocation"):
            record[key] += failed[key]
        record["cost_accounting"] = "Cumulative failed attempt plus one bounded recovery; no charges removed"
        record["aggregate_case_dispatch_cap"] = 8
        record["aggregate_case_token_cap"] = 400000
        write_json(path, record)
        print(json.dumps({"query_id": q["id"], "error": record["error"], "calls_including_failure": record["model_calls_this_invocation"],
            "known_tokens_including_failure": record["known_actual_tokens_this_invocation"], "seconds_including_failure": record["seconds"]}))


if __name__ == "__main__":
    main()

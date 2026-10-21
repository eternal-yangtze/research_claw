"""Retry only structurally invalid blind reviews, retaining their original output.

Some reviewers split one input claim into several atomic judgments. That cannot
be silently aligned to the original claim indices; this asks for one judgment
per original claim, with no target scores or expected answers supplied.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from researchops.core.artifacts import file_hash, write_json, workspace_lock
from researchops.core.config import load_project_env
from researchops.evaluation.rag import ConcurrentPaidCalls
from researchops.harness.runtime import OpenAICompatibleModel


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--workspace", type=Path, default=Path("workspace_rag_incremental_v1_live"))
    args = p.parse_args()
    root = args.workspace.resolve()
    load_project_env()
    with workspace_lock(root):
        cfg = read(root / "source_snapshot/config/benchmarks/rag_incremental_v1.json")
        queries = {q["text"]: q["id"] for idea in cfg["ideas"] for q in idea["queries"]}
        calls = ConcurrentPaidCalls(root, OpenAICompatibleModel(max_attempts=1, timeout=240), 8000000)
        jobs = []
        for path in (root / "calls").glob("*.request.json"):
            request = read(path)
            if request["lease"]["role"] != "blind_answer_evaluation":
                continue
            payload = json.loads(request["messages"][1]["content"])
            qid = queries[payload["question"]]
            if (root / "judgments" / f"{qid}.json").exists():
                continue
            jobs.append((path, request, payload, qid))

        def worker(job):
            path, request, payload, qid = job
            expected = {a["answer_id"]: list(range(len(a["facts"]))) for a in payload["answers"]}
            clarification = ("\nSTRUCTURE REQUIREMENT: one row per ORIGINAL input fact, without splitting compound claims. "
                             "Use these exact answer_id -> fact index sets: " + json.dumps(expected) +
                             ". Judge each entire original claim; use partial if only part is supported. "
                             "Do not create extra rows for subclauses. Scores and science must still be independently assessed.")
            result = calls.call("blind_answer_evaluation_schema_repair", request["messages"][0]["content"] + clarification,
                                payload, 5000)
            if sorted(a.get("answer_id", "") for a in result.get("answers", [])) != sorted(expected):
                raise ValueError("Repair answer IDs incomplete")
            for answer in result["answers"]:
                if sorted(f.get("index") for f in answer.get("facts", [])) != expected[answer["answer_id"]]:
                    raise ValueError("Repair still splits/omits original facts")
                if any(answer.get(k) not in (0, 1, 2, 3, 4) for k in ("correctness", "question_coverage", "design_usefulness")):
                    raise ValueError("Invalid repaired score format")
            modes = list(cfg["generation_modes"])
            random.Random(cfg["seed"] + sum(map(ord, qid))).shuffle(modes)
            aliases = {f"A{i}": mode for i, mode in enumerate(modes)}
            write_json(root / "judgments" / f"{qid}.json", {"aliases": aliases, **result,
                       "schema_repair": {"original_request": str(path), "original_sha256": file_hash(path),
                                         "policy": "Same blind input; exact original claim boundaries; no target scores"}})
            print(f"repaired review {qid}", flush=True)
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(worker, jobs))
        print(json.dumps({"repaired": len(jobs), "charged_tokens": calls.state["total_tokens"]}))


if __name__ == "__main__":
    main()

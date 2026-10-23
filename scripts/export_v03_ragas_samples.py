"""Export complete RAGAS inputs and installed prompt hashes; no model calls."""
from __future__ import annotations

import argparse
import importlib
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from researchops.core.artifacts import canonical_hash, file_hash, write_json


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    rows = []
    for p in sorted((root / "answers").glob("*/*.json")):
        r = read(p)
        reference = read(root / "references" / (r["query_id"]+".json"))
        sample = {"user_input": r["question"], "response": r["response"] or None,
            "retrieved_contexts": [c["text"] for c in r["contexts"]],
            "reference": reference.get("reference"), "reference_contexts": reference.get("reference_contexts")}
        rows.append({"query_id": r["query_id"], "condition": r["condition"], "sample": sample,
            "sample_sha256": canonical_hash(sample), "citations": r["citations"],
            "context_metadata": [{k: v for k, v in c.items() if k != "text"} for c in r["contexts"]],
            "reference_provenance": reference["provenance"], "human_gold": False,
            "generation_error": r["error"], "source_answer_sha256": file_hash(p)})
    path = root / "ragas_samples.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in rows), encoding="utf-8")
    modules = ["ragas.metrics._faithfulness", "ragas.metrics._answer_relevance",
               "ragas.metrics._context_precision", "ragas.metrics._context_recall", "ragas.prompt.pydantic_prompt"]
    prompts = {name: file_hash(importlib.import_module(name).__file__) for name in modules}
    tails = {}
    for split, ids in (("development", {"R01a", "R01b", "R02a", "R02b", "R03a", "R03b"}),
                       ("regression", {"R04a", "R04b", "R05a", "R05b", "R06a", "R06b"})):
        for c in "ABC":
            records = [read(p) for p in (root / "answers").glob("*/"+c+".json") if p.parent.name in ids]
            times = sorted(r["seconds"] for r in records)
            tails[split+"_"+c] = {"count": len(records), "generation_seconds_median": statistics.median(times),
                "generation_seconds_p95_nearest_rank": times[-1], "p95_note": "n=6, nearest-rank p95 equals max; not population tail estimate",
                "max_calls_including_failures": max(r["model_calls_this_invocation"] for r in records),
                "failures": [{"query_id": r["query_id"], "error": r["error"]} for r in records if r["error"]]}
    manifest = {"rows": len(rows), "sample_file": str(path), "sha256": file_hash(path),
        "prompt_source_sha256": prompts, "latency_scope": "generation and local trial tool work; excludes common initial retrieval/model-load and separate evaluation",
        "tails": tails, "human_sampling": "pending; human_review_queue.json is not completed human review"}
    write_json(root / "samples_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Evaluate an immutable, COMPLETED split while another split is generated.

Owns the evaluation ledger lock; never edits generation artifacts, checkpoints
or budgets. At exit rechecks every input hash. Run only one evaluator process,
and use this entrypoint for both splits to share the same process lock.
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from evaluate_v03_ragas import environment, evaluate_async, queries, read, split_ids, verify_pins
from researchops.core.artifacts import file_hash, workspace_lock, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "regression"), required=True)
    args = parser.parse_args()
    environment()
    root = args.output.resolve()
    verify_pins(root)
    paths = [root / name for name in ("config.json", "freeze.json")]
    for q in queries():
        if q["id"] in split_ids(args.split):
            paths.append(root / "retrieval" / (q["id"]+".json"))
            paths.extend(root / "answers" / q["id"] / (c+".json") for c in "ABC")
    if not all(p.is_file() for p in paths):
        raise ValueError("All generation outcomes, including explicit failures, must exist before evaluating a split")
    pinned_inputs = {str(p): file_hash(p) for p in paths}
    with workspace_lock(root / "evaluation_budget"):
        asyncio.run(evaluate_async(args))
        after = {str(p): file_hash(p) for p in paths}
        result = {"split": args.split, "generation_inputs_untouched": after == pinned_inputs,
                  "input_hashes": pinned_inputs}
        write_json(root / "evaluation_input_audits" / (args.split+".json"), result)
        if not result["generation_inputs_untouched"]:
            raise ValueError("Generation inputs changed during evaluation")


if __name__ == "__main__":
    main()

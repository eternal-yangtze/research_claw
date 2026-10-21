"""Recover a single JSON object after a provider's unsolicited prose prefix.

Never changes values or recovers ambiguous/multiple/truncated JSON. Original
requests, responses, charges and failures stay intact. No model calls.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from researchops.core.artifacts import file_hash, write_json
from researchops.core.structured import parse_single_object


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--workspace", type=Path, default=Path("workspace_rag_incremental_v1_live"))
    args = p.parse_args()
    recovered = []
    for response in (args.workspace / "calls").glob("*.response.json"):
        cache = response.with_name(response.name.replace(".response.json", ".json"))
        if cache.exists():
            continue
        value = json.loads(response.read_text(encoding="utf-8"))
        text = value["text"].strip()
        start = text.find("{")
        if start <= 0 or any(c in text[:start] for c in "{}[]"):
            continue
        try:
            parsed = parse_single_object(text[start:])
        except (ValueError, json.JSONDecodeError):
            continue
        record = {"parsed": parsed, "usage": value["usage"], "request_hash": cache.stem,
                  "format_recovery": {"response_sha256": file_hash(response), "removed_prefix": text[:start],
                                      "policy": "Exactly one complete JSON object; no values changed; raw output retained"}}
        write_json(cache, record)
        recovered.append(str(cache))
    print(json.dumps({"recovered": recovered, "new_model_calls": 0}, indent=2))


if __name__ == "__main__":
    main()

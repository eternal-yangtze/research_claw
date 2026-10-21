"""Reuse byte-identical no-context controls across v1/v2, without model calls.

All paid costs stay in v1 and are included in aggregate reporting. Never
overwrites a dispatched/finished target request. No RAG answers or labels copy.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from researchops.core.artifacts import file_hash, write_json


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--baseline", type=Path, default=Path("workspace_rag_incremental_v1_live"))
    p.add_argument("--target", type=Path, default=Path("workspace_rag_incremental_v2_planned"))
    args = p.parse_args()
    if read(args.baseline / "snapshot.json")["pins"]["model"] != read(args.target / "snapshot.json")["pins"]["model"]:
        raise ValueError("Control model differs")
    marker = args.target / "shared_controls.json"
    imports = read(marker)["imports"] if marker.exists() else {}
    for request in (args.baseline / "calls").glob("*.request.json"):
        value = read(request)
        if value["lease"]["role"] != "research_evidence_answer":
            continue
        payload = json.loads(value["messages"][1]["content"])
        if payload.get("passages") != []:
            continue
        name = request.name.replace(".request.json", ".json")
        source, target = args.baseline / "calls" / name, args.target / "calls" / name
        if not source.exists() or target.exists() or (args.target / "calls" / request.name).exists():
            continue
        cached = read(source)
        cached["reused_control"] = {"baseline": str(args.baseline.resolve()), "source_sha256": file_hash(source),
                                    "meaning": "Identical original question, model, prompt, zero passages and output budget"}
        write_json(target, cached)
        imports[name] = {"source": str(source.resolve()), "source_sha256": file_hash(source),
                         "request_sha256": file_hash(request), "question": payload["question"]}
    write_json(marker, {"imports": imports, "cost_policy": "All original usage retained and charged in baseline ledger; target exact-cache hits add zero tokens"})
    print(json.dumps({"shared_no_context_requests": len(imports), "new_model_calls": 0}, indent=2))


if __name__ == "__main__":
    main()

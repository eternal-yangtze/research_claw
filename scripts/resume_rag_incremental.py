"""Bounded infrastructure/format recovery around the unchanged frozen benchmark.

Every launch and exit is logged. Scientific validation errors never change the
questions, labels or scores. At most three restarts, with existing call charges.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from researchops.core.artifacts import write_json


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--workspace", type=Path, default=Path("workspace_rag_incremental_v1_live"))
    p.add_argument("--phase", default="all")
    p.add_argument("--planned", action="store_true", help="Use the v2 query-planned revalidation entry")
    args = p.parse_args()
    root = args.workspace.resolve()
    phases = ((["retrieve", "pool", "generate", "judge", "report"] if args.planned else
               ["acquire", "index", "label", "retrieve", "pool", "generate", "judge", "report"])
              if args.phase == "all" else [args.phase])
    for phase in phases:
        for attempt in range(4):
            stem = root / "recovery" / f"{phase}_{time.time_ns()}"
            stem.parent.mkdir(parents=True, exist_ok=True)
            # Empty BM25 contexts can equal the no-RAG request byte for byte.
            # Serialize generation so those exact requests reuse one cached call.
            workers = "1" if phase == "generate" else "4"
            with stem.with_suffix(".log").open("w", encoding="utf-8") as log:
                script = "scripts/benchmark_rag_planned.py" if args.planned else "scripts/benchmark_rag_incremental.py"
                result = subprocess.run([sys.executable, "-B", script,
                                         "--phase", phase, "--output", str(root), "--workers", workers],
                                        stdout=log, stderr=subprocess.STDOUT)
            text = stem.with_suffix(".log").read_text(encoding="utf-8", errors="replace")
            record = {"phase": phase, "workers": workers, "attempt": attempt, "exit_code": result.returncode,
                      "log": str(stem.with_suffix(".log"))}
            if result.returncode == 0:
                write_json(stem.with_suffix(".json"), record)
                print(f"completed {phase}", flush=True)
                break
            recovered = subprocess.run([sys.executable, "-B", "scripts/recover_rag_format.py", "--workspace", str(root)],
                                       capture_output=True, text=True, encoding="utf-8")
            recovered_json = json.loads(recovered.stdout) if recovered.returncode == 0 else {}
            record["format_recovery"] = recovered_json
            write_json(stem.with_suffix(".json"), record)
            infrastructure = any(s in text for s in ("endpoint was unreachable", "endpoint returned HTTP", "TimeoutError"))
            if not recovered_json.get("recovered") and not infrastructure:
                raise SystemExit(result.returncode)
        else:
            raise SystemExit("Recovery attempts exhausted; inspect saved logs and ledger")


if __name__ == "__main__":
    main()

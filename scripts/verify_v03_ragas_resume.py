"""Cross-process cache replay after all P7 outcomes exist; require unchanged ledgers."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from researchops.core.artifacts import file_hash, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project, root = Path(__file__).resolve().parents[1], args.output.resolve()
    required = {"answers": (list((root / "answers").glob("*/*.json")), 36),
        "metrics": (list((root / "scores").glob("*/*/*.json")), 144),
        "references": (list((root / "references").glob("*.json")), 12),
        "fact_scores": (list((root / "fact_scores").glob("*.json")), 12)}
    if any(len(paths) != count for paths, count in required.values()):
        raise ValueError("Only complete cached runs may enter this no-new-call verification")
    ledgers = [root / name / "ledger.json" for name in ("generation_budget", "evaluation_budget")]
    before = {str(p): file_hash(p) for p in ledgers}
    artifacts_before = {str(p): file_hash(p) for paths, _ in required.values() for p in paths}
    commands = []
    for env, script, phase in ((".venv-v03", "evaluate_v03_ragas.py", ["generate"]),
                               (".venv-ragas", "evaluate_v03_completed_split.py", [])):
        for split in ("development", "regression"):
            cmd = [str(project / env / "Scripts/python.exe"), "-B", str(project / "scripts" / script),
                   *phase, "--output", str(root), "--split", split]
            process = subprocess.run(cmd, cwd=project, capture_output=True, timeout=90)
            if process.returncode:
                raise RuntimeError(f"Replay failed ({script}, {split}): " + process.stderr.decode("utf-8", "replace"))
            commands.append({"script": script, "split": split, "returncode": process.returncode})
    after = {str(p): file_hash(p) for p in ledgers}
    checks = {"root_ledgers_byte_identical": before == after,
              "answers_scores_references_byte_identical": artifacts_before == {str(p): file_hash(p) for paths, _ in required.values() for p in paths}}
    result = {"passed": all(checks.values()), "checks": checks, "commands": commands,
              "ledger_hashes_before": before, "ledger_hashes_after": after,
              "additional_provider_calls": 0 if before == after else None,
              "additional_charged_tokens": 0 if before == after else None}
    write_json(root / "resume_check.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

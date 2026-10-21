"""Supplementary replay of saved model outputs after interface fixes.

Original frozen-run results are never overwritten or reclassified. Generation is
not repeated; optional independent reviews share the original cumulative ledger.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from researchops.core.artifacts import canonical_hash, file_hash, write_json, workspace_lock
from researchops.core.config import load_project_env
from researchops.core.storage import ResearchStore
from researchops.core.structured import parse_single_object
from researchops.evaluation.precoding.gates import RUBRIC, case_gate, mechanical_gate, silver_pass
from researchops.evaluation.precoding.runner import PaidCalls, read_json
from researchops.experiments.planning import HOST_PROTOCOL, compile_contract, read_verified_datasets
from researchops.harness.runtime import OpenAICompatibleModel


def replay(root: Path, *, review=False):
    snapshot = read_json(root / "snapshot.json")
    states = read_verified_datasets(Path(snapshot["source_workspace"]))
    store = ResearchStore(root / "researchops.db")
    results = read_json(root / "results.json")
    suite = read_json("config/benchmarks/precoding_v1.json")
    expectations = {e["idea_id"]: e for e in suite["case_expectations"]}
    ledger = read_json(root / "ledger.json")
    calls = PaidCalls(root, OpenAICompatibleModel(max_attempts=1), ledger["token_cap"]) if review else None
    records = []
    for key, original in results.items():
        if not key.startswith(("A_", "B_")):
            continue
        directory = root / "tasks" / key
        if not (directory / "input.json").exists() or not (directory / "evidence.json").exists():
            continue
        idea, evidence = read_json(directory / "input.json"), read_json(directory / "evidence.json")
        out = root / "supplementary" / key
        record = {"key": key, "original_status": original["status"], "generation_reused": True,
                  "generation_tokens_this_replay": 0, "silver_accepted": original.get("silver_accepted", False),
                  "silver_origin": "original_review" if original.get("silver_accepted") else "unknown_not_rejudged",
                  "original_result": str(directory / "result.json")}
        try:
            proposal_path = directory / "proposal.json"
            if proposal_path.exists():
                proposal = read_json(proposal_path)
            else:
                proposal = None
                for request_path in sorted((root / "calls").glob("*.request.json")):
                    request = read_json(request_path)
                    payload = json.loads(request["messages"][1]["content"])
                    response_path = request_path.with_name(request_path.name.replace(".request.json", ".response.json"))
                    if payload.get("task") == idea and payload.get("fulltext_evidence") == evidence and response_path.exists():
                        proposal = parse_single_object(read_json(response_path)["text"])
                        record["formatting_recovered_from"] = str(response_path)
                        break
                if proposal is None:
                    raise ValueError("No reusable complete model response")
            write_json(out / "proposal.json", proposal)
            spec = compile_contract(idea, proposal, states, {
                "original_evidence": {"path": str(directory / 'evidence.json'), "sha256": file_hash(directory / 'evidence.json')}})
            write_json(out / "frozen_spec.json", spec)
            gate = mechanical_gate(spec, evidence, store)
            constraints = case_gate(spec, expectations[idea["idea_id"]])
            record.update(mechanical=gate, constraints=constraints,
                          metric_name_normalization=spec["metric_name_normalization"],
                          planned_jobs=spec["run_budget"]["total"])
            review_path = out / "review.json"
            if review_path.exists():
                record.update(silver_accepted=silver_pass(read_json(review_path)), silver_origin="supplementary_review")
            elif calls and not record["silver_accepted"] and gate["passed"] and constraints["passed"]:
                try:
                    cited = {c for item in proposal.get("evidence_claims", []) for c in item.get("citations", [])}
                    judged_evidence = [{"chunk_id": n["chunk_id"], "title": n["title"], "text": n["text"]}
                                       for n in evidence if n["chunk_id"] in cited]
                    method_keys = {"method", "rank", "alpha", "dropout", "target_blocks", "weight_decay", "prompt_length",
                                   "bottleneck_dim", "placement", "up_initialization", "equation", "trainable"}
                    judged = calls.call("supplementary_independent_review", RUBRIC, {
                        "task": idea, "proposal": proposal, "evidence": judged_evidence,
                        "compiled_summary": {"host_protocol": HOST_PROTOCOL,
                                             "methods": {k: {f: v for f, v in a.items() if f in method_keys}
                                                         for k, a in spec["methods"].items()},
                                             "metrics": {k: spec["metrics"][k] for k in ("primary", "secondary", "success_rule", "analysis_plan")},
                                             "jobs": spec["run_budget"],
                                             "mechanical_gate": gate,
                                             "manifest_hashes": [d["manifest_sha256"] for d in spec["datasets"]]}}, 900)
                    write_json(review_path, judged)
                    record.update(silver_accepted=silver_pass(judged), silver_origin="supplementary_review")
                except Exception as error:
                    record["review_error"] = f"{type(error).__name__}: {error}"
            acquisition = original.get("acquisition")
            record["success"] = gate["passed"] and constraints["passed"] and record["silver_accepted"] and (
                acquisition is None or acquisition["passed"])
            record["status"] = "replayed"
        except Exception as error:
            record.update(status="error", success=False, error=f"{type(error).__name__}: {error}")
        write_json(out / "result.json", record)
        records.append(record)
    report = {"type": "supplementary_interface_replay_not_original_heldout_score",
              "changes": ["metric spelling canonicalization with preserved endpoints", "single-object closing punctuation recovery",
                          "complete compiled method/metric definitions visible to supplemental judge"],
              "prompt_regenerated": False, "records": records, "tracks": {}}
    for track in ("A", "B"):
        subset = [r for r in records if r["key"].startswith(track + "_")]
        report["tracks"][track] = {"mechanical_passed": sum(r.get("mechanical", {}).get("passed", False) for r in subset),
                                   "silver_accepted": sum(r["silver_accepted"] for r in subset),
                                   "success": sum(r["success"] for r in subset), "total": 12 if track == "A" else 4}
    write_json(root / "supplementary" / "report.json", report)
    return report["tracks"]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("workspace_precoding_v2"))
    parser.add_argument("--review", action="store_true", help="Use remaining original budget for independent supplemental reviews")
    args = parser.parse_args()
    load_project_env()
    with workspace_lock(args.workspace):
        print(json.dumps(replay(args.workspace.resolve(), review=args.review), indent=2))


if __name__ == "__main__":
    main()

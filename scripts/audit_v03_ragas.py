"""Read-only P7 evidence audit. Never invokes a judge or business model."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from researchops.core.artifacts import canonical_hash, file_hash, write_json
from researchops.evaluation.rag_trial import exported_contexts, response_text


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--junit", type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    checks, details = {}, []
    project = Path(__file__).resolve().parents[1]
    def answer_prompt(path):
        return next(ast.literal_eval(node.value) for node in ast.parse(path.read_text(encoding="utf-8")).body
                    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "ANSWER_SYSTEM" for t in node.targets))
    checks["baseline_answer_prompt_matches_original_benchmark"] = answer_prompt(project / "scripts/benchmark_rag_incremental.py") == answer_prompt(project / "src/researchops/evaluation/rag_trial.py")
    config = read(root / "config.json")
    checks["corpus_not_modified"] = file_hash(config["source"]) == config["source_sha256"]
    checks["qrels_not_modified"] = file_hash(config["qrels"]) == config["qrels_sha256"]
    generation, evaluation = (read(root / n / "ledger.json") for n in ("generation_budget", "evaluation_budget"))
    calls = {c["call_id"]: c for ledger in (generation, evaluation) for c in ledger["call_ledger"]}
    for budget, name in ((generation, "generation_budget"), (evaluation, "evaluation_budget")):
        checks[name+"_within_cap"] = budget["total_tokens"] <= budget["token_cap"]
        checks[name+"_ledger_reconciles"] = budget["total_tokens"] == sum(
            c.get("actual_tokens", c["upper_tokens"]) for c in budget["call_ledger"])
        checks[name+"_all_requests_archived"] = all(
            (root / name / "calls" / (c["call_id"]+".request.json")).exists() for c in budget["call_ledger"])
        checks[name+"_all_dispatches_settled"] = all(
            c["status"] == "settled" and "actual_tokens" in c for c in budget["call_ledger"])
    observations = list((root / "generation_calls").rglob("model_observations/*.json"))
    observations += list((root / "evaluation_calls").rglob("model_observations/*.json"))
    mappings = [read(p) for p in observations if read(p)["status"] == "observed"]
    checks["one_exact_dispatch_id_per_observation"] = all(
        len(m.get("provider_call_ids", [])) == 1 and m["provider_call_ids"][0] in calls for m in mappings)
    checks["no_reused_dispatch_ids_across_observations"] = len(mappings) == len({m["provider_call_ids"][0] for m in mappings})
    answers = list((root / "answers").glob("*/*.json"))
    context_matches, reference_isolated, scores_valid = [], [], []
    for path in answers:
        r = read(path)
        qid, c = r["query_id"], r["condition"]
        initial = read(root / "retrieval" / (qid+".json"))["contexts"]
        if c == "A":
            exported = initial
        else:
            state = read(Path(r["result"]["checkpoint_path"]))
            exported = exported_contexts(initial, state["messages"])
        context_matches.append(exported == r["contexts"])
        if not r["error"]:
            context_matches.append(response_text(r["answer"]) == r["response"])
        namespaces = list((root / "generation_calls").glob(qid+"_"+c+"*"))
        first = [read(p) for namespace in namespaces for p in namespace.glob("model_observations/*.json")]
        # Full reference prose is evaluator-only. Exact source evidence can of
        # course overlap retrieved context; that is not reference leakage.
        reference = read(root / "references" / (qid+".json"))
        for observed in first:
            for call_id in observed.get("provider_call_ids", []):
                req = read(root / "generation_budget" / "calls" / (call_id+".request.json"))
                payload = json.loads(req["messages"][1]["content"])
                reference_isolated.append(set(payload) == {"question", "idea", "passages"})
                if reference.get("reference"):
                    reference_isolated.append(reference["reference"] not in json.dumps(req["messages"], ensure_ascii=False))
        for m in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
            score_path = root / "scores" / qid / c / (m+".json")
            scores_valid.append(score_path.exists() and read(score_path)["sample_sha256"] == canonical_hash([r, reference]))
        details.append({"query_id": qid, "condition": c, "context_occurrences": len(r["contexts"]),
            "new_tool_calls": (r.get("result") or {}).get("tool_calls", 0), "generation_error": r["error"]})
    checks["all_36_generator_exports_match_durable_requests"] = len(answers) == 36 and all(context_matches)
    sample_path = root / "ragas_samples.jsonl"
    sample_rows = [json.loads(line) for line in sample_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    checks["all_36_jsonl_samples_match_actual_answers"] = (
        len(sample_rows) == 36 and len({(r["query_id"], r["condition"]) for r in sample_rows}) == 36
        and all(r["source_answer_sha256"] == file_hash(root / "answers" / r["query_id"] / (r["condition"]+".json"))
                and r["sample_sha256"] == canonical_hash(r["sample"])
                and r["sample"]["response"] == (read(root / "answers" / r["query_id"] / (r["condition"]+".json"))["response"] or None)
                and r["sample"]["retrieved_contexts"] == [c["text"] for c in read(root / "answers" / r["query_id"] / (r["condition"]+".json"))["contexts"]]
                for r in sample_rows))
    checks["sample_manifest_file_hash_matches"] = read(root / "samples_manifest.json")["sha256"] == file_hash(sample_path)
    checks["references_stay_evaluator_only"] = bool(reference_isolated) and all(reference_isolated)
    checks["all_144_metric_sample_hashes_match"] = len(scores_valid) == 144 and all(scores_valid)
    metric_records = [read(p) for p in (root / "scores").glob("*/*/*.json")]
    metric_statuses = dict(Counter(r["status"] for r in metric_records))
    checks["all_144_metric_outcomes_accounted_for"] = len(metric_records) == 144 and all(
        (r["status"] == "scored" and isinstance(r.get("score"), (int, float))
         and 0 <= r["score"] <= 1)
        or (r["status"] == "missing_response" and r.get("score") is None and bool(r.get("reason")))
        for r in metric_records)
    checks["generation_answer_costs_match_root_ledger"] = sum(
        r["known_actual_tokens_this_invocation"]+r["unknown_reserved_tokens_this_invocation"]
        for p in answers for r in [read(p)]) == generation["total_tokens"]
    checks["all_36_answers_and_144_metrics_scored"] = read(root / "report.json")["p7_engineering_complete"]
    wire = [json.loads(line) for path in (root / "vector_audit").glob("*.jsonl")
            for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    checks["real_qdrant_queries_archived"] = any(w.get("operation") == "query_points" for w in wire)
    input_audits = [read(p) for s in ("development", "regression")
                    if (p := root / "evaluation_input_audits" / (s+".json")).exists()]
    checks["completed_generation_inputs_untouched_during_evaluation"] = len(input_audits) == 2 and all(a["generation_inputs_untouched"] for a in input_audits)
    replay_path = root / "resume_check.json"
    replay = read(replay_path) if replay_path.exists() else {}
    checks["cross_process_replay_no_new_calls_or_charges"] = (
        replay.get("passed") is True and replay.get("additional_provider_calls") == 0
        and replay.get("additional_charged_tokens") == 0)
    repeated_inputs = {}
    for lease in generation["call_ledger"]:
        req = read(root / "generation_budget/calls" / (lease["call_id"]+".request.json"))
        key = canonical_hash([req["model"], req["messages"], req["tools"], req["max_tokens"]])
        repeated_inputs.setdefault(key, []).append({"call_id": lease["call_id"],
            "prompt_tokens": (lease.get("usage") or {}).get("prompt_tokens")})
    usage_variations = [{"request_input_sha256": key, "calls": leases} for key, leases in repeated_inputs.items()
                        if len(leases) > 1 and len({c["prompt_tokens"] for c in leases}) > 1]
    regression = None
    if args.junit:
        suites = list(ET.parse(args.junit).getroot().iter("testsuite"))
        regression = {key: sum(int(s.get(key, 0)) for s in suites) for key in ("tests", "errors", "failures", "skipped")}
        regression["passed"] = regression["tests"]-regression["errors"]-regression["failures"]-regression["skipped"]
        regression.update(path=str(args.junit.resolve()), sha256=file_hash(args.junit))
        checks["control_regression_passed"] = regression["failures"] == regression["errors"] == 0 and regression["passed"] > 0
    output = {"passed": all(checks.values()), "checks": checks, "details": details,
        "artifact_integrity_passed": all(v for k, v in checks.items() if k != "all_36_answers_and_144_metrics_scored"),
        "business_delivery_passed": checks["all_36_answers_and_144_metrics_scored"],
        "metric_outcomes": {"expected": 144, "recorded": len(metric_records), "statuses": metric_statuses},
        "regression": regression,
        "provider_usage_observations": {"identical_inputs_with_different_reported_prompt_counts": usage_variations,
            "interpretation": "Provider-reported usage retained verbatim; no pricing/cache interpretation or algorithmic compression claim"},
        "budget_summary": {name: {"calls": ledger["turns"], "known_actual_tokens": sum(c.get("actual_tokens", 0) for c in ledger["call_ledger"]),
            "unknown_reserved_tokens": ledger["total_tokens"]-sum(c.get("actual_tokens", 0) for c in ledger["call_ledger"]),
            "charged_tokens": ledger["total_tokens"], "cap": ledger["token_cap"]}
            for name, ledger in (("generation", generation), ("evaluation", evaluation))}}
    write_json(root / "audit.json", output)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if output["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

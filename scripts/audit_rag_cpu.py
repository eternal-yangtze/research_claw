"""Offline audit and paired idea-cluster intervals; never calls an LLM."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from researchops.core.artifacts import canonical_hash, file_hash, write_json
from researchops.core.storage import ResearchStore


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def metrics(ranking, grades):
    ranked = list(dict.fromkeys(ranking))[:10]
    positives = {p for p, g in grades.items() if g > 0}
    dcg = sum((2 ** grades.get(p, 0) - 1) / math.log2(i + 2) for i, p in enumerate(ranked))
    ideal = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(sorted(grades.values(), reverse=True)[:10]))
    return {"recall": sum(p in positives for p in ranked) / len(positives) if positives else None,
            "mrr": next((1 / (i + 1) for i, p in enumerate(ranked) if p in positives), 0) if positives else None,
            "ndcg": dcg / ideal if ideal else None}


def interval(pairs):
    # Resample ideas, retaining the two questions in each cluster as one unit.
    values = np.asarray([statistics.mean(v) for _, v in sorted(pairs.items())])
    rng = np.random.default_rng(20261020)
    boot = values[rng.integers(0, len(values), size=(10000, len(values)))].mean(axis=1)
    return {"mean_delta": float(values.mean()), "idea_clusters": len(values),
            "percentile95": [float(v) for v in np.quantile(boot, [0.025, 0.975])],
            "meaning": "descriptive paired development interval; same-model silver, six ideas"}


def audit(root, study_root, growth_root):
    report, snapshot = read(root / "report.json"), read(root / "snapshot.json")
    for path, expected in snapshot["pins"]["files"].items():
        assert file_hash(path) == expected, f"Current source changed: {path}"
        relative = Path(path).resolve().relative_to(Path.cwd().resolve())
        assert file_hash(root / "source_snapshot" / relative) == expected
    assert file_hash(report["reference"]["path"]) == snapshot["pins"]["qrels_sha256"]
    qrels = read(report["reference"]["path"])
    store = ResearchStore(root / "passages.db")
    rows = store.all_chunks()
    assert canonical_hash([(r["chunk_id"], r["text"], r["source_hash"], r["paper_id"]) for r in rows]) == snapshot["pins"]["corpus_hash"]
    nodes = {r["chunk_id"]: r for r in rows}
    stages = {name: set(ids) for name, ids in snapshot["stages"]}
    assert len(report["retrieval_details"]) == 96
    for item in report["retrieval_details"]:
        record = read(root / "retrieval" / item["stage"] / item["query_id"] / f"{item['mode']}.json")
        assert all(h["paper_id"] in stages[item["stage"]] for h in record["hits"])
        for hit in record["hits"]:
            node = nodes[hit["chunk_id"]]
            assert node["paper_id"] == hit["paper_id"] and node["text"] == hit["text"]
        ranking = list(dict.fromkeys(h["paper_id"] for h in record["hits"]))[:10]
        assert item["ranking"] == ranking
        grades = qrels["paper_grades"][item["query_id"]]
        scopes = {"current": stages[item["stage"]], "fixed94": stages["94"],
                  "old87": stages["87"], "new7": stages["94"] - stages["87"]}
        for scope, ids in scopes.items():
            calculated = metrics(ranking, {p: g for p, g in grades.items() if p in ids})
            for key, value in calculated.items():
                actual = item[scope][key]
                assert (value is None and actual is None) or abs(value - actual) < 1e-12
    for stage in stages:
        for mode in ["legacy", "cpu"]:
            items = [r for r in report["retrieval_details"] if r["stage"] == stage and r["mode"] == mode]
            assert len(items) == 12 and len({r["query_id"] for r in items}) == 12
            for scope in ["current", "fixed94", "old87", "new7"]:
                for key in ["recall", "mrr", "ndcg"]:
                    values = [r[scope][key] for r in items if r[scope][key] is not None]
                    assert abs(statistics.mean(values) - report["retrieval"][stage][mode][scope][key]) < 1e-12
    for mode in ["legacy", "cpu"]:
        deltas = report["growth_regression"][mode]["stages"]
        for row in deltas:
            for key in ["recall", "ndcg"]:
                expected = report["retrieval"][row["stage"]][mode]["old87"][key] - report["retrieval"]["87"][mode]["old87"][key]
                assert abs(row[f"{key}_delta"] - expected) < 1e-12
        passed = all(r["recall_delta"] >= -0.03 and r["ndcg_delta"] >= -0.05 for r in deltas)
        assert (report["growth_regression"][mode]["status"] == "passed") == passed
    assert len(list((root / "answers").glob("*/*.json"))) == 36
    assert len(list((root / "judgments").glob("*.json"))) == 12
    quote_count, rejected_cards, unknown_citations, revisions, gap_searches = 0, 0, 0, 0, 0
    for path in (root / "answers").glob("*/*.json"):
        result = read(path)
        assert sum(len(p["text"]) for p in result["passages"]) <= 10000
        supplied = {p["chunk_id"] for p in result["passages"]}
        for passage in result["passages"]:
            node = nodes[passage["chunk_id"]]
            assert node["paper_id"] == passage["paper_id"] and passage["text"] in node["text"]
        unknown_citations += sum(c not in supplied for f in result["answer"]["facts"] for c in f.get("citations", []))
        judgment = read(root / "judgments" / f"{path.parent.name}.json")
        alias = next(a for a, m in judgment["aliases"].items() if m == path.stem)
        judged = next(r for r in judgment["answers"] if r["answer_id"] == alias)
        facts = result["answer"]["facts"]
        assert sorted(r["index"] for r in judged["facts"]) == list(range(len(facts)))
        locatable = {i for i, fact in enumerate(facts) if fact.get("citations") and all(c in supplied for c in fact["citations"])}
        supported = sum(r["index"] in locatable and r["evidence_support"] == "supported" for r in judged["facts"])
        reported = next(r for r in report["answer_details"] if r["query_id"] == path.parent.name and r["mode"] == path.stem)
        assert reported["facts"] == len(facts) and reported["supported"] == supported
        for key in ["correctness", "question_coverage", "design_usefulness", "design_dimensions"]:
            assert reported[key] == judged[key]
        for packet in [result.get("packet"), result.get("additional_evidence")]:
            if packet:
                rejected_cards += len(packet["rejected_cards"])
                for card in packet["cards"]:
                    assert card["quote"] in nodes[card["chunk_id"]]["text"]
                    quote_count += 1
        revisions += bool(result.get("critique", {}).get("needs_revision"))
        gap_searches += result.get("additional_evidence") is not None
    assert all(r["identical_order"] for r in report["repeat"])
    assert all(r["diagnostics"]["reranked_pairs"] == 0 for r in report["repeat"])
    comparisons = {}
    details = report["retrieval_details"]
    for key in ["recall", "mrr", "ndcg"]:
        pairs = {}
        for row in [r for r in details if r["stage"] == "94" and r["mode"] == "cpu"]:
            old = next(r for r in details if r["stage"] == "94" and r["mode"] == "legacy" and r["query_id"] == row["query_id"])
            pairs.setdefault(row["idea_id"], []).append(row["current"][key] - old["current"][key])
        comparisons[f"retrieval_cpu_minus_legacy_{key}"] = interval(pairs)
    for treatment, control in [("retrieval_only", "legacy"), ("complete", "legacy"), ("complete", "retrieval_only")]:
        for key in ["correctness", "question_coverage", "design_usefulness", "design_dimension_mean", "support_coverage"]:
            def value(r):
                if key == "design_dimension_mean":
                    return statistics.mean(r["design_dimensions"].values())
                if key == "support_coverage":
                    return r["supported"] / r["facts"]
                return r[key]
            pairs = {}
            for row in [r for r in report["answer_details"] if r["mode"] == treatment]:
                old = next(r for r in report["answer_details"] if r["mode"] == control and r["query_id"] == row["query_id"])
                pairs.setdefault(row["idea_id"], []).append(value(row) - value(old))
            comparisons[f"{treatment}_minus_{control}_{key}"] = interval(pairs)
    ledgers = [root / "ledger.json"] + sorted(study_root.glob("*/*/budget.json"))
    # The public PaidCalls ledger is named budget.json in some versions, ledger.json in others.
    ledgers += [p for p in sorted(study_root.glob("*/*/ledger.json")) if p not in ledgers]
    assert len(ledgers) == 5, ledgers
    costs = []
    for path in ledgers:
        state = read(path)
        calls = state["call_ledger"]
        actual = sum(c.get("actual_tokens", 0) for c in calls if c["status"] == "settled")
        unknown = sum(c["upper_tokens"] for c in calls if c["status"] != "settled")
        assert state["total_tokens"] == actual + unknown
        costs.append({"path": str(path), "known_tokens": actual, "unknown_reserved_tokens": unknown,
                      "charged_tokens": state["total_tokens"], "calls": len(calls),
                      "errors": sum(bool(c.get("error")) for c in calls)})
    growth, studies = read(growth_root / "report.json"), read(study_root / "report.json")
    assert growth["old_vectors_identical"] and growth["new_passages_encoded"] == 530
    assert len(studies["results"]) == 4 and all(r["success"] and not r["training_executed"] for r in studies["results"])
    assert report["reference"]["source"] == "silver" and report["reference"]["gold_interface"] is None
    return {"passed": True, "retrieval_records": 96, "answers": 36, "blind_judgments": 12,
            "verified_selected_quotes": quote_count, "rejected_cards": rejected_cards,
            "unknown_answer_citations": unknown_citations, "answer_revisions": revisions, "gap_searches": gap_searches,
            "comparisons": comparisons, "costs": costs,
            "total_known_tokens": sum(c["known_tokens"] for c in costs),
            "total_unknown_reserved_tokens": sum(c["unknown_reserved_tokens"] for c in costs),
            "total_charged_tokens": sum(c["charged_tokens"] for c in costs),
            "public_planner_success": {m: sum(r["success"] for r in studies["results"] if r["retrieval_strategy"] == m) for m in ["legacy", "cpu"]},
            "limitations": "mechanical offline audit; semantic judgments remain same-model silver; vector growth identity taken from separately executed growth check"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("workspace_rag_cpu_v31"))
    args = parser.parse_args()
    result = audit(args.workspace, Path("workspace_cpu_study_v1"), Path("workspace_rag_cpu_growth_v1"))
    write_json(args.workspace / "audit.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))

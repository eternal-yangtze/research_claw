"""Offline recomputation, provenance audit and descriptive idea-cluster intervals."""
from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from audit_rag_cpu import metrics, interval
from researchops.core.artifacts import canonical_hash, file_hash, write_json
from researchops.core.storage import ResearchStore


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    root = Path("workspace_rag_fusion_v1")
    snapshot, report = read(root / "snapshot.json"), read(root / "report.json")
    source = Path(snapshot["source_workspace"])
    for path, expected in snapshot["pins"]["code"].items():
        assert file_hash(path) == expected and file_hash(root / "source_snapshot" / path) == expected
    for path, expected in snapshot["pins"]["controls"].items():
        assert file_hash(source / path) == expected
    assert file_hash(snapshot["reference"]["path"]) == snapshot["pins"]["qrels"]
    qrels = read(snapshot["reference"]["path"])
    store = ResearchStore(root / "passages.db")
    stages = {name: set(ids) for name, ids in snapshot["stages"]}
    source_pins = read(source / "snapshot.json")["pins"]
    rows = store.all_chunks()
    assert canonical_hash([(r["chunk_id"], r["text"], r["source_hash"], r["paper_id"]) for r in rows]) == source_pins["corpus_hash"]
    nodes = {r["chunk_id"]: r for r in rows}
    requests = {}
    for path in (root / "calls").glob("*.request.json"):
        value = read(path)
        role = value["lease"]["role"]
        payload = json.loads(value["messages"][1]["content"])
        if role.startswith("anchored_paper_fusion"):
            assert set(payload) == {"question", "primary_query", "papers"}
            requests[canonical_hash(payload["papers"])] = payload
        if not role.startswith("fusion_blind_judgment"):
            assert "independent_reference_passages" not in payload and "qrels" not in payload
    fallbacks, rejected, retained = 0, 0, 0
    candidate_ceilings = []
    fusion_records = list((root / "retrieval").glob("*/*.json")) + list((root / "retrieval_no_history").glob("*.json"))
    assert len(fusion_records) == 60
    for path in fusion_records:
        record = read(path)
        stage = record.get("stage", "94")
        diagnostic = record["diagnostics"]
        payload = requests[diagnostic["candidate_hash"]]
        assert len(payload["papers"]) <= 32
        for paper in payload["papers"]:
            assert paper["paper_id"] in stages[stage]
            for passage in paper["passages"]:
                assert passage["text"] in nodes[passage["chunk_id"]]["text"]
        for hit in record["hits"]:
            assert hit["paper_id"] in stages[stage] and hit["text"] == nodes[hit["chunk_id"]]["text"]
            assert nodes[hit["chunk_id"]]["paper_id"] == hit["paper_id"]
        old = read(source / "retrieval" / stage / path.stem / "legacy.json")
        assert record["hits"][0]["paper_id"] == old["hits"][0]["paper_id"]
        for assessment in diagnostic["paper_assessments"]:
            if assessment["utility"] > 0:
                paper = next(p for p in payload["papers"] if p["paper_id"] == assessment["paper_id"])
                text = next(p["text"] for p in paper["passages"] if p["chunk_id"] == assessment["chunk_id"])
                assert assessment["quote"] in text and 25 <= len(assessment["quote"]) <= 240
        ranking = list(dict.fromkeys(h["paper_id"] for h in record["hits"]))[:10]
        assert len(diagnostic["retained_papers"]) <= 2
        assert all(p in ranking for p in diagnostic["retained_papers"])
        grades = qrels["paper_grades"][path.stem]
        positives = {p for p, grade in grades.items() if grade > 0 and p in stages[stage]}
        available = positives & set(diagnostic["candidate_papers"])
        candidate_ceilings.append({"stage": stage, "query_id": path.stem,
                                   "mode": "no_history" if path.parent.name == "retrieval_no_history" else "with_history",
                                   "relevant_candidates": len(available), "relevant_total": len(positives),
                                   "recall10_ceiling_ignoring_anchor": min(10, len(available)) / len(positives) if positives else None})
        fallbacks += diagnostic["fallback"]
        rejected += len(diagnostic["rejected_assessments"])
        retained += len(diagnostic["retained_papers"])
    assert len(report["retrieval_details"]) == 144
    for row in report["retrieval_details"]:
        scopes = {"current": stages[row["stage"]], "fixed94": stages["94"], "old87": stages["87"], "new7": stages["94"] - stages["87"]}
        grades = qrels["paper_grades"][row["query_id"]]
        for scope, ids in scopes.items():
            calculated = metrics(row["ranking"], {p: g for p, g in grades.items() if p in ids})
            for key, expected in calculated.items():
                actual = row[scope][key]
                assert (actual is None and expected is None) or abs(actual - expected) < 1e-12
    for row in report["no_history_details"]:
        calculated = metrics(row["ranking"], qrels["paper_grades"][row["query_id"]])
        assert all(abs(row[k] - v) < 1e-12 for k, v in calculated.items())
    for stage in stages:
        for mode in ["legacy", "cpu", "fusion"]:
            group = [r for r in report["retrieval_details"] if r["stage"] == stage and r["mode"] == mode]
            assert len(group) == 12 and len({r["query_id"] for r in group}) == 12
            for scope in ["current", "fixed94", "old87", "new7"]:
                for key in ["recall", "mrr", "ndcg"]:
                    values = [r[scope][key] for r in group if r[scope][key] is not None]
                    assert abs(statistics.mean(values) - report["retrieval"][stage][mode][scope][key]) < 1e-12
    for mode in ["legacy", "cpu", "fusion"]:
        deltas = report["growth"][mode]["stages"]
        for row in deltas:
            for key in ["recall", "ndcg"]:
                expected = report["retrieval"][row["stage"]][mode]["old87"][key] - report["retrieval"]["87"][mode]["old87"][key]
                assert abs(row[f"{key}_delta"] - expected) < 1e-12
        passed = all(r["recall_delta"] >= -0.03 and r["ndcg_delta"] >= -0.05 for r in deltas)
        assert (report["growth"][mode]["status"] == "passed") == passed
    for key in ["recall", "mrr", "ndcg"]:
        assert abs(statistics.mean(r[key] for r in report["no_history_details"]) - report["no_history"][key]) < 1e-12
    assert len(list((root / "answers").glob("*/*.json"))) == 36
    assert len(list((root / "judgments").glob("*.json"))) == 12
    unknown = {m: 0 for m in ["legacy", "cpu", "fusion"]}
    answer_lengths = {m: [] for m in ["legacy", "cpu", "fusion"]}
    revisions, final_invalid, card_count = 0, 0, 0
    for path in (root / "answers").glob("*/*.json"):
        record = read(path)
        assert sum(len(p["text"]) for p in record["passages"]) <= 10000
        supplied = {p["chunk_id"] for p in record["passages"]}
        for p in record["passages"]:
            assert p["text"] in nodes[p["chunk_id"]]["text"]
        facts = record["answer"]["facts"]
        visible = " ".join([f.get("claim", "") for f in facts] + [record["answer"]["design"]]
                           + record["answer"].get("limitations", []))
        answer_lengths[path.stem].append({"visible_characters": len(visible),
                                         "chinese_characters": len(re.findall(r"[\u3400-\u9fff]", visible))})
        unknown[path.stem] += sum(c not in supplied for f in facts for c in f.get("citations", []))
        if path.stem == "fusion":
            final_invalid += len(record["final_invalid_citation_facts"])
            revisions += sum(r["invalid_fact_citations"] != [] for r in record["mechanical_checks"])
            for card in record["packet"]["cards"]:
                assert card["quote"] in nodes[card["chunk_id"]]["text"]
                card_count += 1
        else:
            old_mode = "complete" if path.stem == "cpu" else "legacy"
            assert file_hash(path) == file_hash(source / "answers" / path.parent.name / f"{old_mode}.json")
        judgment = read(root / "judgments" / f"{path.parent.name}.json")
        alias = next(a for a, mode in judgment["aliases"].items() if mode == path.stem)
        item = next(a for a in judgment["answers"] if a["answer_id"] == alias)
        assert sorted(f["index"] for f in item["facts"]) == list(range(len(facts)))
        locatable = {i for i, fact in enumerate(facts) if fact.get("citations") and all(c in supplied for c in fact["citations"])}
        supported = sum(f["index"] in locatable and f["evidence_support"] == "supported" for f in item["facts"])
        result = next(r for r in report["answer_details"] if r["query_id"] == path.parent.name and r["mode"] == path.stem)
        assert result["supported"] == supported and result["facts"] == len(facts)
        for key in ["correctness", "question_coverage", "design_usefulness", "design_dimensions"]:
            assert result[key] == item[key]
    for mode in ["legacy", "cpu", "fusion"]:
        group = [r for r in report["answer_details"] if r["mode"] == mode]
        assert len(group) == 12
        for key in ["correctness", "question_coverage", "design_usefulness"]:
            assert abs(statistics.mean(r[key] for r in group) - report["answers"][mode][key]) < 1e-12
        support = sum(r["supported"] for r in group) / sum(r["facts"] for r in group)
        assert abs(support - report["answers"][mode]["fact_support_coverage"]) < 1e-12
    main = report["majority_check"]
    for key, pair in main["metrics"].items():
        if key in ["recall", "mrr", "ndcg"]:
            assert pair["fusion"] == report["no_history"][key]
            assert pair["legacy"] == report["retrieval"]["94"]["legacy"]["current"][key]
        else:
            assert all(pair[m] == report["answers"][m][key] for m in ["legacy", "fusion"])
    increased = [k for k, v in main["metrics"].items() if v["fusion"] > v["legacy"] + 1e-9]
    decreased = [k for k, v in main["metrics"].items() if v["fusion"] < v["legacy"] - 1e-9]
    assert main["increased"] == increased and main["decreased"] == decreased
    assert main["passed"] == (len(increased) >= 4 and not decreased)
    comparisons = {}
    query_changes = {}
    def counts(deltas):
        return {"improved": sum(v > 1e-9 for v in deltas),
                "unchanged": sum(abs(v) <= 1e-9 for v in deltas),
                "decreased": sum(v < -1e-9 for v in deltas)}
    for key in ["recall", "mrr", "ndcg"]:
        for name, treatment_rows in [("fusion_with_history", [r for r in report["retrieval_details"] if r["stage"] == "94" and r["mode"] == "fusion"]),
                                     ("fusion_no_history", report["no_history_details"])]:
            pairs = {}
            for row in treatment_rows:
                old = next(r for r in report["retrieval_details"] if r["stage"] == "94" and r["mode"] == "legacy" and r["query_id"] == row["query_id"])
                value = row["current"][key] if "current" in row else row[key]
                pairs.setdefault(row["idea_id"], []).append(value - old["current"][key])
            comparisons[f"{name}_minus_legacy_{key}"] = interval(pairs)
            query_changes[f"{name}_minus_legacy_{key}"] = counts([v for values in pairs.values() for v in values])
    for control in ["legacy", "cpu"]:
        for key in ["correctness", "question_coverage", "design_usefulness", "support_coverage"]:
            pairs = {}
            for row in [r for r in report["answer_details"] if r["mode"] == "fusion"]:
                old = next(r for r in report["answer_details"] if r["mode"] == control and r["query_id"] == row["query_id"])
                def value(r):
                    return r["supported"] / r["facts"] if key == "support_coverage" else r[key]
                pairs.setdefault(row["idea_id"], []).append(value(row) - value(old))
            comparisons[f"fusion_minus_{control}_{key}"] = interval(pairs)
            query_changes[f"fusion_minus_{control}_{key}"] = counts([v for values in pairs.values() for v in values])
    costs = []
    for path in [root / "ledger.json"] + sorted(Path("workspace_fusion_study_v1").glob("*/ledger.json")):
        state = read(path)
        actual = sum(c.get("actual_tokens", 0) for c in state["call_ledger"] if c["status"] == "settled")
        unknown_cost = sum(c["upper_tokens"] for c in state["call_ledger"] if c["status"] != "settled")
        assert actual + unknown_cost == state["total_tokens"]
        costs.append({"path": str(path), "known_tokens": actual, "unknown_reserved_tokens": unknown_cost,
                      "charged_tokens": state["total_tokens"], "calls": len(state["call_ledger"]),
                      "errors": sum(bool(c.get("error")) for c in state["call_ledger"])})
    assert len(costs) == 3
    study = read("workspace_fusion_study_v1/report.json")
    assert len(study["results"]) == 2 and all(not r["training_executed"] for r in study["results"])
    for path, expected in read("workspace_fusion_study_v1/source_hashes.json").items():
        assert file_hash(path) == expected
    for result in study["results"]:
        if result["success"]:
            assert result["mechanical"]["passed"] and all(result["mechanical"]["checks"].values())
            assert result["status"] == "ready_for_coding" and result["research_success"] is None
    assert all(r["identical_order"] for r in report["repeat"])
    assert snapshot["reference"]["source"] == "silver" and snapshot["reference"]["gold_interface"] is None
    result = {"passed": True, "retrieval_records": 144, "fusion_rankings": 60, "answers": 36, "blind_judgments": 12,
              "rerank_fallbacks": fallbacks, "rejected_paper_assessments": rejected, "retention_placements": retained,
              "unknown_citations": unknown, "invalid_citation_repair_checks": revisions,
              "answer_lengths": {mode: {key: {"mean": statistics.mean(r[key] for r in values),
                                              "maximum": max(r[key] for r in values)}
                                        for key in ["visible_characters", "chinese_characters"]}
                                  for mode, values in answer_lengths.items()},
              "fusion_final_invalid_citation_facts": final_invalid, "verified_fusion_cards": card_count,
              "comparisons": comparisons, "query_changes": query_changes, "costs": costs, "candidate_ceilings": candidate_ceilings,
              "total_known_tokens": sum(c["known_tokens"] for c in costs),
              "total_unknown_reserved_tokens": sum(c["unknown_reserved_tokens"] for c in costs),
              "total_charged_tokens": sum(c["charged_tokens"] for c in costs),
              "public_success": sum(r["success"] for r in study["results"]),
              "limitation": "mechanical audit; same-model silver development evaluation, no gold or external test"}
    write_json(root / "audit.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

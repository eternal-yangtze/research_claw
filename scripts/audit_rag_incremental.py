"""Offline consistency audit of a completed incremental RAG experiment."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from researchops.core.artifacts import file_hash, write_json
from researchops.core.storage import ResearchStore
from researchops.evaluation.rag import ranking_metrics


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("workspace_rag_incremental_v1_live"))
    args = parser.parse_args()
    root = args.workspace.resolve()
    snapshot, report, qrels = (read(root / name) for name in ("snapshot.json", "report.json", "qrels.json"))
    raw, index = ResearchStore(root / "raw.db"), ResearchStore(root / "passages.db")
    checks, errors = {}, []
    project = Path(__file__).resolve().parents[1]
    checks["frozen_code_unchanged"] = all(file_hash(project / path) == sha for path, sha in snapshot["pins"]["code"].items())
    covered = set()
    for path in (root / "labels").glob("*/*.json"):
        covered.update(read(path)["passage_ids"])
    rows = {r["chunk_id"]: dict(r) for r in index.all_chunks()}
    checks["full_extracted_text_label_coverage"] = covered == set(rows)
    valid_pdfs, hashes, valid_provenance = True, set(), True
    for pid in sorted({r["paper_id"] for r in rows.values()}):
        paper = index.get_paper(pid)
        valid_pdfs &= bool(paper["file_path"] and file_hash(paper["file_path"]) == paper["content_hash"])
        hashes.add(paper["content_hash"])
        original = {r["chunk_id"]: dict(r) for r in raw.paper_chunks(pid)}
        for row in index.paper_chunks(pid):
            provenance = json.loads(row["provenance_json"])
            children = provenance["source_chunk_ids"]
            valid_provenance &= bool(set(children) <= set(original) and
                                     provenance["source_sha256"] == paper["content_hash"] and
                                     all(original[c]["page"] == row["page"] for c in children))
    checks["pdf_hashes_valid"] = valid_pdfs
    checks["paper_hashes_unique"] = len(hashes) == report["corpus"]["after_papers"]
    checks["original_node_page_provenance"] = valid_provenance
    initial = set(snapshot["initial_paper_ids"])
    for row in report["retrieval_details"]:
        grades = qrels["paper_grades"][row["query_id"]]
        if row["corpus"] == "before":
            grades = {pid: grade for pid, grade in grades.items() if pid in initial}
        for k in (5, 10):
            actual = ranking_metrics(row["paper_ranking"], grades, k)
            if actual != row[f"paper_at_{k}"]:
                errors.append(f"metric mismatch {row['query_id']}/{row['mode']}/{k}")
    checks["metrics_recompute_exactly"] = not errors
    ledger = read(root / "ledger.json")
    charged = sum(item["actual_tokens"] if item["status"] == "settled" else item["upper_tokens"] for item in ledger["call_ledger"])
    checks["ledger_reconciles"] = charged == ledger["total_tokens"]
    checks["all_repeated_rankings_identical"] = all(r["identical_order"] for r in report["repeat_search"])
    checks["all_twelve_questions_and_three_answers"] = len(report["answer_details"]) == 36 and report["queries"] == 12
    checks["real_neural_backend_reported"] = all(
        read(path)["hits"][0]["embedding_model"].startswith("sentence-transformers/")
        and read(path)["hits"][0]["reranker_model"].startswith("cross-encoder/")
        for path in (root / "retrieval" / "after").glob("*/full.json"))
    checks["incremental_reuses_old_vectors"] = report["indexing"]["incremental"]["old_embeddings_reused"] > 0
    checks["all_passages_have_vectors"] = index.counts()["chunk_embeddings"] == len(rows)
    checks["fact_judgments_are_typed"] = all(
        isinstance(f.get("correct"), bool) and f.get("evidence_support") in
        ("supported", "partial", "unsupported", "no_citation")
        for path in (root / "judgments").glob("*.json")
        for answer in read(path)["answers"] for f in answer["facts"])
    requests = [read(p) for p in (root / "calls").glob("*.request.json")]
    checks["relevance_judge_blinded"] = all(not any(key in json.loads(r["messages"][1]["content"])
                                             for key in ("ranking", "mode", "answers"))
                                             for r in requests if r["lease"]["role"] in
                                             ("exhaustive_paper_relevance", "pooled_passage_relevance"))
    if (root / "planned_snapshot.json").exists():
        planned = read(root / "planned_snapshot.json")["pins"]
        checks["planned_query_code_unchanged"] = all(file_hash(project / name) == sha
            for name, sha in planned["files"].items())
        checks["original_labels_reused_exactly"] = file_hash(root / "qrels.json") == planned["baseline_qrels_sha256"]
        checks["query_planner_does_not_see_labels"] = all(
            set(json.loads(r["messages"][1]["content"])) == {"question", "idea"}
            for r in requests if r["lease"]["role"] == "retrieval_query_planning")
        if (root / "shared_controls.json").exists():
            shared = read(root / "shared_controls.json")["imports"]
            checks["shared_control_sources_valid"] = all(file_hash(Path(v["source"])) == v["source_sha256"]
                for v in shared.values())
    value = {"passed": all(checks.values()), "checks": checks, "errors": errors,
             "known_actual_tokens": sum(i.get("actual_tokens", 0) for i in ledger["call_ledger"] if i["status"] == "settled"),
             "unknown_reserved_tokens": sum(i["upper_tokens"] for i in ledger["call_ledger"] if i["status"] != "settled"),
             "request_attempts": len(ledger["call_ledger"])}
    acquisition = [read(path) for path in (root / "acquisition").glob("R*.json")]
    value["acquisition"] = {"ideas_with_fresh_search": sum(bool(r["fresh_search_success"]) for r in acquisition),
                            "ideas_with_new_downloads": sum(bool(r["new_ids"]) for r in acquisition),
                            "no_new_download_ids": [r["idea_id"] for r in acquisition if not r["new_ids"]]}
    value["format_recovered_responses"] = sum("format_recovery" in read(path)
        for path in (root / "calls").glob("*.json") if not path.name.endswith((".request.json", ".response.json")))
    from huggingface_hub import try_to_load_from_cache
    models = read(root / "retrieval" / "after" / "R01a" / "full.json")["hits"][0]
    value["local_model_files"] = {}
    for model in (models["embedding_model"].split("@")[0], models["reranker_model"]):
        files = {}
        for name in ("config.json", "model.safetensors", "pytorch_model.bin", "tokenizer.json", "sentence_bert_config.json"):
            cached = try_to_load_from_cache(model, name)
            if isinstance(cached, str) and Path(cached).is_file():
                files[name] = {"sha256": file_hash(cached), "resolved_path": cached}
        value["local_model_files"][model] = files
    write_json(root / "audit.json", value)
    print(json.dumps(value, ensure_ascii=False, indent=2))
    if not value["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

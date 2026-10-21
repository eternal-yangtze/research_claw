"""Reproducible audit on real saved papers/data; no new LLM labels or training.

Run --live separately for a fresh seven-source search/download availability sample.
Known-paper retrieval probes are diagnostic fixtures, NOT human-gold LitQA tasks.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from researchops.config import load_project_env
from researchops.datasets import REGISTRY, file_hash, native_samples
from researchops.literature import LiteratureSearch, download_pdf
from researchops.retrieval import HybridRetriever, _row_source_hash
from researchops.silver_evaluation import silver_metrics
from researchops.storage import ResearchStore


PROBES = [
    ("paper_bb5a94df37f8146c4efa", "Which method extends self-supervised vision pretraining by unfreezing a few transformer blocks and using low rank adapters elsewhere?"),
    ("paper_75e151a0c4939057f178", "How can a frozen vision transformer be adapted to image classification by learning additional input prompt tokens?"),
    ("paper_f19e0664e7ffafebaed3", "Which adaptation approach uses token sparsification and dense feature fusion to reduce both fine tuning and inference costs?"),
    ("paper_7f8304469d6b3583fab8", "How are frequency domain representations used to make vision transformer adaptation parameter efficient?"),
    ("paper_8b6d6de925588a408167", "How can a pretrained point cloud model be adapted using point prompts and geometry aware adapters?"),
    ("paper_e12aa162e2abb3c99e87", "How can federated low rank fine tuning reduce aggregation mismatch through initialization and aggregation refinement?"),
    ("paper_ee4589149bcba165e27a", "How can a frozen vision transformer be adapted to skin lesion segmentation with limited annotated images?"),
    ("paper_291d1dd28cad6cbedfdb", "How can low rank adaptation use the null space of pretrained weights to preserve existing knowledge?"),
]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def rate(n, d):
    return round(n / d, 4) if d else 0.0


def integrity(workspace, store, batch, labels):
    rows = store.all_chunks()
    nodes = {row["chunk_id"]: row for row in rows}
    paper_results = []
    for item in batch["acquisitions"]:
        pdf, parsed = Path(item["pdf_path"]), Path(item["parsed_path"])
        sidecar = read_json(parsed)
        expected = item["detail"]["download"]["sha256"]
        checks = {
            "pdf_hash": file_hash(pdf) == expected,
            "parsed_source_hash": sidecar["source_sha256"] == expected,
            "nonempty_parsed_nodes": bool(sidecar["chunks"]),
            "stored_nodes_match": all(
                node["chunk_id"] in nodes and nodes[node["chunk_id"]]["text"] == node["text"]
                for node in sidecar["chunks"]
            ),
        }
        paper_results.append({"paper_id": item["paper_id"], "checks": checks, "passed": all(checks.values())})
    embeddings = store.chunk_embeddings(
        os.getenv("RESEARCHOPS_EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
        os.getenv("RESEARCHOPS_EMBEDDING_REVISION", "main"),
    )
    valid_embeddings = 0
    for row in rows:
        record = embeddings.get(row["chunk_id"])
        if record is not None and record["source_hash"] == _row_source_hash(row):
            vector = json.loads(record["vector_json"])
            if len(vector) == record["dimensions"] and all(math.isfinite(v) for v in vector):
                valid_embeddings += 1
    cited = [(record["paper_id"], node) for record in labels for node in record.get("evidence_scope", {}).get("valid_chunk_citations", [])]
    valid_cited = sum(node in nodes and nodes[node]["paper_id"] == paper for paper, node in cited)
    trace_path = workspace / "runs" / batch["run_id"] / "trace.jsonl"
    traces = jsonl(trace_path)
    traced = {(event["payload"].get("paper_id"), event["payload"].get("stage")) for event in traces if event["event"].startswith("research.stage.")}
    stages = [(item["paper_id"], stage["stage"]) for item in batch["acquisitions"] for stage in item["dag_stages"]]
    with store.connect() as connection:
        levels = dict(connection.execute("SELECT level, COUNT(*) FROM paper_compressions GROUP BY level").fetchall())
    return {
        "paper_artifact_verifier_pass_rate": rate(sum(p["passed"] for p in paper_results), len(paper_results)),
        "paper_count": len(paper_results), "paper_checks": paper_results,
        "chunks": len(rows), "current_neural_embeddings": valid_embeddings,
        "embedding_coverage": rate(valid_embeddings, len(rows)),
        "parsers": dict(Counter(item["parser"] for item in batch["acquisitions"])),
        "compression_records_all_runs": levels,
        "valid_silver_node_citations": valid_cited, "silver_node_citations": len(cited),
        "citation_locator_validity": rate(valid_cited, len(cited)),
        "citation_semantic_entailment": "not_measured",
        "traced_stages": sum(stage in traced for stage in stages), "stages": len(stages),
        "trace_completeness_rate": rate(sum(stage in traced for stage in stages), len(stages)),
        "formula_transcription_accuracy": "not_measured",
    }


def datasets(workspace):
    results = []
    for dataset_id in REGISTRY:
        started = time.perf_counter()
        directory = workspace / "datasets" / dataset_id
        verification = read_json(directory / "prepared" / "verification.json")
        manifest = jsonl(directory / "prepared" / "samples.jsonl")
        # Decode ALL current native samples again, without replacing frozen manifests.
        decoded, classes = native_samples(dataset_id, Path(verification["root"]))
        by_id = {row["sample_id"]: row for row in decoded}
        matched = sum(
            row["sample_id"] in by_id and all(row.get(key) == by_id[row["sample_id"]].get(key)
                for key in ("label", "source_split", "content_sha256"))
            for row in manifest
        )
        included = [row for row in manifest if row["included"]]
        hashes = {split: {row["content_sha256"] for row in included if row["split"] == split} for split in ("train", "val", "test")}
        overlap = sum(len(hashes[a] & hashes[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test")))
        checks = {
            "all_samples_decoded_and_match": matched == len(manifest) == REGISTRY[dataset_id]["samples"],
            "class_count": len(classes) == REGISTRY[dataset_id]["classes"],
            "manifest_hash": file_hash(directory / "prepared" / "samples.jsonl") == verification["manifest_sha256"],
            "no_effective_split_pixel_leakage": overlap == 0,
            "all_test_samples_preserved": all(row["included"] for row in manifest if row["split"] == "test"),
        }
        result = {"dataset_id": dataset_id, "sample_count": len(decoded), "matched_samples": matched,
                  "excluded_duplicates": len(manifest) - len(included), "cross_split_pixel_overlap": overlap,
                  "checks": checks, "passed": all(checks.values()), "seconds": round(time.perf_counter() - started, 3)}
        results.append(result)
        print(json.dumps({"dataset_verified": result}), flush=True)
    return {"datasets": results, "verified_datasets": sum(r["passed"] for r in results),
            "verifier_pass_rate": rate(sum(r["passed"] for r in results), len(results)),
            "total_samples_redecoded": sum(r["sample_count"] for r in results)}


def retrieval(store):
    import numpy as np

    retriever = HybridRetriever(store)
    rows = store.all_chunks()
    vectors = retriever._vectors(rows)
    matrix = np.asarray([vectors[row["chunk_id"]] for row in rows], dtype=np.float32)
    details = []
    for expected, query in PROBES:
        started = time.perf_counter()
        sparse = retriever._bm25(query, rows)
        bm25_order = sorted(range(len(rows)), key=lambda i: sparse.get(rows[i]["chunk_id"], 0), reverse=True)
        encoded = np.asarray(retriever.embedding.encode([query])[0], dtype=np.float32)
        dense_order = np.argsort(-(matrix @ encoded)).tolist()
        hits = retriever.search(query, limit=20, candidate_limit=80)
        rankings = {
            "bm25": list(dict.fromkeys(rows[i]["paper_id"] for i in bm25_order[:20])),
            "dense": list(dict.fromkeys(rows[i]["paper_id"] for i in dense_order[:20])),
            "hybrid_rerank_mmr": list(dict.fromkeys(hit.paper_id for hit in hits)),
        }
        result = {"query": query, "expected_paper_id": expected,
                  "rankings_from_top20_chunks": rankings,
                  "ranks": {method: ranking.index(expected) + 1 if expected in ranking else None for method, ranking in rankings.items()},
                  "seconds_all_methods": round(time.perf_counter() - started, 3)}
        details.append(result)
        print(json.dumps({"retrieval_probe": result}), flush=True)
    metrics = {}
    for method in ("bm25", "dense", "hybrid_rerank_mmr"):
        ranks = [result["ranks"][method] for result in details]
        metrics[method] = {"hit_at_1": rate(sum(r == 1 for r in ranks), len(ranks)),
                           "hit_at_5": rate(sum(r is not None and r <= 5 for r in ranks), len(ranks)),
                           "mrr_at_10": round(statistics.mean(1 / r if r is not None and r <= 10 else 0 for r in ranks), 4)}
    return {"fixture_type": "8 audit-authored known-paper queries; one designated relevant paper per query, other papers unjudged",
            "caveat": "Diagnostic known-item retrieval; NOT human-gold QA, semantic answer accuracy, or broad generalization.",
            "candidate_limit": 80, "chunk_cutoff": 20, "metrics": metrics, "queries": details,
            "embedding_model": retriever.embedding.model_name, "reranker": retriever.reranker.model_name}


def live_sources(workspace):
    store = ResearchStore(workspace / "source_probe" / "researchops.db")
    def probe(source):
        started = time.perf_counter()
        query = "graph neural network"
        result = LiteratureSearch(store, timeout=20).search(query, "audit_sources", limit_per_source=1, sources=[source])
        row = {"source": source, "query": query, "source_status": result["sources"][0],
               "candidate_count": len(result["papers"]), "download_status": "no_pdf_candidate"}
        paper = next((paper for paper in result["papers"] if paper.get("pdf_url")), None)
        if paper:
            row.update(title=paper["title"], pdf_url=paper["pdf_url"])
            try:
                row["download"] = download_pdf(paper["pdf_url"], workspace / "source_probe" / "papers", f"{source}.pdf")
                row["download_status"] = "downloaded"
            except Exception as error:
                row.update(download_status="failed", download_error=f"{type(error).__name__}: {error}")
        row["seconds"] = round(time.perf_counter() - started, 3)
        save(workspace / "artifacts" / "evaluation" / "source_probes" / f"{source}.json", row)
        print(json.dumps({"live_source": row}), flush=True)
        return row
    with ThreadPoolExecutor(max_workers=7) as pool:
        futures = [pool.submit(probe, source) for source in ("arxiv", "openalex", "openreview", "dblp", "acl", "cvf", "crossref")]
        results = [future.result() for future in as_completed(futures)]
    return {"sources": sorted(results, key=lambda row: row["source"]),
            "search_successes": sum(row["source_status"]["status"] == "success" for row in results),
            "sources_with_candidates": sum(row["candidate_count"] > 0 for row in results),
            "sources_with_downloaded_pdf": sum(row["download_status"] == "downloaded" for row in results),
            "note": "One query per source, one PDF attempted per source. Snapshot availability only; no claim of long-term reliability."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default="workspace_batch_vit_v2")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    load_project_env()
    os.environ["RESEARCHOPS_RETRIEVAL_BACKEND"] = "neural"
    os.environ["RESEARCHOPS_EMBEDDING_LOCAL_ONLY"] = "true"
    os.environ["RESEARCHOPS_RERANKER_LOCAL_ONLY"] = "true"
    workspace = Path(args.workspace).resolve()
    target = workspace / "artifacts" / "evaluation" / ("audit_live_sources.json" if args.live else "audit_current_agent.json")
    report = {"created_at": datetime.now(timezone.utc).isoformat(),
              "schema_version": "researchops-capability-audit-v1",
              "script_sha256": file_hash(Path(__file__)),
              "reference_metrics": {
                  "AERS/eval-harness/README.md": "machine-verifiable artifacts; semantic manual checks remain ungraded",
                  "paper-qa-main/README.md#reproduction": "inspiration for evidence QA evaluation only; these probes do not reproduce LitQA",
                  "project/silver_evaluation.py": ["nDCG@10", "nDCG@30", "shortlist precision/recall"],
              }}
    if args.live:
        report["live"] = live_sources(workspace)
    else:
        batch = read_json(workspace / "artifacts/batch_research_ea4fffe07ca46a653072.json")
        labels = jsonl(workspace / "artifacts/evaluation/silver_vit_peft_small_data_59d209c7101d.jsonl")
        report["input_hashes"] = {
            "batch": file_hash(workspace / "artifacts/batch_research_ea4fffe07ca46a653072.json"),
            "silver_labels": file_hash(workspace / "artifacts/evaluation/silver_vit_peft_small_data_59d209c7101d.jsonl"),
        }
        store = ResearchStore(workspace / "researchops.db")
        report["integrity"] = integrity(workspace, store, batch, labels)
        report["silver_recomputed"] = silver_metrics(labels, [item["paper_id"] for item in batch["acquisitions"]],
                                                      [item["paper_id"] for item in batch["fan_in"]["shortlist"]])
        report["silver_label_distribution"] = dict(Counter(str(item["relevance"]) for item in labels))
        report["datasets"] = datasets(workspace)
        save(target, report)
        report["retrieval"] = retrieval(store)
    save(target, report)
    print(json.dumps({"report_path": str(target), "sections": list(report)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

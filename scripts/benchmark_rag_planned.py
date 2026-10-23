"""Uniform revalidation using a production query planner; original v1 remains intact."""
from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import time
from pathlib import Path

from benchmark_rag_incremental import Benchmark, read
from researchops.core.artifacts import file_hash, progress, workspace_lock, write_json
from researchops.core.config import load_project_env
from researchops.core.storage import ResearchStore
from researchops.knowledge.query_planning import plan_retrieval_query
from researchops.knowledge.retrieval import HybridRetriever


class PlannedBenchmark(Benchmark):
    def __init__(self, args):
        super().__init__(args)
        self.cfg["version"] = "rag-incremental-v2-query-planning"
        base = args.baseline.resolve()
        project = Path(__file__).resolve().parents[1]
        files = [Path(__file__), project / "src/researchops/knowledge/query_planning.py",
                 project / "src/researchops/application/study.py", project / "internal_experiments/evaluation/protocols/RAG_INCREMENTAL_V2_PROTOCOL.md"]
        pins = {"files": {str(p.relative_to(project)): file_hash(p) for p in files},
                "baseline": str(base), "baseline_qrels_sha256": file_hash(base / "qrels.json"),
                "baseline_snapshot_sha256": file_hash(base / "snapshot.json")}
        marker = self.root / "planned_snapshot.json"
        if marker.exists():
            if read(marker)["pins"] != pins:
                raise ValueError("Frozen planned-query protocol changed")
        else:
            if not (base / "qrels.json").exists():
                raise ValueError("Baseline exhaustive labels must complete first")
            for source_name, target_name in (("raw.db", "raw.db"), ("passages.db", "passages.db")):
                with sqlite3.connect(f"{(base / source_name).as_uri()}?mode=ro", uri=True) as src:
                    with sqlite3.connect(self.root / target_name) as dst:
                        src.backup(dst)
            for directory in ("labels", "acquisition"):
                shutil.copytree(base / directory, self.root / directory, dirs_exist_ok=True)
            for name in ("qrels.json", "indexing.json"):
                shutil.copy2(base / name, self.root / name)
            for path in files:
                target = self.root / "source_snapshot" / path.relative_to(project)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
            write_json(marker, {"pins": pins, "meaning": "Same real acquisition and full-text labels, new query-planned retrieval/answers; v1 preserved"})
        self.raw = ResearchStore(self.root / "raw.db")
        self.index = ResearchStore(self.root / "passages.db")

    def retrieve(self):
        import torch
        torch.set_num_threads(4)
        retriever = HybridRetriever(self.index)
        retriever.embedding.encode(["warm model"])
        retriever.reranker.score("warm model", ["vision transformer adaptation"])
        plans = {}
        for q in self.queries:
            path = self.root / "query_plans" / f"{q['id']}.json"
            if not path.exists():
                write_json(path, plan_retrieval_query(q["text"], q["idea"], self.calls))
            plans[q["id"]] = read(path)
        for corpus, ids in (("before", set(self.snapshot["initial_paper_ids"])), ("after", None)):
            for q in self.queries:
                for mode in self.cfg["retrieval_modes"]:
                    path = self.root / "retrieval" / corpus / q["id"] / f"{mode}.json"
                    if path.exists():
                        continue
                    start = time.perf_counter()
                    hits = retriever.search(plans[q["id"]]["retrieval_query"], limit=40,
                                            candidate_limit=self.cfg["candidate_limit"], paper_ids=ids, mode=mode)
                    write_json(path, {"query_id": q["id"], "mode": mode, "corpus": corpus,
                                      "query_plan": plans[q["id"]], "seconds": time.perf_counter() - start,
                                      "hits": [h.to_dict() for h in hits]})
                progress(f"planned retrieval {corpus}/{q['id']}")
        timings = []
        for q in self.queries:
            start = time.perf_counter()
            hits = retriever.search(plans[q["id"]]["retrieval_query"], limit=40,
                                    candidate_limit=self.cfg["candidate_limit"])
            previous = read(self.root / "retrieval" / "after" / q["id"] / "full.json")
            timings.append({"query_id": q["id"], "seconds": time.perf_counter() - start,
                            "identical_order": [h.chunk_id for h in hits] == [h["chunk_id"] for h in previous["hits"]]})
        write_json(self.root / "repeat_search.json", timings)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--suite", type=Path, default=Path("config/benchmarks/rag_incremental_v1.json"))
    p.add_argument("--baseline", type=Path, default=Path("workspace_rag_incremental_v1_live"))
    p.add_argument("--output", type=Path, default=Path("workspace_rag_incremental_v2_planned"))
    p.add_argument("--phase", choices=["retrieve", "pool", "generate", "judge", "report", "all"], default="all")
    p.add_argument("--token-cap", type=int, default=8000000)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    load_project_env()
    os.environ["RESEARCHOPS_RETRIEVAL_BACKEND"] = "neural"
    os.environ["RESEARCHOPS_EMBEDDING_LOCAL_ONLY"] = "true"
    os.environ["RESEARCHOPS_RERANKER_LOCAL_ONLY"] = "true"
    with workspace_lock(args.output):
        benchmark = PlannedBenchmark(args)
        for phase in ("retrieve", "pool", "generate", "judge", "report"):
            if args.phase in (phase, "all"):
                getattr(benchmark, phase)()


if __name__ == "__main__":
    main()

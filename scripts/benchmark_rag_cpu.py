"""CPU RAG + research design, with silver primary reference and four corpus snapshots."""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from benchmark_rag_incremental import ANSWER_SYSTEM, JUDGE_SYSTEM
from researchops.core.artifacts import canonical_hash, file_hash, write_json, workspace_lock, progress
from researchops.core.config import load_project_env
from researchops.core.storage import ResearchStore
from researchops.evaluation.rag import ConcurrentPaidCalls, ranking_metrics
from researchops.evaluation.references import load_reference
from researchops.experiments.evidence_design import build_evidence_packet, improve_research_answer
from researchops.harness.runtime import OpenAICompatibleModel
from researchops.knowledge.evidence_retrieval import EvidenceRetriever
from researchops.knowledge.query_planning import plan_evidence_queries
from researchops.knowledge.retrieval import HybridRetriever


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


DESIGN_JUDGE = """Also include design_dimensions in each answer judgment, with integer 0-4 values for
falsifiability, controls_and_fairness, metrics_and_decision, split_integrity, uncertainty, feasibility_scope.
Assess the actual visible design, not invisible implementation. A concise answer can score 4; do not favor
length or generic checklists. 0=absent/contradictory,1=major gap,2=partial,3=usable minor gap,4=operational.
Require task-appropriate primary metric/decision conditions, fair tuning/data budgets, test isolation,
paired multi-seed uncertainty, and honest implementation/domain boundaries where relevant.
Fact indexes refer to EXACT input facts; never split, merge or invent fact entries. Output one fact verdict
per input index even when a fact has multiple clauses; evidence_support=supported only for the whole fact."""
DIMENSIONS = ["falsifiability", "controls_and_fairness", "metrics_and_decision", "split_integrity", "uncertainty", "feasibility_scope"]


class CPUBenchmark:
    def __init__(self, args):
        self.root = args.output.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.source = Path("workspace_rag_incremental_v2_planned").resolve()
        self.suite = read("config/benchmarks/rag_incremental_v1.json")
        self.queries = [{**q, "idea_id": i["id"], "idea": i["idea"]} for i in self.suite["ideas"] for q in i["queries"]]
        self.workers = args.workers
        self.calls = ConcurrentPaidCalls(self.root, OpenAICompatibleModel(max_attempts=1, timeout=240), args.token_cap)
        self.store = ResearchStore(self.root / "passages.db")
        self.qrels, self.reference_info = load_reference(silver=self.source / "qrels.json")
        self.initial = set(read("workspace_rag_incremental_v1_live/snapshot.json")["initial_paper_ids"])
        all_papers = {r["paper_id"] for r in self.store.all_chunks()}
        self.new = sorted(all_papers - self.initial, key=lambda p: self.store.get_paper(p)["content_hash"])
        self.stages = [(str(len(self.initial) + size), self.initial | set(self.new[:size])) for size in (0, 2, 5, 7)]
        files = [Path(__file__), Path("src/researchops/knowledge/evidence_retrieval.py"),
                 Path("src/researchops/knowledge/query_planning.py"), Path("src/researchops/experiments/evidence_design.py"),
                 Path("src/researchops/knowledge/retrieval.py"), Path("src/researchops/evaluation/references.py"),
                 Path("internal_experiments/evaluation/protocols/RAG_CPU_V3_PROTOCOL.md")]
        pins = {"files": {str(p): file_hash(p) for p in files}, "model": self.calls.model.model,
                "qrels_sha256": self.reference_info["sha256"], "new_paper_order": self.new,
                "corpus_hash": canonical_hash([(r["chunk_id"], r["text"], r["source_hash"], r["paper_id"]) for r in self.store.all_chunks()])}
        marker = self.root / "snapshot.json"
        if marker.exists() and read(marker)["pins"] != pins:
            raise ValueError("Frozen CPU protocol changed; use a new evaluation version")
        if not marker.exists():
            write_json(marker, {"pins": pins, "reference": self.reference_info, "stages": [(n, sorted(ids)) for n, ids in self.stages],
                                "meaning": "development comparison; historical corpus, source downloads and failures preserved"})
            for path in files:
                relative = path.resolve().relative_to(Path(__file__).resolve().parents[1])
                dest = self.root / "source_snapshot" / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dest)

    def parallel(self, jobs, worker):
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = [pool.submit(worker, job) for job in jobs]
            failures = []
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as error:
                    failures.append(f"{type(error).__name__}: {error}")
            if failures:
                raise RuntimeError("; ".join(failures))

    def plan(self):
        def worker(q):
            path = self.root / "query_plans" / f"{q['id']}.json"
            if path.exists():
                return
            primary = read(self.source / "query_plans" / f"{q['id']}.json")
            write_json(path, plan_evidence_queries(q["text"], q["idea"], self.calls, primary=primary))
        self.parallel(self.queries, worker)

    def retrieve(self):
        old, improved = HybridRetriever(self.store), EvidenceRetriever(self.store)
        for stage, ids in self.stages:
            for q in self.queries:
                plan = read(self.root / "query_plans" / f"{q['id']}.json")
                for name, engine in [("legacy", old), ("cpu", improved)]:
                    path = self.root / "retrieval" / stage / q["id"] / f"{name}.json"
                    if path.exists():
                        continue
                    start = time.perf_counter()
                    if name == "legacy":
                        hits = engine.search(plan["retrieval_query"], limit=40, candidate_limit=60, paper_ids=ids)
                        diagnostics = {}
                    else:
                        hits = engine.search(plan["retrieval_query"], queries=[f["query"] for f in plan["facets"]],
                                             limit=40, candidate_limit=72, paper_ids=ids)
                        diagnostics = engine.last_diagnostics
                    assert all(h.paper_id in ids for h in hits)
                    write_json(path, {"query_id": q["id"], "stage": stage, "mode": name,
                                      "seconds": time.perf_counter() - start, "diagnostics": diagnostics,
                                      "hits": [h.to_dict() for h in hits]})
                progress(f"cpu retrieval {stage}/{q['id']}")
        repeat = []
        for q in self.queries:
            plan = read(self.root / "query_plans" / f"{q['id']}.json")
            hits = improved.search(plan["retrieval_query"], queries=[f["query"] for f in plan["facets"]], limit=40)
            previous = read(self.root / "retrieval" / "94" / q["id"] / "cpu.json")
            repeat.append({"query_id": q["id"], "identical_order": [h.chunk_id for h in hits] == [h["chunk_id"] for h in previous["hits"]],
                           "diagnostics": improved.last_diagnostics})
        write_json(self.root / "repeat.json", repeat)

    def generate(self):
        def worker(q):
            for mode in ["legacy", "retrieval_only", "complete"]:
                path = self.root / "answers" / q["id"] / f"{mode}.json"
                if path.exists():
                    continue
                hits = read(self.root / "retrieval" / "94" / q["id"] / ("legacy.json" if mode == "legacy" else "cpu.json"))["hits"]
                if mode == "complete":
                    packet_path = self.root / "packets" / f"{q['id']}.json"
                    if not packet_path.exists():
                        write_json(packet_path, build_evidence_packet(q["text"], q["idea"], hits, self.calls))
                    result = improve_research_answer(q["text"], q["idea"], read(packet_path), self.calls, ANSWER_SYSTEM,
                                                     retriever=EvidenceRetriever(self.store),
                                                     query_plan=read(self.root / "query_plans" / f"{q['id']}.json"))
                else:
                    passages, remaining = [], 10000
                    for hit in hits[:10]:
                        text = hit["text"][:remaining]
                        if not text:
                            break
                        passages.append({"chunk_id": hit["chunk_id"], "paper_id": hit["paper_id"],
                                         "title": hit["paper_title"], "text": text})
                        remaining -= len(text)
                    answer = self.calls.call("research_evidence_answer", ANSWER_SYSTEM,
                                             {"question": q["text"], "idea": q["idea"], "passages": passages}, 2400)
                    result = {"answer": answer, "passages": passages}
                if not isinstance(result["answer"].get("facts"), list) or not result["answer"].get("design"):
                    raise ValueError("Invalid scientific answer")
                write_json(path, result)
                progress(f"cpu answer {q['id']}/{mode}")
        self.parallel(self.queries, worker)

    def judge(self):
        def worker(q):
            path = self.root / "judgments" / f"{q['id']}.json"
            if path.exists():
                return
            modes = ["legacy", "retrieval_only", "complete"]
            random.Random(20261020 + sum(map(ord, q["id"]))).shuffle(modes)
            aliases = {f"A{i}": mode for i, mode in enumerate(modes)}
            records = {alias: read(self.root / "answers" / q["id"] / f"{mode}.json") for alias, mode in aliases.items()}
            candidates = [{"answer_id": alias, **record["answer"]} for alias, record in records.items()]
            sources = [{"answer_id": alias, **p} for alias, record in records.items() for p in record["passages"]]
            refs, seen = [], set()
            for ref in sorted(self.qrels["references"][q["id"]], key=lambda r: (-r["validated_grade"], r["paper_id"])):
                if ref["paper_id"] in seen:
                    continue
                seen.add(ref["paper_id"])
                row = self.store.get_chunk(ref["chunk_id"])
                refs.append({"chunk_id": ref["chunk_id"], "text": row["text"]})
                if len(refs) == 8:
                    break
            payload = {"question": q["text"], "answers": candidates, "independent_reference_passages": refs,
                       "cited_passages": sources,
                       "fact_index_contract": {alias: list(range(len(record["answer"]["facts"]))) for alias, record in records.items()}}
            for attempt in range(2):
                result = self.calls.call("cpu_blind_evaluation_v1" + ("_schema_retry" if attempt else ""),
                                         JUDGE_SYSTEM + "\n" + DESIGN_JUDGE + "\nCited passages are tagged by answer_id; assess citation support using only the passages supplied to that same answer. Obey the exact fact_index_contract.", payload, 7000)
                valid = set(a.get("answer_id") for a in result.get("answers", [])) == set(aliases)
                if valid:
                    for item in result["answers"]:
                        expected = len(records[item["answer_id"]]["answer"]["facts"])
                        valid &= sorted(f.get("index", -1) for f in item.get("facts", [])) == list(range(expected))
                        valid &= all(item.get(key) in range(5) for key in ["correctness", "question_coverage", "design_usefulness"])
                        valid &= all(item.get("design_dimensions", {}).get(key) in range(5) for key in DIMENSIONS)
                if valid:
                    break
            if not valid:
                raise ValueError("Blinded judgment schema invalid; original responses and costs retained")
            write_json(path, {"aliases": aliases, **result})
            progress(f"cpu judged {q['id']}")
        self.parallel(self.queries, worker)

    def report(self):
        details = []
        for stage, ids in self.stages:
            for q in self.queries:
                grades = self.qrels["paper_grades"][q["id"]]
                for mode in ["legacy", "cpu"]:
                    record = read(self.root / "retrieval" / stage / q["id"] / f"{mode}.json")
                    ranking = list(dict.fromkeys(h["paper_id"] for h in record["hits"]))[:10]
                    details.append({"stage": stage, "query_id": q["id"], "idea_id": q["idea_id"], "mode": mode,
                                    "seconds": record["seconds"], "ranking": ranking,
                                    "current": ranking_metrics(ranking, {p: g for p, g in grades.items() if p in ids}, 10),
                                    "fixed94": ranking_metrics(ranking, grades, 10),
                                    "old87": ranking_metrics(ranking, {p: g for p, g in grades.items() if p in self.initial}, 10),
                                    "new7": ranking_metrics(ranking, {p: g for p, g in grades.items() if p in self.new}, 10)})
        def mean(values):
            values = [v for v in values if v is not None]
            return statistics.mean(values) if values else None
        summary = {}
        for stage, _ in self.stages:
            summary[stage] = {}
            for mode in ["legacy", "cpu"]:
                rows = [r for r in details if r["stage"] == stage and r["mode"] == mode]
                summary[stage][mode] = {scope: {metric: mean(r[scope][metric] for r in rows) for metric in ["recall", "mrr", "ndcg"]}
                                        for scope in ["current", "fixed94", "old87", "new7"]}
                summary[stage][mode]["median_seconds"] = statistics.median(r["seconds"] for r in rows)
        answers = []
        for q in self.queries:
            judged = read(self.root / "judgments" / f"{q['id']}.json")
            for item in judged["answers"]:
                mode = judged["aliases"][item["answer_id"]]
                record = read(self.root / "answers" / q["id"] / f"{mode}.json")
                supplied = {p["chunk_id"] for p in record["passages"]}
                facts = record["answer"]["facts"]
                locatable = {i for i, f in enumerate(facts) if f.get("citations") and all(c in supplied for c in f["citations"])}
                supported = sum(f["index"] in locatable and f["evidence_support"] == "supported" for f in item["facts"])
                answers.append({"query_id": q["id"], "idea_id": q["idea_id"], "mode": mode,
                                "facts": len(facts), "supported": supported, "locatable": len(locatable),
                                **{k: item[k] for k in ["correctness", "question_coverage", "design_usefulness", "design_dimensions", "reason"]}})
        answer_summary = {}
        for mode in ["legacy", "retrieval_only", "complete"]:
            rows = [r for r in answers if r["mode"] == mode]
            answer_summary[mode] = {k: mean(r[k] for r in rows) for k in ["correctness", "question_coverage", "design_usefulness"]}
            answer_summary[mode]["design_dimensions"] = {k: mean(r["design_dimensions"][k] for r in rows) for k in DIMENSIONS}
            answer_summary[mode].update(facts=sum(r["facts"] for r in rows), supported=sum(r["supported"] for r in rows),
                                        fact_support_coverage=sum(r["supported"] for r in rows) / sum(r["facts"] for r in rows))
        regression = {}
        for mode in ["legacy", "cpu"]:
            first = summary["87"][mode]["old87"]
            stages = [{"stage": stage, "recall_delta": summary[stage][mode]["old87"]["recall"] - first["recall"],
                       "ndcg_delta": summary[stage][mode]["old87"]["ndcg"] - first["ndcg"]} for stage, _ in self.stages]
            regression[mode] = {"status": "passed" if all(r["recall_delta"] >= -0.03 and r["ndcg_delta"] >= -0.05 for r in stages) else "regression_detected",
                                "thresholds": {"recall_drop": 0.03, "ndcg_drop": 0.05}, "stages": stages}
        report = {"version": "cpu-rag-v3", "reference": self.reference_info, "queries": 12, "ideas": 6,
                  "retrieval": summary, "retrieval_details": details, "answers": answer_summary, "answer_details": answers,
                  "growth_regression": regression, "repeat": read(self.root / "repeat.json"), "cost": self.calls.state,
                  "training_executed": False, "cuda_required": False, "gold_used": False,
                  "limitation": "development silver evaluation; full treatment uses more internal calls; design answer evaluation is distinct from complete frozen-contract acceptance"}
        write_json(self.root / "report.json", report)
        progress(json.dumps({"report": str(self.root / "report.json"), "retrieval94": summary["94"], "answers": answer_summary,
                             "regression": regression, "charged_tokens": self.calls.state["total_tokens"]}, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, default=Path("workspace_rag_cpu_v3"))
    parser.add_argument("--phase", choices=["plan", "retrieve", "generate", "judge", "report", "all"], default="all")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--token-cap", type=int, default=3000000)
    args = parser.parse_args()
    load_project_env()
    os.environ["RESEARCHOPS_RETRIEVAL_BACKEND"] = "neural"
    os.environ["RESEARCHOPS_EMBEDDING_LOCAL_ONLY"] = "true"
    os.environ["RESEARCHOPS_RERANKER_LOCAL_ONLY"] = "true"
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    torch.set_num_threads(4)
    with workspace_lock(args.output):
        benchmark = CPUBenchmark(args)
        for phase in ["plan", "retrieve", "generate", "judge", "report"]:
            if args.phase in [phase, "all"]:
                getattr(benchmark, phase)()

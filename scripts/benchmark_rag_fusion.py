"""Frozen anchored-fusion development comparison; preserves old benchmark artifacts."""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sqlite3
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from benchmark_rag_incremental import ANSWER_SYSTEM, JUDGE_SYSTEM
from benchmark_rag_cpu import DESIGN_JUDGE, DIMENSIONS
from researchops.core.artifacts import canonical_hash, file_hash, write_json, workspace_lock, progress
from researchops.core.config import load_project_env
from researchops.core.storage import ResearchStore
from researchops.evaluation.rag import ConcurrentPaidCalls, ranking_metrics
from researchops.evaluation.references import load_reference
from researchops.harness.runtime import OpenAICompatibleModel
from researchops.knowledge.fusion_retrieval import fuse_candidates
from researchops.experiments.fusion_design import build_fusion_packet, improve_fusion_answer


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def average(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


class FusionBenchmark:
    def __init__(self, args):
        self.root, self.source = args.output.resolve(), Path("workspace_rag_cpu_v31").resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.workers = args.workers
        self.calls = ConcurrentPaidCalls(self.root, OpenAICompatibleModel(max_attempts=1, timeout=240), args.token_cap)
        suite = read("config/benchmarks/rag_incremental_v1.json")
        self.queries = [{**q, "idea_id": i["id"], "idea": i["idea"]} for i in suite["ideas"] for q in i["queries"]]
        source_snapshot = read(self.source / "snapshot.json")
        self.stages = [(name, set(ids)) for name, ids in source_snapshot["stages"]]
        self.qrels, self.reference = load_reference(silver=Path(source_snapshot["reference"]["path"]))
        project = Path(__file__).resolve().parents[1]
        paths = [Path(__file__), Path("scripts/benchmark_rag_incremental.py"), Path("scripts/benchmark_rag_cpu.py"),
                 Path("src/researchops/knowledge/fusion_retrieval.py"), Path("src/researchops/experiments/fusion_design.py"),
                 Path("src/researchops/experiments/evidence_design.py"), Path("src/researchops/evaluation/rag.py"),
                 Path("src/researchops/evaluation/references.py"), Path("src/researchops/knowledge/retrieval.py"),
                 Path("src/researchops/knowledge/evidence_retrieval.py"), Path("evaluation/protocols/RAG_FUSION_V1_PROTOCOL.md")]
        controls = sorted(self.source.glob("retrieval/*/*/*.json")) + sorted(self.source.glob("answers/*/*.json"))
        pins = {"code": {str(p.resolve().relative_to(project)): file_hash(p) for p in paths},
                "controls": {str(p.relative_to(self.source)): file_hash(p) for p in controls},
                "model": self.calls.model.model, "qrels": self.reference["sha256"],
                "query_plans": {p.name: file_hash(p) for p in self.source.glob("query_plans/*.json")},
                "source_snapshot": file_hash(self.source / "snapshot.json")}
        marker = self.root / "snapshot.json"
        if marker.exists():
            if read(marker)["pins"] != pins:
                raise ValueError("Frozen fusion inputs changed: use a new evaluation version")
        else:
            write_json(marker, {"pins": pins, "reference": self.reference, "stages": [(n, sorted(ids)) for n, ids in self.stages],
                                "source_workspace": str(self.source), "kind": "development comparison after observed v3 results"})
            for path in paths:
                target = self.root / "source_snapshot" / path.resolve().relative_to(project)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
            with sqlite3.connect(f"{(self.source / 'passages.db').as_uri()}?mode=ro", uri=True) as src:
                with sqlite3.connect(self.root / "passages.db") as dst:
                    src.backup(dst)
        self.store = ResearchStore(self.root / "passages.db")

    def parallel(self, jobs, worker):
        errors = []
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            pending = [pool.submit(worker, job) for job in jobs]
            for future in as_completed(pending):
                try:
                    future.result()
                except Exception as error:
                    errors.append(f"{type(error).__name__}: {error}")
        if errors:
            raise RuntimeError("; ".join(errors))

    def retrieve(self):
        previous_stage = None
        for stage, allowed in self.stages:
            def worker(q):
                target = self.root / "retrieval" / stage / f"{q['id']}.json"
                if target.exists():
                    return
                inputs = [read(self.source / "retrieval" / stage / q["id"] / f"{mode}.json")["hits"] for mode in ["legacy", "cpu"]]
                prior = read(self.root / "retrieval" / previous_stage / f"{q['id']}.json")["hits"] if previous_stage else []
                for hit in sum(inputs, []) + prior:
                    node = self.store.get_chunk(hit["chunk_id"])
                    assert node and node["text"] == hit["text"] and node["paper_id"] == hit["paper_id"]
                    assert hit["paper_id"] in allowed
                plan = read(self.source / "query_plans" / f"{q['id']}.json")
                hits, diagnostics = fuse_candidates(plan["retrieval_query"], q["text"], *inputs, self.calls, previous_hits=prior)
                write_json(target, {"query_id": q["id"], "stage": stage, "hits": [h.to_dict() for h in hits], "diagnostics": diagnostics})
                progress(f"fusion retrieved {stage}/{q['id']}")
            self.parallel(self.queries, worker)
            previous_stage = stage
        def independent(q):
            target = self.root / "retrieval_no_history" / f"{q['id']}.json"
            if target.exists():
                return
            inputs = [read(self.source / "retrieval/94" / q["id"] / f"{mode}.json")["hits"] for mode in ["legacy", "cpu"]]
            plan = read(self.source / "query_plans" / f"{q['id']}.json")
            hits, diagnostics = fuse_candidates(plan["retrieval_query"], q["text"], *inputs, self.calls)
            write_json(target, {"query_id": q["id"], "hits": [h.to_dict() for h in hits], "diagnostics": diagnostics})
        self.parallel(self.queries, independent)
        repeat = []
        for q in self.queries:
            inputs = [read(self.source / "retrieval/94" / q["id"] / f"{mode}.json")["hits"] for mode in ["legacy", "cpu"]]
            prior = read(self.root / "retrieval/92" / f"{q['id']}.json")["hits"]
            plan = read(self.source / "query_plans" / f"{q['id']}.json")
            hits, _ = fuse_candidates(plan["retrieval_query"], q["text"], *inputs, self.calls, previous_hits=prior)
            saved = read(self.root / "retrieval/94" / f"{q['id']}.json")
            repeat.append({"query_id": q["id"], "identical_order": [h.chunk_id for h in hits] == [h["chunk_id"] for h in saved["hits"]]})
        write_json(self.root / "repeat.json", repeat)

    def generate(self):
        def worker(q):
            directory = self.root / "answers" / q["id"]
            for mode, old_mode in [("legacy", "legacy"), ("cpu", "complete")]:
                target = directory / f"{mode}.json"
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(self.source / "answers" / q["id"] / f"{old_mode}.json", target)
            target = directory / "fusion.json"
            if target.exists():
                return
            # Answer uses the no-history ranking, matching the public default invocation.
            hits = read(self.root / "retrieval_no_history" / f"{q['id']}.json")["hits"]
            packet_path = self.root / "packets" / f"{q['id']}.json"
            if not packet_path.exists():
                write_json(packet_path, build_fusion_packet(q["text"], q["idea"], hits, self.calls))
            value = improve_fusion_answer(q["text"], q["idea"], read(packet_path), self.calls, ANSWER_SYSTEM)
            if not isinstance(value["answer"].get("facts"), list) or not value["answer"].get("design"):
                raise ValueError("Invalid fusion answer")
            write_json(target, value)
            progress(f"fusion answered {q['id']}")
        self.parallel(self.queries, worker)

    def judge(self):
        def worker(q):
            target = self.root / "judgments" / f"{q['id']}.json"
            if target.exists():
                return
            modes = ["legacy", "cpu", "fusion"]
            random.Random(20261021 + sum(map(ord, q["id"]))).shuffle(modes)
            aliases = {f"A{i}": m for i, m in enumerate(modes)}
            records = {a: read(self.root / "answers" / q["id"] / f"{m}.json") for a, m in aliases.items()}
            refs, seen = [], set()
            for ref in sorted(self.qrels["references"][q["id"]], key=lambda r: (-r["validated_grade"], r["paper_id"])):
                if ref["paper_id"] in seen:
                    continue
                seen.add(ref["paper_id"])
                row = self.store.get_chunk(ref["chunk_id"])
                refs.append({"chunk_id": ref["chunk_id"], "text": row["text"]})
                if len(refs) == 8:
                    break
            payload = {"question": q["text"], "answers": [{"answer_id": a, **r["answer"]} for a, r in records.items()],
                       "cited_passages": [{"answer_id": a, **p} for a, r in records.items() for p in r["passages"]],
                       "independent_reference_passages": refs,
                       "fact_index_contract": {a: list(range(len(r["answer"]["facts"]))) for a, r in records.items()}}
            valid = False
            for attempt in range(3):
                try:
                    value = self.calls.call(f"fusion_blind_judgment_v1_attempt{attempt}", JUDGE_SYSTEM + "\n" + DESIGN_JUDGE +
                                            "\nTag cited passages by answer_id; use only that answer's sources for support. Return exactly one JSON object. Put design_dimensions INSIDE each answer object. Obey exact fact_index_contract.", payload, 7000)
                    rows = value.get("answers", [])
                    valid = len(rows) == 3 and {r.get("answer_id") for r in rows} == set(aliases)
                    if valid:
                        for row in rows:
                            valid &= sorted(f.get("index", -1) for f in row.get("facts", [])) == payload["fact_index_contract"][row["answer_id"]]
                            valid &= all(type(row.get(k)) is int and row[k] in range(5) for k in ["correctness", "question_coverage", "design_usefulness"])
                            valid &= all(type(row.get("design_dimensions", {}).get(k)) is int and row["design_dimensions"][k] in range(5) for k in DIMENSIONS)
                    if valid:
                        break
                except ValueError:
                    pass
            if not valid:
                raise ValueError("All blind judgment format attempts failed; costs/raw responses preserved")
            write_json(target, {"aliases": aliases, **value})
            progress(f"fusion judged {q['id']}")
        self.parallel(self.queries, worker)

    def report(self):
        details = []
        for stage, allowed in self.stages:
            for q in self.queries:
                for mode in ["legacy", "cpu", "fusion"]:
                    path = (self.root / "retrieval" / stage / f"{q['id']}.json" if mode == "fusion" else
                            self.source / "retrieval" / stage / q["id"] / f"{mode}.json")
                    record = read(path)
                    ranking = list(dict.fromkeys(h["paper_id"] for h in record["hits"]))[:10]
                    grades = self.qrels["paper_grades"][q["id"]]
                    scopes = {"current": allowed, "fixed94": self.stages[-1][1], "old87": self.stages[0][1],
                              "new7": self.stages[-1][1] - self.stages[0][1]}
                    details.append({"stage": stage, "query_id": q["id"], "idea_id": q["idea_id"], "mode": mode,
                                    "ranking": ranking, **{scope: ranking_metrics(ranking, {p: g for p, g in grades.items() if p in ids}, 10) for scope, ids in scopes.items()}})
        summary = {stage: {mode: {scope: {key: average(r[scope][key] for r in details if r["stage"] == stage and r["mode"] == mode)
                                                          for key in ["recall", "mrr", "ndcg"]} for scope in ["current", "fixed94", "old87", "new7"]}
                          for mode in ["legacy", "cpu", "fusion"]} for stage, _ in self.stages}
        no_history = []
        for q in self.queries:
            record = read(self.root / "retrieval_no_history" / f"{q['id']}.json")
            ranking = list(dict.fromkeys(h["paper_id"] for h in record["hits"]))[:10]
            no_history.append({"query_id": q["id"], "idea_id": q["idea_id"], "ranking": ranking,
                               **ranking_metrics(ranking, self.qrels["paper_grades"][q["id"]], 10)})
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
                answers.append({"query_id": q["id"], "idea_id": q["idea_id"], "mode": mode, "facts": len(facts),
                                "supported": supported, "locatable": len(locatable),
                                **{k: item[k] for k in ["correctness", "question_coverage", "design_usefulness", "design_dimensions", "reason"]}})
        answer_summary = {}
        for mode in ["legacy", "cpu", "fusion"]:
            rows = [r for r in answers if r["mode"] == mode]
            answer_summary[mode] = {k: average(r[k] for r in rows) for k in ["correctness", "question_coverage", "design_usefulness"]}
            answer_summary[mode].update(facts=sum(r["facts"] for r in rows), supported=sum(r["supported"] for r in rows),
                                        fact_support_coverage=sum(r["supported"] for r in rows) / sum(r["facts"] for r in rows),
                                        design_dimensions={k: average(r["design_dimensions"][k] for r in rows) for k in DIMENSIONS})
        growth = {}
        for mode in ["legacy", "cpu", "fusion"]:
            first = summary["87"][mode]["old87"]
            deltas = [{"stage": stage, **{f"{k}_delta": summary[stage][mode]["old87"][k] - first[k] for k in ["recall", "ndcg"]}} for stage, _ in self.stages]
            growth[mode] = {"status": "passed" if all(r["recall_delta"] >= -0.03 and r["ndcg_delta"] >= -0.05 for r in deltas) else "regression_detected", "stages": deltas}
        no_history_summary = {k: average(r[k] for r in no_history) for k in ["recall", "mrr", "ndcg"]}
        seven = {k: {"legacy": summary["94"]["legacy"]["current"][k], "fusion": no_history_summary[k]} for k in ["recall", "mrr", "ndcg"]}
        seven.update({k: {"legacy": answer_summary["legacy"][k], "fusion": answer_summary["fusion"][k]} for k in ["correctness", "question_coverage", "design_usefulness", "fact_support_coverage"]})
        increased = [k for k, v in seven.items() if v["fusion"] > v["legacy"] + 1e-9]
        decreased = [k for k, v in seven.items() if v["fusion"] < v["legacy"] - 1e-9]
        result = {"version": "anchored-fusion-v1", "reference": self.reference, "retrieval": summary, "retrieval_details": details,
                  "no_history": no_history_summary, "no_history_details": no_history, "answers": answer_summary, "answer_details": answers,
                  "growth": growth, "repeat": read(self.root / "repeat.json"), "cost": self.calls.state,
                  "majority_check": {"metrics": seven, "increased": increased, "decreased": decreased,
                                     "passed": len(increased) >= 4 and not decreased},
                  "gold_used": False, "training_executed": False, "coding_executed": False,
                  "limitations": "development set; cached controls newly judged together; same-model silver; extra API tokens"}
        write_json(self.root / "report.json", result)
        progress(json.dumps({"retrieval94": summary["94"], "no_history": no_history_summary, "answers": answer_summary,
                             "majority": result["majority_check"], "growth": growth, "charged_tokens": self.calls.state["total_tokens"]}, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, default=Path("workspace_rag_fusion_v1"))
    parser.add_argument("--phase", choices=["retrieve", "generate", "judge", "report", "all"], default="all")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--token-cap", type=int, default=3000000)
    args = parser.parse_args()
    load_project_env()
    with workspace_lock(args.output):
        benchmark = FusionBenchmark(args)
        for phase in ["retrieve", "generate", "judge", "report"]:
            if args.phase in [phase, "all"]:
                getattr(benchmark, phase)()

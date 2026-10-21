"""Run a version-pinned pre-coding benchmark; never starts coding/training."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from researchops.core.config import load_project_env
from researchops.evaluation.precoding.runner import PrecodingRunner


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source", type=Path, default=Path("workspace_batch_vit_v2"))
    parser.add_argument("--output", type=Path, default=Path("workspace_precoding_v3"))
    parser.add_argument("--suite", type=Path, default=Path("config/benchmarks/precoding_v1.json"))
    parser.add_argument("--phase", choices=["pilot", "A", "B", "boundary", "boundary-batch", "checks", "all", "report", "replay", "replay-review"], default="pilot")
    parser.add_argument("--token-cap", type=int, default=300000)
    parser.add_argument("--use-frozen-code", action="store_true", help="Resume using the preserved source snapshot of this run")
    parser.add_argument("--robust", action="store_true", help="Live full-text checks and bounded critic/repair with two independent acceptance reviews")
    parser.add_argument("--resume-errors", action="store_true", help="Archive infrastructure error records and resume using the same ledger/call cache; never retry semantic failures")
    parser.add_argument("--reuse-acquisition-from", type=Path, help="Reuse actual saved acquisition stages from the identical suite; keep original run and costs intact")
    parser.add_argument("--task-ids", help="Comma-separated task IDs to run in track B; other scheduled tasks remain in the report denominator")
    args = parser.parse_args()
    load_project_env()
    os.environ["RESEARCHOPS_EMBEDDING_LOCAL_ONLY"] = "true"
    os.environ["RESEARCHOPS_RERANKER_LOCAL_ONLY"] = "true"
    os.environ["RESEARCHOPS_PDF_PARSER"] = "pymupdf"
    if args.phase == "boundary-batch":
        from researchops.evaluation.precoding.boundary import run_boundary_batch
        print(json.dumps(run_boundary_batch(args.output.resolve(), args.suite, args.token_cap), indent=2))
        return
    if args.phase in {"replay", "replay-review"}:
        from repair_precoding_contracts import replay
        from researchops.core.artifacts import workspace_lock
        with workspace_lock(args.output):
            print(json.dumps(replay(args.output.resolve(), review=args.phase == "replay-review"), indent=2))
        return
    runner_class = PrecodingRunner
    if args.robust:
        from researchops.evaluation.precoding.robust import RobustPrecodingRunner
        runner_class = RobustPrecodingRunner
    if args.use_frozen_code:
        import importlib
        frozen = args.output.resolve() / "source_snapshot" / "project" / "src"
        if not (frozen / "researchops" / "evaluation" / "precoding" / "runner.py").is_file():
            parser.error("No preserved source snapshot for this run")
        for name in list(sys.modules):
            if name == "researchops" or name.startswith("researchops."):
                del sys.modules[name]
        sys.path.insert(0, str(frozen))
        runner_class = (importlib.import_module("researchops.evaluation.precoding.robust").RobustPrecodingRunner
                        if args.robust else importlib.import_module("researchops.evaluation.precoding.runner").PrecodingRunner)
    runner = runner_class(source=args.source, output=args.output, suite_path=args.suite, cap=args.token_cap)
    if args.use_frozen_code and not runner.suite.get("boundary_cases"):
        # Presentation-only compatibility for historical runners that divided by
        # zero on an intentionally empty boundary track. Scientific code stays frozen.
        def compatible_report():
            from researchops.core.artifacts import write_json
            report = PrecodingRunner.report(runner)
            report["schema_version"] = runner.snapshot_data["pins"]["version"]
            report["report_recovery"] = "Current empty-track-safe formatter; frozen scientific pipeline unchanged"
            for group in report["tracks"].values():
                group["first_pass_success"] = {"passed": sum(bool(r.get("success") and r.get("first_pass_success") and not (args.output / 'infrastructure_recovery' / r['key']).exists()) for r in group["records"]), "total": len(group["records"])}
            write_json(args.output / "report.json", report)
            return report
        runner.report = compatible_report
    preserved = args.output.resolve() / "source_snapshot" / "project"
    if not preserved.exists():
        import shutil
        project = Path(__file__).resolve().parents[1]
        shutil.copytree(project / "src", preserved / "src", ignore=shutil.ignore_patterns("__pycache__"))
        (preserved / "scripts").mkdir(parents=True)
        shutil.copy2(__file__, preserved / "scripts" / "benchmark_precoding.py")
        for item in (args.suite, Path(runner.suite["idea_library"])):
            target = preserved / item
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)
    if args.reuse_acquisition_from:
        import sqlite3
        import time
        from researchops.core.artifacts import write_json, file_hash
        from researchops.core.storage import ResearchStore
        previous = args.reuse_acquisition_from.resolve()
        if previous == args.output.resolve():
            parser.error("Acquisition reuse needs a separate output; do not replace the historical run")
        prior_snapshot = json.loads((previous / "snapshot.json").read_text(encoding="utf-8"))
        for pin in ("suite_sha256", "library_sha256", "dataset_states"):
            if prior_snapshot["pins"][pin] != runner.snapshot_data["pins"][pin]:
                parser.error("Acquisition reuse requires identical suite, ideas and dataset states")
        with sqlite3.connect(f"{(previous / 'researchops.db').as_uri()}?mode=ro", uri=True) as src:
            with sqlite3.connect(args.output.resolve() / "researchops.db") as dst:
                src.backup(dst)
        runner.store = ResearchStore(args.output.resolve() / "researchops.db")
        runner.retriever.store = runner.store
        runner.rows = {r["chunk_id"]: dict(r) for r in runner.store.all_chunks()}
        imported = []
        for idea_id in runner.suite["tracks"]["B"]["idea_ids"]:
            original = previous / "acquisition" / f"{idea_id}.json"
            target = args.output / "acquisition" / f"{idea_id}.json"
            if original.exists() and not target.exists():
                record = json.loads(original.read_text(encoding="utf-8"))
                for paper_id in record["eligible_ids"]:
                    paper = runner.store.get_paper(paper_id)
                    if not paper or file_hash(paper["file_path"]) != paper["content_hash"]:
                        raise ValueError("Imported acquisition source missing or changed")
                record["reused_acquisition"] = {"workspace": str(previous), "record_sha256": file_hash(original),
                    "meaning": "Previously executed real search/download/parse/per-task relevance stage; not a fresh second discovery run"}
                write_json(target, record)
                imported.append({"idea_id": idea_id, "original_sha256": file_hash(original)})
        write_json(args.output / "acquisition_imports" / f"{time.time_ns()}.json", {
            "source": str(previous), "snapshot_sha256": file_hash(previous / "snapshot.json"),
            "database_sha256_after_backup": file_hash(args.output / "researchops.db"), "imports": imported,
            "cost_policy": "New calls use this ledger; real acquisition costs remain in source ledger and must be included in aggregate reporting"})
    recovered = {}
    if args.resume_errors:
        from researchops.core.artifacts import write_json
        import time
        for key, result in list(runner.results.items()):
            if result["status"] == "error":
                event = {"previous_result": result, "resumed_at": time.time(),
                         "reason": "Infrastructure error recovery; retain ledger and source/model/prompt hashes"}
                event_path = args.output / "infrastructure_recovery" / key / f"{time.time_ns()}.json"
                write_json(event_path, event)
                recovered[key] = str(event_path)
                del runner.results[key]
        write_json(runner.results_path, runner.results)
        # Preserve byte-for-byte evidence order for interrupted plan calls, so
        # process hash randomization cannot cause a duplicate model charge.
        saved_evidence = {}
        for key in recovered:
            evidence_path = args.output / "tasks" / key / "evidence.json"
            input_path = args.output / "tasks" / key / "input.json"
            if evidence_path.exists() and input_path.exists():
                task = json.loads(input_path.read_text(encoding="utf-8"))
                saved_evidence[task["idea_id"]] = json.loads(evidence_path.read_text(encoding="utf-8"))
        original_evidence = runner.evidence
        def resumed_evidence(idea, *positional, **kwargs):
            if idea.get("idea_id") in saved_evidence:
                return saved_evidence[idea["idea_id"]]
            return original_evidence(idea, *positional, **kwargs)
        runner.evidence = resumed_evidence
    if args.task_ids:
        from researchops.core.artifacts import workspace_lock
        if args.phase != "B":
            parser.error("--task-ids currently selects only live track B tasks")
        expected = {e["idea_id"]: e for e in runner.suite["case_expectations"]}
        with workspace_lock(args.output):
            for idea_id in args.task_ids.split(","):
                if idea_id not in runner.suite["tracks"]["B"]["idea_ids"]:
                    parser.error("Selected task is not scheduled in track B")
                if args.reuse_acquisition_from and not (args.output / "acquisition" / f"{idea_id}.json").exists():
                    raise ValueError(f"Acquisition checkpoint not yet available for {idea_id}; source task remains pending")
                acquisition = runner.live_acquire(runner.ideas[idea_id])
                runner.plan_task("B_" + idea_id, runner.ideas[idea_id], expected=expected[idea_id], acquisition=acquisition)
            report = runner.report()
    else:
        report = runner.run(args.phase)
    if recovered:
        from researchops.core.artifacts import write_json
        for key, event in recovered.items():
            if key in runner.results:
                record = runner.results[key]
                record["first_pass_plan_success"] = record.get("first_pass_success", False)
                record["first_pass_success"] = False
                record["infrastructure_recovery_event"] = event
                write_json(args.output / "tasks" / key / "result.json", record)
        write_json(runner.results_path, runner.results)
        report = runner.report()
    print(json.dumps({"report": str(args.output / 'report.json'), "cost": report["cost"],
                      "tracks": {k: v["success_at_budget"] for k, v in report["tracks"].items()}}, indent=2))


if __name__ == "__main__":
    main()

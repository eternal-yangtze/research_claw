"""P4 engineering verification, including a real process exit between two commits."""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from researchops.core.artifacts import file_hash, write_json, progress
from researchops.core.models import Paper, Chunk
from researchops.core.storage import ResearchStore
from researchops.evaluation.precoding.gates import CRITERIA
from researchops.experiments.study_workflow import refine_study
from researchops.harness.runtime import ModelTurn, ToolCall
from researchops.knowledge.retrieval import HashingEmbeddingBackend, HybridRetriever, LexicalReranker
from researchops.orchestration.study_calls import StudyCalls
from researchops.orchestration.study_graph import StudyWorkflow


def fixture(root):
    states = {}
    for dataset in ("dtd", "pets"):
        path = root / "data" / dataset
        manifest, subset = path / "manifest.jsonl", path / "subset.json"
        if not manifest.exists():
            path.mkdir(parents=True, exist_ok=True)
            manifest.write_text("\n".join(json.dumps({"sample_id": f"{dataset}-{split}",
                "split": split, "included": True, "content_sha256": f"{dataset}-{split}"})
                for split in ("train", "val", "test")), encoding="utf-8")
            write_json(subset, [f"{dataset}-train"])
        states[dataset] = {"dataset_id": dataset, "status": "verified", "verification": {
            "dataset_id": dataset, "name": dataset, "root": str(path), "class_count": 2,
            "split_counts": {"train": 1, "val": 1, "test": 1},
            "manifest_path": str(manifest), "manifest_sha256": file_hash(manifest),
            "train_fractions": {"0.25": {"path": str(subset), "sha256": file_hash(subset), "count": 1}},
            "exclusions": {}, "split_seed": 42, "official_test_preserved": True}}
    store = ResearchStore(root / "corpus.db")
    if not store.all_chunks():
        store.upsert_paper(Paper("p4-paper", "Low rank method fixture"))
        store.replace_chunks("p4-paper", [Chunk("p4-c1", "p4-paper", "Method",
            "Low rank adaptation trains low rank updates while freezing the backbone.")])
    proposal = {"status": "proposed", "task_family": "image_classification",
        "hypothesis": "unproven rank cost tradeoff", "research_question": "Compare rank 2 and 8",
        "evidence_claims": [{"kind": "paper_fact", "claim": "Low rank updates adapt frozen backbones.", "citations": ["p4-c1"]}],
        "design": {"datasets": ["dtd", "pets"], "fractions": [0.25], "seeds": [42, 43, 44],
            "arms": [{"arm_id": f"r{r}", "method": "lora", "rank": r, "alpha": r,
                      "dropout": 0.0, "weight_decay": 0.05, "target_blocks": [9, 10, 11]} for r in (2, 8)],
            "primary_metrics": ["accuracy"], "secondary_metrics": ["trainable_parameters"],
            "analysis_plan": "Report paired seed differences and uncertainty.",
            "success_rule": "No winner is assumed before measurement.",
            "selection_policy": {"mode": "ordinary", "objective": "validation_accuracy"},
            "controls": ["equal data and tuning budgets"], "limitations": ["three seeds"]}}
    evidence = [{"chunk_id": "p4-c1", "paper_id": "p4-paper", "text": store.get_chunk("p4-c1")["text"]}]
    return states, store, proposal, evidence


class ScriptedModel:
    model = "p4-engineering-scripted-v1"

    def __init__(self, proposal, scenario, *, use_tools=True):
        self.proposal, self.scenario = proposal, scenario
        self.use_tools = use_tools
        self.calls = 0

    def complete_bounded(self, messages, tools, max_tokens, *, timeout):
        self.calls += 1
        system = messages[0]["content"]
        payload = json.loads(messages[1]["content"])
        planner = "research planner for" in system
        if planner:
            if self.scenario == "boundary":
                value = {"status": "needs_adapter", "task_family": "other", "reason": "Time-series forecasting has no verified task adapter.",
                         "alternative": {"requires_constraint_change": True, "changed_constraints": ["task family"]}}
            elif self.scenario == "mechanical_block":
                value = copy.deepcopy(self.proposal)
                value["design"]["primary_metrics"] = ["invented_metric"]
            elif not self.use_tools:
                value = copy.deepcopy(self.proposal)
            else:
                observations = [json.loads(m["content"]) for m in messages if m["role"] == "tool"]
                if not observations:
                    return ModelTurn(tool_calls=(ToolCall("search", "search_study_evidence", {"query": "low rank frozen backbone", "limit": 3}),), usage={"total_tokens": 20})
                if len(observations) == 1:
                    return ModelTurn(tool_calls=(ToolCall("read", "get_study_chunk", {"chunk_id": "p4-c1"}),
                        ToolCall("compile", "compile_study_candidate", {"proposal": self.proposal})), usage={"total_tokens": 20})
                value = copy.deepcopy(self.proposal)
                if "revision_request" in payload:
                    value["design"]["analysis_plan"] += " Decision rule clarified after critic feedback."
        else:
            value = {"criteria": {k: "pass" for k in CRITERIA}, "reasons": ["Engineering fixture review"],
                     "citation_support": [] if self.scenario == "boundary" else [{"chunk_id": "p4-c1", "verdict": "supported"}]}
            if self.scenario == "revision" and "final independent reviewer" not in system and "clarified" not in payload["proposal"]["design"]["analysis_plan"]:
                value["criteria"]["metric_and_decision_rule"] = "fail"
                value["reasons"] = ["Clarify the prospective outcome rule"]
        return ModelTurn(text=json.dumps(value), usage={"total_tokens": 20})


def run_case(root, scenario, mode="langgraph", crash=False):
    states, store, proposal, evidence = fixture(root)
    model = ScriptedModel(proposal, scenario)
    calls = StudyCalls(root, model, 500000)
    retriever = HybridRetriever(store, embedding=HashingEmbeddingBackend(), reranker=LexicalReranker())
    def after(node, patch):
        if crash and node == "planner":
            write_json(root / "crash_marker.json", {"after_node": node, "pid": os.getpid(),
                "budget": calls.summary(), "expected_exit": 86})
            os._exit(86)
    started = time.monotonic()
    engine = StudyWorkflow(idea={"description": "Compare low rank updates" if scenario != "boundary" else "Forecast time series"},
        states=states, store=store, calls=calls, directory=root / "study", evidence=evidence,
        retriever=retriever, orchestration=mode, after_node=after if crash else None)
    result = engine.run()
    summary = {"result": result, "provider_calls_this_process": model.calls, "budget": calls.summary(),
               "wall_seconds": time.monotonic() - started,
               "formal_contract": (engine.directory / "frozen_spec.json").exists(), "pid": os.getpid()}
    write_json(root / "case_report.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("all", "crash", "resume"), default="all")
    args = parser.parse_args()
    root = args.output.resolve()
    if args.phase != "all":
        result = run_case(root, "success", crash=args.phase == "crash")
        progress(json.dumps({"success": result["result"]["success"], "calls": result["provider_calls_this_process"]}))
        return 0
    root.mkdir(parents=True, exist_ok=False)
    cases = {scenario: run_case(root / scenario, scenario) for scenario in
             ("success", "revision", "boundary", "mechanical_block")}
    control = run_case(root / "native_control", "revision", mode="native")
    # Also compare the accepted proposal against the unchanged legacy workflow.
    states, store, proposal, evidence = fixture(root / "legacy_control")
    legacy_calls = StudyCalls(root / "legacy_control", ScriptedModel(proposal, "success", use_tools=False), 500000)
    legacy = refine_study(idea={"description": "Compare low rank updates"}, states=states,
        evidence=evidence, store=store, calls=legacy_calls, directory=root / "legacy_control" / "study")
    recovery_root = root / "process_recovery"
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--output", str(recovery_root)]
    crashed = subprocess.run(command + ["--phase", "crash"], capture_output=True, text=True)
    resumed = subprocess.run(command + ["--phase", "resume"], capture_output=True, text=True)
    if resumed.returncode:
        raise RuntimeError(resumed.stderr[-4000:])
    marker = json.loads((recovery_root / "crash_marker.json").read_text(encoding="utf-8"))
    recovery = json.loads((recovery_root / "case_report.json").read_text(encoding="utf-8"))
    repeated = subprocess.run(command + ["--phase", "resume"], capture_output=True, text=True)
    terminal = json.loads((recovery_root / "case_report.json").read_text(encoding="utf-8"))
    checks = {
        "success_and_mechanical_gate": cases["success"]["result"]["success"] and cases["success"]["result"]["mechanical"]["passed"],
        "revision_after_critic": len(cases["revision"]["result"]["attempts"]) == 2 and cases["revision"]["result"]["success"],
        "boundary_no_formal_contract": cases["boundary"]["result"]["boundary_handled"] and not cases["boundary"]["formal_contract"],
        "mechanical_gate_not_waived": not cases["mechanical_block"]["result"]["success"] and not cases["mechanical_block"]["formal_contract"],
        "micro_native_same_outcome": all(cases["revision"]["result"][k] == control["result"][k]
                                          for k in ("attempts", "mechanical", "planned_jobs", "success")),
        "micro_native_same_call_count": cases["revision"]["budget"]["root"]["model_calls"] == control["budget"]["root"]["model_calls"],
        "legacy_native_gate_agreement": legacy["success"] == cases["success"]["result"]["success"]
                                       and legacy["planned_jobs"] == cases["success"]["result"]["planned_jobs"],
        "actual_process_exit": crashed.returncode == 86 and marker["pid"] != recovery["pid"],
        "recovery_only_remaining_reviews": recovery["provider_calls_this_process"] == 2 and recovery["result"]["success"],
        "no_rebill_completed_planner": recovery["budget"]["root"]["charged_tokens"] == 100,
        "terminal_new_process_no_calls": repeated.returncode == 0 and terminal["provider_calls_this_process"] == 0,
    }
    report = {"passed": all(checks.values()), "checks": checks, "cases": cases,
              "native_control": control, "legacy_control": legacy, "process_recovery": recovery,
              "terminal_resume": terminal, "crash_marker": marker,
              "paid_model_calls": 0, "limitations": "Scripted model + synthetic data + hashing vectors for engineering fault injection. Real service/model checks are separate."}
    write_json(root / "report.json", report)
    progress(json.dumps({"passed": report["passed"], "checks": checks, "report": str(root / "report.json")}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Attach the existing coding loop to one root budget and parent trace."""
from __future__ import annotations

import json
import copy
from pathlib import Path

from researchops.core.artifacts import write_json, canonical_hash, file_hash
from researchops.harness.budget import BudgetedModel, ScopedBudgetedModel
from researchops.harness.runtime import TraceStore


class CodingCancelled(RuntimeError):
    pass


def check_cancelled(app):
    if app.cancellation is not None and app.cancellation.is_set():
        raise CodingCancelled("coding_cancelled")


def attach(app):
    check_cancelled(app)
    root = app.model
    if isinstance(root, ScopedBudgetedModel):
        root = root.parent
    if not isinstance(root, BudgetedModel):
        directory = app.run_dir / "budget"
        root = BudgetedModel(root, directory, app.plan["budget"]["max_total_tokens"],
            max_model_turns=app.plan["budget"]["max_model_turns"], resume=(directory / "ledger.json").is_file())
        root.budget["provider_token_margin"] = app.plan["budget"].get("provider_token_margin", 0)
    scope_id = "coding_" + app.plan["plan_id"]
    app.model = root.scope(scope_id, max_model_turns=app.plan["budget"]["max_model_turns"],
                           max_total_tokens=app.plan["budget"]["max_total_tokens"])
    app.root_budget = root
    path = app.run_dir / "harness_binding.json"
    binding = {"ledger_path": str(root.path.resolve()), "node_run_id": scope_id,
               "parent_run_id": app.parent_run_id, "parent_session_id": app.parent_session_id}
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != binding:
        raise ValueError("Coding resume root budget or parent trace changed")
    checkpoint = app.run_dir / "checkpoint.json"
    # Older coding checkpoints own paid usage already. Import it once into the
    # new root ledger instead of silently giving a historical run a fresh cap.
    if not path.exists() and checkpoint.exists():
        old = json.loads(checkpoint.read_text(encoding="utf-8"))
        if old.get("total_tokens", 0):
            with root._lock:
                node = root.state["scopes"][scope_id]
                if node["call_ids"]:
                    raise ValueError("Ambiguous legacy coding budget association")
                if root.state["total_tokens"] + old["total_tokens"] > root.state["token_cap"]:
                    raise ValueError("Legacy coding spend exceeds remaining parent cap")
                if root.state["turns"] + old["turns"] > root.state["max_model_turns"]:
                    raise ValueError("Legacy coding turns exceed remaining parent cap")
                imported = copy.deepcopy(old.get("call_ledger", []))
                if not imported:
                    imported = [{"status": "legacy_usage_unknown", "upper_tokens": old["total_tokens"]}]
                for index, call in enumerate(imported):
                    call.update(call_id="legacy_"+canonical_hash([scope_id, index, file_hash(checkpoint)])[:24],
                                node_run_id=scope_id, imported_from=str(checkpoint))
                    root.state["call_ledger"].append(call)
                    node["call_ids"].append(call["call_id"])
                root.state["turns"] += old["turns"]
                root.state["total_tokens"] += old["total_tokens"]
                node.update(turns=old["turns"], charged_tokens=old["total_tokens"])
                root._persist()
    write_json(path, binding)
    # Unknown dispatches must be reconciled, never silently reissued on resume.
    scoped_calls = [c for c in root.state["call_ledger"] if c.get("node_run_id") == scope_id]
    if any(c.get("status") != "settled" for c in scoped_calls):
        raise ValueError("Coding has unresolved provider usage; inspect the original ledger before resuming")
    TraceStore(app.workspace / "runs").emit(app.parent_run_id or app.plan["plan_id"],
        "coding.child.started", app.parent_session_id or "coding", binding)
    return binding


def finalize(app, report, binding):
    summary = app.root_budget.summary()
    report = {**report, "harness": {**binding, "budget": summary,
        "accounting": "Root ledger is aggregate cost; local coding ledger is a limit view, never add them together.",
        "provider_call_ids": summary["scopes"][binding["node_run_id"]]["call_ids"]}}
    write_json(app.run_dir / "report.json", report)
    TraceStore(app.workspace / "runs").emit(app.parent_run_id or app.plan["plan_id"],
        "coding.child.completed", app.parent_session_id or "coding", {
            **binding, "status": report["status"], "report_path": report["report_path"],
            "cache_hit": report.get("cache_hit", False)})
    return report

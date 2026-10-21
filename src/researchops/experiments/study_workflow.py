"""Reusable pre-coding critic/repair workflow. No benchmark answers are inputs."""
from __future__ import annotations

from researchops.core.artifacts import file_hash, write_json, progress
from researchops.evaluation.precoding.gates import RUBRIC, mechanical_gate, silver_pass
from researchops.experiments.planning import ROBUST_PLAN_SYSTEM, HOST_PROTOCOL, compile_contract, dataset_catalog, planning_interface
from researchops.papers.validation.automatic import formula_evidence_policy, formula_reproduction_gate


WORKFLOW_ROLE_CONTEXT = """
You are one role inside a host-executed research workflow, not its tool-calling agent.
The host supplies retrieved evidence and verified data, compiles proposals, runs the
independent critics and mechanical gates, and saves real artifact paths afterwards.
User requests for those workflow operations are host responsibilities, not missing
research prerequisites or instructions for this role to call tools. Author a complete
supported design as status=proposed; that status never claims gates or experiments
passed. Assess the proposal and compiled protocol; pending host orchestration alone
is not research infeasibility. Keep genuine missing data/evidence, unsupported design,
scientific constraints and failed mechanical checks as blockers; never waive a gate.
"""


def refine_study(*, idea, states, evidence, store, calls, directory, max_attempts=4, evidence_brief=None):
    directory.mkdir(parents=True, exist_ok=True)
    original_evidence = evidence
    evidence = [{**node, "formula_policy": formula_evidence_policy(store.get_chunk(node["chunk_id"]))}
                for node in evidence if store.get_chunk(node["chunk_id"]) is not None
                and formula_evidence_policy(store.get_chunk(node["chunk_id"]))["eligible_for_planning"]]
    formula_gate = {"withheld_chunk_ids": [n["chunk_id"] for n in original_evidence
                                          if n["chunk_id"] not in {e["chunk_id"] for e in evidence}],
                    "rule": "Unresolved formula nodes cannot establish paper facts; source recovery remains silver and requires original-page review.",
                    "safe_as_sole_reproduction_evidence": False}
    write_json(directory / "formula_evidence_gate.json", formula_gate)
    write_json(directory / "input.json", idea)
    write_json(directory / "evidence.json", evidence)
    base = {"task": idea, "available_data": dataset_catalog(states),
            "host_protocol": HOST_PROTOCOL, "runtime_interface": planning_interface(states),
            "fulltext_evidence": evidence}
    base["formula_evidence_policy"] = formula_gate
    if evidence_brief is not None:
        base["proposed_study_blueprint"] = evidence_brief
    attempts, feedback = [], None
    proposal, spec, review, confirm = {}, None, {}, {}
    for index in range(max_attempts):
        payload = dict(base)
        if feedback:
            payload["revision_request"] = feedback
        attempt_dir = directory / "attempts" / str(index + 1)
        try:
            proposal = calls.call("study_planner" if index == 0 else "study_revision",
                                  ROBUST_PLAN_SYSTEM + WORKFLOW_ROLE_CONTEXT, payload, 6500)
        except (ValueError, TimeoutError) as error:
            attempts.append({"attempt": index + 1, "success": False, "generation_error": str(error)})
            feedback = {"format_error": str(error), "instruction": "Return a complete single JSON object with the required schema."}
            continue
        write_json(attempt_dir / "proposal.json", proposal)
        spec, compile_error = None, None
        if proposal.get("status") == "proposed":
            try:
                spec = compile_contract(idea, proposal, states, {
                    "evidence": {"path": str(directory / "evidence.json"), "sha256": file_hash(directory / "evidence.json")}})
            except (ValueError, KeyError, TypeError) as error:
                compile_error = f"{type(error).__name__}: {error}"
        gate = mechanical_gate(spec, evidence, store)
        formula_check = formula_reproduction_gate(spec, evidence, store)
        write_json(attempt_dir / "formula_gate.json", formula_check)
        gate["checks"]["formulas_not_sole_reproduction_evidence"] = formula_check["passed"]
        gate["passed"] = gate["passed"] and formula_check["passed"]
        gate["formula_violations"] = formula_check["violations"]
        summary = None if spec is None else {
            "host_protocol": HOST_PROTOCOL, "methods": spec["methods"], "metrics": spec["metrics"],
            "design": spec["design"], "jobs": spec["run_budget"], "mechanical_gate": gate,
            "datasets": [{"dataset_id": d["dataset_id"], "manifest_sha256": d["manifest_sha256"],
                          "fractions": list(d["train_fractions"])} for d in spec["datasets"]]}
        review_payload = {"task": idea, "proposal": proposal, "evidence": evidence,
                          "host_protocol": HOST_PROTOCOL,
                          "compiled_summary": summary, "compiler_error": compile_error}
        review = calls.call("study_critic", RUBRIC + WORKFLOW_ROLE_CONTEXT, review_payload, 2300)
        write_json(attempt_dir / "review.json", review)
        boundary = proposal.get("status") in {"blocked", "needs_adapter", "needs_refinement"}
        candidate_ok = (gate["passed"] or boundary) and silver_pass(review)
        confirm = {}
        if candidate_ok:
            confirm = calls.call("study_final_reviewer", RUBRIC + WORKFLOW_ROLE_CONTEXT + "\nYou are the final independent reviewer. No previous review is supplied. Check the entire plan, especially evidence and operational decision rules; do not assume acceptance.", review_payload, 2300)
            write_json(attempt_dir / "confirmation.json", confirm)
        accepted = candidate_ok and silver_pass(confirm)
        attempt = {"attempt": index + 1, "mechanical": gate, "compile_error": compile_error,
                   "critic_pass": silver_pass(review), "confirmation_pass": silver_pass(confirm), "success": accepted}
        attempts.append(attempt)
        if spec:
            write_json(attempt_dir / "frozen_spec.json", spec)
        write_json(directory / "attempts.json", attempts)
        progress(f"study={idea.get('idea_id')} attempt={index+1} mechanical={gate['passed']} critic={silver_pass(review)} final={silver_pass(confirm)}")
        if accepted:
            break
        feedback = {"previous_proposal": proposal, "compiler_error": compile_error,
                    "mechanical_checks": gate, "critic": review, "final_review": confirm,
                    "instruction": "Repair every substantive issue while preserving the user's research question and constraints. Supply the complete revised JSON; never claim a result."}
    write_json(directory / "proposal.json", proposal)
    write_json(directory / "review.json", review)
    write_json(directory / "confirmation.json", confirm)
    if spec:
        write_json(directory / "frozen_spec.json", spec)
    accepted = bool(attempts and attempts[-1]["success"])
    return {"success": accepted, "silver_accepted": accepted,
            "formula_evidence_gate": formula_gate,
            "first_pass_success": bool(attempts and attempts[0]["success"]), "attempts": attempts,
            "mechanical": attempts[-1].get("mechanical", {}) if attempts else {},
            "proposal_status": proposal.get("status"), "boundary_handled": accepted and spec is None,
            "planned_jobs": spec["run_budget"]["total"] if spec else None,
            "citation_support": confirm.get("citation_support", []),
            "citations_total": len((spec or {}).get("baseline_citations", []))}

"""Durable StateGraph study workflow; the host retains gates and budget ownership."""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, TypedDict

from researchops.core.artifacts import canonical_hash, file_hash, write_json, workspace_lock
from researchops.evaluation.precoding.gates import mechanical_gate, silver_pass
from researchops.experiments.planning import HOST_PROTOCOL, compile_contract, dataset_catalog, planning_interface
from researchops.experiments.study_agents import StudyRoles, StudyRoleStopped, VERSION as ROLE_VERSION
from researchops.harness.runtime import TraceStore
from researchops.knowledge.retrieval import HybridRetriever, LexicalReranker
from researchops.papers.validation.automatic import formula_evidence_policy, formula_reproduction_gate

VERSION = "study-graph-v03-1"


class StudyState(TypedDict, total=False):
    input_hash: str
    run_id: str
    attempt: int
    evidence: list[dict]
    evidence_brief: Any
    formula_gate: dict
    proposal: dict
    spec: Any
    gate: dict
    compile_error: Any
    review: dict
    confirmation: dict
    feedback: Any
    attempts: list[dict]
    candidate_ok: bool
    accepted: bool
    fatal: Any
    result: dict


class StudyWorkflow:
    def __init__(self, *, idea, states, store, calls, directory, evidence=None,
                 evidence_brief=None, retrieve=None, retriever=None, max_attempts=4,
                 orchestration="langgraph", after_node=None):
        if orchestration not in {"langgraph", "native"} or not 1 <= max_attempts <= 4:
            raise ValueError("Use langgraph/native and one to four candidate versions")
        self.idea, self.states, self.store, self.calls = idea, states, store, calls
        self.directory, self.max_attempts = Path(directory).resolve(), max_attempts
        self.mode, self.after_node = orchestration, after_node
        self.initial_evidence, self.evidence_brief = evidence or [], evidence_brief
        self.retrieve = retrieve
        self.traces = TraceStore(self.directory / "runs")
        self.rows = {r["chunk_id"]: dict(r) for r in store.all_chunks()
                     if formula_evidence_policy(r)["eligible_for_planning"]}
        self.retriever = retriever or HybridRetriever(store, reranker=LexicalReranker())
        self.roles = StudyRoles(calls=calls, store=store, states=states, rows=self.rows,
                                retriever=self.retriever, directory=self.directory,
                                traces=self.traces, compile_candidate=self.compile_candidate)
        self.base = {"task": idea, "available_data": dataset_catalog(states),
                     "host_protocol": HOST_PROTOCOL, "runtime_interface": planning_interface(states)}
        self.binding = {
            "version": VERSION, "roles": ROLE_VERSION, "project": str(store.path.resolve()),
            "idea": idea, "datasets": states, "corpus_hash": canonical_hash([dict(r) for r in store.all_chunks()]),
            "initial_evidence": self.initial_evidence, "evidence_brief": evidence_brief,
            "mode": orchestration, "model": calls.owner.model, "study_cap": calls.cap,
            "max_model_turns": calls.max_model_turns, "max_attempts": max_attempts,
            "vector_backend": self.retriever.vector_backend.name,
            "vector_binding": {k: getattr(self.retriever.vector_backend, k, None)
                               for k in ("project_id", "corpus_version", "alias", "embedding_key")},
            "code": {name: file_hash(path) for name, path in {
                "graph": Path(__file__), "roles": Path(__file__).parents[1] / "experiments" / "study_agents.py",
                "compiler": Path(__file__).parents[1] / "experiments" / "planning.py",
                "gates": Path(__file__).parents[1] / "evaluation" / "precoding" / "gates.py",
                "calls": Path(__file__).with_name("study_calls.py"),
                "micro": Path(__file__).parents[1] / "harness" / "micro.py",
                "formula_gate": Path(__file__).parents[1] / "papers" / "validation" / "automatic.py",
            }.items()},
        }
        # Dataset files are external to SQLite; changing one invalidates resume
        # even if the graph already has an accepted candidate.
        self.binding["actual_data_hashes"] = {
            str(p): file_hash(p) for value in states.values()
            for p in [Path(value["verification"]["manifest_path"])] + [
                Path(s["path"]) for s in value["verification"]["train_fractions"].values()]
        }
        self.input_hash = canonical_hash(self.binding)
        self.run_id = "study_" + self.input_hash[:24]

    def compile_candidate(self, proposal):
        evidence = self._candidate_evidence(proposal)
        spec, error = None, None
        if proposal.get("status") == "proposed":
            try:
                spec = compile_contract(self.idea, proposal, self.states, {
                    "input_snapshot": {"path": str(self.directory / "binding.json"),
                                       "sha256": file_hash(self.directory / "binding.json")}})
            except (ValueError, KeyError, TypeError, OSError) as exc:
                error = f"{type(exc).__name__}: {exc}"
        gate = mechanical_gate(spec, evidence, self.store)
        formula = formula_reproduction_gate(spec, evidence, self.store)
        gate["checks"]["formulas_not_sole_reproduction_evidence"] = formula["passed"]
        gate["passed"] = gate["passed"] and formula["passed"]
        gate["formula_violations"] = formula["violations"]
        return spec, gate, error

    def _candidate_evidence(self, proposal):
        ids = {node["chunk_id"] for node in self.initial_evidence}
        for claim in proposal.get("evidence_claims", []):
            if isinstance(claim, dict):
                ids.update(c for c in claim.get("citations", []) if isinstance(c, str))
        return [{"chunk_id": cid, "paper_id": self.rows[cid]["paper_id"],
                 "text": self.rows[cid]["text"], "page": self.rows[cid]["page"],
                 "title": self.store.get_paper(self.rows[cid]["paper_id"])["title"],
                 "formula_policy": formula_evidence_policy(self.rows[cid])}
                for cid in sorted(ids) if cid in self.rows]

    def _retrieve(self, state):
        nodes, brief = (self.retrieve() if self.retrieve else (self.initial_evidence, self.evidence_brief))
        self.initial_evidence = nodes
        eligible = [n for n in nodes if n["chunk_id"] in self.rows]
        formula = {"withheld_chunk_ids": [n["chunk_id"] for n in nodes if n["chunk_id"] not in self.rows],
                   "rule": "Unresolved formulas cannot establish paper facts.",
                   "safe_as_sole_reproduction_evidence": False}
        write_json(self.directory / "input.json", self.idea)
        write_json(self.directory / "evidence.json", eligible)
        write_json(self.directory / "formula_evidence_gate.json", formula)
        return {"evidence": eligible, "evidence_brief": brief, "formula_gate": formula}

    def _planner(self, state):
        self.initial_evidence = state["evidence"]
        attempt = state.get("attempt", 0) + 1
        payload = {**self.base, "fulltext_evidence": state["evidence"],
                   "formula_evidence_policy": state["formula_gate"]}
        if state.get("evidence_brief") is not None:
            payload["proposed_study_blueprint"] = state["evidence_brief"]
        if state.get("feedback"):
            payload["revision_request"] = state["feedback"]
        try:
            proposal, runtime = self.roles.run("planner", attempt, payload, self.run_id)
        except StudyRoleStopped as exc:
            return {"attempt": attempt, "fatal": exc.result.to_dict(), "accepted": False}
        write_json(self.directory / "attempts" / str(attempt) / "proposal.json", proposal)
        return {"attempt": attempt, "proposal": proposal, "review": {}, "confirmation": {},
                "accepted": False, "fatal": None, "candidate_ok": False}

    def _compile_gate(self, state):
        self.initial_evidence = state["evidence"]
        spec, gate, error = self.compile_candidate(state["proposal"])
        evidence = self._candidate_evidence(state["proposal"])
        path = self.directory / "attempts" / str(state["attempt"])
        write_json(path / "mechanical_gate.json", gate)
        if spec:
            write_json(path / "evidence.json", evidence)
            spec["sources"]["evidence"] = {"path": str(path / "evidence.json"),
                                           "sha256": file_hash(path / "evidence.json")}
            write_json(path / "candidate_spec.json", spec)
        return {"spec": spec, "gate": gate, "compile_error": error, "evidence": evidence}

    def _review(self, state, role):
        self.initial_evidence = state["evidence"]
        spec = state.get("spec")
        summary = None if not spec else {
            "host_protocol": HOST_PROTOCOL, "methods": spec["methods"], "metrics": spec["metrics"],
            "design": spec["design"], "jobs": spec["run_budget"], "mechanical_gate": state["gate"],
            "datasets": [{"dataset_id": d["dataset_id"], "manifest_sha256": d["manifest_sha256"],
                          "fractions": list(d["train_fractions"])} for d in spec["datasets"]]}
        # This payload is rebuilt for both reviewers: neither receives the
        # previous critic, planner session, or revision_request.
        payload = {"task": self.idea, "proposal": state["proposal"], "evidence": state["evidence"],
                   "host_protocol": HOST_PROTOCOL, "compiled_summary": summary,
                   "compiler_error": state.get("compile_error")}
        try:
            review, runtime = self.roles.run(role, state["attempt"], payload, self.run_id)
        except StudyRoleStopped as exc:
            return {"fatal": exc.result.to_dict(), "accepted": False}
        name = "review" if role == "critic" else "confirmation"
        write_json(self.directory / "attempts" / str(state["attempt"]) / f"{name}.json", review)
        patch = {name: review}
        if role == "critic":
            boundary = state["proposal"].get("status") in {"blocked", "needs_adapter", "needs_refinement"}
            patch["candidate_ok"] = bool((state["gate"]["passed"] or boundary) and silver_pass(review))
        return patch

    def _decide(self, state):
        accepted = bool(not state.get("fatal") and state.get("candidate_ok") and silver_pass(state.get("confirmation", {})))
        attempt = {"attempt": state["attempt"], "mechanical": state.get("gate", {}),
                   "compile_error": state.get("compile_error"), "critic_pass": silver_pass(state.get("review", {})),
                   "confirmation_pass": silver_pass(state.get("confirmation", {})), "success": accepted}
        attempts = state.get("attempts", []) + [attempt]
        write_json(self.directory / "attempts.json", attempts)
        feedback = {"previous_proposal": state.get("proposal", {}), "compiler_error": state.get("compile_error"),
                    "mechanical_checks": state.get("gate", {}), "critic": state.get("review", {}),
                    "final_review": state.get("confirmation", {}),
                    "instruction": "Repair substantive issues, preserve the question and constraints, return the complete JSON."}
        return {"accepted": accepted, "attempts": attempts, "feedback": feedback}

    def _submit(self, state):
        spec, accepted = state.get("spec"), state.get("accepted", False)
        proposal = state.get("proposal", {})
        for filename, value in (("proposal", proposal), ("review", state.get("review", {})),
                                ("confirmation", state.get("confirmation", {})), ("evidence", state["evidence"])):
            write_json(self.directory / f"{filename}.json", value)
        if accepted and spec and state["gate"]["passed"]:
            write_json(self.directory / "frozen_spec.json", spec)
        result = {"success": accepted, "silver_accepted": accepted,
                  "formula_evidence_gate": state["formula_gate"],
                  "first_pass_success": bool(state["attempts"] and state["attempts"][0]["success"]),
                  "attempts": state["attempts"], "mechanical": state.get("gate", {}),
                  "proposal_status": proposal.get("status"), "boundary_handled": accepted and spec is None,
                  "planned_jobs": spec["run_budget"]["total"] if spec else None,
                  "citation_support": state.get("confirmation", {}).get("citation_support", []),
                  "citations_total": len((spec or {}).get("baseline_citations", [])),
                  "orchestration": self.mode, "run_id": self.run_id, "thread_id": self.run_id,
                  "input_hash": self.input_hash, "stop_reason": (state.get("fatal") or {}).get("stop_reason", "accepted" if accepted else "candidate_limit"),
                  "budget": self.calls.summary(), "training_executed": False, "research_success": None}
        write_json(self.directory / "workflow_result.json", result)
        artifacts = [self.directory / name for name in ("proposal.json", "review.json", "confirmation.json", "evidence.json", "workflow_result.json")]
        if (self.directory / "frozen_spec.json").exists():
            artifacts.append(self.directory / "frozen_spec.json")
        write_json(self.directory / "commit.json", {"input_hash": self.input_hash,
            "artifacts": {str(p.relative_to(self.directory)): file_hash(p) for p in artifacts}})
        return {"result": result}

    def _node(self, name, action):
        def execute(state):
            key = canonical_hash([name, state])
            path = self.directory / "node_commits" / f"{name}_{key}.json"
            reused = path.exists()
            if reused:
                patch = json.loads(path.read_text(encoding="utf-8"))["patch"]
            else:
                started = time.monotonic()
                patch = action(state)
                write_json(path, {"node": name, "input_hash": key, "patch": patch,
                                  "seconds": time.monotonic() - started})
            self.traces.emit(self.run_id, "study.graph.node", self.run_id,
                             {"node": name, "attempt": patch.get("attempt", state.get("attempt")),
                              "thread_id": self.run_id, "reused_commit": reused})
            if self.after_node:
                self.after_node(name, patch)
            return patch
        return execute

    def _next_after_decide(self, state):
        return "submit" if state.get("accepted") or state.get("fatal") or state["attempt"] >= self.max_attempts else "planner"

    def run(self, *, interrupt_after=None):
        with workspace_lock(self.directory):
            path = self.directory / "binding.json"
            if path.exists() and json.loads(path.read_text(encoding="utf-8")) != self.binding:
                raise ValueError("Study graph inputs/code/model/limits changed; choose a new output workspace")
            write_json(path, self.binding)
            receipt = self.directory / "commit.json"
            if receipt.exists():
                record = json.loads(receipt.read_text(encoding="utf-8"))
                if record["input_hash"] != self.input_hash or any(file_hash(self.directory / p) != h for p, h in record["artifacts"].items()):
                    raise ValueError("Committed study artifact changed")
            initial = {"run_id": self.run_id, "input_hash": self.input_hash, "attempt": 0, "attempts": [], "accepted": False}
            nodes = {"retrieve": self._retrieve, "planner": self._planner, "compile_gate": self._compile_gate,
                     "critic": lambda s: self._review(s, "critic"), "final_review": lambda s: self._review(s, "final_review"),
                     "decide": self._decide, "submit": self._submit}
            if self.mode == "native":
                state, node = initial, "retrieve"
                while node:
                    state.update(self._node(node, nodes[node])(state))
                    node = {"retrieve": "planner", "planner": "submit" if state.get("fatal") else "compile_gate",
                            "compile_gate": "decide" if state.get("compile_error") else "critic", "critic": "submit" if state.get("fatal") else "final_review" if state.get("candidate_ok") else "decide",
                            "final_review": "decide", "decide": self._next_after_decide(state), "submit": None}[node]
                return state["result"]
            try:
                from langgraph.graph import StateGraph, START, END
                from langgraph.checkpoint.sqlite import SqliteSaver
            except ImportError as exc:
                raise RuntimeError("LangGraph mode requires the agent-v03 extra; select native for the legacy workflow") from exc
            with sqlite3.connect(self.directory / "graph_checkpoints.sqlite", check_same_thread=False) as connection:
                graph = StateGraph(StudyState)
                for name, action in nodes.items():
                    graph.add_node(name, self._node(name, action))
                graph.add_edge(START, "retrieve")
                graph.add_edge("retrieve", "planner")
                graph.add_conditional_edges("planner", lambda s: "submit" if s.get("fatal") else "compile_gate", ["submit", "compile_gate"])
                graph.add_conditional_edges("compile_gate", lambda s: "decide" if s.get("compile_error") else "critic", ["decide", "critic"])
                graph.add_conditional_edges("critic", lambda s: "submit" if s.get("fatal") else "final_review" if s.get("candidate_ok") else "decide", ["submit", "final_review", "decide"])
                graph.add_edge("final_review", "decide")
                graph.add_conditional_edges("decide", self._next_after_decide, ["submit", "planner"])
                graph.add_edge("submit", END)
                compiled = graph.compile(checkpointer=SqliteSaver(connection), interrupt_after=interrupt_after)
                write_json(self.directory / "graph_structure.json", compiled.get_graph().to_json())
                mermaid = compiled.get_graph().draw_mermaid()
                # Keep all generated artifacts atomic using the existing JSON writer.
                write_json(self.directory / "graph_mermaid.json", {"mermaid": mermaid})
                config = {"configurable": {"thread_id": self.run_id}, "recursion_limit": 48}
                previous = compiled.get_state(config)
                if previous.values and not previous.next:
                    return previous.values["result"]
                state = compiled.invoke(None if previous.values else initial, config)
                pending = compiled.get_state(config).next
                if pending:
                    return {"status": "paused", "next": list(pending), "thread_id": self.run_id,
                            "budget": self.calls.summary()}
                return state["result"]

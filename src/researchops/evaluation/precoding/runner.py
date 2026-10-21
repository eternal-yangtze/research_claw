"""Frozen-corpus and live-cache-assisted evaluation with one persistent ledger.

All model calls (including reviews and failed requests) go through admission.
The evaluator manifest is never included in planner requests or method cards.
"""
from __future__ import annotations

import copy
import json
import os
import re
import sqlite3
import time
from dataclasses import asdict
from pathlib import Path

from researchops.coding.budget import BudgetAdmissionError, reserve_call, settle_call
from researchops.core.artifacts import canonical_hash, file_hash, write_json, workspace_lock, progress
from researchops.core.storage import ResearchStore
from researchops.core.structured import parse_single_object
from researchops.evaluation.precoding.gates import RUBRIC, case_gate, mechanical_gate, silver_pass
from researchops.experiments.planning import (PLAN_SYSTEM, HOST_PROTOCOL, compile_contract, dataset_catalog,
                                             read_verified_datasets)
from researchops.harness.runtime import OpenAICompatibleModel
from researchops.knowledge.retrieval import HybridRetriever, tokenize
from researchops.papers.connectors import LiteratureSearch, download_pdf, papers_match
from researchops.papers.ingestion import DocumentIngestor
from researchops.pipelines.batch import _paper_from_dict

VERSION = "precoding-runner-v4.1"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class PaidCalls:
    def __init__(self, root: Path, model, cap: int):
        self.root, self.model = root, model
        self.path = root / "ledger.json"
        self.state = read_json(self.path) if self.path.exists() else {
            "turns": 0, "total_tokens": 0, "call_ledger": [], "cache_hits": 0,
            "token_cap": cap, "wall_seconds": 0.0}
        if self.state["token_cap"] != cap:
            raise ValueError("Cannot reset/increase an existing run budget; use the original cap")
        self.budget = {"max_model_turns": 1000, "max_total_tokens": cap,
                       "provider_token_margin": getattr(model, "provider_token_margin", 0)}

    def call(self, role: str, system: str, payload: dict, output_tokens: int = 2200):
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}]
        key = canonical_hash([VERSION, role, self.model.model, messages, output_tokens])
        cache = self.root / "calls" / f"{key}.json"
        if cache.exists():
            value = read_json(cache)
            self.state["cache_hits"] += 1
            write_json(self.path, self.state)
            return value["parsed"]
        lease = reserve_call(self.state, self.budget, messages, output_tokens)
        lease.update(role=role, input_hash=key, started_at=time.time())
        write_json(self.path, self.state)
        # Save the request before dispatch; interrupted/unknown calls retain reservation.
        request_path = self.root / "calls" / f"{key}.request.json"
        write_json(request_path, {"model": self.model.model, "messages": messages,
                                  "max_output_tokens": output_tokens, "lease": lease})
        started = time.monotonic()
        try:
            turn = self.model.complete_bounded(messages, [], output_tokens, timeout=240)
            lease["usage"] = turn.usage or {}
            write_json(cache.with_suffix(".response.json"), {"text": turn.text, "usage": turn.usage})
            settle_call(self.state, lease, turn.usage or {})
            parsed = parse_single_object(turn.text)
            write_json(cache, {"parsed": parsed, "usage": turn.usage, "request_hash": key})
            return parsed
        except Exception as error:
            lease["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            lease["wall_seconds"] = round(time.monotonic() - started, 3)
            write_json(self.path, self.state)


class PrecodingRunner:
    def __init__(self, *, source: Path, output: Path, suite_path: Path, cap: int = 300000, model=None):
        self.source, self.root = source.resolve(), output.resolve()
        if self.source == self.root or self.source in self.root.parents:
            raise ValueError("Use a separate benchmark workspace, outside the preserved source workspace")
        self.suite_path = suite_path.resolve()
        self.suite = read_json(suite_path)
        self.ideas = {i["idea_id"]: i for i in read_json(self.suite["idea_library"])["ideas"]}
        self.root.mkdir(parents=True, exist_ok=True)
        self.calls = PaidCalls(self.root, model or OpenAICompatibleModel(max_attempts=1), cap)
        self.states = read_verified_datasets(self.source)
        self.store = self.snapshot()
        self.retriever = HybridRetriever(self.store)
        self.rows = {r["chunk_id"]: dict(r) for r in self.store.all_chunks()}
        self.results_path = self.root / "results.json"
        self.results = read_json(self.results_path) if self.results_path.exists() else {}

    def snapshot(self):
        path = self.root / "snapshot.json"
        pins = {"version": VERSION, "suite_sha256": file_hash(self.suite_path),
                "library_sha256": file_hash(self.suite["idea_library"]),
                "planner_prompt": canonical_hash(PLAN_SYSTEM), "rubric": canonical_hash(RUBRIC),
                "model": self.calls.model.model,
                "code": {str(p.relative_to(Path(__file__).parents[2])): file_hash(p)
                         for directory in (Path(__file__).parent, Path(__file__).parents[2] / "experiments")
                         for p in sorted(directory.glob("*.py"))},
                "dataset_states": {k: canonical_hash(v) for k, v in self.states.items()}}
        db = self.root / "researchops.db"
        if path.exists():
            snapshot = read_json(path)
            if snapshot["pins"] != pins:
                raise ValueError("Frozen run inputs/code/model changed; use a new versioned output workspace")
        else:
            if db.exists():
                raise ValueError("Incomplete snapshot; do not overwrite existing database")
            with sqlite3.connect(f"{(self.source / 'researchops.db').as_uri()}?mode=ro", uri=True) as src:
                with sqlite3.connect(db) as dst:
                    src.backup(dst)
            store = ResearchStore(db)
            rows = store.all_chunks()
            ids = sorted({r["paper_id"] for r in rows})
            papers = [dict(store.get_paper(p)) for p in ids]
            valid = []
            for paper in papers:
                p = Path(paper["file_path"] or "")
                if p.is_file() and file_hash(p) == paper["content_hash"]:
                    valid.append(paper["paper_id"])
            snapshot = {"pins": pins, "source_workspace": str(self.source), "created_at": time.time(),
                        "paper_ids": valid, "papers": papers,
                        "source_hashes": [p["content_hash"] for p in papers],
                        "chunks_hash": canonical_hash([(r["chunk_id"], r["source_hash"], r["text"]) for r in rows]),
                        "historical_cost": "not rebilled; original corpus costs excluded and unknown here",
                        "parser_warning": "Existing digital PDFs parsed by PyMuPDF; formulas remain review_required"}
            write_json(path, snapshot)
        self.snapshot_data = snapshot
        return ResearchStore(db)

    def method_cards(self):
        """Extractive, source-pinned cards; no idea-specific relevance or paid re-reading."""
        cards = []
        for paper_id in self.snapshot_data["paper_ids"]:
            p = dict(self.store.get_paper(paper_id))
            rows = [dict(r) for r in self.store.paper_chunks(paper_id)]
            key = canonical_hash([p["content_hash"], p["parser_version"],
                                  [(r["chunk_id"], r["text"]) for r in rows], "extractive-method-card-v1"])
            path = self.root / "method_cards" / f"{key}.json"
            if path.exists():
                cards.append(read_json(path))
                continue
            pattern = r"LoRA|low.rank|visual prompt|bottleneck|adapter|linear prob|calibrat|dataset"
            ranked = sorted(rows, key=lambda r: -len(re.findall(pattern, r["text"], re.I)))
            card = {"paper_id": paper_id, "title": p["title"], "source_sha256": p["content_hash"],
                    "parser_version": p["parser_version"], "cache_key": key,
                    "type": "extractive_navigation_not_semantic_truth",
                    "nodes": [{"chunk_id": r["chunk_id"], "page": r["page"], "text": r["text"][:1000]}
                              for r in ranked[:3]]}
            write_json(path, card)
            cards.append(card)
        return cards

    def evidence(self, idea, mode="full", paper_ids=None):
        ids = set(paper_ids or self.snapshot_data["paper_ids"])
        query = idea.get("research_question", idea.get("description", ""))
        if mode == "bm25":
            rows = [r for r in self.rows.values() if r["paper_id"] in ids]
            scores = self.retriever._bm25(query, rows)
            hits = [{**r, "score": scores.get(r["chunk_id"], 0)} for r in rows]
            hits.sort(key=lambda r: -r["score"])
        elif mode == "abstract":
            # Abstracts are explicitly marked; they never pass as full-text formula evidence.
            hits = []
            terms = set(tokenize(query))
            for paper_id in ids:
                p = dict(self.store.get_paper(paper_id))
                hits.append({"paper_id": paper_id, "chunk_id": f"abstract:{paper_id}",
                             "text": p.get("abstract") or "Abstract unavailable", "page": None,
                             "score": len(terms & set(tokenize(p["title"] + ' ' + (p.get("abstract") or ''))))})
            hits.sort(key=lambda r: -r["score"])
        else:
            hits = [h.to_dict() for h in self.retriever.search(query, limit=16, candidate_limit=36, paper_ids=ids)]
        selected, counts, size = [], {}, 0
        for hit in hits:
            paper_id = hit["paper_id"]
            if counts.get(paper_id, 0) >= 2:
                continue
            paper = dict(self.store.get_paper(paper_id))
            node = {"paper_id": paper_id, "title": paper["title"], "chunk_id": hit["chunk_id"],
                    "page": self.rows.get(hit["chunk_id"], {}).get("page"),
                    "source_sha256": paper["content_hash"], "source_type": "abstract" if mode == "abstract" else "fulltext",
                    "text": hit["text"][:750]}
            length = len(json.dumps(node, ensure_ascii=False))
            if size + length > 6500:
                continue
            selected.append(node)
            counts[paper_id] = counts.get(paper_id, 0) + 1
            size += length
            if len(selected) >= 6:
                break
        return selected

    def plan_task(self, key, idea, *, expected=None, mode="full", acquisition=None, boundary=False):
        if key in self.results and self.results[key]["status"] not in {"unstarted", "running"}:
            return self.results[key]
        started, tokens_before = time.monotonic(), self.calls.state["total_tokens"]
        record = {"key": key, "idea_id": idea["idea_id"], "mode": mode, "boundary": boundary,
                  "status": "running", "started_at": time.time(), "input_hash": canonical_hash(idea)}
        self.results[key] = record
        write_json(self.results_path, self.results)
        directory = self.root / "tasks" / key
        write_json(directory / "input.json", idea)
        try:
            evidence = [] if boundary else self.evidence(idea, mode, acquisition.get("eligible_ids") if acquisition else None)
            write_json(directory / "evidence.json", evidence)
            payload = {"task": idea, "available_data": dataset_catalog(self.states),
                       "host_protocol": HOST_PROTOCOL, "fulltext_evidence": evidence}
            proposal = self.calls.call("planner", PLAN_SYSTEM, payload)
            write_json(directory / "proposal.json", proposal)
            spec = None
            if proposal.get("status") == "proposed":
                spec = compile_contract(idea, proposal, self.states, {
                    "snapshot": {"path": str(self.root / 'snapshot.json'), "sha256": file_hash(self.root / 'snapshot.json')},
                    "evidence": {"path": str(directory / 'evidence.json'), "sha256": file_hash(directory / 'evidence.json')}})
                write_json(directory / "frozen_spec.json", spec)
            gate = mechanical_gate(spec, evidence, self.store)
            constraints = case_gate(spec, expected) if expected else {"passed": True, "checks": {}}
            review_payload = {"task": idea, "proposal": proposal, "evidence": evidence,
                              "compiled_summary": None if spec is None else {
                                  "host_protocol": HOST_PROTOCOL,
                                  "jobs": spec["run_budget"], "mechanical_gate": gate,
                                  "datasets": [{"dataset_id": d["dataset_id"], "manifest_sha256": d["manifest_sha256"],
                                                "fractions": {k: {"sha256": v["sha256"], "count": v["count"]}
                                                              for k, v in d["train_fractions"].items()}}
                                               for d in spec["datasets"]],
                                  "optimizer_groups": spec["training"]["optimizer_groups"],
                                  "method_definitions": spec["methods"],
                                  "metric_definitions": spec["metrics"],
                                  "saved_outputs": spec["metrics"]["saved_outputs"]}}
            # The reviewer sees expectations, but the planner never sees evaluator files.
            if expected:
                review_payload["evaluator_checks"] = expected
            review = self.calls.call("independent_reviewer", RUBRIC, review_payload, 1200)
            write_json(directory / "review.json", review)
            accepted = silver_pass(review)
            acquisition_ok = acquisition is None or acquisition["passed"]
            safe_boundary = boundary and proposal.get("status") in {"blocked", "needs_adapter", "needs_refinement"}
            success = (safe_boundary if boundary else gate["passed"] and constraints["passed"] and acquisition_ok) and accepted
            record.update(status="passed" if success else "failed", mechanical=gate, constraints=constraints,
                          silver_accepted=accepted, boundary_handled=safe_boundary and accepted,
                          success=success, planned_jobs=spec["run_budget"]["total"] if spec else None,
                          primary_metrics=spec["metrics"]["primary"] if spec else None,
                          arms=proposal.get("design", {}).get("arms"), proposal_status=proposal.get("status"),
                          acquisition=acquisition, evidence_count=len(evidence),
                          citations_valid=sum(gate["checks"].get("citation_locators_and_text", False)
                                              for _ in (spec or {}).get("baseline_citations", [])),
                          citations_total=len((spec or {}).get("baseline_citations", [])),
                          citation_support=review.get("citation_support", []))
        except Exception as error:
            record.update(status="budget_exhausted" if isinstance(error, BudgetAdmissionError) else "error",
                          error=f"{type(error).__name__}: {error}", success=False)
        record.update(wall_seconds=round(time.monotonic() - started, 3),
                      charged_tokens=self.calls.state["total_tokens"] - tokens_before)
        write_json(directory / "result.json", record)
        write_json(self.results_path, self.results)
        progress(json.dumps({k: record.get(k) for k in ("key", "status", "charged_tokens", "planned_jobs", "error")}))
        return record

    def live_acquire(self, idea):
        """Fresh per-source discovery; bounded downloads; no old task relevance is reused."""
        path = self.root / "acquisition" / f"{idea['idea_id']}.json"
        if path.exists():
            return read_json(path)
        started = time.monotonic()
        search = LiteratureSearch(self.store, timeout=18)
        outputs = []
        for item in idea.get("search_queries", [])[:2]:
            outputs.append(search.search(item["query"], "precoding_" + idea["idea_id"],
                                         limit_per_source=8, sources=[item["source"]]))
        candidates = {}
        for out in outputs:
            for paper in out["papers"]:
                candidates[paper["paper_id"]] = paper
        # Eligibility and relevance are new judgments, including difficult negative candidates.
        source_ids = set(self.snapshot_data["paper_ids"])
        originals = {p["paper_id"]: p for p in self.snapshot_data["papers"]}
        candidates_list = list(candidates.values())
        for paper_id in source_ids:
            p = originals[paper_id]
            if not any(papers_match(_paper_from_dict(c), self.row_paper(p)) for c in candidates_list):
                candidates_list.append(asdict(self.row_paper(p)))
        selection = self.calls.call("live_relevance", """Select relevant papers for this research problem.
Return JSON {"selected_ids":["id"],"rejected_ids":["id"],"reasons":["specific evidence limitations"]}.
Choose at most 14 papers supporting the research methods/protocol; include promising NEW open
full texts when relevant. Task-adjacent VLM/segmentation papers are inspiration, not direct
classification baselines. Reject unrelated hard negatives. Metadata is for selection only.""",
            {"task": idea, "candidates": [{"paper_id": p["paper_id"], "title": p["title"],
                                          "abstract": (p.get("abstract") or "")[:450],
                                          "cached_fulltext": p["paper_id"] in source_ids,
                                          "open_pdf": bool(p.get("pdf_url"))} for p in candidates_list]}, 1300)
        selected = set(selection.get("selected_ids", []))
        eligible, downloads = [], []
        tried = 0
        for value in sorted(candidates_list, key=lambda p: p["paper_id"] in source_ids):
            if value["paper_id"] not in selected:
                continue
            paper = _paper_from_dict(value)
            original = next((p for p in self.snapshot_data["papers"] if papers_match(paper, self.row_paper(p))), None)
            if original and original["paper_id"] in source_ids:
                eligible.append(original["paper_id"])
                continue
            if not paper.pdf_url or tried >= 4:
                continue
            tried += 1
            try:
                result = download_pdf(paper.pdf_url, self.root / "artifacts" / "papers", filename=paper.paper_id + ".pdf")
                pdf_path = result.get("path") or result.get("file_path")
                ingested = DocumentIngestor(self.store).ingest_file(pdf_path, paper)
                parsed = dict(self.store.get_paper(ingested.paper_id))
                new = parsed["content_hash"] not in self.snapshot_data["source_hashes"]
                eligible.append(ingested.paper_id)
                downloads.append({"paper_id": paper.paper_id, "status": "parsed", "new_unique": new,
                                  "sha256": parsed["content_hash"], "download": result, "ingested": ingested.to_dict()})
            except Exception as error:
                downloads.append({"paper_id": paper.paper_id, "status": "failed", "error": f"{type(error).__name__}: {error}"})
        eligible = sorted(set(eligible))
        target = self.suite["tracks"]["B"]["target_valid_fulltexts_per_idea"]
        minimum = self.suite["tracks"]["B"]["min_new_unique_fulltexts"][idea["idea_id"]]
        new_count = len({d["sha256"] for d in downloads if d["status"] == "parsed" and d["new_unique"]})
        statuses = [s for o in outputs for s in o["sources"]]
        record = {"outputs": outputs, "selection": selection, "candidate_count": len(candidates),
                  "eligible_ids": eligible, "valid_fulltexts": len(eligible), "target": target,
                  "new_unique_fulltexts": new_count, "min_new_unique": minimum, "downloads": downloads,
                  "cached_fulltexts": len(set(eligible) & source_ids),
                  "fresh_search_succeeded": any(s["status"] == "success" for s in statuses),
                  "wall_seconds": round(time.monotonic() - started, 3)}
        record["passed"] = record["fresh_search_succeeded"] and len(eligible) >= target and new_count >= minimum
        write_json(path, record)
        self.rows = {r["chunk_id"]: dict(r) for r in self.store.all_chunks()}
        return record

    @staticmethod
    def row_paper(p):
        from researchops.core.models import Paper
        return Paper(paper_id=p["paper_id"], title=p["title"], doi=p.get("doi"),
                     abstract=p.get("abstract"), pdf_url=p.get("pdf_url"), source=p["source"],
                     external_id=p.get("external_id"), content_hash=p.get("content_hash"))

    def run(self, phase="all"):
        with workspace_lock(self.root):
            self.method_cards()
            expected = {e["idea_id"]: e for e in self.suite["case_expectations"]}
            ids = self.suite["pilot_ids"] if phase == "pilot" else list(self.ideas)
            if phase in {"pilot", "A", "all"}:
                for idea_id in ids:
                    self.plan_task("A_" + idea_id, self.ideas[idea_id], expected=expected[idea_id])
            if phase in {"boundary", "all"}:
                for case in self.suite["boundary_cases"]:
                    self.plan_task(case["case_id"], {"idea_id": case["case_id"], "description": case["prompt"]}, boundary=True)
            if phase in {"B", "all"}:
                for idea_id in self.suite["tracks"]["B"]["idea_ids"]:
                    key = "B_" + idea_id
                    if key in self.results:
                        continue
                    before = self.calls.state["total_tokens"]
                    try:
                        acquisition = self.live_acquire(self.ideas[idea_id])
                        r = self.plan_task(key, self.ideas[idea_id], expected=expected[idea_id], acquisition=acquisition)
                        r["acquisition_charged_tokens"] = self.calls.state["total_tokens"] - before - r["charged_tokens"]
                    except Exception as error:
                        self.results[key] = {"key": key, "status": "error", "success": False,
                                             "error": f"{type(error).__name__}: {error}"}
                    write_json(self.results_path, self.results)
            if phase in {"checks", "all"}:
                self.contrastive_and_ablations()
            return self.report()

    def contrastive_and_ablations(self):
        # New user messages override prior input constraints explicitly. No expected
        # response fields are fed to the planner, and each request has its own hash.
        mutations = [
            ("vit_rank_tradeoff", "Update the study: use rank candidates 4 and 8 ONLY instead of 2,8,16; this overrides every earlier rank list.", "rank", {4, 8}),
            ("vit_vpt_prompt_length", "Update the study: use prompt lengths 5 and 10 ONLY instead of 5,10,20; this overrides every earlier length list.", "prompt_length", {5, 10}),
            ("vit_low_label_lora_vs_full", "Update the study: use 25% labels ONLY instead of 10%; this overrides every earlier label budget.", "fraction", {0.25}),
            ("vit_calibration_comparison", "Update the study: NLL is the SOLE primary metric; keep ECE and accuracy as secondary; update the prospective interpretation rule.", "primary", {"nll"}),
        ]
        checks = []
        for idea_id, change, field, target in mutations:
            task = copy.deepcopy(self.ideas[idea_id])
            task["updated_user_instruction"] = change
            key = "C_" + idea_id
            result = self.plan_task(key, task)
            path = self.root / "tasks" / key / "frozen_spec.json"
            changed = False
            if path.exists():
                spec = read_json(path)
                if field == "fraction":
                    changed = set(spec["design"]["fractions"]) == target
                elif field == "primary":
                    changed = set(spec["metrics"]["primary"]) == target
                else:
                    changed = {a[field] for a in spec["methods"].values() if field in a} == target
                base = self.root / "tasks" / ("A_" + idea_id) / "frozen_spec.json"
                changed &= base.exists() and canonical_hash(read_json(base)) != canonical_hash(spec)
                if field != "primary" and base.exists():
                    changed &= spec["run_budget"]["total"] != read_json(base)["run_budget"]["total"] if field != "fraction" else True
            checks.append({"key": key, "required_change_observed": changed,
                           "success": changed and result.get("success", False)})
        # Four fixed dev tasks. Same model, planner/rubric and input character cap;
        # evidence representation alone changes. Actual paid cost is reported.
        ablations = []
        for idea_id in self.suite["dev_ids"]:
            for mode in ("abstract", "bm25"):
                result = self.plan_task(f"D_{mode}_{idea_id}", self.ideas[idea_id], mode=mode,
                                        expected=next(e for e in self.suite["case_expectations"] if e["idea_id"] == idea_id))
                ablations.append({"idea_id": idea_id, "mode": mode, "result": result})
            ablations.append({"idea_id": idea_id, "mode": "full", "result": self.results.get("A_" + idea_id, {})})
        write_json(self.root / "contrastive.json", {"records": checks, "passed": sum(c["success"] for c in checks), "total": len(checks)})
        write_json(self.root / "ablations.json", {"records": ablations,
            "control": "same model, planner, reviewer, output cap, <=6500 evidence characters; same two-LR protocol",
            "limitation": "input character ceilings equal; actual tokens differ; extractive cards navigate original nodes rather than paid LLM summaries"})

    def report(self):
        scheduled = {"A": ["A_" + i for i in self.ideas], "B": ["B_" + i for i in self.suite["tracks"]["B"]["idea_ids"]],
                     "boundary": [c["case_id"] for c in self.suite["boundary_cases"]]}
        report = {"schema_version": VERSION, "workspace": str(self.root), "model": self.calls.model.model,
                  "snapshot": str(self.root / "snapshot.json"), "ledger": self.calls.state,
                  "evaluation_level": "independent LLM call silver; same model correlated bias; internal authored suite",
                  "training_executed": False, "research_success": None, "tracks": {}}
        for track, keys in scheduled.items():
            records = [self.results.get(k, {"key": k, "status": "unstarted", "success": False}) for k in keys]
            def rate(field):
                n = sum(bool(r.get(field)) for r in records)
                return {"numerator": n, "denominator": len(records), "rate": n / len(records) if records else None}
            for r in records:
                r["mechanical_ready"] = r.get("mechanical", {}).get("passed", False)
            group = {"records": records, "mechanical_ready_rate": rate("mechanical_ready"),
                     "silver_plan_acceptance_rate": rate("silver_accepted"), "success_at_budget": rate("success")}
            for split in ("dev", "test"):
                subset = [r for r in records if r.get("idea_id") in self.suite[split + "_ids"]]
                group[split] = {"passed": sum(r.get("success", False) for r in subset), "total": len(subset)}
            if track == "boundary":
                group["correct_boundary_handling"] = rate("boundary_handled")
            report["tracks"][track] = group
        settled = [c for c in self.calls.state["call_ledger"] if c["status"] == "settled"]
        report["cost"] = {"actual_known_tokens": sum(c["actual_tokens"] for c in settled),
                          "prompt_tokens": sum(c.get("usage", {}).get("prompt_tokens", 0) for c in settled),
                          "completion_tokens": sum(c.get("usage", {}).get("completion_tokens", 0) for c in settled),
                          "charged_with_unknown_reservations": self.calls.state["total_tokens"],
                          "unknown_usage_calls": len(self.calls.state["call_ledger"]) - len(settled),
                          "new_shared_card_llm_tokens": 0,
                          "prior_material_cost": "historical; excluded from incremental cost; not claimed zero",
                          "provider_currency_cost": None,
                          "model_call_wall_seconds": sum(c.get("wall_seconds", 0) for c in self.calls.state["call_ledger"])}
        for name in ("contrastive", "ablations", "recovery"):
            if (self.root / (name + ".json")).exists():
                report[name] = read_json(self.root / (name + ".json"))
        write_json(self.root / "report.json", report)
        return report

"""Live discovery and recoverable scientific planning with fixed quality gates."""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict

from researchops.core.artifacts import canonical_hash, file_hash, write_json, workspace_lock, progress
from researchops.evaluation.precoding.runner import PrecodingRunner, read_json
from researchops.evaluation.precoding.gates import case_gate
from researchops.experiments.study_workflow import refine_study
from researchops.knowledge.retrieval import tokenize
from researchops.papers.connectors import LiteratureSearch, download_pdf, papers_match
from researchops.papers.ingestion import DocumentIngestor
from researchops.pipelines.batch import _paper_from_dict


class RobustPrecodingRunner(PrecodingRunner):
    def evidence(self, idea, mode="full", paper_ids=None):
        """Read several relevant full-text passages per paper, not just six snippets."""
        ids = set(paper_ids or self.snapshot_data["paper_ids"])
        query = json.dumps(idea, ensure_ascii=False)
        terms = set(tokenize(query))
        nodes = []
        def paper_score(paper_id):
            paper = self.store.get_paper(paper_id)
            return len(terms & set(tokenize(paper["title"] + " " + (paper["abstract"] or ""))))
        for paper_id in sorted(ids, key=paper_score, reverse=True)[:16]:
            paper = dict(self.store.get_paper(paper_id))
            if not paper.get("file_path") or file_hash(paper["file_path"]) != paper["content_hash"]:
                continue
            chunks = [dict(c) for c in self.store.paper_chunks(paper_id)]
            def score(c):
                text = c["text"].lower()
                return len(terms & set(tokenize(text))) + 2 * len(re.findall(
                    r"low.rank|visual prompt|bottleneck|calibration|regulariz|few.shot|linear prob|fine.tun", text))
            chunks.sort(key=score, reverse=True)
            for chunk in chunks[:2]:
                nodes.append({"paper_id": paper_id, "title": paper["title"], "chunk_id": chunk["chunk_id"],
                              "page": chunk["page"], "source_sha256": paper["content_hash"],
                              "source_type": "fulltext", "text": chunk["text"][:1500]})
        return nodes

    def live_acquire(self, idea):
        path = self.root / "acquisition" / f"{idea['idea_id']}.json"
        if path.exists():
            return read_json(path)
        directory = path.parent / idea["idea_id"]
        query_plan = self.calls.call("discovery_query_planner", """Plan literature discovery for a small-data pretrained ViT study.
Return JSON {"queries":["query"],"needed_roles":["specific method, baseline or evaluation evidence"]}.
Provide 5 short distinct search queries including direct method family/original paper names,
and evaluation/data/calibration background relevant to the question. Metadata is not evidence.
Avoid combining all constraints into each query. Do not assume proposed outcomes are true.""", {"task": idea}, 1500)
        write_json(directory / "query_plan.json", query_plan)
        search = LiteratureSearch(self.store, timeout=22)
        outputs = []
        for index, query in enumerate(query_plan.get("queries", [])[:5]):
            output_path = directory / f"search_{index}.json"
            if output_path.exists():
                out = read_json(output_path)
            else:
                out = search.search(query, "robust_" + idea["idea_id"], limit_per_source=7,
                                    sources=["arxiv", "openalex"])
                write_json(output_path, out)
            outputs.append(out)
        candidates = {p["paper_id"]: p for out in outputs for p in out["papers"]}
        parsed_ids = {r["paper_id"] for r in self.store.all_chunks()}
        originals = [dict(self.store.get_paper(p)) for p in parsed_ids]
        for p in originals:
            candidates[p["paper_id"]] = asdict(self.row_paper(p))
        selected, selections = set(), []
        values = list(candidates.values())
        for start in range(0, len(values), 32):
            batch = values[start:start + 32]
            selection = self.calls.call("paper_screening", """Screen literature for this research task using metadata.
Return JSON {"papers":[{"paper_id":"id","role":"direct_method|baseline|evaluation|dataset|related_vision_transfer","reason":"specific relevance"}]}.
Select all scientifically useful candidates in this batch, not just exact title matches.
Small-data image-classification PEFT work can support common transfer methodology even
when it studies a different variant. Distinguish direct method from contextual support.
Reject unrelated segmentation/video/time-series/medical-only work unless a specific common
method is demonstrably useful. Do not impose an arbitrary quota or invent evidence.""",
                {"task": idea, "needed_roles": query_plan["needed_roles"], "candidates": [
                    {"paper_id": p["paper_id"], "title": p["title"], "abstract": (p.get("abstract") or "")[:1000],
                     "parsed_fulltext": p["paper_id"] in parsed_ids, "pdf_available": bool(p.get("pdf_url"))} for p in batch]}, 3500)
            selections.append(selection)
            selected.update(p["paper_id"] for p in selection.get("papers", []))
        write_json(directory / "screening.json", selections)
        downloads, eligible = [], []
        # Download before assessing relevance: every eligible paper is checked against text.
        ordered = sorted(values, key=lambda p: p["paper_id"] in parsed_ids)
        tried = 0
        for p in ordered:
            if p["paper_id"] not in selected:
                continue
            paper = _paper_from_dict(p)
            original = next((v for v in originals if papers_match(paper, self.row_paper(v))), None)
            if original:
                eligible.append(original["paper_id"])
                continue
            if not paper.pdf_url or tried >= 14:
                continue
            tried += 1
            download_path = directory / "downloads" / f"{paper.paper_id}.json"
            if download_path.exists():
                downloaded = read_json(download_path)
                downloads.append(downloaded)
                if downloaded["status"] == "parsed":
                    eligible.append(downloaded["ingested"]["paper_id"])
                continue
            try:
                result = download_pdf(paper.pdf_url, self.root / "artifacts" / "papers", filename=paper.paper_id + ".pdf")
                ingested = DocumentIngestor(self.store).ingest_file(result.get("path") or result.get("file_path"), paper)
                row = dict(self.store.get_paper(ingested.paper_id))
                downloaded = {"paper_id": paper.paper_id, "status": "parsed", "sha256": row["content_hash"],
                              "ingested": ingested.to_dict(), "download": result}
                eligible.append(ingested.paper_id)
                originals.append(row)
            except Exception as error:
                downloaded = {"paper_id": paper.paper_id, "status": "failed", "error": f"{type(error).__name__}: {error}"}
            downloads.append(downloaded)
            write_json(download_path, downloaded)
        self.rows = {r["chunk_id"]: dict(r) for r in self.store.all_chunks()}
        assessments, verified = [], []
        for paper_id in sorted(set(eligible)):
            nodes = self.evidence(idea, paper_ids=[paper_id])
            assessment = self.calls.call("fulltext_relevance", """Assess actual supplied full-text excerpts for the research task.
Return JSON {"relevant":true,"role":"direct_method|baseline|evaluation|dataset|related_vision_transfer|irrelevant",
"reason":"specific contribution or limitation","supporting_chunk_ids":["id"]}.
Useful common small-data ViT transfer, PEFT architecture, validation protocol or evaluation
evidence can count as contextual support. It need not study the identical proposed setting.
Metadata alone cannot qualify a paper; identify an actually supporting excerpt.
Exclude merely keyword-matching unrelated tasks. Do not claim exact formulas beyond excerpts.""",
                {"task": idea, "fulltext": nodes}, 1100)
            locators = {n["chunk_id"] for n in nodes}
            valid = bool(assessment.get("relevant") and assessment.get("supporting_chunk_ids") and
                         set(assessment["supporting_chunk_ids"]) <= locators and assessment.get("role") != "irrelevant")
            assessments.append({"paper_id": paper_id, "eligible": valid, **assessment})
            if valid:
                verified.append(paper_id)
        write_json(directory / "fulltext_relevance.json", assessments)
        unique = {self.store.get_paper(p)["content_hash"] for p in verified}
        new_hashes = unique - set(self.snapshot_data["source_hashes"])
        track = self.suite["tracks"]["B"]
        target = track["target_valid_fulltexts_per_idea"]
        minimum = track["min_new_unique_fulltexts"].get(idea["idea_id"], 1)
        statuses = [s for o in outputs for s in o["sources"]]
        record = {"eligible_ids": verified, "valid_fulltexts": len(unique), "target": target,
                  "new_unique_fulltexts": len(new_hashes), "min_new_unique": minimum,
                  "new_hashes": sorted(new_hashes), "downloads": downloads, "candidate_count": len(candidates),
                  "query_plan": query_plan, "search_outputs": [str(directory / f"search_{i}.json") for i in range(len(outputs))],
                  "fulltext_assessments": assessments, "fresh_search_succeeded": any(s["status"] == "success" for s in statuses)}
        record["passed"] = record["fresh_search_succeeded"] and len(unique) >= target and len(new_hashes) >= minimum
        write_json(path, record)
        progress(f"acquisition={idea['idea_id']} relevant_fulltexts={len(unique)} new={len(new_hashes)} passed={record['passed']}")
        return record

    def plan_task(self, key, idea, *, expected=None, mode="full", acquisition=None, boundary=False):
        if key in self.results:
            return self.results[key]
        before, started = self.calls.state["total_tokens"], time.monotonic()
        directory = self.root / "tasks" / key
        try:
            evidence = [] if boundary else self.evidence(idea, paper_ids=acquisition["eligible_ids"] if acquisition else None)
            result = refine_study(idea=idea, states=self.states, evidence=evidence, store=self.store,
                                  calls=self.calls, directory=directory)
            spec = read_json(directory / "frozen_spec.json") if (directory / "frozen_spec.json").exists() else None
            constraints = case_gate(spec, expected) if expected else {"passed": True, "checks": {}}
            # Evaluator expectations are only checked AFTER the workflow terminates.
            result["constraints"] = constraints
            result["success"] &= constraints["passed"] and (acquisition is None or acquisition["passed"])
            result.update(acquisition=acquisition, evidence_count=len(evidence),
                          status="passed" if result["success"] else "failed")
        except Exception as error:
            result = {"status": "error", "success": False, "error": f"{type(error).__name__}: {error}"}
        result.update(key=key, idea_id=idea["idea_id"], wall_seconds=round(time.monotonic() - started, 3),
                      charged_tokens=self.calls.state["total_tokens"] - before)
        self.results[key] = result
        write_json(directory / "result.json", result)
        write_json(self.results_path, self.results)
        progress(json.dumps({k: result.get(k) for k in ("key", "status", "charged_tokens", "error")}))
        return result

    def run(self, phase="B"):
        return super().run(phase)

    def report(self):
        report = super().report()
        report["quality_protocol"] = "Critic plus independent final review; same model silver, all 7 criteria in both; <=4 plan attempts. No evaluator checks supplied to generation/repair."
        report["performance_excludes"] = ["wall_seconds", "coding", "training_outcomes"]
        for group in report["tracks"].values():
            rows = group["records"]
            group["first_pass_success"] = {"passed": sum(bool(r.get("success") and r.get("first_pass_success")) for r in rows), "total": len(rows)}
            group["attempt_counts"] = {r["key"]: len(r.get("attempts", [])) for r in rows}
        write_json(self.root / "report.json", report)
        return report

"""Recompute final integrity and citation evidence from saved v4 artifacts."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from researchops.core.artifacts import file_hash, write_json
from researchops.core.storage import ResearchStore
from researchops.evaluation.precoding.gates import mechanical_gate, silver_pass


def audit(root):
    read = lambda p: json.loads(p.read_text(encoding="utf-8"))
    results = read(root / "results.json")
    snapshot = read(root / "snapshot.json")
    store = ResearchStore(root / "researchops.db")
    tasks = []
    pdfs, new_pdfs = set(), set()
    for key, result in results.items():
        directory = root / "tasks" / key
        if not (directory / "proposal.json").exists():
            tasks.append({"key": key, "status": result["status"], "success": False})
            continue
        spec = read(directory / "frozen_spec.json") if (directory / "frozen_spec.json").exists() else None
        evidence = read(directory / "evidence.json")
        integrity = mechanical_gate(spec, evidence, store) if spec else None
        critic, final = read(directory / "review.json"), read(directory / "confirmation.json")
        references = (spec or {}).get("baseline_citations", [])
        cited = {n["chunk_id"]: n for n in evidence if n["chunk_id"] in references}
        hashes_valid = all(file_hash(store.get_paper(n["paper_id"])["file_path"]) == n["source_sha256"] for n in cited.values())
        support = [s for s in final.get("citation_support", []) if s["chunk_id"] in references]
        paper_fact_citations = {c for claim in (spec or {}).get("evidence_claims", []) if claim["kind"] == "paper_fact" for c in claim["citations"]}
        support_map = {s["chunk_id"]: s["verdict"] for s in support}
        supported = {c for c in paper_fact_citations if support_map.get(c) == "supported"}
        acquisition = result.get("acquisition")
        acquisition_check = None
        if acquisition:
            actual = set()
            for paper_id in acquisition["eligible_ids"]:
                p = store.get_paper(paper_id)
                if file_hash(p["file_path"]) == p["content_hash"]:
                    actual.add(p["content_hash"])
            new = actual - set(snapshot["source_hashes"])
            pdfs |= actual
            new_pdfs |= new
            acquisition_check = {"valid_unique_fulltexts": len(actual), "new_vs_initial": len(new),
                                 "passed": len(actual) >= acquisition["target"] and len(new) >= acquisition["min_new_unique"] and acquisition["fresh_search_succeeded"]}
        tasks.append({"key": key, "success": result.get("success", False),
                      "first_pass_success": result.get("first_pass_success", False) and not (root / "infrastructure_recovery" / key).exists(),
                      "first_pass_plan_success": result.get("first_pass_plan_success", result.get("first_pass_success", False)),
                      "infrastructure_recovered": (root / "infrastructure_recovery" / key).exists(),
                      "attempt_count": len(result.get("attempts", [])),
                      "mechanical_recheck": integrity, "critic_pass": silver_pass(critic),
                      "final_review_pass": silver_pass(final), "cited_pdf_hashes_valid": hashes_valid,
                      "citations_locatable": len(cited), "citations_total": len(references),
                      "fact_citations_supported": len(supported), "fact_citations_total": len(paper_fact_citations),
                      "acquisition_recheck": acquisition_check})
    leakage = []
    for request in (root / "calls").glob("*.request.json"):
        content = request.read_text(encoding="utf-8")
        if any(token in content for token in ('"case_expectations"', '"arm_requirements"', '"required_variants"')):
            leakage.append(request.name)
    ledger = read(root / "ledger.json")
    actual = sum(c.get("actual_tokens", 0) for c in ledger["call_ledger"])
    audit = {"workspace": str(root), "tasks": tasks, "task_count": len(tasks),
             "successes": sum(t["success"] for t in tasks), "unique_eligible_pdf_hashes": len(pdfs),
             "new_eligible_pdf_hashes_vs_initial": len(new_pdfs), "evaluator_manifest_leakage_requests": leakage,
             "actual_known_tokens": actual, "charged_tokens": ledger["total_tokens"],
             "request_count": ledger["turns"],
             "unknown_usage_calls": sum(c["status"] != "settled" for c in ledger["call_ledger"]),
             "limitations": "Semantic support is same-model independent-call silver; audit verifies saved judgments and source integrity, not human gold correctness."}
    write_json(root / "audit_v4.json", audit)
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("workspaces", nargs="+", type=Path)
    args = parser.parse_args()
    for root in args.workspaces:
        report = audit(root.resolve())
        print(json.dumps({k: v for k, v in report.items() if k not in {"tasks"}}, ensure_ascii=False, indent=2))

"""Audit joint P8 artifacts without issuing any model requests."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from importlib.metadata import distributions, version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from researchops.core.artifacts import canonical_hash, file_hash, write_json
from verify_v03_p8 import check_pins, contexts_from_messages, read
from verify_v03_p5_p6 import inspect_wire


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--junit", type=Path, required=True)
    parser.add_argument("--pytest-root", type=Path, required=True)
    args = parser.parse_args()
    args.pytest_root = args.pytest_root.resolve(strict=True)
    args.junit = args.junit.resolve(strict=True)
    root = args.output.resolve()
    cfg = check_pins(root)
    paper_result, study, coding = (read(root / (n+"_result.json")) for n in ("paper", "study", "coding"))
    paper = read(paper_result["path"])
    state = read(paper["micro_agent"]["checkpoint_path"])
    sample, reference = read(root/"ragas_input.json"), read(root/"reference.json")
    exported={"user_input":sample["question"],"response":sample["response"],
        "retrieved_contexts":[c["text"] for c in sample["contexts"]],
        "reference":reference["reference"],"reference_contexts":reference["reference_contexts"]}
    # Runtime artifact export, not a second generation or substitute answer.
    (root/"ragas_samples.jsonl").write_text(json.dumps(exported,ensure_ascii=False)+"\n",encoding="utf-8")
    wires = [inspect_wire(Path(r["audit_path"])) for r in paper["micro_agent"]["mcp_reads"]]
    fault_paths = sorted(args.pytest_root.rglob("*wire.jsonl"))
    if not fault_paths:
        raise OSError(f"No protocol test transcripts found under {args.pytest_root}; audit cannot omit fault evidence")
    fault_wires = [inspect_wire(p) for p in fault_paths]
    ledgers = {n: read(root/n/"ledger.json") for n in ("generation_budget", "evaluation_budget")}
    requests = {n: [read(p) for p in (root/n/"calls").glob("*.request.json")] for n in ledgers}
    graph = Path(study["artifacts"])
    graph_events = [json.loads(line) for p in (graph/"runs").rglob("trace.jsonl")
                    for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    graph_nodes = sorted({e["payload"]["node"] for e in graph_events if e["event"]=="study.graph.node"})
    query_rows = [json.loads(line) for p in root.rglob("vector_audit/*.jsonl")
                  for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    source_queries = [r for r in query_rows if r.get("operation")=="query_points"]
    checks = {
        "pinned_source_code_and_p7_unchanged": True,
        "paper_completed_with_real_mcp": paper_result["status"] in {"completed", "completed_with_gaps"} and bool(wires),
        "formal_paper_hash_matches_evaluated_original": file_hash(paper_result["path"])==paper_result["sha256"]==sample["source_artifact"]["sha256"],
        "mcp_handshake_catalog_call_correlated": bool(wires) and all(not w["unanswered_ids"] and
            all(m in w["methods"] for m in ("initialize", "notifications/initialized", "tools/list", "tools/call")) for w in wires),
        "mcp_text_consumed_by_real_provider": any(m.get("role")=="tool" and "mcp_stdio" in str(m.get("content"))
            for r in requests["generation_budget"] for m in r["messages"]),
        "upstream_analysis_consumed_by_study_model": any(
            "literature_analysis" in str(m.get("content")) and paper_result["sha256"] in str(m.get("content"))
            for r in requests["generation_budget"] for m in r["messages"] if m.get("role")=="user"),
        "langgraph_all_business_nodes_executed": set(graph_nodes) >= {"retrieve","planner","compile_gate","critic","final_review","decide","submit"},
        "langgraph_persistent_checkpoint": (graph/"graph_checkpoints.sqlite").is_file(),
        "study_formal_gate_accepted": study["status"]=="ready_for_coding" and study["mechanical"]["passed"],
        "coding_docker_verifier_passed": coding["status"]=="coding_verified" and coding["verification"]["passed"],
        "coding_shares_root_ledger_and_parent": Path(coding["harness"]["ledger_path"]).resolve()==root/"generation_budget/ledger.json"
            and coding["harness"]["parent_run_id"]=="p8-joint" and coding["harness"]["parent_session_id"]==study["thread_id"],
        "real_qdrant_queries": bool(source_queries),
        "qdrant_restart_persistence": read(root/"qdrant_persistence.json")["passed"],
        "ragas_actual_contexts": sample["contexts"]==contexts_from_messages(state["messages"]),
        "silver_reference_not_leaked": all(reference["reference"] not in json.dumps(r["messages"],ensure_ascii=False) for r in requests["generation_budget"]),
        "reference_is_explicitly_silver": reference["human_reviewed"] is False,
        "evaluation_inputs_untouched": read(root/"evaluation_input_audit.json")["checks"]["generation_inputs_untouched"],
        "all_four_ragas_scores_for_same_real_input": all(
            (s:=read(root/"scores"/(m+".json")))["status"]=="scored" and s["sample_sha256"]==canonical_hash([sample,reference])
            for m in ("faithfulness","answer_relevancy","context_precision","context_recall")),
        "cross_process_resume_zero_new_calls": read(root/"resume_check.json")["passed"] and read(root/"resume_check.json")["additional_provider_calls"]==0,
        "no_research_success_inferred": coding["research_success"] is None and study["research_success"] is None,
        "mcp_fault_cancellation_test_recorded": any("notifications/cancelled" in w["methods"] for w in fault_wires),
        "mcp_fault_errors_test_recorded": any(w["error_responses"]>=4 for w in fault_wires),
    }
    # In-process and MCP read the same underlying scoped SQLite rows. Check every
    # actual full-text output, not a newly invented ideal evidence collection.
    from researchops.core.storage import ResearchStore
    store = ResearchStore(root/"paper/researchops.db")
    full_reads = []
    for message in state["messages"]:
        if message.get("role")!="tool":
            continue
        value=json.loads(message["content"])
        output=value.get("output") or {}
        if value.get("status")=="success" and output.get("full_text"):
            row=store.get_chunk(output["chunk_id"])
            full_reads.append(row is not None and row["text"]==output["text"] and row["paper_id"]==output["paper_id"])
    checks["mcp_and_inprocess_source_text_equal"]=bool(full_reads) and all(full_reads)
    comparison=read(root/"comparison.json")
    checks["native_same_role_engine_accepted"]=comparison["native_success"] and comparison["native_mechanical_pass"] and comparison["same_role_engine"]
    checks["local_qdrant_fixed_query_agreement"]=comparison["vector_comparison"]["overlap_at_10"]>=0.9
    settlement={}
    for name,b in ledgers.items():
        checks[name+"_ledger_reconciles"]=b["total_tokens"]==sum(c.get("actual_tokens",c["upper_tokens"]) for c in b["call_ledger"])
        checks[name+"_within_cap"]=b["total_tokens"]<=b["token_cap"]
        settlement[name]={"all_dispatches_settled":all(c["status"]=="settled" and "actual_tokens" in c for c in b["call_ledger"]),
            "unknown_calls":[{"call_id":c["call_id"],"node_run_id":c["node_run_id"],"reserved_tokens":c["upper_tokens"],"error":c.get("error")} for c in b["call_ledger"] if c["status"]!="settled"]}
        # Unknown failed requests stay charged. A live/abandoned unrecorded
        # dispatch is not acceptable; a preserved error reservation is explicit.
        checks[name+"_all_dispatch_outcomes_accounted_for"]=all(
            (c["status"]=="settled" and "actual_tokens" in c) or
            (c["status"]=="reserved" and bool(c.get("error")) and
             (root/name/"calls"/(c["call_id"]+".request.json")).is_file()) for c in b["call_ledger"])
    if (root/"coding_recovery_binding.json").exists():
        recovery=read(root/"coding_recovery_binding.json")
        checks["original_coding_failure_preserved"]=file_hash(root/"failed_attempts/coding.initial.json")==recovery["initial_result_sha256"]
        aggregate_coding=sum(v["charged_tokens"] for k,v in ledgers["generation_budget"]["scopes"].items() if k.startswith("coding_"))
        checks["recovery_did_not_reset_coding_allowance"]=aggregate_coding<=recovery["aggregate_coding_cap"]
        checks["coding_cumulative_view_matches_root_scopes"]=coding["total_tokens"]+recovery["original_charge"]==aggregate_coding
        if (root/"coding_continuation_receipt.json").exists():
            receipt=read(root/"coding_continuation_receipt.json")
            checks["continuation_inherits_without_host_source_repairs"]=receipt["host_repaired_project_code"] is False and coding["implementation_origin"]["host_repaired_project_code"] is False
            checks["continuation_does_not_add_task_attempts"]=all(n==3 for n in receipt["unchanged_task_attempt_limits"].values())
    if (root/"stability_amendment.json").exists():
        amendment=read(root/"stability_amendment.json")
        checks["metadata_fix_has_zero_new_calls_and_keeps_original_answer"]=(amendment["new_model_calls"]==0
            and amendment["root_ledgers_byte_identical"] and amendment["answers_changed"] is False
            and amendment["original_analysis_sha256_restored"]==paper_result["sha256"])
    suites=list(ET.parse(args.junit).getroot().iter("testsuite"))
    tests={k:sum(int(s.get(k,0)) for s in suites) for k in ("tests","failures","errors","skipped")}
    tests["passed"]=tests["tests"]-tests["failures"]-tests["errors"]-tests["skipped"]
    checks["control_regression_passed"]=tests["passed"]>0 and tests["failures"]==tests["errors"]==0
    versions={name:version(name) for name in ("langgraph","langgraph-checkpoint-sqlite","qdrant-client","mcp","torch","timm")}
    evaluation_versions={d.metadata["Name"]:d.version for d in distributions(path=[str(Path(__file__).resolve().parents[1]/".venv-ragas/Lib/site-packages")])
        if d.metadata["Name"].lower() in {"ragas","langchain-core","langchain-community","langchain-openai","sentence-transformers"}}
    write_json(root/"ragas_manifest.json",{"rows":1,"sample_sha256":file_hash(root/"ragas_samples.jsonl"),
        "versions":evaluation_versions,"model":cfg["model"],"provider_options":cfg["provider_options"],
        "metric_workers":1,"answer_relevancy_strictness":3,"ragas_transport_attempts":1,"parser_repair_attempts":3,
        "provider_timeout_seconds":120,"embedding_revision":cfg["embedding_revision"],"human_reviewed":False})
    images={}
    for name,cmd in (("qdrant",["docker","inspect","researchops-qdrant-v03"]),
                     ("coding",["docker","image","inspect","researchops-coding:cpu-v1"])):
        p=subprocess.run(cmd,capture_output=True,text=True,timeout=30)
        if p.returncode:
            raise RuntimeError("Docker inspection failed: "+p.stderr)
        x=json.loads(p.stdout)[0]
        images[name]={k:x.get(k) for k in ("Id","Image","RepoDigests","Mounts") if x.get(k) is not None}
    out={"engineering_audit_passed":all(checks.values()),"checks":checks,"control_regression":tests,
         "junit_sha256":file_hash(args.junit),"graph_nodes":graph_nodes,"qdrant_query_count":len(source_queries),
         "mcp_full_text_reads":len(full_reads),"live_wire_sessions":wires,"fault_wire_sessions":fault_wires,
         "versions":versions,"evaluation_versions":evaluation_versions,"docker":images,"comparison":comparison,
         "request_settlement":settlement,
         "quality_release_passed":False,"quality_note":"Original P7 release criteria remain unmet; user authorized engineering P8 without further P7 tuning.",
         "artifact_hashes":{name:file_hash(root/name) for name in ("config.json","report.json","ragas_input.json","reference.json","resume_check.json","comparison.json","ragas_samples.jsonl","ragas_manifest.json")}}
    out["tool_source_hashes"]={p.name:file_hash(p) for p in Path(__file__).parent.glob("*v03_p8*.py")}
    write_json(root/"audit.json",out)
    print(json.dumps({k:v for k,v in out.items() if k in {"engineering_audit_passed","checks","control_regression","graph_nodes","qdrant_query_count","mcp_full_text_reads","quality_release_passed"}},ensure_ascii=False,indent=2))
    return 0 if out["engineering_audit_passed"] else 1


if __name__=="__main__":
    raise SystemExit(main())

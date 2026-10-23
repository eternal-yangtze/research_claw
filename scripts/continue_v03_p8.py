"""One explicit continuation of P8's partial code after a recorded TLS failure.

Inherit source, attempts, local accounting, and verification. Unknown root
charges remain in their original scopes. No host writes generated project code,
no extra task attempts, and no reset of the original aggregate Coding budget.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import time
from pathlib import Path

from verify_v03_p8 import (CODING_CAP, canonical_hash, check_pins, environment,
    file_hash, owner, pin, progress, read, workspace_lock, write_json)


def stage(root, model):
    path=root/"coding_continuation_plan.json"
    if path.exists():
        return read(path)
    result=read(root/"coding_recovery_result.json")
    if result["status"]!="needs_attention" or "UNEXPECTED_EOF_WHILE_READING" not in result["stop_reason"]:
        raise ValueError("This one-time continuation only handles the archived TLS failure")
    source=Path(result["report_path"]).parent
    checkpoint=read(source/"checkpoint.json")
    plan=read(root/"implementation_recovery/artifacts/coding"/(result["plan_id"]+".json"))
    inherited=checkpoint["total_tokens"]
    original=read(root/"coding_recovery_binding.json")
    if inherited+original["original_charge"]>=CODING_CAP:
        raise ValueError("Aggregate original Coding budget exhausted")
    # Preserve the original failed local state; acknowledge the lack of a
    # response only in the NEW continuation, without inventing an observation.
    reconciled=[]
    for task_id,row in checkpoint["tasks"].items():
        index=row.get("pending_call")
        if index is None:
            continue
        if (source/"calls"/f"{index:03d}_response.json").exists():
            raise ValueError("Observed responses must use normal journal recovery")
        lease=checkpoint["call_ledger"][index-1]
        if lease["status"]!="failed_usage_unknown_charged_upper_bound":
            raise ValueError("Only explicit failed transport reservations may be acknowledged")
        reconciled.append({"task":task_id,"local_call":index,"tokens_still_charged":lease["upper_tokens"],
            "resolution":"No response available; original files unchanged; retain full reservation. A NEW bounded request may repair the retained verifier failure."})
        row["pending_call"]=None
    if not reconciled:
        raise ValueError("No recorded failed pending dispatch")
    plan={k:v for k,v in plan.items() if k not in {"plan_path","cache_hit"}}
    plan["continuation_of"]={"plan_id":result["plan_id"],"checkpoint_sha256":file_hash(source/"checkpoint.json"),
        "inherited_tokens":inherited,"inherited_turns":checkpoint["turns"],"reconciliation":reconciled,
        "reason":"Explicit TLS-error continuation; unchanged attempts/caps; primary files are inherited LLM output, not a template."}
    plan["plan_id"]="study_coding_"+canonical_hash(plan)[:20]+"_continued"
    target=root/"implementation_continued/coding"/plan["plan_id"]
    if target.exists():
        raise ValueError("Uncommitted continuation target already exists")
    (target/"project").mkdir(parents=True)
    source_hashes={}
    for name in plan["allowed_outputs"]:
        src=source/"project"/name
        if src.is_file():
            dst=target/"project"/name
            dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(src,dst)
            source_hashes[name]=file_hash(src)
            if source_hashes[name]!=file_hash(dst):
                raise ValueError("Generated source changed during copy")
    shutil.copytree(source/"calls",target/"calls")
    shutil.copy2(source/"task_trace.jsonl",target/"task_trace.jsonl")
    checkpoint.update(plan_hash=canonical_hash(plan),status="in_progress")
    write_json(target/"checkpoint.json",checkpoint)
    scope="coding_"+plan["plan_id"]
    model.scope(scope,max_model_turns=plan["budget"]["max_model_turns"],max_total_tokens=plan["budget"]["max_total_tokens"])
    # These inherited charges already exist in the shared root. Bind explicitly
    # so legacy checkpoint import cannot count them for a second time.
    write_json(target/"harness_binding.json",{"ledger_path":str(model.path.resolve()),"node_run_id":scope,
        "parent_run_id":"p8-joint","parent_session_id":read(root/"study_result.json")["thread_id"]})
    pin(root/"coding_continuation_receipt.json",{"source_checkpoint_sha256":file_hash(source/"checkpoint.json"),
        "inherited_source_hashes":source_hashes,"host_repaired_project_code":False,"reconciliation":reconciled,
        "root_import_skipped":"Inherited local charges already belong to original root scopes; no duplicate debit.",
        "inherited_tokens":inherited,"inherited_turns":checkpoint["turns"],"original_coding_cap":CODING_CAP,
        "unchanged_task_attempt_limits":{t["id"]:t["max_attempts"] for t in plan["execution"]["tasks"]}})
    pin(path,plan)
    return plan


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("run","resume"))
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    environment()
    root=args.output.resolve()
    cfg=check_pins(root)
    with workspace_lock(root):
        from researchops.coding.agent import CodingApplication
        model=owner(root)
        before=file_hash(model.path)
        if args.phase=="run" and (root/"coding_continuation_result.json").exists():
            raise ValueError("One-time continuation already ended")
        if args.phase=="resume":
            from researchops.application.study import plan_study
            from verify_v03_p8 import STUDY_CAP, paper_stage
            import os
            paper_stage(root,model)
            os.environ["RESEARCHOPS_QDRANT_COLLECTION"]="researchops_p8_"+root.name.lower()+"_study"
            study=plan_study(read(root/"study_idea.json"),source=root/"paper",output=root/"study",model=model,
                dataset_source=Path(cfg["datasets"]),token_cap=STUDY_CAP,orchestration="langgraph")
            initial_study=read(root/"study_result.json")
            # The public result intentionally reports the CURRENT shared root
            # usage. Downstream Coding increases it without changing the study.
            stable_study=lambda v:{k:x for k,x in v.items() if k!="charged_tokens_cumulative"}
            if stable_study(study)!=stable_study(initial_study):
                raise ValueError("Study changed on replay")
        plan=stage(root,model)
        started=time.monotonic()
        result=CodingApplication(workspace=root/"implementation_continued",model=model,
            parent_run_id="p8-joint",parent_session_id=read(root/"study_result.json")["thread_id"]).run(plan,resume=True)
        summary={"status":result["status"],"stop_reason":result["stop_reason"],"turns_including_parent":result["turns"],
            "charged_tokens_including_parent":result["total_tokens"],"verification":result["verification"],
            "host_repaired_project_code":result["implementation_origin"]["host_repaired_project_code"]}
        if args.phase=="run":
            write_json(root/"coding_continuation_result.json",result)
            write_json(root/"coding_continuation_run.json",{"seconds":time.monotonic()-started,"summary":summary})
            if result["status"]=="coding_verified":
                write_json(root/"coding_result.json",result)
        else:
            checks={"root_ledger_byte_identical":before==file_hash(model.path),"coding_cached":result.get("cache_hit") is True,
                    "coding_still_verified":result["status"]=="coding_verified","same_study":True}
            value={"passed":all(checks.values()),"checks":checks,"additional_provider_calls":0 if checks["root_ledger_byte_identical"] else None,
                "additional_charged_tokens":0 if checks["root_ledger_byte_identical"] else None,
                "note":"Both prior unknown dispatches remain charged in their original root scopes. Study operational cumulative usage reflects the current shared root; formal study outputs are identical.",
                "study_cumulative_tokens":{"at_submission":initial_study["charged_tokens_cumulative"],"on_replay":study["charged_tokens_cumulative"]}}
            write_json(root/"resume_check.json",value)
            summary["resume"]=value
        progress(json.dumps(summary,ensure_ascii=False,indent=2))
        if result["status"]!="coding_verified":
            raise ValueError("Continuation did not pass all verifier gates")


if __name__=="__main__":
    main()

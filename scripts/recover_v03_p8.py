"""Explicit one-time Coding recovery; preserve unknown spend in the SAME P8 root.

The original failed attempt stays immutable. A separate scope/workspace receives
only the remaining original Coding allowance. Never resend an uncertain journal.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from verify_v03_p8 import (CODING_CAP, STUDY_CAP, check_pins, environment, file_hash,
    owner, paper_stage, pin, progress, read, report, workspace_lock, write_json)


def binding(root):
    path=root/"coding_recovery_binding.json"
    if path.exists():
        return read(path)
    initial=read(root/"coding_result.json")
    if initial["status"]=="coding_verified":
        raise ValueError("Recovery is only for the archived failed initial Coding attempt")
    node=initial["harness"]["node_run_id"]
    budget=read(root/"generation_budget/ledger.json")
    charge=budget["scopes"][node]["charged_tokens"]
    if charge>=CODING_CAP:
        raise ValueError("No remaining original Coding allowance")
    pin(root/"failed_attempts/coding.initial.json",initial)
    value={"original_node":node,"original_charge":charge,"remaining_cap":CODING_CAP-charge,
        "aggregate_coding_cap":CODING_CAP,"initial_status":initial["status"],"initial_stop_reason":initial["stop_reason"],
        "reason":"One explicit recovery in a new scope after provider SSE error; unknown original dispatch remains charged and is not retried in place.",
        "initial_result_sha256":file_hash(root/"failed_attempts/coding.initial.json")}
    pin(path,value)
    return value


def coding(root):
    from researchops.coding.agent import CodingApplication
    from researchops.coding.study_adapter import build_study_coding_plan
    with workspace_lock(root):
        config=binding(root)
        if (root/"coding_recovery_attempt.json").exists():
            raise ValueError("Recovery attempt already recorded; use resume for a completed attempt")
        model=owner(root)
        study=read(root/"study_result.json")
        plan=build_study_coding_plan(root/"implementation_recovery",Path(study["artifacts"])/"result.json",token_cap=config["remaining_cap"])
        started=time.monotonic()
        result=CodingApplication(workspace=root/"implementation_recovery",model=model,
            parent_run_id="p8-joint",parent_session_id=study["thread_id"]).run(plan)
        write_json(root/"coding_recovery_attempt.json",{"seconds":time.monotonic()-started,"result":result})
        write_json(root/"coding_recovery_result.json",result)
        scope=model.summary()["scopes"][result["harness"]["node_run_id"]]
        progress(json.dumps({"status":result["status"],"stop_reason":result["stop_reason"],"scope":scope,
            "aggregate_coding_charge":config["original_charge"]+scope["charged_tokens"]},ensure_ascii=False,indent=2))
        if result["status"]!="coding_verified":
            raise ValueError("Explicit Coding recovery did not verify")
        write_json(root/"coding_result.json",result)


def resume(root):
    import os
    from researchops.application.study import plan_study
    from researchops.coding.agent import CodingApplication
    from researchops.coding.study_adapter import build_study_coding_plan
    cfg=check_pins(root)
    config=read(root/"coding_recovery_binding.json")
    if read(root/"coding_recovery_result.json")["status"]!="coding_verified":
        raise ValueError("Only a verified recovery may enter zero-new-charge replay")
    before=file_hash(root/"generation_budget/ledger.json")
    with workspace_lock(root):
        model=owner(root)
        paper_stage(root,model)
        os.environ["RESEARCHOPS_QDRANT_COLLECTION"]="researchops_p8_"+root.name.lower()+"_study"
        study=plan_study(read(root/"study_idea.json"),source=root/"paper",output=root/"study",model=model,
            dataset_source=Path(cfg["datasets"]),token_cap=STUDY_CAP,orchestration="langgraph")
        plan=build_study_coding_plan(root/"implementation_recovery",Path(study["artifacts"])/"result.json",token_cap=config["remaining_cap"])
        result=CodingApplication(workspace=root/"implementation_recovery",model=model,
            parent_run_id="p8-joint",parent_session_id=study["thread_id"]).run(plan,resume=True)
        checks={"root_ledger_byte_identical":before==file_hash(root/"generation_budget/ledger.json"),
            "same_study":study==read(root/"study_result.json"),"coding_cached":result.get("cache_hit") is True,
            "coding_still_verified":result["status"]=="coding_verified"}
        value={"passed":all(checks.values()),"checks":checks,"additional_provider_calls":0 if checks["root_ledger_byte_identical"] else None,
            "additional_charged_tokens":0 if checks["root_ledger_byte_identical"] else None,
            "note":"Unknown original provider dispatch is still charged; completed recovery scope is resumed without resending it."}
        write_json(root/"resume_check.json",value)
        progress(json.dumps(value,ensure_ascii=False,indent=2))
        if not value["passed"]:
            raise ValueError("Recovery replay failed")


def finalize(args):
    root=args.output.resolve()
    report(args)
    value=read(root/"report.json")
    code=read(root/("coding_continuation_result.json" if (root/"coding_continuation_result.json").exists() else "coding_recovery_result.json"))
    b=read(root/"coding_recovery_binding.json")
    ledger=read(root/"generation_budget/ledger.json")
    recovery_charge=sum(v["charged_tokens"] for k,v in ledger["scopes"].items() if k.startswith("coding_") and k!=b["original_node"])
    value["initial_joint_attempt_passed"]=False
    value["coding_recovery"]={**b,"result_status":code["status"],"aggregate_charged_tokens":b["original_charge"]+recovery_charge,
        "seconds":read(root/"coding_recovery_attempt.json")["seconds"],
        "first_recovery_status":read(root/"coding_recovery_result.json")["status"],
        "continued_after_tls_error":(root/"coding_continuation_result.json").exists()}
    value["checks"]["paper_study_coding_completed"]=all(read(root/"stages"/(n+".json"))["status"]=="completed" for n in ("paper","study")) and code["status"]=="coding_verified"
    value["checks"]["coding_failures_and_unknown_reservation_retained"]=(root/"failed_attempts/coding.initial.json").exists() and b["original_node"] in ledger["scopes"]
    value["checks"]["aggregate_coding_within_original_cap"]=b["original_charge"]+recovery_charge<=b["aggregate_coding_cap"]
    value["engineering_passed"]=all(value["checks"].values())
    write_json(root/"report.json",value)
    progress(json.dumps({k:v for k,v in value.items() if k!="scores"},ensure_ascii=False,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("coding","resume","report"))
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    environment()
    root=args.output.resolve()
    check_pins(root)
    if args.phase=="report":
        finalize(args)
    else:
        globals()[args.phase](root)


if __name__=="__main__":
    main()

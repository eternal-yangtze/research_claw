"""Record the P8 completion-metadata fix without new calls or changed answers.

Restore the exact ORIGINAL analysis bytes from its recorded completion budget;
verify its original SHA before writing. Keep the intermediate live-budget view
and old code/config. Only one explicitly named source hash may change.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from verify_v03_p8 import PROJECT, file_hash, pin, read, workspace_lock, write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    root=args.output.resolve()
    with workspace_lock(root):
        cfg=read(root/"config.json")
        old_source=root/"source_snapshot_before_stability/analysis_agent.py"
        source_name=str(Path("src/researchops/papers/analysis_agent.py"))
        if (root/"stability_amendment.json").exists():
            raise ValueError("One-time stability amendment already applied")
        assert file_hash(old_source)==cfg["code"][source_name]
        assert all(file_hash(PROJECT/n)==h for n,h in cfg["code"].items() if n!=source_name)
        ledgers={n:file_hash(root/n/"ledger.json") for n in ("generation_budget","evaluation_budget")}
        first=read(root/"stages/paper.json")["result"]
        result=first["result"]
        artifact_path=Path(result["path"])
        artifact=read(artifact_path)
        pin(root/"metadata_history/paper_after_downstream.json",artifact)
        pin(root/"metadata_history/paper_result_after_downstream.json",read(root/"paper_result.json"))
        artifact["micro_agent"]["budget"]=first["budget_after"]
        # Match write_json/Path.write_text platform newline translation. The
        # archived Windows artifact uses CRLF, including its recorded byte hash.
        original_bytes=json.dumps(artifact,ensure_ascii=False,indent=2).replace("\n",os.linesep).encode("utf-8")
        assert hashlib.sha256(original_bytes).hexdigest()==result["sha256"], "Reconstruction must match original analysis SHA exactly"
        checkpoint=Path(artifact["micro_agent"]["checkpoint_path"])
        pin(checkpoint.with_name("completion_budget.json"),{"checkpoint_sha256":file_hash(checkpoint),"budget":first["budget_after"]})
        write_json(artifact_path,artifact)
        write_json(root/"paper_result.json",result)
        assert file_hash(artifact_path)==result["sha256"]
        pin(root/"config.before_stability.json",cfg)
        old_digest=cfg["code"][source_name]
        cfg["code"][source_name]=file_hash(PROJECT/source_name)
        write_json(root/"config.json",cfg)
        after={n:file_hash(root/n/"ledger.json") for n in ledgers}
        assert ledgers==after
        out={"source_file":source_name,"old_source_sha256":old_digest,"new_source_sha256":cfg["code"][source_name],
            "reason":"Freeze analysis completion budget metadata so downstream shared-root usage cannot alter formal artifact hashes.",
            "original_analysis_sha256_restored":file_hash(artifact_path),"checkpoint_sha256_unchanged":file_hash(checkpoint),
            "answers_changed":False,"new_model_calls":0,"root_ledgers_byte_identical":ledgers==after,
            "ledger_hashes":ledgers,"old_config_sha256":file_hash(root/"config.before_stability.json"),
            "new_config_sha256":file_hash(root/"config.json"),"intermediate_metadata_preserved":True}
        write_json(root/"stability_amendment.json",out)
        print(json.dumps(out,indent=2))


if __name__=="__main__":
    main()

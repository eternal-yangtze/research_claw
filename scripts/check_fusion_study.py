"""Exercise the real public fusion planner on two tasks; no coding/training."""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from researchops.core.artifacts import write_json, workspace_lock, file_hash
from researchops.core.config import load_project_env
from researchops.application.study import plan_study


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    load_project_env()
    os.environ["RESEARCHOPS_EMBEDDING_LOCAL_ONLY"] = "true"
    os.environ["RESEARCHOPS_RERANKER_LOCAL_ONLY"] = "true"
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    torch.set_num_threads(4)
    root = Path("workspace_fusion_study_v1")
    ideas = json.loads(Path("config/idea_library_vit_peft_v1.json").read_text(encoding="utf-8"))["ideas"][:2]
    def worker(idea):
        output = root / idea["idea_id"]
        marker = output / "public_result.json"
        if marker.exists():
            return json.loads(marker.read_text(encoding="utf-8"))
        result = plan_study(idea, source=Path("workspace_rag_cpu_v31"), output=output,
                            dataset_source=Path("workspace_batch_vit_v2"), retrieval_strategy="fusion", token_cap=250000)
        write_json(marker, result)
        return result
    with workspace_lock(root):
        write_json(root / "source_hashes.json", {str(p): file_hash(p) for p in [Path(__file__), Path("src/researchops/application/study.py"),
                                                                          Path("src/researchops/experiments/study_workflow.py")]})
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(worker, ideas))
        write_json(root / "report.json", {"tasks": [i["idea_id"] for i in ideas], "results": results,
                                          "same_corpus_papers": 94, "coding_executed": False, "training_executed": False})
        print([(r["status"], r["first_pass_success"], r["charged_tokens_cumulative"]) for r in results])

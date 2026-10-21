"""Real model routes a dependent import -> evidence task through project skills."""
from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from researchops.application.workflow import ResearchOpsApp
from researchops.core.config import load_project_env
from researchops.core.artifacts import file_hash, write_json, workspace_lock
from benchmark_skills import run_task, summarize


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    load_project_env()
    for key in ("RESEARCHOPS_EMBEDDING_LOCAL_ONLY", "RESEARCHOPS_RERANKER_LOCAL_ONLY"):
        os.environ[key] = "true"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["RESEARCHOPS_PDF_PARSER"] = "pymupdf"
    import torch
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-only", action="store_true")
    args = parser.parse_args()
    if args.study_only:
        root = Path("workspace_skills_study_v2").resolve()
        source, datasets = Path("workspace_rag_cpu_v31").resolve(), Path("workspace_batch_vit_v2").resolve()
        with workspace_lock(root):
            app = ResearchOpsApp(root, allowed_roots=[source, datasets])
            record = run_task(app, "study_role_repair",
                f"已有论文库{source}和已验证数据清单{datasets}。请制定ViT图像分类中LoRA rank=2/8/16的精度与参数量比较方案。"
                "保持其他LoRA因素相同，用验证集选参、测试集隔离，配对种子，不训练不coding。"
                "采用fusion检索，完成规划、独立评审和冻结规格，报告实际门禁结果。研究组件token上限250000。",
                cap=350000)
            successful = [r for r in record["invocations"] if r["name"] == "study-plan" and r["status"] == "success"]
            output = successful[-1]["result"]["output"] if successful else {}
            report = {"run_id": record["run_id"], "workflow_completed": "result" in record,
                      "study_output": output, "budget": summarize(root), "error": record.get("error"),
                      "limitation": "Targeted development repair; not a held-out generalization result"}
            write_json(root / "report.json", report)
            print(report, flush=True)
        return
    root = Path("workspace_skills_composition_v1").resolve()
    pdf = Path("workspace_batch_vit_v2/artifacts/papers/2502.17920.pdf").resolve()
    with workspace_lock(root):
        app = ResearchOpsApp(root, allowed_roots=[pdf])
        record = run_task(app, "import_then_retrieve",
            f"先把现有PDF {pdf} 导入当前知识库，再从入库后的知识库检索 C-LoRA 如何缓解灾难性遗忘，"
            "给出可定位原文引用。不要联网、设计新实验或写代码。", cap=250000)
        events = []
        trace = root / "runs" / record["run_id"] / "trace.jsonl"
        if trace.exists():
            import json
            events = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
        ordered = [e["payload"]["name"] for e in events if e["event"] == "skill.invocation.completed"]
        retrieval = [r for r in record["invocations"] if r["name"] == "evidence-retrieve" and r["status"] == "success"]
        good = ("result" in record and bool(ordered) and ordered[0] == "document-ingest" and "evidence-retrieve" in ordered[1:]
                and bool(retrieval) and all(r["result"]["output"]["citations"] for r in retrieval))
        report = {"run_id": record["run_id"], "ordered_completed_skills": ordered,
                  "dependent_execution_pass": good, "input_pdf_sha256": file_hash(pdf),
                  "knowledge_base_counts": app.store.counts(),
                  "citation_counts": [len(r["result"]["output"]["citations"]) for r in retrieval],
                  "budget": summarize(root), "error": record.get("error"),
                  "limitation": "One development workflow; locator checks are not independent semantic accuracy"}
        write_json(root / "report.json", report)
        print(report, flush=True)


if __name__ == "__main__":
    main()

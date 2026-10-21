"""Real-provider skill routing, production execution, and eager/lazy costs."""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from researchops.application.workflow import ResearchOpsApp
from researchops.core.artifacts import write_json, file_hash, canonical_hash, workspace_lock
from researchops.core.config import load_project_env
from researchops.core.structured import parse_single_object


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def invoke_records(root, run_id):
    return [load(p) for p in (root / "artifacts" / "skills" / run_id).glob("*.json")]


def run_task(app, key, task, *, context_mode="lazy", selection_only=False, cap=400000, limits=None):
    marker = app.workspace / "benchmark_results" / (key + ".json")
    if marker.exists():
        return load(marker)
    # Each retry is a new charged attempt. Never erase old reservations/failures.
    run_id = key + "_" + uuid.uuid4().hex[:8]
    try:
        result = app.run_skill_agent(task, run_id=run_id, token_cap=cap,
                    context_mode=context_mode, selection_only=selection_only, research_limits=limits)
        record = {"run_id": run_id, "result": result, "invocations": invoke_records(app.workspace, run_id)}
        write_json(marker, record)
        print(key, "completed", result["budget"], flush=True)
        return record
    except Exception as error:
        write_json(app.workspace / "benchmark_failures" / (run_id + ".json"), {"error": str(error), "run_id": run_id})
        print(key, "failed", str(error), flush=True)
        return {"run_id": run_id, "error": str(error), "invocations": invoke_records(app.workspace, run_id)}


def summarize(root):
    ledgers = [load(p) for p in root.glob("**/skill_runs/*/ledger.json")]
    known = sum(c.get("actual_tokens", 0) for l in ledgers for c in l["call_ledger"])
    charged = sum(l["total_tokens"] for l in ledgers)
    return {"runs_including_failures": len(ledgers), "known_actual_tokens": known,
            "unknown_reserved_tokens": charged - known, "charged_tokens": charged,
            "note": "Nested component ledgers are not added again"}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="workspace_skills_v1")
    parser.add_argument("--stage", choices=["routing", "execution", "context", "all"], default="all")
    args = parser.parse_args()
    load_project_env()
    os.environ["RESEARCHOPS_EMBEDDING_LOCAL_ONLY"] = "true"
    os.environ["RESEARCHOPS_RERANKER_LOCAL_ONLY"] = "true"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["RESEARCHOPS_PDF_PARSER"] = "pymupdf"
    import torch
    torch.set_num_threads(4)
    root = Path(args.output).resolve()
    suite_path = Path("config/benchmarks/skills_v1.json")
    suite = load(suite_path)
    source = Path("workspace_rag_cpu_v31").resolve()
    datasets = Path("workspace_batch_vit_v2").resolve()
    pdf = datasets / "artifacts/papers/2502.17920.pdf"
    protocol = {"suite_sha256": file_hash(suite_path), "script_sha256": file_hash(Path(__file__)),
        "skill_hashes": {str(p): file_hash(p) for p in Path("skills").glob("*/SKILL.md")},
        "source_corpus": str(source), "input_pdf_sha256": file_hash(pdf),
        "selection_expectations_not_in_model_context": True}
    with workspace_lock(root):
        if (root / "protocol.json").exists() and load(root / "protocol.json") != protocol:
            raise ValueError("Frozen benchmark changed; choose a new output version")
        write_json(root / "protocol.json", protocol)
        report = load(root / "report.json") if (root / "report.json").exists() else {}
        if args.stage in {"routing", "all"}:
            app = ResearchOpsApp(root / "routing")
            rows = []
            for case in suite["cases"]:
                record = run_task(app, case["id"], case["task"], selection_only=True, cap=80000)
                result = record.get("result", {})
                try:
                    selected = parse_single_object(result["answer"])["selected_skills"]
                except (KeyError, ValueError):
                    selected = None
                loaded = set(result.get("loaded_skills", {}))
                correct = selected is not None and set(selected) == set(case["expected"]) and loaded == set(case["expected"])
                rows.append({"id": case["id"], "selected": selected, "loaded": sorted(loaded),
                             "expected": case["expected"], "correct": correct, "run_id": record["run_id"]})
            negative = [r for r in rows if not r["expected"]]
            report["routing"] = {"cases": rows, "correct": sum(r["correct"] for r in rows), "total": len(rows),
                                 "negative_false_triggers": sum(bool(r["loaded"]) for r in negative), "negative_total": len(negative)}
            write_json(root / "report.json", report)
        if args.stage in {"execution", "all"}:
            app = ResearchOpsApp(root / "execution", allowed_roots=[pdf, source, datasets])
            rows = []
            ingest = run_task(app, "ingest", f"把这篇现有PDF导入知识库并报告解析结果，保留质量标记：{pdf}。只做入库。")
            rows.append(ingest)
            good = [r for r in ingest["invocations"] if r["name"] == "document-ingest" and r["status"] == "success"]
            if good:
                direct = app.call("ingest_documents", good[0]["arguments"])
                before, after = good[0]["result"]["output"], direct["output"]
                parity = lambda records: [(r["paper_id"], r["chunks"], r["parser"], r["quality_status"]) for r in records]
                report["ingestion_parity"] = parity(before) == parity(after)
                write_json(root / "ingestion_direct_control.json", direct)
            question = "现有知识库中，C-LoRA如何处理持续学习的灾难性遗忘？给出可定位的原文证据，不要联网。"
            retrieval = run_task(app, "retrieve", question)
            rows.append(retrieval)
            good = [r for r in retrieval["invocations"] if r["name"] == "evidence-retrieve" and r["status"] == "success"]
            if good:
                direct = app.call("retrieve_evidence", good[0]["arguments"])
                a, b = good[0]["result"]["output"], direct["output"]
                report["retrieval_parity"] = all(a[k] == b[k] for k in ["answer", "citations", "verified", "unsupported_claims"])
                report["retrieval_citations"] = len(a["citations"])
                report["retrieval_verified"] = a["verified"]
                write_json(root / "retrieval_direct_control.json", direct)
            study_task = (f"已有论文库{source}和已验证数据清单{datasets}。请制定ViT图像分类中LoRA rank=2/8/16的精度与参数量比较方案。"
                "保持其他LoRA因素相同，用验证集选参，测试集隔离；使用已有可用数据和配对种子，不做训练或coding。"
                "采用fusion检索，调用规划和独立评审，报告冻结规格、计划状态及引用。研究组件token上限250000。")
            study = run_task(app, "study", study_task, cap=600000)
            rows.append(study)
            good = [r for r in study["invocations"] if r["name"] == "study-plan" and r["status"] == "success"]
            if good:
                # Identical public operation, reuse the real generated call cache.
                direct = app.call("plan_study", good[0]["arguments"])
                report["study_cached_direct_parity"] = good[0]["result"]["output"] == direct["output"]
                report["study_result"] = good[0]["result"]["output"]
                write_json(root / "study_direct_control.json", direct)
            acquire = run_task(app, "acquire", "新研究idea：ViT少样本图像分类中的参数高效微调，比较LoRA与视觉提示。"
                "请搜索并获取论文，最多1条arXiv查询、每条最多2个候选、最多下载1篇。报告真实获取结果和失败，停在文献准备。",
                limits={"max_queries": 1, "limit_per_query": 2, "max_downloads": 1})
            rows.append(acquire)
            good = [r for r in acquire["invocations"] if r["name"] == "literature-acquire" and r["status"] == "success"]
            if good:
                report["acquisition_counts"] = good[0]["result"]["output"]["pipeline_counts"]
                report["acquisition_artifact"] = good[0]["result"]["output"]["artifact"]
            report["execution"] = [{"run_id": r["run_id"], "error": r.get("error"),
                "invoked_skills": [x["name"] for x in r["invocations"]],
                "successful_invocations": sum(x["status"] == "success" for x in r["invocations"])} for r in rows]
            write_json(root / "report.json", report)
        if args.stage in {"context", "all"}:
            context = []
            for mode in ["lazy", "eager"]:
                for i in range(2):
                    app = ResearchOpsApp(root / "context" / mode / str(i))
                    task = "只在现有知识库检索LoRA的低秩更新机制并引用原文。如果库里没有内容，明确报告证据不足，不要下载或规划。"
                    record = run_task(app, "comparison", task, context_mode=mode, cap=100000)
                    ledger = load(app.workspace / "skill_runs" / record["run_id"] / "ledger.json")
                    first = ledger["call_ledger"][0]
                    context.append({"mode": mode, "repeat": i, "run_id": record["run_id"],
                        "first_prompt_tokens": first.get("usage", {}).get("prompt_tokens"),
                        "actual_tokens": sum(c.get("actual_tokens", 0) for c in ledger["call_ledger"]),
                        "charged_tokens": ledger["total_tokens"], "model_calls": ledger["turns"],
                        "workflow_completed": "result" in record,
                        "error": record.get("error"),
                        "pass": "result" in record and bool(record["invocations"]) and all(x["name"] == "evidence-retrieve" and x["status"] == "success" and
                            not x["result"]["output"]["citations"] for x in record["invocations"])})
            report["context_comparison"] = context
        report["budget"] = summarize(root)
        report["coding_executed"] = report["training_executed"] = False
        report["limitations"] = suite["limitations"] + ["Context comparison is two repeats of an empty-corpus refusal, not general quality evidence"]
        write_json(root / "report.json", report)
        print(json.dumps({"report_path": str(root / "report.json"), "budget": report["budget"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

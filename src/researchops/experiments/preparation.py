"""Grounded dataset recommendations and an immutable pre-coding contract.

The contract defines a proposed controlled experiment, not a verbatim reproduction
of every cited paper. Nothing in this module generates or executes training code.
"""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from typing import Any

from researchops.experiments.datasets import DATASET_VERSION, REGISTRY, DatasetManager
from researchops.core.artifacts import canonical_hash, file_hash, write_json
from researchops.harness.runtime import ModelPort
from researchops.evaluation.silver import _parse_json_object
from researchops.core.storage import ResearchStore

SPEC_VERSION = "researchops-experiment-contract-v1"
RECOMMEND_SYSTEM = """You select datasets for a research experiment from supplied full-text evidence and synthesis.
Return only one JSON object: {"task_family":"image_classification|other", "recommendations":[
{"dataset_id":"registry id", "priority":1, "role":"primary|reserve", "reason":"specific suitability and limitations",
"citations":["supplied chunk_id"]}], "manual_recommendations":[{"name":"dataset name", "landing_url":"https://...",
"reason":"why needed"}], "baseline_citations":["supplied chunk_id"], "limitations":["..."]}.
Rank all suitable registry datasets (up to 5), with exactly two primary datasets if possible.
Favor small-data transfer and complementary domains; distinguish full datasets from VTAB-1K protocols.
Only cite supplied nodes. Do not claim a paper used a dataset if its excerpt does not say so.
Explain design choices as proposals when they are not paper claims. No fabricated URLs.
Never recommend image datasets for a time-series or knowledge-graph task.
Manual recommendations are optional; do not require inaccessible data when public alternatives suffice.
Choose standard linear probing, full fine tuning, LoRA, prompt tuning and adapters as comparison families
only if the supplied paper evidence supports those families. Report limitations honestly."""


def newest_matching(directory: Path, pattern: str, idea_id: str) -> Path:
    for path in sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("idea_id") == idea_id:
            return path
    raise FileNotFoundError(f"No {pattern} for Idea {idea_id}")


def evidence_for_datasets(store: ResearchStore, batch: dict, max_chars: int = 23000) -> list[dict]:
    aliases = [alias for item in REGISTRY.values() for alias in item["aliases"]]
    pattern = re.compile("|".join(re.escape(alias) for alias in aliases), re.I)
    baseline_pattern = re.compile(r"linear probing|linear tuning|full fine.tuning|LoRA|VPT|bottleneck|AdaptFormer", re.I)
    candidates = []
    for paper in batch.get("acquisitions", []):
        for row in store.paper_chunks(paper["paper_id"]):
            text = str(row["text"])
            if re.search(r"^references$|bibliography", str(row["section"]), re.I):
                continue
            matched = pattern.findall(text)
            baseline_matches = baseline_pattern.findall(text)
            if not matched and not baseline_matches:
                continue
            score = len(set(x.lower() for x in matched)) * 4 + min(len(baseline_matches), 4)
            candidates.append((score, {
                "chunk_id": row["chunk_id"], "paper_id": paper["paper_id"], "title": paper["title"],
                "page": row["page"], "section": row["section"], "node_type": row["node_type"],
                "text": text[:2400],
            }))
    candidates.sort(key=lambda pair: (-pair[0], pair[1]["chunk_id"]))
    # Avoid a single paper monopolizing the prompt. Second pass fills remaining space.
    selected, counts, size = [], {}, 0
    for cap in (2, 5):
        for _, node in candidates:
            paper_id = node["paper_id"]
            if counts.get(paper_id, 0) >= cap or any(n["chunk_id"] == node["chunk_id"] for n in selected):
                continue
            length = len(json.dumps(node, ensure_ascii=False))
            if size + length > max_chars:
                continue
            selected.append(node)
            counts[paper_id] = counts.get(paper_id, 0) + 1
            size += length
    return selected


class ExperimentPreparation:
    def __init__(self, *, workspace: Path, store: ResearchStore, model: ModelPort):
        self.workspace = workspace.resolve()
        self.store = store
        self.model = model
        self.output = self.workspace / "artifacts" / "experiments"
        self.output.mkdir(parents=True, exist_ok=True)
        self.datasets = DatasetManager(self.workspace)

    def run(self, idea: dict, *, workers: int = 3, download: bool = True, dataset_source: str = "auto") -> dict:
        self.datasets.source = dataset_source
        idea_id = idea["idea_id"]
        batch_path = newest_matching(self.workspace / "artifacts", "batch_research_*.json", idea_id)
        silver_path = newest_matching(self.workspace / "artifacts" / "evaluation", "silver_report_*.json", idea_id)
        batch = json.loads(batch_path.read_text(encoding="utf-8"))
        silver = json.loads(silver_path.read_text(encoding="utf-8"))
        if Path(silver.get("source_batch_report_path", "")).resolve() != batch_path.resolve():
            raise ValueError("Silver evaluation does not correspond to the selected literature batch")
        evidence = evidence_for_datasets(self.store, batch)
        if len(evidence) < 2:
            raise ValueError("Insufficient full-text evidence to justify dataset and baseline choices")
        recommendation = self.recommend(idea, batch, evidence)
        # Runtime cache/usage counters must not change an otherwise identical contract.
        stable_recommendation = {key: value for key, value in recommendation.items()
                                 if key not in {"cache_hit", "usage_this_run"}}
        rec_path = self.output / f"datasets_{idea_id}_{canonical_hash(stable_recommendation)[:16]}.json"
        write_json(rec_path, stable_recommendation)
        dataset_ids = [item["dataset_id"] for item in recommendation["recommendations"]]
        if download:
            states = self.datasets.acquire_many(dataset_ids, workers=workers)
        else:
            states = [self.datasets.acquire(dataset_id, offline=True) for dataset_id in dataset_ids]
        verified = {state["dataset_id"]: state for state in states if state["status"] == "verified"}
        primary = [r["dataset_id"] for r in recommendation["recommendations"]
                   if r["role"] == "primary" and r["dataset_id"] in verified]
        # A failed optional source does not impose manual work if two suitable alternatives passed.
        replacements = []
        for dataset_id in dataset_ids:
            if len(primary) >= 2:
                break
            if dataset_id in verified and dataset_id not in primary:
                primary.append(dataset_id)
                replacements.append(dataset_id)
        baseline_citations = recommendation["baseline_citations"]
        baseline_support = baseline_evidence(evidence)
        gates = {
            "literature_synthesis_available": bool(batch.get("synthesis", {}).get("final_synthesis")),
            "silver_quality_passed": silver.get("quality_gate", {}).get("status") == "pass"
                and silver.get("silver_ready") is True,
            "two_verified_primary_datasets": len(primary) == 2,
            "grounded_dataset_recommendations": all(
                bool(item["citations"]) for item in recommendation["recommendations"]
                if item["dataset_id"] in primary
            ),
            "baseline_evidence_available": len(baseline_citations) >= 2 and all(baseline_support.values()),
            "supported_task_family": recommendation["task_family"] == "image_classification",
        }
        spec = None
        if all(gates.values()):
            spec = build_vision_contract(
                idea, primary, verified, {**recommendation, "baseline_support": baseline_support},
                {"batch": {"path": str(batch_path), "sha256": file_hash(batch_path)},
                 "silver": {"path": str(silver_path), "sha256": file_hash(silver_path)},
                 "recommendations": {"path": str(rec_path), "sha256": file_hash(rec_path)}},
            )
            spec_hash = canonical_hash(spec)
            spec_path = self.output / f"frozen_{idea_id}_{spec_hash[:20]}.json"
            write_json(spec_path, spec)
            spec = {"path": str(spec_path), "sha256": file_hash(spec_path), "contract_hash": spec_hash,
                    "status": "ready_for_coding", "planned_jobs": spec["run_budget"]}
        report = {
            "schema_version": SPEC_VERSION, "idea_id": idea_id,
            "status": "ready_for_coding" if spec else "blocked_before_coding",
            "recommendation_path": str(rec_path), "recommendation_cache_hit": recommendation["cache_hit"],
            "llm_usage_this_run": recommendation["usage_this_run"],
            "recommended_count": len(dataset_ids), "verified_count": len(verified),
            "original_sample_count": sum(s["verification"]["sample_count"] for s in verified.values()),
            "dataset_metrics": {
                "acquisition_success_rate": len(verified) / len(dataset_ids) if dataset_ids else 0.0,
                "fully_decoded_datasets": sum(s["verification"]["checks"]["every_sample_readable"] for s in verified.values()),
                "official_source_splits_verified": sum(bool(s["verification"].get("source_split_counts")) for s in verified.values()),
                "excluded_duplicate_samples": sum(sum(s["verification"]["exclusions"].values()) for s in verified.values()),
                "manual_required_count": sum(s["status"] == "manual_required" for s in states),
            },
            "downloaded_bytes": sum(f["bytes"] for s in states for f in s["files"]),
            "primary_datasets": primary, "fallback_primary_datasets": replacements,
            "datasets": states, "manual_recommendations": recommendation.get("manual_recommendations", []),
            "quality_gates": gates, "frozen_spec": spec,
            "human_gold_completed": silver.get("human_gold_ready", False),
            "evaluation_level": "LLM silver; human gold reserved, not claimed",
            "training_executed": False, "hardware": hardware_profile(),
            "required_user_actions_before_coding": [] if spec else [
                "Resolve failed quality gates; import missing required data with dataset-import, then rerun prepare-experiment."
            ],
            "limitations": [
                "Native verified download/format adapters currently cover five image-classification benchmarks.",
                "Other task families are not silently mapped to vision; they require task-specific registry/adapters/contracts.",
                "The experiment is a new controlled protocol, not a claim to reproduce published results or VTAB-1K.",
                "Pretrained model weight acquisition and training dependency installation are Coding Agent acceptance tasks.",
            ],
        }
        report_path = self.output / f"preparation_{idea_id}.json"
        report["report_path"] = str(report_path)
        write_json(report_path, report)
        return {
            "status": report["status"], "report_path": str(report_path),
            "recommended_count": report["recommended_count"], "verified_count": report["verified_count"],
            "original_sample_count": report["original_sample_count"], "downloaded_bytes": report["downloaded_bytes"],
            "dataset_metrics": report["dataset_metrics"],
            "primary_datasets": primary, "quality_gates": gates, "frozen_spec": spec,
            "llm_usage_this_run": report["llm_usage_this_run"],
            "datasets": [{"dataset_id": s["dataset_id"], "status": s["status"],
                          "split_counts": s.get("verification", {}).get("split_counts"),
                          "sample_count": s.get("verification", {}).get("sample_count"),
                          "error": s.get("error"), "landing_url": s.get("landing_url"),
                          "import_command": s.get("import_command")} for s in states],
            "required_user_actions_before_coding": report["required_user_actions_before_coding"],
            "hardware": report["hardware"],
        }

    def recommend(self, idea: dict, batch: dict, evidence: list[dict]) -> dict:
        request = {"idea": idea, "fulltext_evidence": evidence,
                   "synthesis": batch.get("synthesis", {}).get("final_synthesis"),
                   "dataset_registry": REGISTRY}
        prompt_hash = canonical_hash({"version": SPEC_VERSION, "system": RECOMMEND_SYSTEM})
        source_hash = canonical_hash(request)
        model_name = getattr(self.model, "model", type(self.model).__name__)
        key = canonical_hash([prompt_hash, source_hash, model_name])
        cache_path = self.output / "recommendation_cache" / f"{key}.json"
        if cache_path.exists():
            value = json.loads(cache_path.read_text(encoding="utf-8"))
            value.update(cache_hit=True, usage_this_run={"total_tokens": 0})
            return value
        turn = self.model.complete(
            [{"role": "system", "content": RECOMMEND_SYSTEM},
             {"role": "user", "content": json.dumps(request, ensure_ascii=False)}],
            tools=[], max_tokens=4500,
        )
        value = _parse_json_object(turn.text)
        allowed = {node["chunk_id"] for node in evidence}
        recommendations = value.get("recommendations", [])
        if len(recommendations) > 5 or len({r["dataset_id"] for r in recommendations}) != len(recommendations):
            raise ValueError("Duplicate or excessive dataset recommendations")
        for item in recommendations:
            if item["dataset_id"] not in REGISTRY or item.get("role") not in {"primary", "reserve"}:
                raise ValueError("Invalid dataset id or role from model")
            citations = item.get("citations", [])
            if not citations or any(citation not in allowed for citation in citations):
                raise ValueError("Dataset recommendation must cite supplied full-text nodes")
        value["baseline_citations"] = [c for c in value.get("baseline_citations", []) if c in allowed]
        for item in value.get("manual_recommendations", []):
            if not re.match(r"^https://[^\s/]+", item.get("landing_url", "")):
                raise ValueError("Manual recommendation requires an HTTPS source page")
            item["url_status"] = "LLM recommendation; not verified and not auto-executed"
        if value.get("task_family") != "image_classification":
            # No mismatched downloads; unsupported family remains a visible blocking gate.
            value["recommendations"] = []
        value.update(schema_version=SPEC_VERSION, prompt_hash=prompt_hash, source_hash=source_hash,
                     model=model_name, cache_hit=False, usage_this_run=turn.usage or {},
                     generation_usage=turn.usage or {}, evidence=evidence)
        write_json(cache_path, value)
        return value


def hardware_profile() -> dict:
    import torch
    return {"torch_version": torch.__version__, "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "missing_training_packages": [name for name in ("timm",) if importlib.util.find_spec(name) is None],
            "note": "Data/spec preparation works on CPU; full ViT experiments target a CUDA GPU (about 12+ GB)."}


def baseline_evidence(evidence: list[dict]) -> dict:
    patterns = {
        "linear_probe": r"linear probing|linear tuning|linear.head|head.only",
        "full_finetune": r"full fine.tun|full tuning|\bFFT\b",
        "lora": r"\bLoRA\b|low.rank adapt",
        "vpt_shallow": r"\bVPT\b|visual prompt",
        "bottleneck_adapter": r"adapter.based|bottleneck|AdaptFormer",
    }
    return {method: [node["chunk_id"] for node in evidence if re.search(pattern, node["text"], re.I)][:4]
            for method, pattern in patterns.items()}


def build_vision_contract(idea: dict, primary: list[str], verified: dict, recommendation: dict, sources: dict) -> dict:
    if len(primary) != 2 or any(verified[p]["status"] != "verified" for p in primary):
        raise ValueError("Freeze requires two verified datasets")
    data = []
    for dataset_id in primary:
        validation = verified[dataset_id]["verification"]
        path = Path(validation["manifest_path"])
        if file_hash(path) != validation["manifest_sha256"]:
            raise ValueError("Dataset manifest changed since validation")
        for subset in validation["train_fractions"].values():
            if file_hash(Path(subset["path"])) != subset["sha256"]:
                raise ValueError("Training subset changed since validation")
        data.append({key: validation[key] for key in (
            "dataset_id", "name", "root", "class_count", "split_counts", "manifest_path",
            "manifest_sha256", "train_fractions", "exclusions", "split_seed", "official_test_preserved")})
    methods = {
        "linear_probe": {"trainable": ["classification head"], "lr_candidates": [0.001, 0.01]},
        "full_finetune": {"trainable": ["all model parameters"], "lr_candidates": [0.00001, 0.0001]},
        "lora": {"trainable": ["LoRA A/B on attention query and value", "classification head"],
                 "rank": 8, "alpha": 8, "dropout": 0.0, "lr_candidates": [0.0001, 0.001],
                 "equation": "y = W0 x + (alpha/r) B A dropout(x); A~Kaiming, B=0; W0 frozen",
                 "implementation": "For timm fused qkv, add low-rank updates to Q and V slices only; K frozen."},
        "vpt_shallow": {"trainable": ["10 prompt tokens", "classification head"],
                        "lr_candidates": [0.001, 0.01], "prompt_init": "Normal(0,0.02)",
                        "placement": "After CLS before image tokens, after adding original image position embeddings; retain prompts through all blocks; classify CLS."},
        "bottleneck_adapter": {"trainable": ["post-MLP adapters", "classification head"],
                               "bottleneck_dim": 64, "lr_candidates": [0.0001, 0.001],
                               "equation": "h <- h + W_up GELU(W_down h); W_up and up bias zero initialized; down Kaiming, down bias zero",
                               "placement": "Sequential residual adapter after each block's original MLP residual update; adapter has biases; backbone LayerNorm frozen."},
    }
    jobs = []
    for dataset_id in primary:
        for method, config in methods.items():
            # Shared validation-only LR pilot on 100% training subset; no test evaluation.
            for lr in config["lr_candidates"]:
                jobs.append({"stage": "validation_tuning", "dataset_id": dataset_id, "method": method,
                             "fraction": 1.0, "seed": 42, "lr": lr, "evaluate_test": False})
            for fraction in (0.1, 0.25, 1.0):
                for seed in (42, 43, 44):
                    jobs.append({"stage": "main", "dataset_id": dataset_id, "method": method,
                                 "fraction": fraction, "seed": seed, "lr": "best_validation_lr_for_dataset_method",
                                 "evaluate_test": True})
        # A small rank/regularization ablation at 25% data; main LoRA r=8,d=0 is the control.
        for rank, dropout in ((2, 0.0), (16, 0.0), (8, 0.1)):
            for seed in (42, 43, 44):
                jobs.append({"stage": "ablation", "dataset_id": dataset_id, "method": "lora",
                             "fraction": 0.25, "seed": seed, "rank": rank, "alpha": rank,
                             "dropout": dropout, "lr": "best_validation_lr_for_dataset_lora", "evaluate_test": True})
    for index, job in enumerate(jobs):
        job["job_id"] = f"job_{index:03d}_{canonical_hash(job)[:10]}"
    return {
        "schema_version": SPEC_VERSION, "status": "frozen_ready_for_coding", "idea": idea,
        "design_origin": "Agent-proposed controlled study grounded in cited method families; not published-result reproduction.",
        "sources": sources, "dataset_schema_version": DATASET_VERSION,
        "baseline_citations": recommendation["baseline_citations"],
        "baseline_family_support": recommendation.get("baseline_support", {}),
        "dataset_recommendations": recommendation["recommendations"], "datasets": data,
        "hypothesis": "At 10%, 25% and 100% labels, LoRA may approach full fine-tuning accuracy while training fewer parameters; test, do not assume.",
        "model": {"library": "timm", "name": "vit_base_patch16_224.augreg_in21k_ft_in1k",
                  "pretrained": True, "initialization": "Same pretrained state and identically seeded classification head across methods for each dataset/seed.",
                  "weights": "Coding Agent must acquire pinned weights before training and record sha256; offline/manual fallback if inaccessible.",
                  "normalization": "Use pretrained timm resolve_model_data_config; persist mean/std/interpolation and package version.",
                  "image_size": 224, "backbone_norm": "Frozen for all methods except full fine tuning."},
        "preprocessing": {"train": "RGB; RandomResizedCrop(224,scale=(0.8,1.0),ratio=(0.75,1.333333)); horizontal flip p=0.5; tensor; pretrained normalization",
                          "val_test": "RGB; resize shorter edge to 256; center crop 224; tensor; same normalization",
                          "no_augmentation_fitted_on_test": True},
        "methods": methods,
        "training": {"optimizer": "AdamW", "betas": [0.9, 0.999], "eps": 1e-8,
                     "weight_decay": 0.05, "weight_decay_exclusions": "biases, normalization scales, prompt tokens",
                     "loss": "Mean cross entropy, label_smoothing=0", "batch_size": 32,
                     "effective_batch_size": 32, "max_epochs": 20, "warmup_epochs": 2,
                     "scheduler": "linear warmup then cosine to 0.01*initial_lr; per optimizer update",
                     "early_stopping": {"patience": 5, "metric": "validation_accuracy", "min_delta": 0},
                     "checkpoint_selection": "highest validation accuracy; ties choose earliest epoch",
                     "lr_selection": "Highest validation accuracy from tuning jobs; ties choose smaller LR; no test access. Reuse at every fraction.",
                     "seeds": [42, 43, 44], "split_seed": 42, "grad_clip_norm": 1.0,
                     "precision": "fp32", "workers": 2,
                     "resource_policy": "One GPU job at a time. If memory insufficient, halve microbatch with accumulation to effective batch 32; log changes. Never silently change model or epochs."},
        "metrics": {"accuracy": "top-1 fraction correct", "macro_f1": "mean per-class F1 over all classes, zero_division=0",
                    "ece": "15 equal-width confidence bins on [0,1]; sum bin fraction * abs(bin accuracy-bin confidence)",
                    "trainable_parameters": "count requires_grad parameters excluding optimizer state",
                    "peak_memory": "CUDA max_memory_allocated after reset, synchronized; CPU value null with explicit reason",
                    "training_seconds": "wall clock training/validation excluding download and initial model load; synchronized CUDA",
                    "aggregation": "mean and sample standard deviation across 3 seeds; retain all seed results and predictions",
                    "success_rule": "LoRA mean accuracy within 0.02 absolute of full fine tune AND <=0.10 trainable-parameter ratio on BOTH datasets at fraction=1; otherwise mixed or unsupported. Report all fractions and uncertainty; no global novelty claim."},
        "data_policy": {"fixed_nested_fractions": [0.1, 0.25, 1.0],
                        "fraction_rounding": "Per class ceil(n*f), minimum 1; actual counts in hash-pinned subsets",
                        "official_splits": "Preserve official validation/test when supplied; CIFAR/Pets use fixed stratified training-only validation.",
                        "leakage": "Exclude pixel-identical train/val duplicates of test and train duplicates of val; no fitted statistics from test. CIFAR resizing is an explicit transfer design difference.",
                        "not_vtab_1k": True},
        "run_budget": {"main": sum(j["stage"] == "main" for j in jobs),
                       "validation_tuning": sum(j["stage"] == "validation_tuning" for j in jobs),
                       "ablation": sum(j["stage"] == "ablation" for j in jobs), "total": len(jobs),
                       "execution_order": "CPU shape/gradient smoke -> one-dataset one-seed pilot -> validation tuning -> main -> ablation",
                       "automatic_training_authorized_by_this_preparation": False},
        "jobs": jobs,
        "coding_acceptance": [
            "Pin dependencies, obtain pretrained model weights, record software/hardware/weight hashes.",
            "Verify sample/subset hashes; enforce fixed official test and train-only fractions.",
            "Load samples by the manifest: PIL RGB for path entries, numpy.load(mmap_mode='r',allow_pickle=False)[index] for numpy_rgb entries; keep source data read-only.",
            "CPU smoke test on synthetic images: finite loss, valid logits, gradient only in intended trainable groups; no benchmark result claims.",
            "Test zero-init LoRA reproduces original backbone outputs, adapters are identity at init, frozen parameters remain unchanged after one update.",
            "Run all methods with identical backbone, preprocessing, splits and validation tuning budget.",
            "Checkpoint model/optimizer/scheduler/RNG, resume per job, preserve immutable config and raw metrics/predictions.",
            "Stop on NaN, empty classes, hash mismatch, leakage or OOM that cannot preserve effective batch; record failure, never fabricate scores.",
        ],
        "evidence_limitations": recommendation.get("limitations", []),
        "human_gold": "Optional future audit; current gate is explicitly based on LLM silver.",
        "training_executed": False,
    }

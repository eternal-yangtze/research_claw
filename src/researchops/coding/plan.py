"""Evidence-grounded, versioned handoff to the research coding application."""
from __future__ import annotations

import json
from pathlib import Path

from researchops.pipelines.batch import BatchResearchPipeline, _idea_from_entry
from researchops.core.artifacts import canonical_hash, file_hash, write_json
from researchops.core.models import Paper
from researchops.papers.ingestion import DocumentIngestor
from researchops.papers.connectors import LiteratureSearch
from researchops.harness.runtime import TraceStore
from researchops.evaluation.silver import _parse_json_object
from researchops.coding.tasks import compile_tasks

PLAN_VERSION = "researchops-coding-plan-v5"
API_CONTRACT = {
    "models.py": {
        "build_model(spec, method, num_classes, *, smoke=False)":
            "Return torch.nn.Module. smoke=True: timm VisionTransformer(img_size=32,patch_size=8,embed_dim=48,depth=2,num_heads=3,mlp_ratio=2), random weights only. "
            "smoke=False: exact frozen timm backbone, pretrained=True, never silently substitute. All five methods from spec. "
            "Use spec method rank/alpha/dropout. Initialize common backbone/head identically for equal torch seed. "
            "Expose net.adaptation_method string. Named LoRA params contain lora_; prompt params contain prompt; adapter params contain adapter. "
            "Linear/full share unchanged base forward; LoRA zero B identity; bottleneck zero up identity. VPT prompt count=10. "
            "Frozen norm/backbone parameters remain frozen. LoRA Q,V only, K unchanged. Q and V use independent A/B matrices; no shared or tied updates.",
    },
    "metrics.py": {
        "classification_metrics(logits, labels, num_classes)":
            "Return JSON-serializable accuracy, macro_f1, ece. All configured classes in macro F1; zero_division=0; 15 equal-width ECE bins including confidence 1.0. No sklearn required.",
    },
    "train.py": {
        "run_job(spec, job, output_dir, *, smoke=False, max_steps=None, resume=False)":
            "Return dict and save checkpoint.pt and result.json. Use researchops.experiment_support.ManifestDataset, seed_everything, save_checkpoint, load_checkpoint. "
            "Honor frozen manifest/subset hashes, preprocessing, all five methods, AdamW decay groups, warmup/cosine, grad clipping, epochs, validation best checkpoint/early stopping. "
            "No test access in validation_tuning or smoke. Full mode uses spec backbone and full schedule. "
            "smoke mode must use real manifest train/val images, resized to32, max_steps as cap, no pretrained downloads; never claim research scores. "
            "Record spec_hash, job_id, global_step, mode=smoke|full, test_evaluated, trainable_parameters, peak_memory (null CPU), training_seconds, metrics. "
            "Resume restores model/optimizer/scheduler/RNG and validates job+spec hash; training continues rather than restarts. "
            "Write val_logits.npy and val_labels.npy at output_dir; metrics must describe these exact arrays. "
            "Return result['metrics'] as validation metrics, including accuracy, macro_f1, ece. "
            "Full test metrics use separate test_metrics and test_logits.npy/test_labels.npy. max_steps is total-step cap, not extra steps.",
    },
    "run_experiment.py": {
        "CLI":
            "--spec PATH --output PATH [--smoke] [--resume] [--max-jobs N]. Frozen jobs in order; tune using validation only; select smaller LR on ties; resolve main/ablation LR from tuning results. "
            "Per-job checkpoint/resume; skip only validated completed jobs; report mean/sample std across 3 seeds; evaluate success rule only with complete full jobs. "
            "Implement ablation rank/dropout overrides via deep-copy spec. Never report smoke as hypothesis support. "
            "Use researchops.experiment_support.spec_hash for the exact per-job copied spec, not a different JSON hash scheme. "
            "Fresh run_job return['metrics']['accuracy'] supplies validation LR selection; consume its return dict in memory. "
            "Existing cached jobs additionally require validated checkpoint/prediction artifacts before skipping. "
            "Keep fresh result handling separate from on-disk resume validation. A dispatcher unit-test trainer can return metrics without full training artifacts.",
    },
    "README.md": {"documentation": "Install versions, smoke/full commands, dataset provenance, CPU smoke limitations, failure/resume procedure and research metrics."},
    "tests/test_experiment.py": {"tests": "Nontrivial model/metrics/resume tests. External harness tests are authoritative and cannot be edited."},
}


def validate_implementation(value):
    tasks = value.get("tasks") or []
    ids = [task.get("id") for task in tasks]
    if not ids or any(not isinstance(item, str) or not item for item in ids) or len(set(ids)) != len(ids):
        raise ValueError("Plan task IDs must be nonempty and unique")
    dependencies = {}
    outputs = set()
    for task in tasks:
        dependencies[task["id"]] = set(task.get("depends_on") or [])
        if not dependencies[task["id"]] <= set(ids) or task["id"] in dependencies[task["id"]]:
            raise ValueError("Plan has invalid task dependencies")
        if not task.get("verification") or not task.get("description"):
            raise ValueError("Every plan task needs description and verification")
        current = set(task.get("outputs") or [])
        if not current <= set(API_CONTRACT):
            raise ValueError("Plan output outside allowed files")
        outputs |= current
    resolved = set()
    while len(resolved) < len(tasks):
        ready = {name for name, parents in dependencies.items() if parents <= resolved} - resolved
        if not ready:
            raise ValueError("Plan dependency cycle")
        resolved |= ready
    if outputs != set(API_CONTRACT):
        raise ValueError("Plan does not cover all required files")


def build_coding_plan(workspace: Path, store, model, idea_id: str, *, refresh_rag: bool = True, from_scratch: bool = False) -> dict:
    prep_path = workspace / "artifacts" / "experiments" / f"preparation_{idea_id}.json"
    preparation = json.loads(prep_path.read_text(encoding="utf-8"))
    if preparation.get("status") != "ready_for_coding":
        raise ValueError("Prepare and verify datasets before coding")
    frozen = preparation["frozen_spec"]
    spec_path = Path(frozen["path"])
    if file_hash(spec_path) != frozen["sha256"]:
        raise ValueError("Frozen experiment specification hash mismatch")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    batch_path = Path(spec["sources"]["batch"]["path"])
    if file_hash(batch_path) != spec["sources"]["batch"]["sha256"]:
        raise ValueError("Frozen source batch changed")
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    template_root = Path(__file__).with_name("templates") / "vit_peft"
    template = None
    if not from_scratch and set(spec["methods"]) == {"linear_probe", "full_finetune", "lora", "vpt_shallow", "bottleneck_adapter"} and spec["model"]["name"] == "vit_base_patch16_224.augreg_in21k_ft_in1k":
        template = {"id": "vit_peft_v1", "files": {name: file_hash(template_root / name) for name in API_CONTRACT},
            "origin": "LLM draft plus reviewed integration repairs; reusable implementation, not fresh autonomous synthesis"}
    execution_tasks = compile_tasks(API_CONTRACT)
    key = canonical_hash({"version": PLAN_VERSION, "spec": frozen["sha256"], "rag": "hybrid-v2", "api": API_CONTRACT, "template": template, "execution_tasks": execution_tasks})
    target = workspace / "artifacts" / "coding" / f"plan_{idea_id}_{key[:16]}.json"
    cached_synthesis = None
    cached_implementation = None
    if target.is_file():
        value = json.loads(target.read_text(encoding="utf-8"))
        if all(Path(item["path"]).is_file() and file_hash(Path(item["path"])) == item["sha256"]
               for item in value["source_artifacts"]):
            try:
                validate_implementation(value["implementation"])
            except ValueError:
                cached_synthesis = value.get("synthesis")
            else:
                return {**value, "plan_path": str(target), "cache_hit": True}
    if not cached_synthesis:
        for previous in sorted(target.parent.glob(f"plan_{idea_id}_*.json"), reverse=True):
            old = json.loads(previous.read_text(encoding="utf-8"))
            if old.get("frozen_spec", {}).get("sha256") != frozen["sha256"]:
                continue
            if not all(Path(a["path"]).is_file() and file_hash(a["path"]) == a["sha256"] for a in old.get("source_artifacts", [])):
                continue
            if old.get("synthesis", {}).get("map_count"):
                cached_synthesis = old["synthesis"]
                try:
                    validate_implementation(old["implementation"])
                except ValueError:
                    pass
                else:
                    cached_implementation = old["implementation"]
                break
    entry = spec["idea"]
    if not entry.get("subproblems"):
        entry = {**entry, "subproblems": ["Reproducible parameter efficient vision adaptation methods",
                 "Datasets splits and validation-only comparison protocols", "Accuracy memory and compute limitations"]}
    pipeline = BatchResearchPipeline(model=model, store=store, literature=LiteratureSearch(store),
        ingestor=DocumentIngestor(store), workspace=workspace, traces=TraceStore(workspace / "runs"))
    selected = {item["paper_id"] for item in batch["fan_in"]["shortlist"]}
    papers = []
    for paper_id in sorted(selected):
        row = store.get_paper(paper_id)
        if row is None:
            raise ValueError(f"Missing shortlisted paper: {paper_id}")
        papers.append(Paper(paper_id, row["title"], content_hash=row["content_hash"]))
    # _map_reduce now consumes paper-filtered BM25+dense+cross-encoder+MMR evidence.
    synthesis = cached_synthesis or (pipeline._map_reduce(_idea_from_entry(entry), entry, papers) if refresh_rag else batch["synthesis"])
    if synthesis.get("status") != "completed":
        raise ValueError("Evidence synthesis is incomplete")
    source_artifacts = [{"path": str(spec_path), "sha256": frozen["sha256"]},
                        {"path": str(batch_path), "sha256": file_hash(batch_path)},
                        {"path": synthesis["path"], "sha256": synthesis["sha256"]}]
    request = {
        "task": "Produce an implementation plan for a coding agent. Return JSON only.",
        "idea": entry, "synthesis": synthesis["final_synthesis"],
        "subproblem_syntheses": synthesis["subproblem_syntheses"],
        "frozen_design": {key: spec[key] for key in ("model", "methods", "training", "metrics", "coding_acceptance")},
        "required_api": API_CONTRACT,
        "rules": ["Do not change the frozen experiment design or test splits.",
                  "Cite supplied paper_id/node_id for method claims. Separate proposed engineering choices.",
                  "Each task must list outputs, dependencies and concrete verification.",
                  "Task outputs MUST be selected ONLY from required_api keys; together cover all six files. No other files, directories or requirements.txt.",
                  "Trusted support handles data/checkpoints; environment is already provided. Put setup documentation in README.md.",
                  "Plan implementation and engineering smoke only. Full study is a generated CLI, not a task to run now."],
        "output_schema": {"objective": "string", "tasks": [{"id": "string", "description": "string",
            "depends_on": ["task id"], "outputs": ["relative file"], "verification": ["string"]}],
            "evidence": [{"paper_id": "string", "node_id": "string", "use": "string"}], "risks": ["string"]},
    }
    turn = None
    if cached_implementation:
        implementation = cached_implementation
    else:
        turn = model.complete([{"role": "system", "content": "Plan evidence-grounded reproducible research code; output JSON only."},
                               {"role": "user", "content": json.dumps(request, ensure_ascii=False)}], [], 3500)
        implementation = _parse_json_object(turn.text)
    valid = {(str(ev.get("paper_id")), str(ev.get("node_id")))
             for reduction in synthesis["subproblem_syntheses"] for ev in reduction.get("evidence", [])}
    implementation["evidence"] = [ev for ev in implementation.get("evidence", [])
        if (str(ev.get("paper_id")), str(ev.get("node_id"))) in valid]
    if not implementation.get("tasks") or not implementation["evidence"]:
        raise ValueError("Coding plan must contain tasks and valid retrieved evidence")
    validate_implementation(implementation)
    plan = {"schema_version": PLAN_VERSION, "plan_id": "coding_" + key[:20], "idea_id": idea_id,
        "objective": implementation.get("objective", entry["title"]), "source_artifacts": source_artifacts,
        "frozen_spec": frozen, "implementation": implementation, "api_contract": API_CONTRACT,
        "allowed_outputs": list(API_CONTRACT),
        "execution": {"strategy": "task_dag_v1", "tasks": execution_tasks,
                      "repair_format": "exact unique search/replace; atomic per-file; rollback on failed repair",
                      "max_integration_repairs": 3},
        "acceptance": {"syntax": True, "all_five_model_methods": True, "identity_and_freezing": True,
            "metrics_recomputed": True, "real_primary_data_smoke": True, "checkpoint_resume": True,
            "no_test_access_in_smoke": True},
        "budget": {"max_model_turns": 24, "max_tool_calls": 90, "max_verification_rounds": 6,
                   "max_total_tokens": 120000, "max_wall_seconds": 1800, "command_timeout_seconds": 180},
        "finish_condition": "All external verifier gates pass on unchanged code, with artifacts and trace; coding_verified only, not research success.",
        "stop_conditions": ["all_verifier_gates_pass", "budget_exhausted", "repeated_identical_failure_3_times", "unavailable_required_dependency"],
        "manual_supplement": {"paper_import": "reserved", "dataset_import": "reserved"},
        "template": template,
        "synthesis": synthesis, "planning_usage": turn.usage if turn else {"cached_implementation": True}}
    write_json(target, plan)
    return {**plan, "plan_path": str(target), "cache_hit": False}

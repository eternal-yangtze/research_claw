from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from researchops.core.models import Idea, Paper
from researchops.harness.runtime import ModelPort
from researchops.core.storage import ResearchStore


ANALYZER_VERSION = "baseline-readiness-v2-formula-policy"
UNKNOWN = "Not established from the supplied full text."


class PaperAnalyzer:
    """Build a citation-bound paper analysis for baseline decisions."""

    def __init__(self, model: ModelPort, store: ResearchStore, workspace: Path) -> None:
        self.model = model
        self.store = store
        self.workspace = workspace

    def analyze(self, task_id: str, idea: Idea, paper: Paper) -> dict[str, Any]:
        from researchops.papers.validation.automatic import formula_evidence_policy
        rows = self.store.paper_chunks(paper.paper_id)
        if not rows:
            return self._insufficient(task_id, paper, "No parsed full-text chunks are available.")
        policy = {row["chunk_id"]: formula_evidence_policy(row) for row in rows}
        chosen = _analysis_chunks([row for row in rows if policy[row["chunk_id"]]["eligible_for_planning"]])
        if not chosen:
            return self._insufficient(task_id, paper, "Only unresolved formula nodes are available; original-page review required.")
        chunk_payload = [
            {
                "chunk_id": row["chunk_id"],
                "section": row["section"],
                "node_type": row["node_type"],
                "page": row["page"],
                "provenance": json.loads(row["provenance_json"] or "{}"),
                "formula_policy": policy[row["chunk_id"]],
                "text": row["text"][:6000],
            }
            for row in chosen
        ]
        request = {
            "task": "Analyze this paper for method understanding, reproduction, and baseline selection.",
            "current_idea": {
                "title": idea.title,
                "research_question": idea.research_question,
                "atomic_claims": list(idea.atomic_claims),
                "success_criteria": list(idea.success_criteria),
            },
            "paper": {"paper_id": paper.paper_id, "title": paper.title, "doi": paper.doi},
            "rules": [
                "Use only the supplied full-text chunks; do not rely on outside knowledge.",
                "Every evidence_field must be {text, citations}; citations must contain supplied chunk_id values.",
                "Use the literal text 'Not established from the supplied full text.' when evidence is missing.",
                "Do not infer implementation details from the title, abstract, or common practice.",
                "Assess baseline suitability for the current Idea, not the paper in isolation.",
                "Formula source_recovered is inferred silver alignment, not verified OCR; cite its selected_latex with uncertainty. No formula alone establishes exact reproduction correctness.",
            ],
            "output_schema": {
                "paper_summary": "evidence_field",
                "research_problem": "evidence_field",
                "inputs_outputs": "evidence_field",
                "method_overview": "evidence_field",
                "architecture_components": ["evidence_field"],
                "knowledge_or_graph_role": "evidence_field",
                "training_objective": "evidence_field",
                "inference_procedure": "evidence_field",
                "datasets": ["evidence_field"],
                "data_splits_preprocessing": "evidence_field",
                "metrics": ["evidence_field"],
                "paper_baselines": ["evidence_field"],
                "reported_results": ["evidence_field"],
                "implementation_availability": "evidence_field",
                "compute_requirements": "evidence_field",
                "limitations": ["evidence_field"],
                "baseline_assessment": {
                    "verdict": "eligible|partial|not_eligible|insufficient_evidence",
                    "rationale": "evidence_field",
                    "task_alignment": "evidence_field",
                    "input_output_compatibility": "evidence_field",
                    "dataset_metric_comparability": "evidence_field",
                    "reproduction_requirements": ["evidence_field"],
                    "adaptations_required": ["evidence_field"],
                },
                "missing_information": ["string"],
            },
            "evidence_field": {"text": "string", "citations": ["chunk_id"]},
            "chunks": chunk_payload,
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict research-method analyst. Return one JSON object only. "
                    "Unsupported claims must be marked unknown rather than guessed."
                ),
            },
            {"role": "user", "content": json.dumps(request, ensure_ascii=False)},
        ]
        raw: dict[str, Any] | None = None
        for attempt in range(2):
            turn = self.model.complete(messages, [], 3500)
            try:
                raw = _parse_json_object(turn.text)
                break
            except json.JSONDecodeError:
                if attempt:
                    raise
                messages.extend([
                    {"role": "assistant", "content": turn.text},
                    {
                        "role": "user",
                        "content": "The previous response was invalid JSON. Return the same grounded analysis as one syntactically valid JSON object only.",
                    },
                ])
        assert raw is not None
        valid_ids = {row["chunk_id"] for row in chosen}
        normalized, unsupported = _normalize_analysis(raw, valid_ids)
        normalized, missing = _ensure_analysis_schema(normalized)
        unsupported.extend(missing)
        normalized.update({
            "paper_id": paper.paper_id,
            "paper_title": paper.title,
            "task_id": task_id,
            "analyzer_version": ANALYZER_VERSION,
            "evidence_chunk_count": len(chosen),
            "source_pdf_path": rows[0]["file_path"],
            "citation_index": {
                row["chunk_id"]: {
                    "section": row["section"],
                    "page": row["page"],
                    "node_type": row["node_type"],
                    "provenance": json.loads(row["provenance_json"] or "{}"),
                }
                for row in chosen
            },
            "unsupported_fields": unsupported,
            "formula_policy": policy,
            "formula_nodes_withheld": [key for key, value in policy.items() if not value["eligible_for_planning"]],
        })
        status = "completed" if not unsupported else "completed_with_gaps"
        return self._persist(task_id, paper, status, normalized)

    def failed(self, task_id: str, paper: Paper, error: Exception) -> dict[str, Any]:
        return self._insufficient(task_id, paper, f"{type(error).__name__}: {error}", status="analysis_failed")

    def mark_insufficient(self, task_id: str, paper: Paper, reason: str) -> dict[str, Any]:
        return self._insufficient(task_id, paper, reason)

    def _insufficient(
        self,
        task_id: str,
        paper: Paper,
        reason: str,
        *,
        status: str = "insufficient_full_text",
    ) -> dict[str, Any]:
        analysis = {
            "paper_id": paper.paper_id,
            "paper_title": paper.title,
            "task_id": task_id,
            "analyzer_version": ANALYZER_VERSION,
            "status": status,
            "reason": reason,
            "baseline_assessment": {
                "verdict": "insufficient_evidence",
                "rationale": {"text": UNKNOWN, "citations": []},
            },
        }
        return self._persist(task_id, paper, status, analysis)

    def _persist(self, task_id: str, paper: Paper, status: str, analysis: dict[str, Any]) -> dict[str, Any]:
        analysis["status"] = status
        target_dir = self.workspace / "artifacts" / "analyses" / task_id
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{paper.paper_id}.json"
        target.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
        self.store.save_paper_analysis(
            task_id, paper.paper_id, status, analysis, str(target), ANALYZER_VERSION
        )
        return {
            "status": status,
            "path": str(target),
            "baseline_verdict": analysis.get("baseline_assessment", {}).get(
                "verdict", "insufficient_evidence"
            ),
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        }


class ResearchExecutionPlanner:
    """Synthesize research and coding plans from completed paper analyses."""

    def __init__(self, model: ModelPort, workspace: Path) -> None:
        self.model = model
        self.workspace = workspace

    def build(
        self,
        task_id: str,
        idea: Idea,
        analyses: list[dict[str, Any]],
        *,
        use_model: bool = True,
    ) -> dict[str, Any]:
        usable: list[dict[str, Any]] = []
        valid_evidence: set[tuple[str, str]] = set()
        for record in analyses:
            if record.get("status") not in {"completed", "completed_with_gaps"}:
                continue
            path = record.get("path")
            if not path or not Path(path).is_file():
                continue
            value = json.loads(Path(path).read_text(encoding="utf-8"))
            paper_id = str(value.get("paper_id") or "")
            for chunk_id in value.get("citation_index", {}):
                valid_evidence.add((paper_id, str(chunk_id)))
            usable.append(_analysis_for_planning(value))
        if not usable:
            return self._persist(task_id, {
                "task_id": task_id,
                "status": "insufficient_full_text",
                "idea_restatement": idea.description,
                "reason": "No completed full-text paper analysis is available for grounded planning.",
                "research_plan": None,
                "coding_plan": None,
                "source_papers": [],
            })
        request = {
            "task": "Create an executable research plan and coding plan grounded in parsed papers.",
            "idea": {
                "title": idea.title,
                "description": idea.description,
                "research_question": idea.research_question,
                "atomic_claims": list(idea.atomic_claims),
                "success_criteria": list(idea.success_criteria),
            },
            "paper_analyses": usable,
            "rules": [
                "Clearly distinguish facts from papers and newly proposed experiment decisions.",
                "Only select a paper as a baseline when its supplied analysis supports the choice.",
                "Every baseline must cite at least one {paper_id, chunk_id} pair from the supplied analysis.",
                "Do not invent datasets, reported scores, code repositories, or implementation details.",
                "The coding plan must be implementation-ready but must not claim code has already been written.",
            ],
            "output_schema": {
                "idea_restatement": "string",
                "paper_grounding_summary": ["string"],
                "baseline_selection": [{
                    "paper_id": "string",
                    "role": "primary|secondary|method_reference|not_selected",
                    "justification": "string",
                    "evidence": [{"paper_id": "string", "chunk_id": "string"}],
                    "adaptation_required": ["string"],
                }],
                "research_plan": {
                    "objective": "string",
                    "hypotheses": ["string"],
                    "datasets": ["string"],
                    "data_protocol": ["string"],
                    "baselines": ["string"],
                    "metrics": ["string"],
                    "ablations": ["string"],
                    "experiment_matrix": ["string"],
                    "reproducibility_controls": ["string"],
                    "expected_artifacts": ["string"],
                    "decision_rules": ["string"],
                },
                "coding_plan": {
                    "architecture": ["string"],
                    "modules": [{"path": "string", "responsibility": "string"}],
                    "implementation_steps": ["string"],
                    "interfaces": ["string"],
                    "tests": ["string"],
                    "commands": ["string"],
                    "outputs": ["string"],
                },
                "open_questions": ["string"],
                "risks": ["string"],
            },
        }
        try:
            if not use_model:
                raise RuntimeError("Deterministic planning was requested")
            turn = self.model.complete(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are a research and software planning agent. Return one JSON object only. "
                            "Plans must remain grounded in the supplied parsed-paper analyses."
                        ),
                    },
                    {"role": "user", "content": json.dumps(request, ensure_ascii=False)},
                ],
                [],
                3500,
            )
            value = _parse_json_object(turn.text)
        except Exception as error:
            value = _fallback_execution_plan(idea, usable, error)
        invalid_evidence = _sanitize_plan_evidence(value, valid_evidence)
        generated_status = value.pop("_generated_status", None)
        value.update({
            "task_id": task_id,
            "status": generated_status or (
                "completed" if not invalid_evidence else "completed_with_gaps"
            ),
            "source_papers": [item["paper_id"] for item in usable],
            "invalid_evidence_removed": invalid_evidence,
        })
        return self._persist(task_id, value)

    def failed(self, task_id: str, idea: Idea, error: Exception) -> dict[str, Any]:
        return self._persist(task_id, {
            "task_id": task_id,
            "status": "planning_failed",
            "idea_restatement": idea.description,
            "reason": f"{type(error).__name__}: {error}",
            "research_plan": None,
            "coding_plan": None,
            "source_papers": [],
        })

    def _persist(self, task_id: str, value: dict[str, Any]) -> dict[str, Any]:
        target_dir = self.workspace / "artifacts" / "plans"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{task_id}.json"
        target.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            "status": value["status"],
            "path": str(target),
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "source_papers": value.get("source_papers", []),
        }


def _analysis_chunks(rows: list[Any], max_chars: int = 18_000) -> list[Any]:
    terms = re.compile(
        r"method|model|architecture|algorithm|train|loss|objective|inference|dataset|experiment|"
        r"evaluation|metric|baseline|result|ablation|implementation|code|complexity|limitation|"
        r"方法|模型|训练|损失|推理|数据集|实验|指标|基线|结果|消融|实现|局限",
        re.I,
    )
    ranked = sorted(
        enumerate(rows),
        key=lambda pair: (
            0 if pair[1]["node_type"] in {"formula", "table"} else 1,
            -bool(terms.search(f"{pair[1]['section']} {pair[1]['text']}")),
            pair[0],
        ),
    )
    selected: list[tuple[int, Any]] = []
    size = 0
    for ordinal, row in ranked:
        length = min(len(row["text"]), 6000)
        if selected and size + length > max_chars:
            continue
        selected.append((ordinal, row))
        size += length
        if size >= max_chars:
            break
    return [row for _, row in sorted(selected, key=lambda item: item[0])]


def _analysis_for_planning(value: dict[str, Any]) -> dict[str, Any]:
    keep = (
        "paper_id",
        "paper_title",
        "paper_summary",
        "research_problem",
        "inputs_outputs",
        "method_overview",
        "architecture_components",
        "knowledge_or_graph_role",
        "training_objective",
        "inference_procedure",
        "datasets",
        "data_splits_preprocessing",
        "metrics",
        "paper_baselines",
        "reported_results",
        "implementation_availability",
        "compute_requirements",
        "limitations",
        "baseline_assessment",
        "missing_information",
        "citation_index",
        "formula_policy",
        "formula_nodes_withheld",
    )
    return {name: value.get(name) for name in keep}


def _fallback_execution_plan(
    idea: Idea,
    analyses: list[dict[str, Any]],
    error: Exception,
) -> dict[str, Any]:
    baselines: list[dict[str, Any]] = []
    dataset_notes: list[str] = []
    metric_notes: list[str] = []
    source_titles: list[str] = []
    for analysis in analyses:
        paper_id = str(analysis.get("paper_id") or "")
        title = str(analysis.get("paper_title") or paper_id)
        source_titles.append(title)
        assessment = analysis.get("baseline_assessment") or {}
        verdict = str(assessment.get("verdict") or "insufficient_evidence")
        rationale = assessment.get("rationale") or {}
        citations = [
            {"paper_id": paper_id, "chunk_id": str(chunk_id)}
            for chunk_id in rationale.get("citations") or []
        ]
        adaptations = [
            item.get("text")
            for item in assessment.get("adaptations_required") or []
            if isinstance(item, dict) and item.get("text") and item.get("text") != UNKNOWN
        ]
        baselines.append({
            "paper_id": paper_id,
            "role": "primary" if verdict in {"eligible", "partial"} and citations else "not_selected",
            "justification": str(rationale.get("text") or UNKNOWN),
            "evidence": citations,
            "adaptation_required": adaptations,
        })
        dataset_notes.extend(_field_texts(analysis.get("datasets")))
        metric_notes.extend(_field_texts(analysis.get("metrics")))
    return {
        "_generated_status": "completed_with_fallback",
        "fallback_reason": f"{type(error).__name__}: {error}",
        "idea_restatement": idea.description,
        "paper_grounding_summary": [
            f"Use the verified analysis for {title}; treat missing fields as unresolved rather than inferred."
            for title in source_titles
        ],
        "baseline_selection": baselines,
        "research_plan": {
            "objective": idea.research_question,
            "hypotheses": list(idea.atomic_claims) or [idea.description],
            "datasets": dataset_notes or [
                "Select a public in-domain dataset only after recording version, license, labels, and checksum."
            ],
            "data_protocol": [
                "Freeze preprocessing, window length, stride, feature scaling, and train/validation/test boundaries in configuration.",
                "Fit preprocessing and thresholds without reading test labels; record any protocol deviation explicitly.",
                "Preserve raw data checksums and materialize split manifests before training.",
            ],
            "baselines": [
                f"Reproduce the evidence-supported configuration of {title}." for title in source_titles
            ],
            "metrics": metric_notes + list(idea.success_criteria),
            "ablations": [
                "Run the reproduced method without changing its documented components.",
                "Remove the paper's principal proposed mechanism while preserving parameter and training budgets where possible.",
                "Separate scoring-rule and threshold changes from architecture changes.",
            ],
            "experiment_matrix": [
                "Cross product of dataset, baseline/ablation variant, and random seed with a frozen evaluation protocol.",
                "Include a smoke run before full training and retain failures as results.",
            ],
            "reproducibility_controls": [
                "Pin dependency versions, seeds, device, precision mode, and configuration hash.",
                "Run multiple seeds and report per-run metrics plus aggregate uncertainty.",
                "Recompute final metrics from saved predictions and labels in a separate verification step.",
            ],
            "expected_artifacts": [
                "dataset_manifest.json",
                "resolved_config.json",
                "checkpoints/",
                "predictions.parquet",
                "metrics_per_run.json",
                "aggregate_metrics.json",
                "run_trace.jsonl",
            ],
            "decision_rules": [
                "Do not claim improvement unless the fixed protocol, repeated runs, and recomputed metrics agree.",
                "Stop and mark the baseline non-reproducible when a required method detail cannot be recovered without guessing.",
            ],
        },
        "coding_plan": {
            "architecture": [
                "Configuration-driven data, model, training, scoring, evaluation, and artifact layers.",
                "A common model interface so the reproduced baseline and ablations share the same runner.",
            ],
            "modules": [
                {"path": "src/data.py", "responsibility": "Dataset loading, checksums, splits, scaling, and windows."},
                {"path": "src/models/baseline.py", "responsibility": "Evidence-supported baseline model and tensor contracts."},
                {"path": "src/models/ablations.py", "responsibility": "Controlled component removal or replacement variants."},
                {"path": "src/training.py", "responsibility": "Optimization, checkpoints, seeds, and failure capture."},
                {"path": "src/scoring.py", "responsibility": "Anomaly scores, aggregation, and threshold fitting."},
                {"path": "src/evaluation.py", "responsibility": "F1/AUROC and protocol-specific metric recomputation."},
                {"path": "scripts/run_experiment.py", "responsibility": "Execute the frozen experiment matrix."},
            ],
            "implementation_steps": [
                "Encode the paper-derived tensor shapes, equations, and scoring flow as tests before model training.",
                "Implement and smoke-test the data pipeline on a small deterministic slice.",
                "Implement the baseline, then compare intermediate tensors against documented equations or official code when available.",
                "Add ablations behind configuration flags and run the frozen matrix.",
                "Verify metrics from saved predictions independently of the training process.",
            ],
            "interfaces": [
                "DatasetBuilder.build(config) -> train, validation, test, manifest",
                "Model.forward(batch) -> reconstruction and method-specific evidence",
                "Scorer.score(model_output, batch) -> point-wise anomaly score",
                "Evaluator.compute(scores, labels, protocol) -> metrics",
            ],
            "tests": [
                "Unit tests for windows, split isolation, tensor shapes, losses, scores, and threshold fitting.",
                "Leakage test proving test labels do not affect training or threshold selection.",
                "End-to-end deterministic smoke test that writes every expected artifact.",
            ],
            "commands": [
                "python -m pytest -q",
                "python scripts/run_experiment.py --config configs/smoke.yaml",
                "python scripts/run_experiment.py --config configs/full.yaml",
            ],
            "outputs": [
                "Machine-readable per-run predictions, metrics, resolved configuration, logs, checkpoints, and provenance."
            ],
        },
        "open_questions": [
            "Which exact dataset versions and evaluation adjustments match the source paper?",
            "Which missing formulas or hyperparameters require original-page or official-code review?",
        ],
        "risks": [
            "Reported scores may depend on undocumented preprocessing, thresholding, or point adjustment.",
            "A fallback plan is operational guidance, not evidence that reproduction has succeeded.",
        ],
    }


def _field_texts(value: Any) -> list[str]:
    items = value if isinstance(value, list) else [value]
    return [
        str(item["text"])
        for item in items
        if isinstance(item, dict) and item.get("text") and item.get("text") != UNKNOWN
    ]


def _sanitize_plan_evidence(value: dict[str, Any], valid: set[tuple[str, str]]) -> list[dict[str, str]]:
    invalid: list[dict[str, str]] = []
    selections = value.get("baseline_selection")
    if not isinstance(selections, list):
        value["baseline_selection"] = []
        return invalid
    for selection in selections:
        if not isinstance(selection, dict):
            continue
        evidence = selection.get("evidence")
        if not isinstance(evidence, list):
            selection["evidence"] = []
            continue
        cleaned = []
        for citation in evidence:
            if not isinstance(citation, dict):
                continue
            pair = (str(citation.get("paper_id") or ""), str(citation.get("chunk_id") or ""))
            if pair in valid:
                cleaned.append({"paper_id": pair[0], "chunk_id": pair[1]})
            else:
                invalid.append({"paper_id": pair[0], "chunk_id": pair[1]})
        selection["evidence"] = cleaned
        if selection.get("role") != "not_selected" and not cleaned:
            selection["role"] = "not_selected"
            selection["justification"] = (
                "Not selected because the generated justification lacked valid parsed-paper evidence."
            )
    return invalid


def _normalize_analysis(value: Any, valid_ids: set[str], path: str = "") -> tuple[Any, list[str]]:
    unsupported: list[str] = []
    if isinstance(value, dict) and "text" in value and "citations" in value:
        citations = [str(item) for item in value.get("citations", []) if str(item) in valid_ids]
        text = str(value.get("text") or "").strip() or UNKNOWN
        if text != UNKNOWN and not citations:
            unsupported.append(path or "root")
            text = UNKNOWN
        return {"text": text, "citations": citations}, unsupported
    if isinstance(value, dict):
        output = {}
        for key, item in value.items():
            normalized, gaps = _normalize_analysis(item, valid_ids, f"{path}.{key}".strip("."))
            output[str(key)] = normalized
            unsupported.extend(gaps)
        return output, unsupported
    if isinstance(value, list):
        output = []
        for index, item in enumerate(value):
            normalized, gaps = _normalize_analysis(item, valid_ids, f"{path}[{index}]")
            output.append(normalized)
            unsupported.extend(gaps)
        return output, unsupported
    return value, unsupported


def _ensure_analysis_schema(value: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    missing: list[str] = []
    field_names = (
        "paper_summary",
        "research_problem",
        "inputs_outputs",
        "method_overview",
        "knowledge_or_graph_role",
        "training_objective",
        "inference_procedure",
        "data_splits_preprocessing",
        "implementation_availability",
        "compute_requirements",
    )
    list_names = (
        "architecture_components",
        "datasets",
        "metrics",
        "paper_baselines",
        "reported_results",
        "limitations",
    )
    for name in field_names:
        if not _is_evidence_field(value.get(name)):
            value[name] = {"text": UNKNOWN, "citations": []}
            missing.append(name)
    for name in list_names:
        if not isinstance(value.get(name), list):
            value[name] = []
            missing.append(name)
    baseline = value.get("baseline_assessment")
    if not isinstance(baseline, dict):
        baseline = {}
        value["baseline_assessment"] = baseline
        missing.append("baseline_assessment")
    verdicts = {"eligible", "partial", "not_eligible", "insufficient_evidence"}
    if baseline.get("verdict") not in verdicts:
        baseline["verdict"] = "insufficient_evidence"
        missing.append("baseline_assessment.verdict")
    for name in (
        "rationale",
        "task_alignment",
        "input_output_compatibility",
        "dataset_metric_comparability",
    ):
        if not _is_evidence_field(baseline.get(name)):
            baseline[name] = {"text": UNKNOWN, "citations": []}
            missing.append(f"baseline_assessment.{name}")
    for name in ("reproduction_requirements", "adaptations_required"):
        if not isinstance(baseline.get(name), list):
            baseline[name] = []
            missing.append(f"baseline_assessment.{name}")
    if not isinstance(value.get("missing_information"), list):
        value["missing_information"] = []
        missing.append("missing_information")
    return value, missing


def _is_evidence_field(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("text"), str)
        and isinstance(value.get("citations"), list)
    )


def _parse_json_object(text: str) -> dict[str, Any]:
    value = str(text or "").strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.I)
        value = re.sub(r"\s*```$", "", value)
    start, end = value.find("{"), value.rfind("}")
    if start < 0 or end < start:
        raise ValueError("paper analyzer did not return a JSON object")
    payload = json.loads(value[start:end + 1])
    if not isinstance(payload, dict):
        raise ValueError("paper analyzer response must be a JSON object")
    return payload

from __future__ import annotations

import hashlib
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from researchops.knowledge.retrieval import tokenize
from researchops.harness.runtime import ModelPort
from researchops.core.storage import ResearchStore


SILVER_SCHEMA_VERSION = "researchops-silver-v1"
SILVER_PROMPT_VERSION = "silver-judge-v1"
GOLD_REVIEW_SCHEMA_VERSION = "researchops-human-review-v1"
ALLOWED_ROLES = {"baseline", "method", "evaluation", "weak", "irrelevant"}


class SilverEvaluationPipeline:
    """Evidence-grounded LLM judgments kept explicitly separate from human gold."""

    def __init__(
        self,
        *,
        model: ModelPort,
        store: ResearchStore,
        workspace: Path,
        max_workers: int = 3,
    ) -> None:
        self.model = model
        self.store = store
        self.workspace = workspace
        self.max_workers = min(max(1, int(max_workers)), 8)
        self.prompt_hash = hashlib.sha256(SILVER_PROMPT_VERSION.encode("utf-8")).hexdigest()

    def run(
        self,
        idea_entry: dict[str, Any],
        *,
        provisional_path: str | Path,
        batch_report_path: str | Path,
        max_papers: int = 40,
        max_fulltext_chars: int = 14_000,
    ) -> dict[str, Any]:
        if not 30 <= max_papers <= 50:
            raise ValueError("max_papers must be between 30 and 50")
        provisional_path = Path(provisional_path).resolve()
        batch_report_path = Path(batch_report_path).resolve()
        raw_provisional = _read_jsonl(provisional_path)
        if len(raw_provisional) < 30:
            raise ValueError("silver evaluation requires at least 30 provisional records")
        if len({item.get("paper_id") for item in raw_provisional}) != len(raw_provisional):
            raise ValueError("provisional records contain duplicate paper_id values")
        report = json.loads(batch_report_path.read_text(encoding="utf-8"))
        selected_ids = [
            str(item["paper_id"]) for item in report.get("acquisitions") or []
        ]
        shortlist_ids = [
            str(item["paper_id"])
            for item in (report.get("fan_in") or {}).get("shortlist") or []
        ]
        provisional = self._evaluation_pool(
            raw_provisional, selected_ids, max_papers
        )
        output_dir = self.workspace / "artifacts" / "evaluation"
        cache_dir = output_dir / "silver_cache" / str(idea_entry["idea_id"])
        cache_dir.mkdir(parents=True, exist_ok=True)

        records: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(
                    self._judge,
                    idea_entry,
                    source,
                    cache_dir,
                    max_fulltext_chars,
                ): source
                for source in provisional
            }
            for future in as_completed(futures):
                source = futures[future]
                try:
                    records.append(future.result())
                except Exception as error:
                    errors.append({
                        "paper_id": str(source.get("paper_id") or ""),
                        "error": f"{type(error).__name__}: {error}",
                    })
                    records.append(_failed_silver_record(idea_entry, source, error, self.model))
        order = {str(item["paper_id"]): index for index, item in enumerate(provisional)}
        records.sort(key=lambda item: order[item["paper_id"]])

        metrics = silver_metrics(records, selected_ids, shortlist_ids)
        distribution = {
            str(label): sum(int(item["relevance"]) == label for item in records)
            for label in range(4)
        }
        usage = _sum_usage(records)
        generation_usage = _sum_usage(records, "generation_usage")
        completed = sum(item["silver_status"] == "completed" for item in records)
        fulltext = sum(bool(item["evidence_scope"]["full_text_available"]) for item in records)
        cited = sum(bool(item["evidence_scope"]["valid_chunk_citations"]) for item in records)
        cache_hits = sum(bool(item.get("cache_hit")) for item in records)
        quality_gate = _quality_gate(metrics, completed, len(records))

        digest_seed = json.dumps(
            {
                "schema": SILVER_SCHEMA_VERSION,
                "idea_id": idea_entry["idea_id"],
                "records": records,
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
        digest = hashlib.sha256(digest_seed).hexdigest()
        judgments_path = output_dir / f"silver_{idea_entry['idea_id']}_{digest[:12]}.jsonl"
        judgments_path.write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
            encoding="utf-8",
        )
        review_path = output_dir / f"human_review_{idea_entry['idea_id']}_{digest[:12]}.jsonl"
        GoldReviewManager.create_template(records, review_path)
        summary = {
            "schema_version": SILVER_SCHEMA_VERSION,
            "idea_id": idea_entry["idea_id"],
            "judge_model": getattr(self.model, "model", type(self.model).__name__),
            "prompt_version": SILVER_PROMPT_VERSION,
            "prompt_hash": self.prompt_hash,
            "source_provisional_path": str(provisional_path),
            "source_batch_report_path": str(batch_report_path),
            "judgment_count": len(records),
            "selected_corpus_count": sum(
                item["paper_id"] in set(selected_ids) for item in records
            ),
            "completed_count": completed,
            "failed_count": len(errors),
            "cache_hit_count": cache_hits,
            "full_text_available_count": fulltext,
            "valid_chunk_citation_count": cited,
            "label_distribution": distribution,
            "metrics": metrics,
            "quality_gate": quality_gate,
            "usage": usage,
            "usage_this_run": usage,
            "generation_usage_for_final_corpus": generation_usage,
            "errors": errors,
            "silver_ready": completed == len(records),
            "human_gold_ready": False,
            "judgments_path": str(judgments_path),
            "human_review_template_path": str(review_path),
            "note": (
                "These are LLM silver labels, not human gold. The review template is reserved "
                "for later human verification and is not counted as completed gold."
            ),
        }
        summary_path = output_dir / f"silver_report_{idea_entry['idea_id']}_{digest[:12]}.json"
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summary["report_path"] = str(summary_path)
        summary["sha256"] = hashlib.sha256(summary_path.read_bytes()).hexdigest()
        return summary

    def _evaluation_pool(
        self,
        provisional: list[dict[str, Any]],
        selected_ids: list[str],
        limit: int,
    ) -> list[dict[str, Any]]:
        by_id = {str(item["paper_id"]): item for item in provisional}
        ordered: list[dict[str, Any]] = []
        seen: set[str] = set()
        for paper_id in [*selected_ids, *(str(item["paper_id"]) for item in provisional)]:
            if paper_id in seen:
                continue
            source = by_id.get(paper_id)
            if source is None:
                row = self.store.get_paper(paper_id)
                if row is None:
                    continue
                source = {
                    "idea_id": "",
                    "paper_id": paper_id,
                    "title": row["title"],
                    "relevance": None,
                    "role": "unjudged",
                    "reason": "Added from the 30-paper screened corpus for full-text silver evaluation.",
                }
            ordered.append(source)
            seen.add(paper_id)
            if len(ordered) >= limit:
                break
        if len(ordered) < 30:
            raise ValueError("Could not construct a 30-paper silver evaluation pool")
        return ordered

    def _judge(
        self,
        idea_entry: dict[str, Any],
        source: dict[str, Any],
        cache_dir: Path,
        max_fulltext_chars: int,
    ) -> dict[str, Any]:
        paper_id = str(source["paper_id"])
        row = self.store.get_paper(paper_id)
        if row is None:
            raise KeyError(f"Paper metadata is missing: {paper_id}")
        abstract = str(row["abstract"] or "").strip()
        evidence = _select_fulltext_evidence(
            self.store.paper_chunks(paper_id), idea_entry, max_fulltext_chars
        )
        request = _judge_request(idea_entry, row, abstract, evidence)
        input_hash = hashlib.sha256(
            json.dumps(request, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        cache_path = cache_dir / f"{paper_id}_{input_hash[:16]}.json"
        if cache_path.is_file():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            cached["cache_hit"] = True
            cached["generation_usage"] = cached.get("generation_usage") or cached.get("usage") or {}
            cached["usage"] = {}
            return cached

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a conservative research relevance judge. Use only supplied evidence. "
                    "Return one syntactically valid JSON object and never call an LLM judgment human gold."
                ),
            },
            {"role": "user", "content": json.dumps(request, ensure_ascii=False)},
        ]
        parsed: dict[str, Any] | None = None
        usage: dict[str, Any] = {}
        for attempt in range(2):
            turn = self.model.complete(messages, [], 1500)
            usage = turn.usage
            try:
                parsed = _parse_json_object(turn.text)
                break
            except json.JSONDecodeError:
                if attempt:
                    raise
                messages.extend([
                    {"role": "assistant", "content": turn.text},
                    {
                        "role": "user",
                        "content": "The response was invalid JSON. Return the judgment as one valid JSON object only.",
                    },
                ])
        assert parsed is not None
        record = _normalize_judgment(
            idea_entry,
            row,
            source,
            parsed,
            abstract,
            evidence,
            input_hash,
            self.prompt_hash,
            self.model,
            usage,
        )
        cache_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return record


class GoldReviewManager:
    """Reserved human-review path; it never upgrades silver labels by itself."""

    @staticmethod
    def create_template(silver_records: list[dict[str, Any]], path: str | Path) -> Path:
        target = Path(path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        values = []
        for item in silver_records:
            values.append({
                "schema_version": GOLD_REVIEW_SCHEMA_VERSION,
                "idea_id": item["idea_id"],
                "paper_id": item["paper_id"],
                "title": item["title"],
                "silver_relevance": item["relevance"],
                "silver_role": item["role"],
                "silver_reason": item["reason"],
                "human_relevance": None,
                "human_role": None,
                "human_reason": "",
                "review_status": "pending_human",
            })
        target.write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in values),
            encoding="utf-8",
        )
        return target

    @staticmethod
    def validate_verified(path: str | Path, minimum: int = 30) -> list[dict[str, Any]]:
        values = _read_jsonl(Path(path))
        verified = [item for item in values if item.get("review_status") == "human_verified"]
        if len(verified) < minimum:
            raise ValueError(f"Only {len(verified)} human-verified records; {minimum} required")
        for item in verified:
            relevance = item.get("human_relevance")
            role = item.get("human_role")
            if not isinstance(relevance, int) or relevance not in range(4):
                raise ValueError(f"Invalid human_relevance for {item.get('paper_id')}")
            if role not in ALLOWED_ROLES:
                raise ValueError(f"Invalid human_role for {item.get('paper_id')}")
            if not str(item.get("human_reason") or "").strip():
                raise ValueError(f"Missing human_reason for {item.get('paper_id')}")
        return verified


def _judge_request(
    idea: dict[str, Any], row: Any, abstract: str, evidence: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "task": "Assign one evidence-grounded silver relevance judgment for this paper and Idea.",
        "idea": {
            "title": idea["title"],
            "research_question": idea["research_question"],
            "claims": idea.get("atomic_claims") or [],
            "success_criteria": idea.get("success_criteria") or [],
        },
        "rubric": {
            "3": "Directly addresses the proposed task/method or is a strong reproducible baseline.",
            "2": "Useful competing method, component, dataset, metric, or evaluation protocol with a material mismatch.",
            "1": "Only weak or indirect inspiration.",
            "0": "Not useful for this Idea.",
        },
        "roles": sorted(ALLOWED_ROLES),
        "rules": [
            "Judge relevance to the supplied Idea, not general paper quality.",
            "Use abstract for topical screening and full-text nodes for method, baseline, dataset, and reproducibility claims.",
            "A baseline role requires full-text support; otherwise use method or weak and state what is missing.",
            "Cite only 'abstract' or supplied chunk_id values.",
            "Formula/table extraction may be imperfect; do not infer unreadable details.",
            "State the strongest mismatch or counterargument.",
        ],
        "paper": {
            "paper_id": row["paper_id"],
            "title": row["title"],
            "year": row["year"],
            "abstract": abstract,
            "full_text_available": bool(evidence),
            "evidence_nodes": evidence,
        },
        "output_schema": {
            "relevance": "integer 0..3",
            "role": "baseline|method|evaluation|weak|irrelevant",
            "confidence": "number 0..1",
            "reason": "concise evidence-grounded string",
            "counterargument": "strongest mismatch or missing evidence",
            "citations": ["abstract or chunk_id"],
        },
    }


def _select_fulltext_evidence(
    rows: list[Any], idea: dict[str, Any], max_chars: int
) -> list[dict[str, Any]]:
    if not rows or max_chars <= 0:
        return []
    category_patterns = {
        "introduction": re.compile(r"\b(?:introduction|motivation|background)\b", re.I),
        "method": re.compile(r"\b(?:method|approach|architecture|algorithm|objective|loss|training)\b", re.I),
        "data": re.compile(r"\b(?:dataset|data set|split|preprocess|augmentation)\b", re.I),
        "experiment": re.compile(r"\b(?:experiment|evaluation|metric|baseline|implementation|setting)\b", re.I),
        "result": re.compile(r"\b(?:result|ablation|comparison|accuracy|f1|memory|runtime)\b", re.I),
        "limitation": re.compile(r"\b(?:limitation|discussion|conclusion|future work)\b", re.I),
    }
    idea_terms = set(tokenize(
        f"{idea['title']} {idea['research_question']} {' '.join(idea.get('atomic_claims') or [])}"
    ))
    candidates = []
    for ordinal, row in enumerate(rows):
        text = str(row["text"] or "").strip()
        if not text:
            continue
        searchable = f"{row['section']} {text}"
        overlap = len(idea_terms & set(tokenize(searchable)))
        categories = [name for name, pattern in category_patterns.items() if pattern.search(searchable)]
        score = overlap + 3 * len(categories) + (2 if row["node_type"] in {"formula", "table"} else 0)
        candidates.append((score, ordinal, row, categories))
    selected: list[tuple[Any, list[str]]] = []
    seen: set[str] = set()
    for category in category_patterns:
        matches = [item for item in candidates if category in item[3]]
        if matches:
            _, _, row, categories = max(matches, key=lambda item: (item[0], -item[1]))
            if row["chunk_id"] not in seen:
                selected.append((row, categories))
                seen.add(row["chunk_id"])
    for _, _, row, categories in sorted(candidates, key=lambda item: (-item[0], item[1])):
        if row["chunk_id"] in seen:
            continue
        selected.append((row, categories))
        seen.add(row["chunk_id"])
        if len(selected) >= 10:
            break
    output = []
    used = 0
    for row, categories in selected:
        text = str(row["text"] or "")[:3000]
        if output and used + len(text) > max_chars:
            continue
        output.append({
            "chunk_id": row["chunk_id"],
            "section": row["section"],
            "page": row["page"],
            "node_type": row["node_type"],
            "categories": categories,
            "text": text,
        })
        used += len(text)
        if used >= max_chars:
            break
    return output


def _normalize_judgment(
    idea: dict[str, Any],
    row: Any,
    source: dict[str, Any],
    value: dict[str, Any],
    abstract: str,
    evidence: list[dict[str, Any]],
    input_hash: str,
    prompt_hash: str,
    model: ModelPort,
    usage: dict[str, Any],
) -> dict[str, Any]:
    relevance = min(3, max(0, int(value.get("relevance", 0))))
    role = str(value.get("role") or "weak").strip().lower()
    if role not in ALLOWED_ROLES:
        role = "weak" if relevance else "irrelevant"
    if relevance == 0:
        role = "irrelevant"
    confidence = min(1.0, max(0.0, float(value.get("confidence") or 0.0)))
    valid = {str(item["chunk_id"]) for item in evidence}
    allowed = valid | ({"abstract"} if abstract else set())
    citations = list(dict.fromkeys(
        str(item) for item in (value.get("citations") or []) if str(item) in allowed
    ))
    if role == "baseline" and not any(item in valid for item in citations):
        role = "method"
        confidence = min(confidence, 0.7)
    return {
        "schema_version": SILVER_SCHEMA_VERSION,
        "idea_id": idea["idea_id"],
        "paper_id": row["paper_id"],
        "title": row["title"],
        "relevance": relevance,
        "role": role,
        "confidence": round(confidence, 4),
        "reason": str(value.get("reason") or "No reason supplied.").strip(),
        "counterargument": str(value.get("counterargument") or "Not supplied.").strip(),
        "citations": citations,
        "evidence_scope": {
            "abstract_available": bool(abstract),
            "full_text_available": bool(evidence),
            "selected_node_count": len(evidence),
            "selected_char_count": sum(len(item["text"]) for item in evidence),
            "valid_chunk_citations": [item for item in citations if item in valid],
        },
        "previous_provisional": {
            "relevance": source.get("relevance"),
            "role": source.get("role"),
            "reason": source.get("reason"),
        },
        "annotation_source": "llm_silver",
        "silver_status": "completed",
        "review_status": "not_human_reviewed",
        "judge_model": getattr(model, "model", type(model).__name__),
        "prompt_hash": prompt_hash,
        "input_hash": input_hash,
        "cache_hit": False,
        "usage": usage,
        "generation_usage": usage,
    }


def _failed_silver_record(
    idea: dict[str, Any], source: dict[str, Any], error: Exception, model: ModelPort
) -> dict[str, Any]:
    relevance = min(3, max(0, int(source.get("relevance") or 0)))
    return {
        "schema_version": SILVER_SCHEMA_VERSION,
        "idea_id": idea["idea_id"],
        "paper_id": str(source.get("paper_id") or ""),
        "title": str(source.get("title") or ""),
        "relevance": relevance,
        "role": str(source.get("role") or "weak"),
        "confidence": 0.0,
        "reason": "Silver judge failed; retained provisional label for audit only.",
        "counterargument": f"{type(error).__name__}: {error}",
        "citations": [],
        "evidence_scope": {
            "abstract_available": False,
            "full_text_available": False,
            "selected_node_count": 0,
            "selected_char_count": 0,
            "valid_chunk_citations": [],
        },
        "previous_provisional": source,
        "annotation_source": "silver_fallback_provisional",
        "silver_status": "failed",
        "review_status": "not_human_reviewed",
        "judge_model": getattr(model, "model", type(model).__name__),
        "prompt_hash": "",
        "input_hash": "",
        "cache_hit": False,
        "usage": {},
        "generation_usage": {},
    }


def silver_metrics(
    records: list[dict[str, Any]], selected_ids: list[str], shortlist_ids: list[str]
) -> dict[str, Any]:
    relevance = {item["paper_id"]: int(item["relevance"]) for item in records}

    def ndcg(ranking: list[str], k: int) -> float:
        gains = [relevance.get(paper_id, 0) for paper_id in ranking[:k]]
        dcg = sum((2 ** gain - 1) / math.log2(index + 2) for index, gain in enumerate(gains))
        ideal = sorted(relevance.values(), reverse=True)[:k]
        idcg = sum((2 ** gain - 1) / math.log2(index + 2) for index, gain in enumerate(ideal))
        return round(dcg / idcg, 4) if idcg else 0.0

    relevant = {paper_id for paper_id, label in relevance.items() if label >= 2}
    shortlist_relevant = relevant & set(shortlist_ids)
    paired = []
    for item in records:
        previous = item["previous_provisional"].get("relevance")
        if isinstance(previous, int) and previous in range(4):
            paired.append((previous, int(item["relevance"])))
    return {
        "ndcg_at_10": ndcg(selected_ids, 10),
        "ndcg_at_30": ndcg(selected_ids, 30),
        "shortlist_precision": round(len(shortlist_relevant) / max(1, len(shortlist_ids)), 4),
        "shortlist_recall": round(len(shortlist_relevant) / max(1, len(relevant)), 4),
        "relevant_count": len(relevant),
        "mean_confidence": round(sum(float(item["confidence"]) for item in records) / max(1, len(records)), 4),
        "evidence_citation_rate": round(
            sum(bool(item["citations"]) for item in records) / max(1, len(records)), 4
        ),
        "provisional_exact_agreement": round(
            sum(first == second for first, second in paired) / max(1, len(paired)), 4
        ),
        "provisional_within_one": round(
            sum(abs(first - second) <= 1 for first, second in paired) / max(1, len(paired)), 4
        ),
        "provisional_mean_absolute_difference": round(
            sum(abs(first - second) for first, second in paired) / max(1, len(paired)), 4
        ),
        "provisional_quadratic_weighted_kappa": _quadratic_weighted_kappa(paired),
        "provisional_comparison_count": len(paired),
        "note": "Metrics use LLM silver labels and are not human-gold results.",
    }


def _quality_gate(metrics: dict[str, Any], completed: int, total: int) -> dict[str, Any]:
    checks = {
        "all_judgments_completed": completed == total,
        "ndcg_at_10_at_least_0_50": metrics["ndcg_at_10"] >= 0.50,
        "shortlist_precision_at_least_0_50": metrics["shortlist_precision"] >= 0.50,
        "evidence_citation_rate_at_least_0_80": metrics["evidence_citation_rate"] >= 0.80,
        "mean_confidence_at_least_0_70": metrics["mean_confidence"] >= 0.70,
    }
    return {
        "status": "pass" if all(checks.values()) else "needs_improvement",
        "checks": checks,
        "policy": "Silver quality gate only; it does not replace human approval.",
    }


def _quadratic_weighted_kappa(pairs: list[tuple[int, int]]) -> float:
    if not pairs:
        return 0.0
    size = 4
    observed = [[0.0] * size for _ in range(size)]
    first_counts = [0.0] * size
    second_counts = [0.0] * size
    for first, second in pairs:
        observed[first][second] += 1.0
        first_counts[first] += 1.0
        second_counts[second] += 1.0
    count = float(len(pairs))
    weighted_observed = 0.0
    weighted_expected = 0.0
    for first in range(size):
        for second in range(size):
            weight = ((first - second) ** 2) / ((size - 1) ** 2)
            weighted_observed += weight * observed[first][second]
            weighted_expected += weight * first_counts[first] * second_counts[second] / count
    if weighted_expected == 0:
        return 1.0 if weighted_observed == 0 else 0.0
    return round(1.0 - weighted_observed / weighted_expected, 4)


def _sum_usage(
    records: list[dict[str, Any]], field: str = "usage"
) -> dict[str, int]:
    keys = ("prompt_tokens", "completion_tokens", "total_tokens")
    return {
        key: sum(int((item.get(field) or {}).get(key) or 0) for item in records)
        for key in keys
    }


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    values = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL line {line_number} is not an object")
        values.append(value)
    return values


def _parse_json_object(text: str) -> dict[str, Any]:
    value = str(text or "").strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*|\s*```$", "", value, flags=re.I)
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        start, end = value.find("{"), value.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(value[start:end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("silver judge output must be a JSON object")
    return parsed

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from researchops.core.models import CompressionRecord, Idea, Paper
from researchops.harness.runtime import ModelPort
from researchops.core.storage import ResearchStore
from researchops.knowledge.retrieval import HybridRetriever


SCHEMA_VERSION = "paper-compression-v1"
COMPRESSOR_VERSION = "hierarchical-compressor-v2-formula-policy"
LEVELS = ("L0", "L1", "L2", "L3")

PROMPTS = {
    "L0": "Bibliographic identity and one-sentence discovery card; no inferred claims.",
    "L1": "Compact screening card for relevance, baseline candidacy, and inspiration from metadata/abstract only.",
    "L2": "Reproduction dossier grounded in parsed nodes, preserving formulas, tables, methods, data, metrics, and citations.",
    "L3": "Question-conditioned evidence synthesis grounded in supplied nodes and explicit node citations.",
}


class PaperCompressor:
    """Versioned hierarchical compression with source- and prompt-addressed caching."""

    def __init__(self, store: ResearchStore, model: ModelPort | None = None,
                 retriever: HybridRetriever | None = None) -> None:
        self.store = store
        self.model = model
        self.retriever = retriever or HybridRetriever(store)

    def l0(self, paper: Paper) -> dict[str, Any]:
        content = {
            "paper_id": paper.paper_id,
            "title": paper.title,
            "authors": list(paper.authors[:12]),
            "year": paper.year,
            "doi": paper.doi,
            "identity": asdict(paper.identity) if paper.identity else None,
            "sources": [asdict(item) for item in paper.source_records],
            "discovery_card": _sentences(paper.abstract or "", 1, 420),
            "evidence_level": "abstract" if paper.abstract else "title_only",
        }
        return self._save(paper.paper_id, "L0", "global", content, _paper_source_hash(paper))

    def l1(self, paper: Paper, idea: Idea) -> dict[str, Any]:
        idea_terms = _terms(f"{idea.research_question} {' '.join(idea.atomic_claims)}")
        paper_text = f"{paper.title} {paper.abstract or ''}"
        matched = sorted(term for term in idea_terms if term in paper_text.casefold())
        content = {
            "paper_id": paper.paper_id,
            "title": paper.title,
            "screening_summary": _sentences(paper.abstract or paper.title, 4, 1400),
            "matched_idea_terms": matched[:20],
            "relevance_score": round(len(matched) / max(1, len(idea_terms)), 4),
            "full_text_available": bool(paper.file_path or paper.pdf_url),
            "possible_roles": [
                "baseline_candidate",
                "method_inspiration",
                "evaluation_inspiration",
            ],
            "limitations": ["Screening uses title/abstract only; method claims require L2 full-text evidence."],
        }
        scope = hashlib.sha256(idea.research_question.encode("utf-8")).hexdigest()[:20]
        return self._save(paper.paper_id, "L1", scope, content, _paper_source_hash(paper))

    def l2_from_analysis(self, paper: Paper, analysis_path: str | Path) -> dict[str, Any]:
        path = Path(analysis_path)
        value = json.loads(path.read_text(encoding="utf-8"))
        keep = (
            "paper_id", "paper_title", "paper_summary", "research_problem", "inputs_outputs",
            "method_overview", "architecture_components", "knowledge_or_graph_role",
            "training_objective", "inference_procedure", "datasets", "data_splits_preprocessing",
            "metrics", "paper_baselines", "reported_results", "implementation_availability",
            "compute_requirements", "limitations", "baseline_assessment", "missing_information",
            "citation_index",
            "formula_policy", "formula_nodes_withheld",
        )
        content = {key: value.get(key) for key in keep}
        content["formula_nodes"] = self._node_refs(paper.paper_id, "formula")
        content["table_nodes"] = self._node_refs(paper.paper_id, "table")
        source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        return self._save(paper.paper_id, "L2", "global", content, source_hash)

    def l3(self, paper: Paper, subproblem: str, *, max_nodes: int = 24) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("L3 compression requires a configured model")
        scope = hashlib.sha256(subproblem.strip().encode("utf-8")).hexdigest()[:20]
        rows = self.store.paper_chunks(paper.paper_id)
        from researchops.papers.validation.automatic import formula_evidence_policy
        hits = self.retriever.search(subproblem, limit=max_nodes, candidate_limit=max(40, max_nodes * 3),
                                     paper_ids={paper.paper_id})
        by_id = {row["chunk_id"]: row for row in rows}
        ranked = [by_id[hit.chunk_id] for hit in hits if hit.chunk_id in by_id
                  and formula_evidence_policy(by_id[hit.chunk_id])["eligible_for_planning"]]
        retrieval = {
            "strategy": "bm25_dense_crossencoder_mmr",
            "embedding": self.retriever.embedding.model_name,
            "embedding_revision": self.retriever.embedding.model_revision,
            "reranker": self.retriever.reranker.model_name,
            "mmr_lambda": self.retriever.mmr_lambda,
            "max_nodes": max_nodes,
            "nodes": [hit.to_dict(include_text=False) for hit in hits],
        }
        source_hash = hashlib.sha256(json.dumps({
            "all_nodes": _rows_source_hash(rows), "retrieval": retrieval,
        }, sort_keys=True).encode()).hexdigest()
        prompt_hash = _prompt_hash("L3", subproblem)
        cached = self.store.get_compression(
            paper.paper_id, "L3", scope, COMPRESSOR_VERSION, prompt_hash, source_hash
        )
        if cached:
            return _row_result(cached, cache_hit=True)
        request = {
            "task": PROMPTS["L3"],
            "paper": {"paper_id": paper.paper_id, "title": paper.title},
            "subproblem": subproblem,
            "rules": [
                "Use only supplied nodes.",
                "Every substantive statement must cite node_id values.",
                "Preserve formulas verbatim and mark uncertain extraction for original-page review.",
                "Source-recovered formulas are inferred silver alignment; use selected_latex with uncertainty. Never treat one formula as sole reproduction truth.",
            ],
            "nodes": [
                {
                    "node_id": row["chunk_id"], "node_type": row["node_type"],
                    "section": row["section"], "page": row["page"], "text": row["text"],
                    "formula_policy": formula_evidence_policy(row),
                }
                for row in ranked
            ],
            "output_schema": {
                "answer": "string", "evidence": [{"node_id": "string", "use": "string"}],
                "formulas": [{"node_id": "string", "latex": "string"}],
                "tables": [{"node_id": "string", "finding": "string"}],
                "unknowns": ["string"],
            },
        }
        turn = self.model.complete(
            [
                {"role": "system", "content": "Return one grounded JSON object only."},
                {"role": "user", "content": json.dumps(request, ensure_ascii=False)},
            ],
            [],
            3000,
        )
        content = _parse_json(turn.text)
        content["retrieval"] = retrieval
        valid = {row["chunk_id"] for row in ranked}
        content["invalid_citations_removed"] = _remove_invalid_node_refs(content, valid)
        return self._save(
            paper.paper_id, "L3", scope, content, source_hash,
            prompt_override=subproblem, model_name=getattr(self.model, "model", None),
        )

    def _node_refs(self, paper_id: str, node_type: str) -> list[dict[str, Any]]:
        return [
            {
                "node_id": row["chunk_id"], "page": row["page"],
                "text": row["text"], "provenance": json.loads(row["provenance_json"] or "{}"),
            }
            for row in self.store.paper_chunks(paper_id)
            if row["node_type"] == node_type
        ]

    def _save(
        self,
        paper_id: str,
        level: str,
        scope_key: str,
        content: dict[str, Any],
        source_hash: str,
        *,
        prompt_override: str = "",
        model_name: str | None = None,
    ) -> dict[str, Any]:
        if level not in LEVELS:
            raise ValueError(f"unsupported compression level: {level}")
        prompt_hash = _prompt_hash(level, prompt_override)
        cached = self.store.get_compression(
            paper_id, level, scope_key, COMPRESSOR_VERSION, prompt_hash, source_hash
        )
        if cached:
            return _row_result(cached, cache_hit=True)
        seed = "\x1f".join((paper_id, level, scope_key, COMPRESSOR_VERSION, prompt_hash, source_hash))
        record = CompressionRecord(
            compression_id="compression_" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:24],
            paper_id=paper_id,
            level=level,
            scope_key=scope_key,
            content=content,
            schema_version=SCHEMA_VERSION,
            compressor_version=COMPRESSOR_VERSION,
            prompt_hash=prompt_hash,
            source_hash=source_hash,
            model_name=model_name,
            token_estimate=max(1, len(json.dumps(content, ensure_ascii=False)) // 4),
        )
        self.store.save_compression(record)
        return {**asdict(record), "cache_hit": False}


def _paper_source_hash(paper: Paper) -> str:
    value = paper.content_hash or json.dumps({
        "title": paper.title, "abstract": paper.abstract, "identity": asdict(paper.identity) if paper.identity else None,
    }, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _rows_source_hash(rows: list[Any]) -> str:
    value = "\x1f".join(str(row["source_hash"] or row["content_hash"] or "") + row["provenance_json"] for row in rows)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _prompt_hash(level: str, extra: str = "") -> str:
    value = f"{SCHEMA_VERSION}\n{COMPRESSOR_VERSION}\n{PROMPTS[level]}\n{extra.strip()}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _terms(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{3,}|[\u3400-\u9fff]{2,}", value.casefold()))


def _sentences(value: str, maximum: int, max_chars: int) -> str:
    pieces = [item.strip() for item in re.split(r"(?<=[.!?。！？])\s+", value) if item.strip()]
    return " ".join(pieces[:maximum])[:max_chars]


def _parse_json(value: str) -> dict[str, Any]:
    text = value.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("compression model output must be a JSON object")
    return payload


def _remove_invalid_node_refs(value: Any, valid: set[str]) -> list[str]:
    removed: list[str] = []
    if isinstance(value, dict):
        for key, item in list(value.items()):
            if key == "node_id" and isinstance(item, str) and item not in valid:
                removed.append(item)
                value[key] = None
            else:
                removed.extend(_remove_invalid_node_refs(item, valid))
    elif isinstance(value, list):
        for item in value:
            removed.extend(_remove_invalid_node_refs(item, valid))
    return removed


def _row_result(row: Any, *, cache_hit: bool) -> dict[str, Any]:
    return {
        "compression_id": row["compression_id"],
        "paper_id": row["paper_id"],
        "level": row["level"],
        "scope_key": row["scope_key"],
        "content": json.loads(row["content_json"]),
        "schema_version": row["schema_version"],
        "compressor_version": row["compressor_version"],
        "prompt_hash": row["prompt_hash"],
        "source_hash": row["source_hash"],
        "model_name": row["model_name"],
        "token_estimate": row["token_estimate"],
        "status": row["status"],
        "cache_hit": cache_hit,
    }

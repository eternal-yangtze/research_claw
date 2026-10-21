from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass

from researchops.core.models import Claim, Evidence, Idea, NoveltyDecision, OverlapLabel
from researchops.knowledge.retrieval import HybridRetriever, SearchHit, lexical_similarity
from researchops.core.storage import ResearchStore


@dataclass(frozen=True)
class PriorWorkMatch:
    paper_id: str
    paper_title: str
    chunk_id: str
    locator: str
    overlap: OverlapLabel
    similarity: float
    excerpt: str
    citation: str


@dataclass(frozen=True)
class ClaimAudit:
    claim_id: str
    claim: str
    matches: tuple[PriorWorkMatch, ...]


class NoveltyAuditor:
    def __init__(self, store: ResearchStore, retriever: HybridRetriever) -> None:
        self.store = store
        self.retriever = retriever

    def refine(self, task_id: str, idea: Idea) -> list[Claim]:
        raw_claims = list(idea.atomic_claims) or _split_claims(idea.description)
        claims = [
            Claim(
                claim_id=_id("claim", task_id, str(index), text),
                task_id=task_id,
                text=text.strip(),
                ordinal=index,
            )
            for index, text in enumerate(raw_claims, start=1)
            if text.strip()
        ]
        if not claims:
            raise ValueError("idea did not yield any atomic claims")
        self.store.save_claims(task_id, claims)
        return claims

    def audit(self, task_id: str, idea: Idea, top_k: int = 3) -> dict:
        claims = self.refine(task_id, idea)
        audits: list[ClaimAudit] = []
        evidence_rows: list[Evidence] = []
        for claim in claims:
            hits = self.retriever.search(claim.text, limit=max(top_k * 30, 90), candidate_limit=240)
            primary_hits = [hit for hit in hits if _is_primary_evidence(hit.section)]
            hits = primary_hits or hits
            distinct: list[SearchHit] = []
            seen_papers: set[str] = set()
            for hit in hits:
                if hit.paper_id in seen_papers:
                    continue
                seen_papers.add(hit.paper_id)
                distinct.append(hit)
                if len(distinct) == top_k:
                    break
            matches: list[PriorWorkMatch] = []
            for hit in distinct:
                similarity = lexical_similarity(claim.text, f"{hit.section} {hit.text}")
                label = _classify(claim.text, hit, similarity)
                excerpt = _excerpt(hit.text, claim.text)
                match = PriorWorkMatch(
                    paper_id=hit.paper_id,
                    paper_title=hit.paper_title,
                    chunk_id=hit.chunk_id,
                    locator=hit.locator,
                    overlap=label,
                    similarity=round(similarity, 4),
                    excerpt=excerpt,
                    citation=hit.citation,
                )
                matches.append(match)
                evidence_rows.append(Evidence(
                    evidence_id=_id("evidence", claim.claim_id, hit.chunk_id),
                    claim_id=claim.claim_id,
                    chunk_id=hit.chunk_id,
                    relation=label.value,
                    score=similarity,
                    excerpt=excerpt,
                    locator=hit.locator,
                ))
            audits.append(ClaimAudit(claim.claim_id, claim.text, tuple(matches)))
        self.store.save_evidence(evidence_rows)
        decision, reasons = _decision(audits)
        return {
            "decision": decision.value,
            "decision_scope": "automated local-corpus screening; human approval required",
            "reasons": reasons,
            "claims": [
                {
                    "claim_id": audit.claim_id,
                    "claim": audit.claim,
                    "matches": [
                        {**asdict(match), "overlap": match.overlap.value}
                        for match in audit.matches
                    ],
                }
                for audit in audits
            ],
        }


class EvidenceAnswerer:
    def __init__(self, store: ResearchStore, retriever: HybridRetriever) -> None:
        self.store = store
        self.retriever = retriever

    def answer(self, question: str, top_k: int = 5) -> dict:
        candidates = self.retriever.search(question, limit=max(top_k * 15, 60), candidate_limit=200)
        primary = [hit for hit in candidates if _is_primary_evidence(hit.section)]
        hits = (primary or candidates)[:top_k]
        if not hits:
            return {
                "answer": "证据不足：当前知识库未检索到可支持回答的内容。",
                "citations": [],
                "verified": True,
                "unsupported_claims": 0,
            }
        statements = []
        citations = []
        for hit in hits:
            excerpt = _excerpt(hit.text, question, limit=260)
            statements.append(f"- {hit.paper_title}（{hit.locator}）：{excerpt} {hit.citation}")
            citations.append({"citation": hit.citation, "chunk_id": hit.chunk_id, "locator": hit.locator, "excerpt": excerpt})
        answer = "当前知识库中与问题最相关的可定位证据如下：\n" + "\n".join(statements)
        verified = True
        for item in citations:
            row = self.store.get_chunk(item["chunk_id"])
            clean_source = " ".join(str(row["text"]).split()) if row is not None else ""
            clean_excerpt = item["excerpt"].removeprefix("...").removesuffix("...")
            if row is None or item["citation"] not in answer or clean_excerpt not in clean_source:
                verified = False
                break
        return {"answer": answer, "citations": citations, "verified": verified, "unsupported_claims": 0 if verified else 1}


def _classify(claim: str, hit: SearchHit, similarity: float) -> OverlapLabel:
    combined = f"{hit.section} {hit.text}".lower()
    if similarity >= 0.88:
        return OverlapLabel.EXACT_DUPLICATE
    if similarity >= 0.58:
        return OverlapLabel.METHOD_OVERLAP
    if similarity >= 0.38:
        if any(term in combined for term in ("dataset", "evaluation", "metric", "实验", "评估", "指标")):
            return OverlapLabel.EVALUATION_OVERLAP
        return OverlapLabel.SETTING_OVERLAP
    if similarity >= 0.2:
        return OverlapLabel.INCREMENTAL_EXTENSION
    return OverlapLabel.UNRESOLVED_GAP


def _is_primary_evidence(section: str) -> bool:
    normalized = section.lower()
    secondary_markers = (
        "relevance", "current research", "research gap", "innovation extraction",
        "novelty", "rq collision", "research question collision", "workload and optimization analysis",
        "for this thesis", "limitations inferred", "writing evidence", "literature table",
        "takeaway",
        "开题", "当前研究", "相关性", "创新提取",
    )
    return not any(marker in normalized for marker in secondary_markers)


def _decision(audits: list[ClaimAudit]) -> tuple[NoveltyDecision, list[str]]:
    labels = [match.overlap for audit in audits for match in audit.matches]
    if not labels or all(label is OverlapLabel.UNRESOLVED_GAP for label in labels):
        return NoveltyDecision.INSUFFICIENT_EVIDENCE, ["The indexed corpus did not provide enough overlapping evidence."]
    if OverlapLabel.EXACT_DUPLICATE in labels:
        return NoveltyDecision.REJECT, ["At least one atomic claim has a high-similarity candidate requiring human duplicate review."]
    strong_claims = sum(
        any(match.overlap in {OverlapLabel.METHOD_OVERLAP, OverlapLabel.SETTING_OVERLAP} for match in audit.matches)
        for audit in audits
    )
    if strong_claims >= max(2, len(audits) // 2 + 1):
        return NoveltyDecision.REFINE, ["Several claims overlap prior methods or settings; narrow the contribution before experimentation."]
    return NoveltyDecision.PROCEED, ["No exact duplicate was detected in the indexed corpus; proceed only after broader search and human review."]


def _split_claims(text: str) -> list[str]:
    parts = re.split(r"[。；;]\s*|(?<=[.!?])\s+", text)
    return [part.strip(" ，,\n") for part in parts if len(part.strip()) >= 12]


def _excerpt(text: str, query: str, limit: int = 360) -> str:
    clean = " ".join(text.split())
    terms = sorted(set(re.findall(r"[a-zA-Z]{3,}|[\u3400-\u9fff]{2,}", query)), key=len, reverse=True)
    positions = [clean.lower().find(term.lower()) for term in terms]
    positions = [position for position in positions if position >= 0]
    start = max(0, (min(positions) if positions else 0) - 80)
    excerpt = clean[start:start + limit]
    if start:
        excerpt = "..." + excerpt
    if start + limit < len(clean):
        excerpt += "..."
    return excerpt


def _id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:20]
    return f"{prefix}_{digest}"

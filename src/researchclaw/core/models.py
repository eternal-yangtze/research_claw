from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class Domain(StringEnum):
    TIME_SERIES = "time_series"
    AI_ML = "ai_ml"
    KNOWLEDGE_GRAPH = "knowledge_graph"


class TaskStatus(StringEnum):
    INTAKE = "INTAKE"
    PROBLEM_REFINED = "PROBLEM_REFINED"
    SEARCH_PLANNED = "SEARCH_PLANNED"
    SEARCH_COMPLETED = "SEARCH_COMPLETED"
    PAPERS_SELECTED = "PAPERS_SELECTED"
    FULL_TEXT_PARTIAL = "FULL_TEXT_PARTIAL"
    FULL_TEXT_READY = "FULL_TEXT_READY"
    NOVELTY_REVIEWED = "NOVELTY_REVIEWED"
    FEASIBILITY_APPROVED = "FEASIBILITY_APPROVED"
    CORPUS_INGESTED = "CORPUS_INGESTED"
    KNOWLEDGE_READY = "KNOWLEDGE_READY"


class NoveltyDecision(StringEnum):
    PROCEED = "PROCEED"
    REFINE = "REFINE"
    REJECT = "REJECT"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class OverlapLabel(StringEnum):
    EXACT_DUPLICATE = "EXACT_DUPLICATE"
    METHOD_OVERLAP = "METHOD_OVERLAP"
    SETTING_OVERLAP = "SETTING_OVERLAP"
    EVALUATION_OVERLAP = "EVALUATION_OVERLAP"
    INCREMENTAL_EXTENSION = "INCREMENTAL_EXTENSION"
    UNRESOLVED_GAP = "UNRESOLVED_GAP"


BOUNDARIES = (
    "PDF equations cannot be guaranteed to parse perfectly; provenance, quality flags, original-page review, and a human gate are required.",
    "A single result above a baseline is not success; reruns, metric recomputation, controls, leakage checks, and artifact verification are required.",
)


@dataclass(frozen=True)
class Idea:
    title: str
    domain: Domain
    description: str
    research_question: str
    atomic_claims: tuple[str, ...] = ()
    success_criteria: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Idea":
        return cls(
            title=str(value["title"]),
            domain=Domain(value["domain"]),
            description=str(value["description"]),
            research_question=str(value["research_question"]),
            atomic_claims=tuple(str(x) for x in value.get("atomic_claims", [])),
            success_criteria=tuple(str(x) for x in value.get("success_criteria", [])),
        )


@dataclass(frozen=True)
class PaperIdentity:
    doi: str | None = None
    arxiv_id: str | None = None
    openalex_id: str | None = None
    openreview_id: str | None = None
    dblp_key: str | None = None
    acl_id: str | None = None
    cvf_id: str | None = None
    title_fingerprint: str = ""


@dataclass(frozen=True)
class PaperSourceRecord:
    source: str
    external_id: str | None = None
    source_url: str | None = None
    pdf_url: str | None = None
    version: str | None = None


@dataclass(frozen=True)
class Paper:
    paper_id: str
    title: str
    authors: tuple[str, ...] = ()
    year: int | None = None
    doi: str | None = None
    source: str = "local"
    source_url: str | None = None
    file_path: str | None = None
    content_hash: str | None = None
    parser_version: str | None = None
    quality_status: str = "accepted"
    abstract: str | None = None
    pdf_url: str | None = None
    external_id: str | None = None
    identity: PaperIdentity | None = None
    source_records: tuple[PaperSourceRecord, ...] = ()
    source_version: str | None = None
    source_archive_url: str | None = None
    source_archive_path: str | None = None


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    paper_id: str
    section: str
    text: str
    page: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    line_start: int | None = None
    line_end: int | None = None
    content_hash: str | None = None
    vector: tuple[float, ...] = ()
    node_type: str = "paragraph"
    parent_chunk_id: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    source_hash: str | None = None


@dataclass(frozen=True)
class CompressionRecord:
    compression_id: str
    paper_id: str
    level: str
    content: dict[str, Any]
    schema_version: str
    compressor_version: str
    prompt_hash: str
    source_hash: str
    scope_key: str = "global"
    model_name: str | None = None
    token_estimate: int = 0
    status: str = "completed"


@dataclass(frozen=True)
class Claim:
    claim_id: str
    task_id: str
    text: str
    ordinal: int


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    claim_id: str
    chunk_id: str
    relation: str
    score: float
    excerpt: str
    locator: str


@dataclass(frozen=True)
class SourceStatus:
    source: str
    status: str
    query: str
    result_count: int = 0
    error: str | None = None
    searched_at: str = ""


@dataclass
class ToolResult:
    status: str
    output: Any = None
    error_type: str | None = None
    error_message: str | None = None
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Protocol, Sequence

from researchops.core.storage import ResearchStore


LATIN_WORD = re.compile(r"[a-zA-Z][a-zA-Z0-9_.-]+")
CJK_RUN = re.compile(r"[\u3400-\u9fff]+")
STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "are", "was", "into",
    "一种", "可以", "进行", "以及", "通过", "方法", "研究", "数据", "时间序列",
}


def tokenize(text: str) -> list[str]:
    lowered = text.lower()
    tokens = [word for word in LATIN_WORD.findall(lowered) if word not in STOPWORDS]
    for run in CJK_RUN.findall(lowered):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[index:index + 2] for index in range(len(run) - 1))
    return tokens


def hashing_vector(text: str, dimensions: int = 384) -> tuple[float, ...]:
    values = [0.0] * dimensions
    for token, count in Counter(tokenize(text)).items():
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        number = int.from_bytes(digest, "big")
        index = number % dimensions
        sign = 1.0 if number & 1 else -1.0
        values[index] += sign * (1.0 + math.log(count))
    return _normalize(values)


def cosine(left: Iterable[float], right: Iterable[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def lexical_similarity(query: str, document: str) -> float:
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return 0.0
    document_tokens = set(tokenize(document))
    coverage = len(query_tokens & document_tokens) / len(query_tokens)
    phrase_bonus = 0.15 if query.strip().lower() in document.lower() else 0.0
    return min(1.0, coverage + phrase_bonus)


class EmbeddingBackend(Protocol):
    model_name: str
    model_revision: str

    def encode(self, texts: Sequence[str]) -> list[tuple[float, ...]]: ...


class RerankerBackend(Protocol):
    model_name: str

    def score(self, query: str, documents: Sequence[str]) -> list[float]: ...


class SentenceTransformerEmbedding:
    """Lazy, local SentenceTransformer inference with a persistent vector cache."""

    def __init__(
        self,
        model_name: str | None = None,
        *,
        revision: str | None = None,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name or os.getenv(
            "RESEARCHOPS_EMBEDDING_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
        self.model_revision = revision or os.getenv("RESEARCHOPS_EMBEDDING_REVISION", "main")
        self.device = device or os.getenv("RESEARCHOPS_EMBEDDING_DEVICE", "cpu")
        self.batch_size = max(1, int(os.getenv("RESEARCHOPS_EMBEDDING_BATCH_SIZE", "32")))
        self._model = None
        self._lock = threading.Lock()

    def encode(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        if not texts:
            return []
        with self._lock:
            if self._model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as error:
                    raise RuntimeError(
                        "Neural retrieval requires sentence-transformers; install the 'retrieval' extra"
                    ) from error
                local_only = _env_bool("RESEARCHOPS_EMBEDDING_LOCAL_ONLY", False)
                self._model = SentenceTransformer(
                    self.model_name,
                    revision=self.model_revision,
                    device=self.device,
                    local_files_only=local_only,
                )
            vectors = self._model.encode(
                list(texts),
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        return [tuple(float(value) for value in vector) for vector in vectors]


class CrossEncoderReranker:
    """Lazy local cross-encoder scoring; raw logits are normalized by the retriever."""

    def __init__(self, model_name: str | None = None, *, device: str | None = None) -> None:
        self.model_name = model_name or os.getenv(
            "RESEARCHOPS_RERANKER_MODEL",
            "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        )
        self.device = device or os.getenv("RESEARCHOPS_RERANKER_DEVICE", "cpu")
        self.batch_size = max(1, int(os.getenv("RESEARCHOPS_RERANKER_BATCH_SIZE", "16")))
        self._model = None
        self._lock = threading.Lock()

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []
        with self._lock:
            if self._model is None:
                try:
                    from sentence_transformers import CrossEncoder
                except ImportError as error:
                    raise RuntimeError(
                        "Neural reranking requires sentence-transformers; install the 'retrieval' extra"
                    ) from error
                local_only = _env_bool("RESEARCHOPS_RERANKER_LOCAL_ONLY", False)
                self._model = CrossEncoder(
                    self.model_name,
                    device=self.device,
                    local_files_only=local_only,
                )
            values = self._model.predict(
                [(query, document) for document in documents],
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
        if hasattr(values, "tolist"):
            values = values.tolist()
        return [float(value[0] if isinstance(value, list) else value) for value in values]


class HashingEmbeddingBackend:
    """Explicit test/offline fallback; never reported as a neural model."""

    model_name = "hashing-vector"
    model_revision = "v1"

    def encode(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        return [hashing_vector(text) for text in texts]


class LexicalReranker:
    model_name = "lexical-overlap"

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        return [lexical_similarity(query, document) for document in documents]


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    paper_id: str
    paper_title: str
    section: str
    text: str
    score: float
    sparse_score: float
    dense_score: float
    rerank_score: float
    mmr_score: float
    locator: str
    embedding_model: str
    reranker_model: str

    @property
    def citation(self) -> str:
        return f"[{self.paper_id}:{self.chunk_id}]"

    def to_dict(self, include_text: bool = True) -> dict:
        value = {
            "chunk_id": self.chunk_id,
            "paper_id": self.paper_id,
            "paper_title": self.paper_title,
            "section": self.section,
            "score": round(self.score, 6),
            "sparse_score": round(self.sparse_score, 6),
            "dense_score": round(self.dense_score, 6),
            "rerank_score": round(self.rerank_score, 6),
            "mmr_score": round(self.mmr_score, 6),
            "embedding_model": self.embedding_model,
            "reranker_model": self.reranker_model,
            "locator": self.locator,
            "citation": self.citation,
        }
        if include_text:
            value["text"] = self.text
        return value


class HybridRetriever:
    """BM25 + neural dense retrieval + cross-encoder reranking + MMR selection."""

    def __init__(
        self,
        store: ResearchStore,
        *,
        embedding: EmbeddingBackend | None = None,
        reranker: RerankerBackend | None = None,
        mmr_lambda: float = 0.75,
    ) -> None:
        self.store = store
        mode = os.getenv("RESEARCHOPS_RETRIEVAL_BACKEND", "neural").strip().lower()
        if mode not in {"neural", "hashing"}:
            raise ValueError("RESEARCHOPS_RETRIEVAL_BACKEND must be neural or hashing")
        self.embedding = embedding or (
            HashingEmbeddingBackend() if mode == "hashing" else SentenceTransformerEmbedding()
        )
        self.reranker = reranker or (
            LexicalReranker() if mode == "hashing" else CrossEncoderReranker()
        )
        if not 0.0 <= mmr_lambda <= 1.0:
            raise ValueError("mmr_lambda must be between 0 and 1")
        self.mmr_lambda = mmr_lambda
        self._index_lock = threading.Lock()

    def search(self, query: str, limit: int = 8, candidate_limit: int = 40,
               *, paper_ids: set[str] | None = None, mode: str = "full") -> list[SearchHit]:
        if mode not in {"bm25", "dense", "hybrid", "full"}:
            raise ValueError("mode must be bm25, dense, hybrid or full")
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        rows = self.store.all_chunks()
        if paper_ids is not None:
            rows = [row for row in rows if row["paper_id"] in paper_ids]
        if not rows:
            return []

        sparse = self._bm25(query, rows)
        fts = self._fts(query, candidate_limit)
        allowed_ids = {row["chunk_id"] for row in rows}
        for chunk_id, score in fts.items():
            if chunk_id not in allowed_ids:
                continue
            sparse[chunk_id] = max(sparse.get(chunk_id, 0.0), score)

        vectors = self._vectors(rows) if mode != "bm25" else {}
        query_vector = self.embedding.encode([query])[0] if mode != "bm25" else ()
        dense = {
            row["chunk_id"]: max(0.0, cosine(query_vector, vectors[row["chunk_id"]]))
            for row in rows
        } if mode != "bm25" else {}
        sparse_rank = self._rank(sparse, candidate_limit)
        dense_rank = self._rank(dense, candidate_limit)
        candidate_ids = sorted(
            set(sparse_rank) | set(dense_rank),
            key=lambda chunk_id: (
                sparse_rank.get(chunk_id, candidate_limit + 1)
                + dense_rank.get(chunk_id, candidate_limit + 1),
                chunk_id,
            ),
        )[:candidate_limit]
        if mode in {"bm25", "dense"}:
            candidate_ids = list(sparse_rank if mode == "bm25" else dense_rank)
        by_id = {row["chunk_id"]: row for row in rows}
        documents = [f"{by_id[chunk_id]['section']} {by_id[chunk_id]['text']}" for chunk_id in candidate_ids]
        raw_rerank = self.reranker.score(query, documents) if mode == "full" else []
        rerank = {
            chunk_id: _sigmoid(score)
            for chunk_id, score in zip(candidate_ids, raw_rerank)
        }
        relevance = {
            chunk_id: (
                0.25 * sparse.get(chunk_id, 0.0)
                + 0.35 * dense.get(chunk_id, 0.0)
                + 0.40 * rerank.get(chunk_id, 0.0)
            )
            for chunk_id in candidate_ids
        } if mode == "full" else {
            chunk_id: (sparse.get(chunk_id, 0.0) if mode == "bm25" else
                       dense.get(chunk_id, 0.0) if mode == "dense" else
                       0.5 * sparse.get(chunk_id, 0.0) + 0.5 * dense.get(chunk_id, 0.0))
            for chunk_id in candidate_ids
        }
        selected = (self._mmr(candidate_ids, relevance, vectors, limit) if mode in {"full", "hybrid"}
                    else [(cid, relevance[cid]) for cid in candidate_ids[:max(0, limit)]])

        hits: list[SearchHit] = []
        for chunk_id, mmr_score in selected:
            row = by_id[chunk_id]
            hits.append(SearchHit(
                chunk_id=chunk_id,
                paper_id=row["paper_id"],
                paper_title=row["paper_title"],
                section=row["section"],
                text=row["text"],
                score=relevance[chunk_id],
                sparse_score=sparse.get(chunk_id, 0.0),
                dense_score=dense.get(chunk_id, 0.0),
                rerank_score=rerank.get(chunk_id, 0.0),
                mmr_score=mmr_score,
                locator=_locator(row),
                embedding_model=f"{self.embedding.model_name}@{self.embedding.model_revision}" if mode != "bm25" else "none",
                reranker_model=self.reranker.model_name if mode == "full" else "none",
            ))
        return hits

    def _vectors(self, rows: list) -> dict[str, tuple[float, ...]]:
        with self._index_lock:
            cached = self.store.chunk_embeddings(
                self.embedding.model_name, self.embedding.model_revision
            )
            missing = []
            vectors: dict[str, tuple[float, ...]] = {}
            for row in rows:
                source_hash = _row_source_hash(row)
                record = cached.get(row["chunk_id"])
                if record is not None and record["source_hash"] == source_hash:
                    vectors[row["chunk_id"]] = tuple(json.loads(record["vector_json"]))
                else:
                    missing.append((row, source_hash))
            if missing:
                encoded = self.embedding.encode([
                    f"{row['section']}\n{row['text']}" for row, _ in missing
                ])
                records = [
                    (row["chunk_id"], source_hash, vector)
                    for (row, source_hash), vector in zip(missing, encoded)
                ]
                self.store.save_chunk_embeddings(
                    self.embedding.model_name, self.embedding.model_revision, records
                )
                vectors.update({chunk_id: tuple(vector) for chunk_id, _, vector in records})
        return vectors

    def _mmr(
        self,
        candidates: list[str],
        relevance: dict[str, float],
        vectors: dict[str, tuple[float, ...]],
        limit: int,
    ) -> list[tuple[str, float]]:
        remaining = set(candidates)
        selected: list[tuple[str, float]] = []
        selected_ids: list[str] = []
        while remaining and len(selected) < max(0, limit):
            best_id = ""
            best_score = float("-inf")
            for chunk_id in sorted(remaining):
                redundancy = max(
                    (cosine(vectors[chunk_id], vectors[chosen]) for chosen in selected_ids),
                    default=0.0,
                )
                score = self.mmr_lambda * relevance[chunk_id] - (1.0 - self.mmr_lambda) * redundancy
                if score > best_score:
                    best_id, best_score = chunk_id, score
            selected.append((best_id, best_score))
            selected_ids.append(best_id)
            remaining.remove(best_id)
        return selected

    def _fts(self, query: str, limit: int) -> dict[str, float]:
        terms = list(dict.fromkeys(LATIN_WORD.findall(query.lower())))
        if not terms:
            return {}
        match_query = " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms[:12])
        try:
            rows = self.store.fts_search(match_query, limit)
        except Exception:
            return {}
        return {row["chunk_id"]: 1.0 / (1.0 + max(0.0, float(row["rank"]))) for row in rows}

    @staticmethod
    def _bm25(query: str, rows: list, k1: float = 1.5, b: float = 0.75) -> dict[str, float]:
        query_terms = list(dict.fromkeys(tokenize(query)))
        documents = {row["chunk_id"]: tokenize(f"{row['section']} {row['text']}") for row in rows}
        average_length = sum(map(len, documents.values())) / max(1, len(documents))
        doc_frequency = defaultdict(int)
        for terms in documents.values():
            for term in set(terms):
                doc_frequency[term] += 1
        count = len(documents)
        scores: dict[str, float] = {}
        for chunk_id, terms in documents.items():
            frequencies = Counter(terms)
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                inverse = math.log(1 + (count - doc_frequency[term] + 0.5) / (doc_frequency[term] + 0.5))
                denominator = frequency + k1 * (1 - b + b * len(terms) / max(1.0, average_length))
                score += inverse * frequency * (k1 + 1) / denominator
            if score:
                scores[chunk_id] = score
        maximum = max(scores.values(), default=1.0)
        return {key: value / maximum for key, value in scores.items()}

    @staticmethod
    def _rank(scores: dict[str, float], limit: int) -> dict[str, int]:
        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]
        return {chunk_id: rank for rank, (chunk_id, _) in enumerate(ordered, start=1)}


def _row_source_hash(row) -> str:
    value = str(row["source_hash"] or row["content_hash"] or "")
    if value:
        return value
    return hashlib.sha256(f"{row['section']}\n{row['text']}".encode("utf-8")).hexdigest()


def _normalize(values: Iterable[float]) -> tuple[float, ...]:
    output = [float(value) for value in values]
    norm = math.sqrt(sum(value * value for value in output))
    return tuple(value / norm for value in output) if norm else tuple(output)


def _sigmoid(value: float) -> float:
    if value >= 0:
        factor = math.exp(-value)
        return 1.0 / (1.0 + factor)
    factor = math.exp(value)
    return factor / (1.0 + factor)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _locator(row) -> str:
    if row["page"] is not None:
        suffix = f", bbox={row['bbox_json']}" if row["bbox_json"] else ""
        return f"page {row['page']}{suffix}"
    if row["line_start"] is not None:
        return f"lines {row['line_start']}-{row['line_end']}"
    return "location unavailable"

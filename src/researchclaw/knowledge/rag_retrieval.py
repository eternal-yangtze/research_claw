"""Published retrieval: Qdrant dense/BM25, RRF, reranking; PG text authority."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Sequence

from qdrant_client.http import models

from researchclaw.knowledge.retrieval import CrossEncoderReranker, SentenceTransformerEmbedding


@dataclass(frozen=True)
class RAGHit:
    chunk_id: str
    parent_id: str
    paper_id: str
    title: str
    page: int | None
    child_text: str
    parent_text: str
    dense_rank: int | None
    lexical_rank: int | None
    rrf_score: float
    rerank_score: float
    quality_grade: str

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class PublishedRAGRetriever:
    def __init__(self, store, *, corpus_version: str, embedding=None, reranker=None,
                 rrf_k: int = 60, parent_word_budget: int = 500,
                 max_hits_per_paper: int = 1):
        self.store = store
        self.corpus_version = corpus_version
        self.embedding = embedding or SentenceTransformerEmbedding()
        self.reranker = reranker or CrossEncoderReranker()
        self.rrf_k = int(rrf_k)
        self.parent_word_budget = max(100, int(parent_word_budget))
        self.max_hits_per_paper = max(1, int(max_hits_per_paper))

    def _focused_parent(self, parent_text: str, child_text: str) -> str:
        words = parent_text.split()
        if len(words) <= self.parent_word_budget:
            return parent_text
        child_words = child_text.split()
        needle = " ".join(child_words[:12])
        joined = " ".join(words)
        prefix = joined.find(needle)
        word_index = len(joined[:prefix].split()) if prefix >= 0 else 0
        start = max(0, word_index - self.parent_word_budget // 3)
        end = min(len(words), start + self.parent_word_budget)
        start = max(0, end - self.parent_word_budget)
        return " ".join(words[start:end])

    def _state(self):
        key = f"{self.embedding.model_name}@{self.embedding.model_revision}"
        with self.store.database.connect(read_only=True) as db:
            row = db.execute("""SELECT * FROM vector_sync_state
                WHERE project_id=? AND corpus_version=? AND embedding_key=? AND status='ready'
                ORDER BY updated_at DESC LIMIT 1""", (self.store.project_id, self.corpus_version, key)).fetchone()
        if not row:
            raise RuntimeError("Published Qdrant vector index is not ready for this corpus/model")
        return dict(row)

    def search(self, query: str, *, limit: int = 6, candidate_limit: int = 40,
               mode: str = 'full', max_hits_per_paper: int | None = None,
               paper_ids: Sequence[str] | None = None, rerank_windowed: bool = False) -> list[RAGHit]:
        if mode not in {'dense', 'bm25', 'rrf', 'full', 'balanced'}:
            raise ValueError('Unknown retrieval mode')
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        state = self._state()
        query_vector = self.embedding.encode([query])[0]
        response = self.store.client.query_points(
            state["version_collection"], query=list(map(float, query_vector)),
            query_filter=models.Filter(must=[
                models.FieldCondition(key="project_id", match=models.MatchValue(value=self.store.project_id)),
                models.FieldCondition(key="corpus_version", match=models.MatchValue(value=self.corpus_version)),
                models.FieldCondition(key="embedding_key", match=models.MatchValue(value=state["embedding_key"])),
                *([models.FieldCondition(key="paper_id", match=models.MatchAny(any=list(paper_ids)))] if paper_ids else []),
            ]), limit=candidate_limit, with_payload=True)
        dense_ids = [str((point.payload or {}).get("chunk_id")) for point in response.points]
        from researchclaw.knowledge.qdrant_lexical import lexical_candidates
        lexical = lexical_candidates(self.store, state, query, candidate_limit, paper_ids=paper_ids)
        self.lexical_backend = 'qdrant_bm25'
        if lexical is None:
            # Compatibility only for historical workspaces not yet indexed.
            lexical = [dict(row) for row in self.store.fts_search(query, candidate_limit)]
            self.lexical_backend = 'postgresql_fts_legacy'
        lexical_ids = [row["chunk_id"] for row in lexical if row.get("chunk_id")]
        if mode == 'dense':
            lexical_ids = []
        elif mode == 'bm25':
            dense_ids = []
        dense_rank = {value: index for index, value in enumerate(dense_ids, 1)}
        lexical_rank = {value: index for index, value in enumerate(lexical_ids, 1)}
        ids = set(dense_rank) | set(lexical_rank)
        fused = {value: (1 / (self.rrf_k + dense_rank[value]) if value in dense_rank else 0)
                        + (1 / (self.rrf_k + lexical_rank[value]) if value in lexical_rank else 0)
                 for value in ids}
        ordered = [value for value, _ in sorted(fused.items(), key=lambda item: (-item[1], item[0]))[:candidate_limit]]
        if not ordered:
            return []
        with self.store.database.connect(read_only=True) as db:
            placeholders = ",".join("?" for _ in ordered)
            rows = [dict(row) for row in db.execute(f"""SELECT c.*,p.title AS paper_title
                FROM chunks c JOIN papers p ON p.paper_id=c.paper_id
                WHERE c.chunk_id IN ({placeholders}) AND c.node_type='child'""", tuple(ordered))]
            by_id = {row["chunk_id"]: row for row in rows}
            parents = {row["chunk_id"]: dict(row) for row in db.execute(
                f"SELECT * FROM chunks WHERE chunk_id IN ({','.join('?' for _ in {r['parent_chunk_id'] for r in rows})})",
                tuple({r["parent_chunk_id"] for r in rows}))} if rows else {}
        candidates = [value for value in ordered if value in by_id]
        documents = [f"{by_id[value]['section']}\n{by_id[value]['text']}" for value in candidates]
        if mode == 'balanced':
            documents = [f"{by_id[value]['paper_title']}\n{text}" for value, text in zip(candidates, documents)]
        scorer = self.reranker.score_windowed if rerank_windowed else self.reranker.score
        raw = scorer(query, documents) if mode in {'full', 'balanced'} else [fused[value] for value in candidates]
        if mode == 'balanced':
            cross_order = sorted(range(len(candidates)), key=lambda i: -float(raw[i]))
            cross_rank = {candidates[i]: rank for rank, i in enumerate(cross_order, 1)}
            # Rank-only fusion avoids comparing logits to reciprocal-rank scores.
            raw = [.75 / (self.rrf_k + rank) + .25 / (self.rrf_k + cross_rank[value])
                   for rank, value in enumerate(candidates, 1)]
        scored = sorted(zip(candidates, raw), key=lambda item: (-float(item[1]), -fused[item[0]], item[0]))
        hits = []
        used_parents = set()
        paper_counts: dict[str, int] = {}
        for chunk_id, score in scored:
            row = by_id[chunk_id]
            parent_id = row["parent_chunk_id"]
            paper_id = row["paper_id"]
            if (parent_id in used_parents
                    or paper_counts.get(paper_id, 0) >= (max_hits_per_paper or self.max_hits_per_paper)):
                continue
            used_parents.add(parent_id)
            paper_counts[paper_id] = paper_counts.get(paper_id, 0) + 1
            provenance = json.loads(row.get("provenance_json") or "{}")
            parent = parents.get(parent_id, {})
            hits.append(RAGHit(chunk_id, parent_id, row["paper_id"], row["paper_title"], row.get("page"),
                               row["text"], self._focused_parent(parent.get("text", row["text"]), row["text"]),
                               dense_rank.get(chunk_id),
                               lexical_rank.get(chunk_id), fused[chunk_id], float(score),
                               provenance.get("quality_grade", "unknown")))
            if len(hits) >= limit:
                break
        return hits

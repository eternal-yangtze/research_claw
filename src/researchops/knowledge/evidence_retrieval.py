"""CPU evidence retrieval: multi-query rank fusion and bounded per-paper reranking.

Uses the project's existing embedding/reranker adapters. No relevance labels,
model training, CUDA, or downloaded framework runtime are required.
"""
from __future__ import annotations

import json
import math
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path

from researchops.core.artifacts import canonical_hash
from .retrieval import HybridRetriever, SearchHit, tokenize, _locator, _sigmoid

VERSION = "cpu-evidence-retrieval-v1"


def checkpoint_key(backend) -> str:
    revision = getattr(backend, "model_revision", "main")
    try:
        from huggingface_hub import try_to_load_from_cache
        cached = try_to_load_from_cache(backend.model_name, "config.json", revision=revision)
        if isinstance(cached, str):
            revision = Path(cached).parent.name
    except (ImportError, ValueError):
        pass
    return f"{backend.model_name}@{revision}"


class EvidenceRetriever:
    def __init__(self, store, *, embedding=None, reranker=None, vector_backend=None):
        self.store = store
        self.base = HybridRetriever(
            store, embedding=embedding, reranker=reranker,
            vector_backend=vector_backend,
        )
        self.embedding, self.reranker = self.base.embedding, self.base.reranker
        self.vector_backend = self.base.vector_backend
        self.embedding_key = checkpoint_key(self.embedding)
        self.reranker_key = checkpoint_key(self.reranker)
        self.signature = None
        self.last_diagnostics = {}
        with sqlite3.connect(store.path) as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS evidence_pair_cache (cache_key TEXT PRIMARY KEY, logit REAL NOT NULL)")

    def _refresh(self):
        import numpy as np
        started = time.perf_counter()
        rows = [dict(r) for r in self.store.all_chunks()]
        keys = [canonical_hash([r["chunk_id"], r["source_hash"], r["section"], r["text"], self.embedding_key]) for r in rows]
        signature = canonical_hash(keys)
        if signature == self.signature:
            return {"changed": False, "encoded_passages": 0, "reused_passages": len(rows), "seconds": time.perf_counter() - started}
        namespace = f"{VERSION}:{self.embedding_key}"
        cached = self.store.chunk_embeddings(self.embedding.model_name, namespace)
        vectors, missing = {}, []
        for row, key in zip(rows, keys):
            record = cached.get(row["chunk_id"])
            if record is not None and record["source_hash"] == key:
                vectors[row["chunk_id"]] = json.loads(record["vector_json"])
            else:
                missing.append((row, key))
        if missing:
            values = self.embedding.encode([f"{r['section']}\n{r['text']}" for r, _ in missing])
            if len(values) != len(missing):
                raise ValueError("Embedding adapter returned incomplete vectors")
            self.store.save_chunk_embeddings(self.embedding.model_name, namespace,
                                            [(r["chunk_id"], key, v) for (r, key), v in zip(missing, values)])
            vectors.update({r["chunk_id"]: v for (r, _), v in zip(missing, values)})
        self.rows, self.keys = rows, dict(zip((r["chunk_id"] for r in rows), keys))
        self.by_id = {r["chunk_id"]: i for i, r in enumerate(rows)}
        self.matrix = np.asarray([vectors[r["chunk_id"]] for r in rows], dtype=np.float32)
        self.vectors = {chunk_id: tuple(vector) for chunk_id, vector in vectors.items()}
        self.postings = defaultdict(list)
        self.lengths = []
        self.title_tokens = []
        for i, row in enumerate(rows):
            terms = Counter(tokenize(f"{row['section']} {row['text']}"))
            self.lengths.append(sum(terms.values()))
            self.title_tokens.append(set(tokenize(row["paper_title"])))
            for term, frequency in terms.items():
                self.postings[term].append((i, frequency))
        self.signature = signature
        return {"changed": True, "encoded_passages": len(missing), "reused_passages": len(rows) - len(missing),
                "seconds": time.perf_counter() - started, "matrix_bytes": int(self.matrix.nbytes)}

    def search(self, query: str, *, queries: list[str] | None = None, limit=40,
               candidate_limit=72, paper_ids: set[str] | None = None) -> list[SearchHit]:
        import numpy as np
        started = time.perf_counter()
        if not query.strip():
            raise ValueError("Query must not be empty")
        if not 1 <= candidate_limit <= 200:
            raise ValueError("candidate_limit must be between 1 and 200")
        refresh = self._refresh()
        if not self.rows or limit <= 0:
            if not self.rows and self.vector_backend.name == "qdrant":
                self.vector_backend.sync([], {})
            self.last_diagnostics = {"refresh": refresh, "returned": 0}
            return []
        query_list = list(dict.fromkeys([query.strip()] + [q.strip() for q in (queries or []) if q.strip()]))[:4]
        allowed = [i for i, r in enumerate(self.rows) if paper_ids is None or r["paper_id"] in paper_ids]
        allowed_set = set(allowed)
        if not allowed:
            return []
        # Corpus statistics use only the declared snapshot, even when the DB holds future papers.
        count = len(allowed)
        average = sum(self.lengths[i] for i in allowed) / count
        dense_vectors = np.asarray(self.embedding.encode(query_list), dtype=np.float32)
        rrf, best_sparse, best_dense = defaultdict(float), defaultdict(float), defaultdict(float)
        def rank(scores):
            return sorted(scores, key=lambda i: (-scores[i], self.rows[i]["chunk_id"]))[:candidate_limit]
        for q_index, q in enumerate(query_list):
            sparse = defaultdict(float)
            for term in dict.fromkeys(tokenize(q)):
                posting = [(i, f) for i, f in self.postings.get(term, []) if i in allowed_set]
                idf = math.log(1 + (count - len(posting) + 0.5) / (len(posting) + 0.5))
                for i, frequency in posting:
                    sparse[i] += idf * frequency * 2.5 / (frequency + 1.5 * (0.25 + 0.75 * self.lengths[i] / max(1, average)))
            dense_by_id = self.base.dense_scores(
                dense_vectors[q_index], rows=self.rows, vectors=self.vectors,
                limit=candidate_limit, paper_ids=paper_ids,
            )
            dense = {
                self.by_id[chunk_id]: score for chunk_id, score in dense_by_id.items()
                if chunk_id in self.by_id and self.by_id[chunk_id] in allowed_set
            }
            sparse_rank, dense_rank = rank(sparse), rank(dense)
            for order in (sparse_rank, dense_rank):
                for position, i in enumerate(order, 1):
                    rrf[i] += 1 / (60 + position)
            maximum = max(sparse.values(), default=1)
            for i, score in sparse.items():
                best_sparse[i] = max(best_sparse[i], score / maximum)
            for i, score in dense.items():
                best_dense[i] = max(best_dense[i], score)
        # Avoid spending the whole cross-encoder budget on many snippets from one paper.
        counts, candidates = Counter(), []
        for i in sorted(rrf, key=lambda j: (-rrf[j], self.rows[j]["chunk_id"])):
            pid = self.rows[i]["paper_id"]
            if counts[pid] >= 3:
                continue
            candidates.append(i)
            counts[pid] += 1
            if len(candidates) >= candidate_limit:
                break
        cache_keys = {i: canonical_hash([VERSION, self.reranker_key, query, self.keys[self.rows[i]["chunk_id"]],
                                         self.rows[i]["paper_title"]]) for i in candidates}
        with sqlite3.connect(self.store.path) as connection:
            parameters = list(cache_keys.values())
            cached = dict(connection.execute("SELECT cache_key, logit FROM evidence_pair_cache WHERE cache_key IN ("
                                             + ",".join("?" for _ in parameters) + ")", parameters))
        missing = [i for i in candidates if cache_keys[i] not in cached]
        if missing:
            scores = self.reranker.score(query, [f"{self.rows[i]['paper_title']}\n{self.rows[i]['section']}\n{self.rows[i]['text']}" for i in missing])
            if len(scores) != len(missing):
                raise ValueError("Reranker adapter returned incomplete scores")
            with sqlite3.connect(self.store.path) as connection:
                connection.executemany("INSERT OR REPLACE INTO evidence_pair_cache VALUES (?, ?)",
                                       [(cache_keys[i], value) for i, value in zip(missing, scores)])
            cached.update({cache_keys[i]: value for i, value in zip(missing, scores)})
        # Combine ranks, not incomparable BM25 maxima / neural logits.
        rerank_order = sorted(candidates, key=lambda i: (-cached[cache_keys[i]], self.rows[i]["chunk_id"]))
        fused_order = sorted(candidates, key=lambda i: (-rrf[i], self.rows[i]["chunk_id"]))
        rerank_positions = {i: p for p, i in enumerate(rerank_order, 1)}
        fused_positions = {i: p for p, i in enumerate(fused_order, 1)}
        score = {i: 0.7 / (60 + rerank_positions[i]) + 0.3 / (60 + fused_positions[i]) for i in candidates}
        ordered = sorted(candidates, key=lambda i: (-score[i], self.rows[i]["chunk_id"]))
        # First round provides distinct papers. Later rounds retain supporting context.
        paper_order, grouped = [], defaultdict(list)
        for i in ordered:
            pid = self.rows[i]["paper_id"]
            if pid not in grouped:
                paper_order.append(pid)
            grouped[pid].append(i)
        selected = [grouped[pid][round_] for round_ in range(3) for pid in paper_order if len(grouped[pid]) > round_][:limit]
        self.last_diagnostics = {"version": VERSION, "snapshot_signature": self.signature, "queries": query_list,
                                 "refresh": refresh, "candidate_passages": len(candidates), "candidate_papers": len(grouped),
                                 "reranked_pairs": len(missing), "cached_pairs": len(candidates) - len(missing),
                                 "returned": len(selected), "seconds": time.perf_counter() - started,
                                 "embedding_checkpoint": self.embedding_key, "reranker_checkpoint": self.reranker_key,
                                 "vector_backend": self.vector_backend.name,
                                 "vector_backend_status": (
                                     self.vector_backend.last_status.__dict__.copy()
                                     if self.vector_backend.last_status is not None else None
                                 )}
        return [SearchHit(chunk_id=(r := self.rows[i])["chunk_id"], paper_id=r["paper_id"],
                          paper_title=r["paper_title"], section=r["section"], text=r["text"],
                          score=score[i], sparse_score=best_sparse[i], dense_score=best_dense[i],
                          rerank_score=_sigmoid(cached[cache_keys[i]]), mmr_score=score[i], locator=_locator(r),
                          embedding_model=self.embedding_key, reranker_model=self.reranker_key) for i in selected]

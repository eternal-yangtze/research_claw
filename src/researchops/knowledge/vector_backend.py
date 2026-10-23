"""Selectable exact-local and Qdrant dense-vector backends."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import uuid
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, Sequence

from researchops.core.artifacts import canonical_hash, workspace_lock


_LOCKS: dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


@dataclass(frozen=True)
class VectorBackendStatus:
    backend: str
    collection: str | None
    version_collection: str | None
    project_id: str
    corpus_version: str
    embedding_key: str
    dimensions: int
    status: str
    indexed_points: int = 0
    upserted_points: int = 0
    deleted_points: int = 0
    error: str | None = None


class DenseVectorBackend(Protocol):
    name: str
    last_status: VectorBackendStatus | None

    def search(
        self,
        query_vector: Sequence[float],
        *,
        rows: list[dict],
        vectors: dict[str, Sequence[float]],
        limit: int,
        paper_ids: set[str] | None = None,
    ) -> dict[str, float]: ...


class LocalExactVectorBackend:
    name = "local"

    def __init__(self) -> None:
        self.last_status = None

    def search(self, query_vector, *, rows, vectors, limit, paper_ids=None):
        if limit <= 0:
            return {}
        allowed = [row for row in rows if paper_ids is None or row["paper_id"] in paper_ids]
        scores = {
            row["chunk_id"]: sum(float(a) * float(b) for a, b in zip(
                query_vector, vectors[row["chunk_id"]]
            ))
            for row in allowed
        }
        ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
        ordered = ordered[:limit]
        return {chunk_id: scores[chunk_id] for chunk_id in ordered}


class QdrantVectorBackend:
    """Qdrant ANN with versioned collections and SQLite sync audit state."""

    name = "qdrant"

    def __init__(
        self,
        store,
        embedding,
        *,
        url: str | None = None,
        alias: str | None = None,
        project_id: str | None = None,
        corpus_version: str | None = None,
        distance: str = "cosine",
        hnsw_m: int = 16,
        hnsw_ef_construct: int = 100,
        search_hnsw_ef: int = 64,
        client=None,
    ) -> None:
        self.store = store
        self.embedding = embedding
        self.url = url or os.getenv("RESEARCHOPS_QDRANT_URL", "http://127.0.0.1:6333")
        self.project_id = project_id or os.getenv("RESEARCHOPS_PROJECT_ID") or _project_id(store.path)
        self.alias = _safe_name(alias or os.getenv("RESEARCHOPS_QDRANT_COLLECTION") or (
            "researchops_" + hashlib.sha256(self.project_id.encode()).hexdigest()[:12]
        ))
        self.corpus_version = corpus_version or os.getenv("RESEARCHOPS_CORPUS_VERSION", "v1")
        self.embedding_key = f"{embedding.model_name}@{embedding.model_revision}"
        self.distance = distance.lower()
        if self.distance != "cosine":
            raise ValueError("P2 currently supports cosine distance only")
        self.hnsw_m = int(hnsw_m)
        self.hnsw_ef_construct = int(hnsw_ef_construct)
        self.search_hnsw_ef = int(search_hnsw_ef)
        if min(self.hnsw_m, self.hnsw_ef_construct, self.search_hnsw_ef) < 1:
            raise ValueError("HNSW parameters must be positive")
        self._client = client
        with _LOCKS_GUARD:
            self._lock = _LOCKS.setdefault(self._backend_key(), threading.RLock())
        self.audit_path = Path(store.path).parent / "vector_audit" / (self._backend_key() + ".jsonl")
        self.last_status: VectorBackendStatus | None = None
        self._ready_signature = None
        self._ensure_state_table()

    @property
    def client(self):
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
            except ImportError as error:
                raise RuntimeError("Qdrant backend requires the 'agent-v03' extra") from error
            self._client = QdrantClient(url=self.url, timeout=30)
        return self._client

    def health(self) -> dict:
        collections = self.client.get_collections().collections
        return {
            "backend": self.name,
            "url": self.url,
            "reachable": True,
            "collections": sorted(item.name for item in collections),
            "active_alias": self.alias,
            "last_status": asdict(self.last_status) if self.last_status else None,
        }

    def search(self, query_vector, *, rows, vectors, limit, paper_ids=None):
        if limit <= 0:
            return {}
        with self._lock:
            status = self.sync(rows, vectors)
            if not rows:
                return {}
            if len(query_vector) != status.dimensions:
                raise ValueError("Query vector dimensions differ from the index")
            models = _qdrant_models()
            conditions = [
                models.FieldCondition(key="project_id", match=models.MatchValue(value=self.project_id)),
                models.FieldCondition(key="corpus_version", match=models.MatchValue(value=self.corpus_version)),
                models.FieldCondition(key="embedding_key", match=models.MatchValue(value=self.embedding_key)),
            ]
            if paper_ids is not None:
                if not paper_ids:
                    return {}
                conditions.append(models.FieldCondition(
                    key="paper_id", match=models.MatchAny(any=sorted(paper_ids))))
            # Pin this request to its version: another project/version may
            # legitimately change the public alias after synchronization.
            query_filter = models.Filter(must=conditions)
            response = self.client.query_points(
                collection_name=status.version_collection,
                query=list(map(float, query_vector)),
                query_filter=query_filter,
                search_params=models.SearchParams(hnsw_ef=self.search_hnsw_ef, exact=False),
                limit=max(1, int(limit)),
                with_payload=True,
            )
            allowed = {row["chunk_id"]: row["paper_id"] for row in rows}
            by_id = {row["chunk_id"]: row for row in rows}
            scores: dict[str, float] = {}
            for point in response.points:
                payload = point.payload or {}
                chunk_id = str(payload.get("chunk_id") or "")
                paper_id = str(payload.get("paper_id") or "")
                if chunk_id not in allowed or allowed[chunk_id] != paper_id:
                    raise RuntimeError("Qdrant returned a point outside the active SQLite corpus")
                if paper_ids is not None and paper_id not in paper_ids:
                    raise RuntimeError("Qdrant server-side paper filter was not enforced")
                if (payload.get("project_id") != self.project_id
                        or payload.get("corpus_version") != self.corpus_version
                        or payload.get("embedding_key") != self.embedding_key
                        or payload.get("content_hash") != _row_hash(by_id[chunk_id])):
                    raise RuntimeError("Qdrant result payload is stale or outside the bound scope")
                scores[chunk_id] = float(point.score)
            self._audit("query_points", {
                "collection": status.version_collection,
                "filter": query_filter.model_dump(mode="json"),
                "query_vector_hash": canonical_hash(list(map(float, query_vector))),
                "hnsw_ef": self.search_hnsw_ef, "limit": limit,
                "results": [{"id": str(p.id), "score": p.score, "payload": p.payload}
                            for p in response.points],
            })
            self.last_status = status
            return scores

    def sync(self, rows: list[dict], vectors: dict[str, Sequence[float]]) -> VectorBackendStatus:
        with self._lock, workspace_lock(
            Path(self.store.path).parent / "vector_locks" / self._backend_key()
        ):
            return self._sync(rows, vectors)

    def _sync(self, rows, vectors):
        if not rows:
            previous = self.sync_state()
            if not previous or not previous["version_collection"]:
                return VectorBackendStatus(
                    self.name, self.alias, None, self.project_id, self.corpus_version,
                    self.embedding_key, 0, "ready")
            existing = self._scroll_project(previous["version_collection"])
            if existing:
                self.client.delete(previous["version_collection"],
                    _qdrant_models().PointIdsList(points=list(existing)), wait=True)
            status = VectorBackendStatus(
                self.name, self.alias, previous["version_collection"], self.project_id,
                self.corpus_version, self.embedding_key, previous["dimensions"], "ready",
                deleted_points=len(existing))
            self._save_status(status, signature=canonical_hash([]))
            self.last_status = status
            return status
        missing = [row["chunk_id"] for row in rows if row["chunk_id"] not in vectors]
        if missing:
            raise ValueError(f"Missing vectors for {len(missing)} chunk(s)")
        dimensions = len(vectors[rows[0]["chunk_id"]])
        if dimensions < 1 or any(len(vectors[row["chunk_id"]]) != dimensions for row in rows):
            raise ValueError("All indexed vectors must have the same positive dimensions")
        if any(not math.isfinite(float(v)) for vector in vectors.values() for v in vector):
            raise ValueError("Index vectors must contain only finite values")
        version_collection = self._version_collection(dimensions)
        vector_hashes = {cid: canonical_hash(list(map(float, vector)))
                         for cid, vector in vectors.items()}
        signature = canonical_hash(sorted(
            (row["chunk_id"], _row_hash(row), vector_hashes[row["chunk_id"]]) for row in rows
        ))
        started = VectorBackendStatus(
            self.name, self.alias, version_collection, self.project_id,
            self.corpus_version, self.embedding_key, dimensions, "syncing",
        )
        self._save_status(started, signature=None)
        try:
            self._ensure_collection(version_collection, dimensions)
            if self._ready_signature == signature:
                info = self.client.get_collection(version_collection)
                if info.points_count == len(rows):
                    self._activate(version_collection)
                    status = VectorBackendStatus(
                        self.name, self.alias, version_collection, self.project_id,
                        self.corpus_version, self.embedding_key, dimensions, "ready",
                        indexed_points=len(rows))
                    self._save_status(status, signature=signature)
                    self.last_status = status
                    return status
            existing = self._scroll_project(version_collection)
            desired_ids = {_point_id(self.project_id, row["chunk_id"]): row for row in rows}
            models = _qdrant_models()
            points = []
            for point_id, row in desired_ids.items():
                content_hash = _row_hash(row)
                current = existing.get(point_id)
                vector_hash = vector_hashes[row["chunk_id"]]
                if (current and (current.payload or {}).get("content_hash") == content_hash
                        and (current.payload or {}).get("vector_hash") == vector_hash
                        and (current.payload or {}).get("paper_id") == row["paper_id"]):
                    continue
                points.append(models.PointStruct(
                    id=point_id,
                    vector=list(map(float, vectors[row["chunk_id"]])),
                    payload={
                        "project_id": self.project_id,
                        "paper_id": row["paper_id"],
                        "chunk_id": row["chunk_id"],
                        "corpus_version": self.corpus_version,
                        "content_hash": content_hash,
                        "vector_hash": vector_hash,
                        "embedding_model": self.embedding.model_name,
                        "embedding_revision": self.embedding.model_revision,
                        "embedding_key": self.embedding_key,
                    },
                ))
            for offset in range(0, len(points), 128):
                self.client.upsert(version_collection, points[offset:offset + 128], wait=True)
            stale_ids = sorted(set(existing) - set(desired_ids))
            if stale_ids:
                self.client.delete(
                    version_collection,
                    models.PointIdsList(points=stale_ids),
                    wait=True,
                )
            self._activate(version_collection)
            status = VectorBackendStatus(
                self.name, self.alias, version_collection, self.project_id,
                self.corpus_version, self.embedding_key, dimensions, "ready",
                indexed_points=len(desired_ids), upserted_points=len(points),
                deleted_points=len(stale_ids),
            )
            self._save_status(status, signature=signature)
            self._audit("sync", {"status": asdict(status), "corpus_signature": signature})
            self.last_status = status
            self._ready_signature = signature
            return status
        except Exception as error:
            self._ready_signature = None
            failed = VectorBackendStatus(
                self.name, self.alias, version_collection, self.project_id,
                self.corpus_version, self.embedding_key, dimensions, "failed",
                error=f"{type(error).__name__}: {error}",
            )
            self._save_status(failed, signature=None)
            self.last_status = failed
            raise

    def delete_chunks(self, chunk_ids: Sequence[str]) -> int:
        values = list(dict.fromkeys(str(value) for value in chunk_ids if str(value)))
        if not values:
            return 0
        models = _qdrant_models()
        aliases = {item.alias_name: item.collection_name for item in self.client.get_aliases().aliases}
        current = self.sync_state()
        collection = current["version_collection"] if current else aliases.get(self.alias, self.alias)
        self.client.delete(
            collection,
            models.PointIdsList(points=[_point_id(self.project_id, value) for value in values]),
            wait=True,
        )
        return len(values)

    def sync_state(self) -> dict | None:
        with sqlite3.connect(self.store.path) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT * FROM vector_sync_state WHERE backend_key=?",
                (self._backend_key(),),
            ).fetchone()
        return dict(row) if row else None

    def _version_collection(self, dimensions: int) -> str:
        digest = hashlib.sha256(json.dumps({
            "schema": "researchops-qdrant-v2",
            "alias": self.alias,
            "project_id": self.project_id,
            "embedding": self.embedding_key,
            "dimensions": dimensions,
            "distance": self.distance,
            "corpus_version": self.corpus_version,
            "hnsw_m": self.hnsw_m,
            "hnsw_ef_construct": self.hnsw_ef_construct,
        }, sort_keys=True).encode("utf-8")).hexdigest()[:12]
        return _safe_name(f"{self.alias}_{digest}")

    def _ensure_collection(self, collection: str, dimensions: int) -> None:
        models = _qdrant_models()
        if not self.client.collection_exists(collection):
            self.client.create_collection(
                collection_name=collection,
                vectors_config=models.VectorParams(
                    size=dimensions,
                    distance=models.Distance.COSINE,
                ),
                hnsw_config=models.HnswConfigDiff(
                    m=self.hnsw_m,
                    ef_construct=self.hnsw_ef_construct,
                    full_scan_threshold=10,
                ),
                optimizers_config=models.OptimizersConfigDiff(indexing_threshold=1),
                metadata={
                    "schema": "researchops-qdrant-v2",
                    "embedding_key": self.embedding_key,
                    "dimensions": dimensions,
                    "distance": self.distance,
                    "corpus_version": self.corpus_version,
                    "project_id": self.project_id,
                },
            )
            for field in ("project_id", "paper_id", "chunk_id", "corpus_version", "embedding_key"):
                self.client.create_payload_index(
                    collection, field, models.PayloadSchemaType.KEYWORD, wait=True)
        info = self.client.get_collection(collection)
        params = info.config.params.vectors
        size = getattr(params, "size", None)
        distance = str(getattr(params, "distance", "")).lower()
        if size != dimensions or "cosine" not in distance:
            raise ValueError(
                f"Qdrant collection contract mismatch: size={size}, distance={distance}"
            )
        metadata = info.config.metadata or {}
        if (metadata.get("embedding_key") != self.embedding_key
                or metadata.get("corpus_version") != self.corpus_version
                or metadata.get("project_id") != self.project_id):
            raise ValueError("Qdrant collection model/version metadata mismatch")

    def _scroll_project(self, collection: str):
        models = _qdrant_models()
        scope = models.Filter(must=[
            models.FieldCondition(key="project_id", match=models.MatchValue(value=self.project_id)),
            models.FieldCondition(key="corpus_version", match=models.MatchValue(value=self.corpus_version)),
            models.FieldCondition(key="embedding_key", match=models.MatchValue(value=self.embedding_key)),
        ])
        output, offset = {}, None
        while True:
            records, offset = self.client.scroll(
                collection, scroll_filter=scope, limit=256, offset=offset,
                with_payload=True, with_vectors=False,
            )
            output.update({str(record.id): record for record in records})
            if offset is None:
                return output

    def _activate(self, collection: str) -> None:
        models = _qdrant_models()
        aliases = {item.alias_name: item.collection_name for item in self.client.get_aliases().aliases}
        active = aliases.get(self.alias)
        if active == collection:
            return
        operations = []
        if active:
            operations.append(models.DeleteAliasOperation(
                delete_alias=models.DeleteAlias(alias_name=self.alias)))
        operations.append(models.CreateAliasOperation(create_alias=models.CreateAlias(
            collection_name=collection, alias_name=self.alias)))
        self.client.update_collection_aliases(operations)
        self._audit("alias_switch", {"alias": self.alias, "previous": active, "active": collection})

    def _audit(self, operation, payload):
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self.audit_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"timestamp": time.time(), "operation": operation,
                                     "payload": payload}, ensure_ascii=False) + "\n")

    def _backend_key(self) -> str:
        return canonical_hash([
            self.url, self.alias, self.project_id, self.corpus_version, self.embedding_key,
        ])

    def _ensure_state_table(self) -> None:
        with sqlite3.connect(self.store.path) as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS vector_sync_state (
                    backend_key TEXT PRIMARY KEY,
                    backend TEXT NOT NULL,
                    collection_name TEXT,
                    version_collection TEXT,
                    project_id TEXT NOT NULL,
                    corpus_version TEXT NOT NULL,
                    embedding_key TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    indexed_points INTEGER NOT NULL DEFAULT 0,
                    upserted_points INTEGER NOT NULL DEFAULT 0,
                    deleted_points INTEGER NOT NULL DEFAULT 0,
                    corpus_signature TEXT,
                    error TEXT,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def _save_status(self, status: VectorBackendStatus, *, signature: str | None) -> None:
        with sqlite3.connect(self.store.path) as connection:
            connection.execute("""
                INSERT INTO vector_sync_state(
                    backend_key,backend,collection_name,version_collection,project_id,
                    corpus_version,embedding_key,dimensions,status,indexed_points,
                    upserted_points,deleted_points,corpus_signature,error,updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)
                ON CONFLICT(backend_key) DO UPDATE SET
                    collection_name=excluded.collection_name,
                    version_collection=excluded.version_collection,
                    dimensions=excluded.dimensions,status=excluded.status,
                    indexed_points=excluded.indexed_points,
                    upserted_points=excluded.upserted_points,
                    deleted_points=excluded.deleted_points,
                    corpus_signature=excluded.corpus_signature,error=excluded.error,
                    updated_at=CURRENT_TIMESTAMP
            """, (
                self._backend_key(), status.backend, status.collection,
                status.version_collection, status.project_id, status.corpus_version,
                status.embedding_key, status.dimensions, status.status,
                status.indexed_points, status.upserted_points, status.deleted_points,
                signature, status.error,
            ))


def build_vector_backend(store, embedding, name: str | None = None) -> DenseVectorBackend:
    selected = (name or os.getenv("RESEARCHOPS_VECTOR_BACKEND", "local")).strip().lower()
    if selected == "local":
        return LocalExactVectorBackend()
    if selected == "qdrant":
        return QdrantVectorBackend(store, embedding)
    raise ValueError("RESEARCHOPS_VECTOR_BACKEND must be local or qdrant")


def _qdrant_models():
    try:
        from qdrant_client.http import models
    except ImportError as error:
        raise RuntimeError("Qdrant backend requires the 'agent-v03' extra") from error
    return models


def _safe_name(value: str) -> str:
    output = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    if not output:
        raise ValueError("Qdrant collection name must contain letters or digits")
    return output[:200]


def _project_id(path: str | Path) -> str:
    return "project_" + hashlib.sha256(str(Path(path).resolve()).encode("utf-8")).hexdigest()[:16]


def _point_id(project_id: str, chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"researchops:{project_id}:{chunk_id}"))


def _row_hash(row) -> str:
    return canonical_hash([row["paper_id"], row["section"], row["text"],
                           row.get("source_hash"), row.get("content_hash")])

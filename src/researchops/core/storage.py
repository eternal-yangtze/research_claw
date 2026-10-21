from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

from researchops.core.models import Chunk, Claim, CompressionRecord, Evidence, Paper, SourceStatus, TaskStatus


SCHEMA = """
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    domain TEXT NOT NULL,
    idea_json TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS papers (
    paper_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    authors_json TEXT NOT NULL,
    year INTEGER,
    doi TEXT,
    source TEXT NOT NULL,
    source_url TEXT,
    file_path TEXT,
    content_hash TEXT,
    parser_version TEXT,
    quality_status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    abstract TEXT,
    pdf_url TEXT,
    external_id TEXT,
    identity_json TEXT,
    source_records_json TEXT,
    source_version TEXT,
    source_archive_url TEXT,
    source_archive_path TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_papers_doi ON papers(doi) WHERE doi IS NOT NULL;

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    section TEXT NOT NULL,
    text TEXT NOT NULL,
    page INTEGER,
    bbox_json TEXT,
    line_start INTEGER,
    line_end INTEGER,
    content_hash TEXT,
    vector_json TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    node_type TEXT NOT NULL DEFAULT 'paragraph',
    parent_chunk_id TEXT,
    provenance_json TEXT NOT NULL DEFAULT '{}',
    source_hash TEXT
);
CREATE INDEX IF NOT EXISTS idx_chunks_paper ON chunks(paper_id, ordinal);

CREATE TABLE IF NOT EXISTS chunk_embeddings (
    chunk_id TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    model_name TEXT NOT NULL,
    model_revision TEXT NOT NULL,
    source_hash TEXT NOT NULL,
    vector_json TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(chunk_id,model_name,model_revision)
);
CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_model
ON chunk_embeddings(model_name,model_revision);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    paper_id UNINDEXED,
    section,
    text,
    tokenize='unicode61 remove_diacritics 2'
);

CREATE TABLE IF NOT EXISTS claims (
    claim_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    UNIQUE(task_id, ordinal)
);

CREATE TABLE IF NOT EXISTS evidence (
    evidence_id TEXT PRIMARY KEY,
    claim_id TEXT NOT NULL REFERENCES claims(claim_id) ON DELETE CASCADE,
    chunk_id TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    relation TEXT NOT NULL,
    score REAL NOT NULL,
    excerpt TEXT NOT NULL,
    locator TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS source_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    source TEXT NOT NULL,
    status TEXT NOT NULL,
    query TEXT NOT NULL,
    result_count INTEGER NOT NULL,
    error TEXT,
    searched_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    task_id TEXT,
    session_id TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    summary_json TEXT
);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id TEXT PRIMARY KEY,
    task_id TEXT,
    run_id TEXT,
    kind TEXT NOT NULL,
    path TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS task_papers (
    task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    paper_id TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    selection_reason TEXT,
    acquired_status TEXT NOT NULL,
    added_at TEXT NOT NULL,
    PRIMARY KEY(task_id, paper_id)
);

CREATE TABLE IF NOT EXISTS paper_analyses (
    task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    paper_id TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    status TEXT NOT NULL,
    analysis_json TEXT NOT NULL,
    artifact_path TEXT,
    analyzer_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(task_id, paper_id)
);

CREATE TABLE IF NOT EXISTS paper_compressions (
    compression_id TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    level TEXT NOT NULL,
    scope_key TEXT NOT NULL,
    content_json TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    compressor_version TEXT NOT NULL,
    prompt_hash TEXT NOT NULL,
    source_hash TEXT NOT NULL,
    model_name TEXT,
    token_estimate INTEGER NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(paper_id,level,scope_key,compressor_version,prompt_hash,source_hash)
);
CREATE INDEX IF NOT EXISTS idx_compressions_lookup
ON paper_compressions(paper_id,level,scope_key,source_hash);

CREATE TABLE IF NOT EXISTS acquisition_stages (
    task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    paper_id TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    stage TEXT NOT NULL,
    status TEXT NOT NULL,
    input_hash TEXT NOT NULL,
    output_json TEXT,
    attempt_count INTEGER NOT NULL,
    run_id TEXT NOT NULL,
    error TEXT,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(task_id,paper_id,stage)
);
CREATE INDEX IF NOT EXISTS idx_acquisition_stages_status
ON acquisition_stages(task_id,status,stage);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ResearchStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(SCHEMA)
            existing = {row[1] for row in connection.execute("PRAGMA table_info(papers)")}
            for name, column_type in (
                ("abstract", "TEXT"),
                ("pdf_url", "TEXT"),
                ("external_id", "TEXT"),
                ("identity_json", "TEXT"),
                ("source_records_json", "TEXT"),
                ("source_version", "TEXT"),
                ("source_archive_url", "TEXT"),
                ("source_archive_path", "TEXT"),
            ):
                if name not in existing:
                    connection.execute(f"ALTER TABLE papers ADD COLUMN {name} {column_type}")
            chunk_columns = {row[1] for row in connection.execute("PRAGMA table_info(chunks)")}
            for name, definition in (
                ("node_type", "TEXT NOT NULL DEFAULT 'paragraph'"),
                ("parent_chunk_id", "TEXT"),
                ("provenance_json", "TEXT NOT NULL DEFAULT '{}'"),
                ("source_hash", "TEXT"),
            ):
                if name not in chunk_columns:
                    connection.execute(f"ALTER TABLE chunks ADD COLUMN {name} {definition}")
            legacy_tasks = connection.execute(
                "SELECT task_id FROM tasks WHERE status='PAPERS_ACQUIRED'"
            ).fetchall()
            for row in legacy_tasks:
                statuses = [
                    value[0]
                    for value in connection.execute(
                        "SELECT acquired_status FROM task_papers WHERE task_id=? AND role='selected'",
                        (row[0],),
                    )
                ]
                ingested = sum(value == "ingested" for value in statuses)
                if statuses and ingested == len(statuses):
                    status = TaskStatus.FULL_TEXT_READY.value
                elif ingested:
                    status = TaskStatus.FULL_TEXT_PARTIAL.value
                elif statuses:
                    status = TaskStatus.PAPERS_SELECTED.value
                else:
                    status = TaskStatus.SEARCH_COMPLETED.value
                connection.execute(
                    "UPDATE tasks SET status=?,updated_at=? WHERE task_id=?",
                    (status, utc_now(), row[0]),
                )

    def create_task(self, task_id: str, title: str, domain: str, idea: dict) -> None:
        now = utc_now()
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO tasks(task_id,title,domain,idea_json,status,created_at,updated_at)
                   VALUES(?,?,?,?,?,?,?)
                   ON CONFLICT(task_id) DO UPDATE SET title=excluded.title,
                   domain=excluded.domain,idea_json=excluded.idea_json,updated_at=excluded.updated_at""",
                (task_id, title, domain, json.dumps(idea, ensure_ascii=False), TaskStatus.INTAKE.value, now, now),
            )

    def update_task_status(self, task_id: str, status: TaskStatus) -> None:
        with self.connect() as connection:
            cursor = connection.execute(
                "UPDATE tasks SET status=?,updated_at=? WHERE task_id=?",
                (status.value, utc_now(), task_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Unknown task: {task_id}")

    def get_task(self, task_id: str) -> sqlite3.Row | None:
        with self.connect() as connection:
            return connection.execute(
                "SELECT * FROM tasks WHERE task_id=?",
                (task_id,),
            ).fetchone()

    def upsert_paper(self, paper: Paper) -> None:
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO papers(
                       paper_id,title,authors_json,year,doi,source,source_url,file_path,
                       content_hash,parser_version,quality_status,created_at,abstract,pdf_url,external_id,
                       identity_json,source_records_json,source_version,source_archive_url,source_archive_path
                   ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(paper_id) DO UPDATE SET
                   title=excluded.title,authors_json=excluded.authors_json,year=excluded.year,
                   doi=COALESCE(excluded.doi,papers.doi),source=excluded.source,
                   source_url=COALESCE(excluded.source_url,papers.source_url),
                   file_path=COALESCE(excluded.file_path,papers.file_path),
                   content_hash=COALESCE(excluded.content_hash,papers.content_hash),
                   parser_version=COALESCE(excluded.parser_version,papers.parser_version),
                   quality_status=excluded.quality_status,
                   abstract=COALESCE(excluded.abstract,papers.abstract),
                   pdf_url=COALESCE(excluded.pdf_url,papers.pdf_url),
                   external_id=COALESCE(excluded.external_id,papers.external_id),
                   identity_json=COALESCE(excluded.identity_json,papers.identity_json),
                   source_records_json=COALESCE(excluded.source_records_json,papers.source_records_json),
                   source_version=COALESCE(excluded.source_version,papers.source_version),
                   source_archive_url=COALESCE(excluded.source_archive_url,papers.source_archive_url),
                   source_archive_path=COALESCE(excluded.source_archive_path,papers.source_archive_path)""",
                (
                    paper.paper_id,
                    paper.title,
                    json.dumps(paper.authors, ensure_ascii=False),
                    paper.year,
                    paper.doi,
                    paper.source,
                    paper.source_url,
                    paper.file_path,
                    paper.content_hash,
                    paper.parser_version,
                    paper.quality_status,
                    utc_now(),
                    paper.abstract,
                    paper.pdf_url,
                    paper.external_id,
                    json.dumps(asdict(paper.identity), ensure_ascii=False) if paper.identity else None,
                    json.dumps([asdict(item) for item in paper.source_records], ensure_ascii=False),
                    paper.source_version,
                    paper.source_archive_url,
                    paper.source_archive_path,
                ),
            )

    def link_task_paper(
        self,
        task_id: str,
        paper_id: str,
        *,
        role: str,
        acquired_status: str,
        selection_reason: str = "",
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO task_papers(task_id,paper_id,role,selection_reason,acquired_status,added_at)
                   VALUES(?,?,?,?,?,?)
                   ON CONFLICT(task_id,paper_id) DO UPDATE SET
                   role=excluded.role,selection_reason=excluded.selection_reason,
                   acquired_status=excluded.acquired_status""",
                (task_id, paper_id, role, selection_reason, acquired_status, utc_now()),
            )

    def task_papers(self, task_id: str) -> list[sqlite3.Row]:
        with self.connect() as connection:
            return list(connection.execute(
                """SELECT p.*,tp.role,tp.selection_reason,tp.acquired_status
                   FROM task_papers tp JOIN papers p ON p.paper_id=tp.paper_id
                   WHERE tp.task_id=? ORDER BY tp.role DESC,p.year DESC,p.title""",
                (task_id,),
            ))

    def get_paper(self, paper_id: str) -> sqlite3.Row | None:
        with self.connect() as connection:
            return connection.execute(
                "SELECT * FROM papers WHERE paper_id=?",
                (paper_id,),
            ).fetchone()

    def replace_chunks(self, paper_id: str, chunks: Iterable[Chunk]) -> int:
        values = list(chunks)
        with self.connect() as connection:
            old_ids = [row[0] for row in connection.execute("SELECT chunk_id FROM chunks WHERE paper_id=?", (paper_id,))]
            if old_ids:
                connection.executemany("DELETE FROM chunks_fts WHERE chunk_id=?", ((x,) for x in old_ids))
            connection.execute("DELETE FROM chunks WHERE paper_id=?", (paper_id,))
            for ordinal, chunk in enumerate(values):
                connection.execute(
                    """INSERT INTO chunks(
                        chunk_id,paper_id,section,text,page,bbox_json,line_start,line_end,
                        content_hash,vector_json,ordinal,node_type,parent_chunk_id,
                        provenance_json,source_hash
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        chunk.chunk_id,
                        chunk.paper_id,
                        chunk.section,
                        chunk.text,
                        chunk.page,
                        json.dumps(chunk.bbox) if chunk.bbox else None,
                        chunk.line_start,
                        chunk.line_end,
                        chunk.content_hash,
                        json.dumps(chunk.vector),
                        ordinal,
                        chunk.node_type,
                        chunk.parent_chunk_id,
                        json.dumps(chunk.provenance, ensure_ascii=False),
                        chunk.source_hash,
                    ),
                )
                connection.execute(
                    "INSERT INTO chunks_fts(chunk_id,paper_id,section,text) VALUES(?,?,?,?)",
                    (chunk.chunk_id, chunk.paper_id, chunk.section, chunk.text),
                )
        return len(values)

    def all_chunks(self) -> list[sqlite3.Row]:
        with self.connect() as connection:
            return list(connection.execute(
                """SELECT c.*,p.title AS paper_title,p.doi,p.file_path
                   FROM chunks c JOIN papers p ON p.paper_id=c.paper_id"""
            ))

    def paper_chunks(self, paper_id: str) -> list[sqlite3.Row]:
        with self.connect() as connection:
            return list(connection.execute(
                """SELECT c.*,p.title AS paper_title,p.file_path
                   FROM chunks c JOIN papers p ON p.paper_id=c.paper_id
                   WHERE c.paper_id=? ORDER BY c.ordinal""",
                (paper_id,),
            ))

    def get_chunk(self, chunk_id: str) -> sqlite3.Row | None:
        with self.connect() as connection:
            return connection.execute(
                """SELECT c.*,p.title AS paper_title,p.doi,p.file_path
                   FROM chunks c JOIN papers p ON p.paper_id=c.paper_id WHERE c.chunk_id=?""",
                (chunk_id,),
            ).fetchone()

    def chunk_embeddings(self, model_name: str, model_revision: str) -> dict[str, sqlite3.Row]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT * FROM chunk_embeddings
                   WHERE model_name=? AND model_revision=?""",
                (model_name, model_revision),
            ).fetchall()
        return {str(row["chunk_id"]): row for row in rows}

    def save_chunk_embeddings(
        self,
        model_name: str,
        model_revision: str,
        records: Iterable[tuple[str, str, Iterable[float]]],
    ) -> int:
        values = [
            (chunk_id, source_hash, tuple(float(value) for value in vector))
            for chunk_id, source_hash, vector in records
        ]
        if not values:
            return 0
        now = utc_now()
        with self.connect() as connection:
            connection.executemany(
                """INSERT INTO chunk_embeddings(
                       chunk_id,model_name,model_revision,source_hash,vector_json,
                       dimensions,created_at,updated_at
                   ) VALUES(?,?,?,?,?,?,?,?)
                   ON CONFLICT(chunk_id,model_name,model_revision) DO UPDATE SET
                   source_hash=excluded.source_hash,vector_json=excluded.vector_json,
                   dimensions=excluded.dimensions,updated_at=excluded.updated_at""",
                (
                    (
                        chunk_id,
                        model_name,
                        model_revision,
                        source_hash,
                        json.dumps(vector),
                        len(vector),
                        now,
                        now,
                    )
                    for chunk_id, source_hash, vector in values
                ),
            )
        return len(values)

    def fts_search(self, match_query: str, limit: int) -> list[sqlite3.Row]:
        with self.connect() as connection:
            return list(connection.execute(
                """SELECT f.chunk_id,f.paper_id,f.section,f.text,bm25(chunks_fts) AS rank,
                          p.title AS paper_title,p.doi,p.file_path,c.page,c.bbox_json,c.line_start,c.line_end,c.vector_json
                   FROM chunks_fts f JOIN papers p ON p.paper_id=f.paper_id
                   JOIN chunks c ON c.chunk_id=f.chunk_id
                   WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?""",
                (match_query, limit),
            ))

    def save_claims(self, task_id: str, claims: Iterable[Claim]) -> None:
        with self.connect() as connection:
            connection.execute("DELETE FROM claims WHERE task_id=?", (task_id,))
            connection.executemany(
                "INSERT INTO claims(claim_id,task_id,text,ordinal) VALUES(?,?,?,?)",
                ((x.claim_id, x.task_id, x.text, x.ordinal) for x in claims),
            )

    def save_evidence(self, evidence: Iterable[Evidence]) -> None:
        values = list(evidence)
        if not values:
            return
        claim_ids = sorted({item.claim_id for item in values})
        with self.connect() as connection:
            connection.executemany("DELETE FROM evidence WHERE claim_id=?", ((x,) for x in claim_ids))
            connection.executemany(
                "INSERT INTO evidence VALUES(?,?,?,?,?,?,?)",
                ((x.evidence_id, x.claim_id, x.chunk_id, x.relation, x.score, x.excerpt, x.locator) for x in values),
            )

    def record_source_status(self, run_id: str, status: SourceStatus) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO source_runs(run_id,source,status,query,result_count,error,searched_at) VALUES(?,?,?,?,?,?,?)",
                (run_id, status.source, status.status, status.query, status.result_count, status.error, status.searched_at),
            )

    def start_run(self, run_id: str, task_id: str, session_id: str) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO runs(run_id,task_id,session_id,status,started_at) VALUES(?,?,?,?,?)",
                (run_id, task_id, session_id, "running", utc_now()),
            )

    def finish_run(self, run_id: str, status: str, summary: dict) -> None:
        with self.connect() as connection:
            connection.execute(
                "UPDATE runs SET status=?,completed_at=?,summary_json=? WHERE run_id=?",
                (status, utc_now(), json.dumps(summary, ensure_ascii=False), run_id),
            )

    def recover_running_runs(self, task_id: str, summary: dict) -> list[str]:
        with self.connect() as connection:
            run_ids = [
                row[0]
                for row in connection.execute(
                    "SELECT run_id FROM runs WHERE task_id=? AND status='running'",
                    (task_id,),
                )
            ]
            connection.executemany(
                "UPDATE runs SET status='completed_recovered',completed_at=?,summary_json=? WHERE run_id=?",
                (
                    (utc_now(), json.dumps(summary, ensure_ascii=False), run_id)
                    for run_id in run_ids
                ),
            )
        return run_ids

    def save_artifact(
        self,
        artifact_id: str,
        task_id: str,
        run_id: str,
        kind: str,
        path: str,
        content_hash: str,
        metadata: dict,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO artifacts VALUES(?,?,?,?,?,?,?,?)
                   ON CONFLICT(artifact_id) DO UPDATE SET path=excluded.path,
                   content_hash=excluded.content_hash,metadata_json=excluded.metadata_json""",
                (artifact_id, task_id, run_id, kind, path, content_hash, json.dumps(metadata, ensure_ascii=False), utc_now()),
            )

    def save_paper_analysis(
        self,
        task_id: str,
        paper_id: str,
        status: str,
        analysis: dict,
        artifact_path: str | None,
        analyzer_version: str,
    ) -> None:
        now = utc_now()
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO paper_analyses(
                       task_id,paper_id,status,analysis_json,artifact_path,
                       analyzer_version,created_at,updated_at
                   ) VALUES(?,?,?,?,?,?,?,?)
                   ON CONFLICT(task_id,paper_id) DO UPDATE SET
                   status=excluded.status,analysis_json=excluded.analysis_json,
                   artifact_path=excluded.artifact_path,
                   analyzer_version=excluded.analyzer_version,updated_at=excluded.updated_at""",
                (
                    task_id,
                    paper_id,
                    status,
                    json.dumps(analysis, ensure_ascii=False),
                    artifact_path,
                    analyzer_version,
                    now,
                    now,
                ),
            )

    def paper_analysis(self, task_id: str, paper_id: str) -> sqlite3.Row | None:
        with self.connect() as connection:
            return connection.execute(
                "SELECT * FROM paper_analyses WHERE task_id=? AND paper_id=?",
                (task_id, paper_id),
            ).fetchone()

    def get_compression(
        self,
        paper_id: str,
        level: str,
        scope_key: str,
        compressor_version: str,
        prompt_hash: str,
        source_hash: str,
    ) -> sqlite3.Row | None:
        with self.connect() as connection:
            return connection.execute(
                """SELECT * FROM paper_compressions
                   WHERE paper_id=? AND level=? AND scope_key=? AND compressor_version=?
                   AND prompt_hash=? AND source_hash=?""",
                (paper_id, level, scope_key, compressor_version, prompt_hash, source_hash),
            ).fetchone()

    def save_compression(self, record: CompressionRecord) -> None:
        now = utc_now()
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO paper_compressions(
                    compression_id,paper_id,level,scope_key,content_json,schema_version,
                    compressor_version,prompt_hash,source_hash,model_name,token_estimate,
                    status,created_at,updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(compression_id) DO UPDATE SET
                    content_json=excluded.content_json,model_name=excluded.model_name,
                    token_estimate=excluded.token_estimate,status=excluded.status,
                    updated_at=excluded.updated_at""",
                (
                    record.compression_id,
                    record.paper_id,
                    record.level,
                    record.scope_key,
                    json.dumps(record.content, ensure_ascii=False),
                    record.schema_version,
                    record.compressor_version,
                    record.prompt_hash,
                    record.source_hash,
                    record.model_name,
                    record.token_estimate,
                    record.status,
                    now,
                    now,
                ),
            )

    def paper_compressions(self, paper_id: str, level: str | None = None) -> list[sqlite3.Row]:
        with self.connect() as connection:
            if level is None:
                rows = connection.execute(
                    """SELECT * FROM paper_compressions
                       WHERE paper_id=? ORDER BY level,created_at""",
                    (paper_id,),
                ).fetchall()
            else:
                rows = connection.execute(
                    """SELECT * FROM paper_compressions
                       WHERE paper_id=? AND level=? ORDER BY created_at""",
                    (paper_id, level),
                ).fetchall()
        return list(rows)

    def acquisition_stage(self, task_id: str, paper_id: str, stage: str) -> sqlite3.Row | None:
        with self.connect() as connection:
            return connection.execute(
                """SELECT * FROM acquisition_stages
                   WHERE task_id=? AND paper_id=? AND stage=?""",
                (task_id, paper_id, stage),
            ).fetchone()

    def start_acquisition_stage(
        self,
        task_id: str,
        paper_id: str,
        stage: str,
        input_hash: str,
        run_id: str,
    ) -> None:
        now = utc_now()
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO acquisition_stages(
                       task_id,paper_id,stage,status,input_hash,output_json,
                       attempt_count,run_id,error,started_at,completed_at,updated_at
                   ) VALUES(?,?,?,?,?,?,1,?,?,?,NULL,?)
                   ON CONFLICT(task_id,paper_id,stage) DO UPDATE SET
                   status='running',input_hash=excluded.input_hash,output_json=NULL,
                   attempt_count=acquisition_stages.attempt_count+1,
                   run_id=excluded.run_id,error=NULL,started_at=excluded.started_at,
                   completed_at=NULL,updated_at=excluded.updated_at""",
                (
                    task_id, paper_id, stage, "running", input_hash, None,
                    run_id, None, now, now,
                ),
            )

    def finish_acquisition_stage(
        self,
        task_id: str,
        paper_id: str,
        stage: str,
        output: dict,
    ) -> None:
        now = utc_now()
        with self.connect() as connection:
            connection.execute(
                """UPDATE acquisition_stages
                   SET status='completed',output_json=?,error=NULL,
                       completed_at=?,updated_at=?
                   WHERE task_id=? AND paper_id=? AND stage=?""",
                (json.dumps(output, ensure_ascii=False), now, now, task_id, paper_id, stage),
            )

    def fail_acquisition_stage(
        self,
        task_id: str,
        paper_id: str,
        stage: str,
        error: str,
    ) -> None:
        now = utc_now()
        with self.connect() as connection:
            connection.execute(
                """UPDATE acquisition_stages
                   SET status='failed',error=?,completed_at=?,updated_at=?
                   WHERE task_id=? AND paper_id=? AND stage=?""",
                (error, now, now, task_id, paper_id, stage),
            )

    def acquisition_stages(self, task_id: str) -> list[sqlite3.Row]:
        with self.connect() as connection:
            return list(connection.execute(
                """SELECT * FROM acquisition_stages WHERE task_id=?
                   ORDER BY paper_id,stage""",
                (task_id,),
            ))

    def counts(self) -> dict[str, int]:
        with self.connect() as connection:
            return {
                table: int(connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0])
                for table in (
                    "tasks", "papers", "chunks", "claims", "evidence", "runs",
                    "artifacts", "task_papers", "paper_analyses", "paper_compressions",
                    "chunk_embeddings", "acquisition_stages",
                )
            }

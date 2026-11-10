"""PostgreSQL authority with an idempotent, transactional Qdrant outbox.

Historical collection and point identities are preserved. SQLite is supported
only by the explicit, read-only importer; no runtime projection is created.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import threading
import uuid

from psycopg import sql

from researchclaw.core.artifacts import write_json
from researchclaw.core.postgres import PostgresDatabase, preflight, encode_text
from researchclaw.core.storage import ResearchStore, SCHEMA

KNOWLEDGE_TABLES = ('papers', 'chunks', 'chunk_embeddings', 'paper_compressions')
KEYS = {'papers': ['paper_id'], 'chunks': ['chunk_id'],
        'chunk_embeddings': ['chunk_id', 'model_name', 'model_revision'],
        'paper_compressions': ['compression_id']}
COPY_TABLES = ('tasks', 'papers', 'chunks', 'chunk_embeddings', 'claims', 'evidence',
               'source_runs', 'runs', 'artifacts', 'task_papers', 'paper_analyses',
               'paper_compressions', 'acquisition_stages')
_STORES, _GUARD = {}, threading.RLock()


def record_id(table, row, keys):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, json.dumps([table, [row[k] for k in keys]], ensure_ascii=False)))


OUTBOX_SCHEMA = """
CREATE TABLE knowledge_outbox (
 id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
 table_name TEXT NOT NULL, operation TEXT NOT NULL, record JSONB NOT NULL,
 status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending','processing','completed','failed')),
 attempts INTEGER NOT NULL DEFAULT 0, error TEXT,
 created_at TIMESTAMPTZ NOT NULL DEFAULT now(), updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_outbox_pending ON knowledge_outbox(id) WHERE status <> 'completed';
CREATE TABLE storage_binding (id INTEGER PRIMARY KEY CHECK(id=1), collection TEXT NOT NULL, imported BOOLEAN NOT NULL DEFAULT false);
CREATE OR REPLACE FUNCTION enqueue_knowledge_change() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
 INSERT INTO knowledge_outbox(table_name,operation,record)
 VALUES(TG_TABLE_NAME,TG_OP,CASE WHEN TG_OP='DELETE' THEN to_jsonb(OLD) ELSE to_jsonb(NEW) END);
 RETURN NULL;
END $$;
""" + '\n'.join(
    f'CREATE TRIGGER {table}_outbox AFTER INSERT OR UPDATE OR DELETE ON {table} '
    'FOR EACH ROW EXECUTE FUNCTION enqueue_knowledge_change();' for table in KNOWLEDGE_TABLES)


class QdrantResearchStore(ResearchStore):
    def __init__(self, workspace, *, client=None, recover=True):
        from qdrant_client import QdrantClient
        self.root = Path(workspace).resolve()
        marker = self.root / 'knowledge-storage.json'
        manifest = json.loads(marker.read_text(encoding='utf-8')) if marker.is_file() else {}
        self.collection = manifest.get('collection') or 'researchclaw_knowledge_' + hashlib.sha256(str(self.root).encode()).hexdigest()[:16]
        self.project_id = self.collection
        self.database = PostgresDatabase(self.root, identity=self.collection)
        self.path = self.root / 'runtime-binding.json'
        self.client = client or QdrantClient(url=os.getenv('RESEARCHCLAW_QDRANT_URL', 'http://127.0.0.1:6333'), timeout=30)
        # Fail before migrations if a bound historical corpus has not been restored.
        if not self.client.collection_exists(self.collection):
            if manifest:
                raise RuntimeError('Bound Qdrant collection missing; restore historical backup first: ' + self.collection)
            self.client.create_collection(self.collection, vectors_config={})
        self.root.mkdir(parents=True, exist_ok=True)
        self.database.migrate('001-research', SCHEMA)
        self.database.migrate('002-knowledge-outbox', OUTBOX_SCHEMA)
        with self.database.connect() as db:
            db.execute('INSERT INTO storage_binding(id,collection) VALUES(1,?) ON CONFLICT DO NOTHING', (self.collection,))
            bound = db.execute('SELECT * FROM storage_binding WHERE id=1').fetchone()
            if bound['collection'] != self.collection:
                raise RuntimeError('PostgreSQL/Qdrant workspace binding mismatch')
            count = db.execute('SELECT count(*) FROM papers').fetchone()[0]
        if not bound['imported'] and not count:
            self.import_restored_knowledge()
        write_json(self.path, {'backend': 'postgresql', 'schema': self.database.schema, 'collection': self.collection})
        if not marker.is_file():
            write_json(marker, {'backend': 'postgresql+qdrant', 'collection': self.collection, 'schema': 2})
        if recover:
            self.flush_outbox()

    @contextmanager
    def connect(self):
        with self.database.connect() as db:
            yield db
        # PG has committed. A Qdrant error is durable, visible, and retryable;
        # it must never roll back or falsely report an atomic cross-store write.
        self.flush_outbox()

    def _records(self):
        offset = None
        while True:
            page, offset = self.client.scroll(self.collection, limit=256, offset=offset, with_payload=True, with_vectors=False)
            for point in page:
                payload = point.payload or {}
                if payload.get('kind') == 'knowledge' and payload.get('table') in KNOWLEDGE_TABLES:
                    yield payload
            if offset is None:
                break

    @staticmethod
    def _insert(db, table, row):
        columns = list(row)
        statement = sql.SQL('INSERT INTO {} ({}) VALUES ({}) ON CONFLICT DO NOTHING').format(
            sql.Identifier(table), sql.SQL(',').join(map(sql.Identifier, columns)),
            sql.SQL(',').join(sql.Placeholder() for _ in columns))
        db.raw.execute(statement, [encode_text(row[column]) for column in columns])

    @staticmethod
    def _insert_many(db, table, rows):
        groups = {}
        for row in rows:
            columns = tuple(sorted(row))
            groups.setdefault(columns, []).append([encode_text(row[column]) for column in columns])
        with db.raw.cursor() as cursor:
            for columns, values in groups.items():
                statement = sql.SQL('INSERT INTO {} ({}) VALUES ({}) ON CONFLICT DO NOTHING').format(
                    sql.Identifier(table), sql.SQL(',').join(map(sql.Identifier, columns)),
                    sql.SQL(',').join(sql.Placeholder() for _ in columns))
                # psycopg executemany pipelines the batch to PostgreSQL.
                cursor.executemany(statement, values)

    def import_restored_knowledge(self):
        """One-time Qdrant -> PG authority bootstrap; never rewrite restored points."""
        records = list(self._records())
        with self.database.connect() as db:
            db.execute('SELECT pg_advisory_xact_lock(hashtext(?))', (self.database.schema + ':bootstrap',))
            if db.execute('SELECT imported FROM storage_binding WHERE id=1').fetchone()[0]:
                return
            for table in KNOWLEDGE_TABLES:
                # Triggers are disabled only inside this import transaction to
                # preserve every original Qdrant payload and revision.
                db.raw.execute(sql.SQL('ALTER TABLE {} DISABLE TRIGGER {}').format(sql.Identifier(table), sql.Identifier(table + '_outbox')))
            for table in KNOWLEDGE_TABLES:
                self._insert_many(db, table, [record['row'] for record in records if record['table'] == table])
            for table in KNOWLEDGE_TABLES:
                db.raw.execute(sql.SQL('ALTER TABLE {} ENABLE TRIGGER {}').format(sql.Identifier(table), sql.Identifier(table + '_outbox')))
            db.execute('UPDATE storage_binding SET imported=true WHERE id=1')

    def flush_outbox(self, *, limit=10000, blocking=False):
        from qdrant_client import models
        processed = 0
        # Serialize all dispatchers in PG (including separate processes).
        with self.database.connect() as db:
            if blocking:
                db.execute('SELECT pg_advisory_xact_lock(hashtext(?))', (self.database.schema + ':outbox',))
            elif not db.execute('SELECT pg_try_advisory_xact_lock(hashtext(?))', (self.database.schema + ':outbox',)).fetchone()[0]:
                return 0
            events = db.execute("SELECT * FROM knowledge_outbox WHERE status <> 'completed' ORDER BY id LIMIT ? FOR UPDATE", (limit,)).fetchall()
            for event in events:
                db.execute("UPDATE knowledge_outbox SET status='processing',attempts=attempts+1,updated_at=now() WHERE id=?", (event['id'],))
                try:
                    table, row = event['table_name'], event['record']
                    identity = record_id(table, row, KEYS[table])
                    if event['operation'] == 'DELETE':
                        self.client.delete(self.collection, models.PointIdsList(points=[identity]), wait=True)
                        if table == 'chunks':
                            self._delete_dense(db, row['chunk_id'])
                    else:
                        self.client.upsert(self.collection, [models.PointStruct(id=identity, vector={}, payload={
                            'kind': 'knowledge', 'table': table, 'row': row})], wait=True)
                    db.execute("UPDATE knowledge_outbox SET status='completed',error=NULL,updated_at=now() WHERE id=?", (event['id'],))
                    processed += 1
                except Exception as error:
                    db.execute("UPDATE knowledge_outbox SET status='failed',error=?,updated_at=now() WHERE id=?", (type(error).__name__, event['id']))
                    break  # Preserve event order; a delete must not overtake an upsert.
        return processed

    def _delete_dense(self, db, chunk_id):
        tables = db.execute("SELECT to_regclass('vector_sync_state')").fetchone()[0]
        if tables is None:
            return
        from qdrant_client import models
        from researchclaw.knowledge.vector_backend import _point_id
        for state in db.execute('SELECT * FROM vector_sync_state'):
            collection = state['version_collection']
            if collection and self.client.collection_exists(collection):
                self.client.delete(collection, models.PointIdsList(points=[_point_id(state['project_id'], chunk_id)]), wait=True)

    def index_status(self):
        with self.database.connect(read_only=True) as db:
            return [dict(row) for row in db.execute(
                "SELECT status,count(*) AS count,max(attempts) AS attempts FROM knowledge_outbox GROUP BY status ORDER BY status")]

    def require_index_ready(self):
        # A concurrent workbench poll may own dispatch. Wait for its transaction
        # instead of reporting transient pending events as a failed deletion.
        self.flush_outbox(blocking=True)
        incomplete = [row for row in self.index_status() if row['status'] != 'completed']
        if incomplete:
            raise RuntimeError('PostgreSQL committed; Qdrant index operations are pending/failed. Retry synchronization; metadata was retained.')

    def copy_from(self, source):
        with source.database.connect(read_only=True) as old:
            data = {table: [dict(row) for row in old.execute('SELECT * FROM ' + table)] for table in COPY_TABLES}
        with self.connect() as db:
            for table in COPY_TABLES:
                self._insert_many(db, table, data[table])
            db.execute("SELECT setval(pg_get_serial_sequence('source_runs','id'),COALESCE(max(id),1),max(id) IS NOT NULL) FROM source_runs")

    def import_database(self, source):
        """Explicit historical import only. No automatic legacy runtime reads."""
        import sqlite3
        from contextlib import closing
        source = Path(source).resolve()
        with closing(sqlite3.connect(source.as_uri() + '?mode=ro', uri=True)) as old:
            old.row_factory = sqlite3.Row
            tables = {row[0] for row in old.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            with self.connect() as db:
                for table in COPY_TABLES:
                    if table not in tables:
                        continue
                    allowed = {row[0] for row in db.execute('SELECT column_name FROM information_schema.columns WHERE table_schema=? AND table_name=?', (self.database.schema, table))}
                    for row in old.execute('SELECT * FROM ' + table):
                        self._insert(db, table, {key: row[key] for key in row.keys() if key in allowed})
                db.execute("SELECT setval(pg_get_serial_sequence('source_runs','id'),COALESCE(max(id),1),max(id) IS NOT NULL) FROM source_runs")

    def import_runtime_database(self, source):
        """Explicit runtime-only history import; never resurrect deleted knowledge."""
        import sqlite3
        from contextlib import closing
        from psycopg.errors import ForeignKeyViolation
        source = Path(source).resolve()
        wal = Path(str(source) + '-wal')
        uri = source.as_uri() + '?mode=ro' + ('&immutable=1' if not wal.exists() or not wal.stat().st_size else '')
        imported, skipped = {}, {}
        with closing(sqlite3.connect(uri, uri=True)) as old:
            old.row_factory = sqlite3.Row
            available = {row[0] for row in old.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            with self.database.connect() as db:
                db.execute('SELECT pg_advisory_xact_lock(hashtext(?))', (self.database.schema + ':runtime-import',))
                for table in (*COPY_TABLES, 'vector_sync_state', 'evidence_pair_cache'):
                    if table in KNOWLEDGE_TABLES or table not in available:
                        continue
                    if table == 'vector_sync_state':
                        from researchclaw.knowledge.vector_backend import VECTOR_STATE_SCHEMA
                        db.executescript(VECTOR_STATE_SCHEMA)
                    elif table == 'evidence_pair_cache':
                        db.execute('CREATE TABLE IF NOT EXISTS evidence_pair_cache (cache_key TEXT PRIMARY KEY, logit REAL NOT NULL)')
                    allowed = {row[0] for row in db.execute('SELECT column_name FROM information_schema.columns WHERE table_schema=? AND table_name=?', (self.database.schema, table))}
                    imported[table], skipped[table] = 0, 0
                    for original in old.execute('SELECT * FROM ' + table):
                        row = {key: original[key] for key in original.keys() if key in allowed}
                        if table in {'runs', 'acquisition_stages'} and row.get('status') == 'running':
                            row['status'] = 'interrupted'
                        try:
                            with db.raw.transaction():
                                self._insert(db, table, row)
                            imported[table] += 1
                        except ForeignKeyViolation:
                            # A paper/chunk may have legitimately been deleted since
                            # the old runtime snapshot; keep the old row in the archive.
                            skipped[table] += 1
                db.execute("SELECT setval(pg_get_serial_sequence('source_runs','id'),COALESCE(max(id),1),max(id) IS NOT NULL) FROM source_runs")
        return {'rows_seen': imported, 'archived_orphans_not_resurrected': skipped}


def build_research_store(workspace):
    mode = os.getenv('RESEARCHCLAW_KNOWLEDGE_BACKEND', 'qdrant').strip()
    if mode not in {'qdrant', 'postgresql+qdrant'}:
        raise RuntimeError('Runtime requires PostgreSQL + Qdrant; SQLite fallback is disabled')
    root = Path(workspace).resolve()
    if (root / 'researchops.db').is_file() and not (root / 'knowledge-storage.json').is_file():
        raise RuntimeError('Unmapped historical SQLite workspace: use an explicit read-only import into a new workspace')
    with _GUARD:
        key = (str(root), os.getenv('RESEARCHCLAW_QDRANT_URL', 'http://127.0.0.1:6333'))
        if key not in _STORES:
            preflight()
            _STORES[key] = QdrantResearchStore(root)
        return _STORES[key]

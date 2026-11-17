"""Isolated, sealed RAG snapshots and transactional PostgreSQL channel pointers.

Publication never updates an existing workspace or a shared Qdrant alias.
Qdrant is not transactional with PG: validate a complete immutable candidate
first, then switch one PG pointer. Readers retain the resolved snapshot.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

from psycopg import sql

from researchclaw.core.artifacts import canonical_hash, file_hash, write_json
from researchclaw.core.postgres import PostgresDatabase
from researchclaw.core.qdrant_store import QdrantResearchStore
from researchclaw.knowledge.vector_backend import _row_hash

ROOT = Path(__file__).resolve().parents[3]
ISOLATED = ROOT / 'self-revisev0.5'
SCHEMA = '''
CREATE TABLE rag_releases (
 release_id TEXT PRIMARY KEY, manifest_json TEXT NOT NULL,
 manifest_hash TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT now());
CREATE TABLE rag_channels (
 channel TEXT PRIMARY KEY, release_id TEXT REFERENCES rag_releases(release_id),
 generation BIGINT NOT NULL DEFAULT 0);
CREATE TABLE rag_release_events (
 id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, channel TEXT NOT NULL,
 previous_release TEXT, release_id TEXT NOT NULL, generation BIGINT NOT NULL,
 reason TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT now());
'''
SEALED_TABLES = ('papers', 'chunks', 'chunk_embeddings', 'paper_compressions',
                 'vector_sync_state', 'storage_binding')
CODE_FILES = ('rag_release.py', 'rag_retrieval.py', 'qdrant_lexical.py',
              'paper_layout.py', 'token_safety.py', 'embedding_contract.py',
              'coverage_retrieval.py', 'evidence_coverage.py', 'evidence_planner.py',
              'evidence_projection.py', 'research_rag.py', 'vector_backend.py',
              'gpu_client.py', 'gpu_service.py', 'trained_reranker.py',
              'neural_models.py', 'qwen_reranker.py', 'retrieval.py')


class ReleaseError(RuntimeError):
    pass


class SealedResearchStore(QdrantResearchStore):
    """Open a verified existing binding without migrations, inserts or recovery."""
    def __init__(self, workspace):
        from qdrant_client import QdrantClient
        import os
        self.root = isolated_path(workspace)
        marker = json.loads((self.root / 'knowledge-storage.json').read_text(encoding='utf8'))
        self.collection = self.project_id = marker['collection']
        self.database = PostgresDatabase(self.root, identity=self.collection)
        self.path = self.root / 'runtime-binding.json'
        self.client = QdrantClient(url=os.getenv('RESEARCHCLAW_QDRANT_URL', 'http://127.0.0.1:6333'), timeout=30)
        binding = json.loads(self.path.read_text(encoding='utf8'))
        if binding['schema'] != self.database.schema or binding['collection'] != self.collection:
            raise ReleaseError('Read-only workspace binding mismatch')

    def connect(self):
        return self.database.connect(read_only=True)

    def flush_outbox(self, *args, **kwargs):
        raise ReleaseError('Sealed snapshot does not perform outbox recovery')


def isolated_path(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ISOLATED.resolve()) or path == ISOLATED.resolve():
        raise ReleaseError('Release paths must be inside a dedicated self-revisev0.5 directory')
    return path


def _name(value):
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]{0,95}', value):
        raise ReleaseError('Invalid release/channel name')
    return value


def _plain(row):
    return dict(row)


def _points(client, collection):
    result, offset = [], None
    while True:
        points, offset = client.scroll(collection, offset=offset, limit=256,
                                       with_payload=True, with_vectors=True)
        result.extend({'id': str(p.id), 'payload': p.payload,
                       'vector': {k: v.model_dump() if hasattr(v, 'model_dump') else v
                                  for k, v in p.vector.items()} if isinstance(p.vector, dict) else p.vector}
                      for p in points)
        if offset is None:
            break
    return sorted(result, key=lambda p: p['id'])


def inventory(store, corpus_version, embedding_key):
    """Full pilot integrity scan, including actual vectors rather than payload claims."""
    with store.database.connect(read_only=True) as db:
        papers = [_plain(r) for r in db.execute('SELECT * FROM papers ORDER BY paper_id')]
        chunks = [_plain(r) for r in db.execute('SELECT * FROM chunks ORDER BY chunk_id')]
        states = [_plain(r) for r in db.execute('''SELECT * FROM vector_sync_state
            WHERE project_id=? AND corpus_version=? AND embedding_key=? AND status='ready' ''',
            (store.project_id, corpus_version, embedding_key))]
        pending = db.execute("SELECT count(*) FROM knowledge_outbox WHERE status<>'completed'").fetchone()[0]
    if len(states) != 1 or pending:
        raise ReleaseError('Exactly one ready dense state and a fully drained outbox required')
    state = states[0]
    children = {r['chunk_id']: r for r in chunks if r['node_type'] == 'child'}
    parents = {r['chunk_id']: r for r in chunks if r['node_type'] == 'parent'}
    bypaper = {r['paper_id']: r for r in papers}
    if not children or not parents or not papers:
        raise ReleaseError('Empty release is not publishable')
    for c in chunks:
        paper = bypaper.get(c['paper_id'])
        if not paper or c['source_hash'] != paper['content_hash']:
            raise ReleaseError('Chunk/PDF source hash mismatch')
        if c['node_type'] == 'child':
            parent = parents.get(c['parent_chunk_id'])
            if not parent or parent['paper_id'] != c['paper_id'] or parent['source_hash'] != c['source_hash']:
                raise ReleaseError('Cross-paper or missing context parent')
        provenance = json.loads(c['provenance_json'])
        if not provenance.get('structure', {}).get('source_spans'):
            raise ReleaseError('Missing structural source spans')
    lexical_path = store.root / 'rag-search-index.json'
    lexical = json.loads(lexical_path.read_text(encoding='utf8'))
    for key, expected in {'project_id': store.project_id, 'corpus_version': corpus_version,
                          'embedding_key': embedding_key, 'dense_collection': state['version_collection'],
                          'document_count': len(children)}.items():
        if lexical.get(key) != expected:
            raise ReleaseError('Dense/sparse/PG release binding mismatch: ' + key)
    dense = _points(store.client, state['version_collection'])
    sparse = _points(store.client, lexical['sparse_collection'])
    dense_ids = [(p.get('payload') or {}).get('chunk_id') for p in dense]
    sparse_ids = [(p.get('payload') or {}).get('chunk_id') for p in sparse]
    if (len(dense_ids) != len(children) or len(sparse_ids) != len(children)
            or set(dense_ids) != set(children) or set(sparse_ids) != set(children)):
        raise ReleaseError('Dense/sparse IDs differ from PG authority')
    for point in dense:
        payload = point['payload']; row = children[payload['chunk_id']]
        if any(payload.get(k) != v for k, v in {
            'content_hash': _row_hash(row), 'project_id': store.project_id,
            'corpus_version': corpus_version, 'embedding_key': embedding_key,
            'paper_id': row['paper_id']}.items()):
            raise ReleaseError('Dense payload/text revision mismatch')
    for point in sparse:
        payload = point['payload']; row = children[payload['chunk_id']]
        if payload.get('paper_id') != row['paper_id'] or payload.get('project_id') != store.project_id:
            raise ReleaseError('Sparse source identity mismatch')
    files = {}
    for p in papers:
        pdf = Path(p['file_path']).resolve()
        if file_hash(pdf) != p['content_hash']:
            raise ReleaseError('Original PDF hash mismatch')
        files[str(pdf)] = p['content_hash']
    for path in sorted((store.root / 'paper-structure').glob('*.json')):
        files[str(path.resolve())] = file_hash(path)
    if len(list((store.root / 'paper-structure').glob('*.json'))) != len(papers):
        raise ReleaseError('Each paper requires a persisted structure artifact')
    files[str(lexical_path.resolve())] = file_hash(lexical_path)
    for marker in ('knowledge-storage.json', 'runtime-binding.json'):
        path = store.root / marker
        files[str(path.resolve())] = file_hash(path)
    return {'schema': store.database.schema, 'project_id': store.project_id,
            'corpus_version': corpus_version, 'embedding_key': embedding_key,
            'dense_collection': state['version_collection'], 'sparse_collection': lexical['sparse_collection'],
            'papers': len(papers), 'parents': len(parents), 'children': len(children),
            'pg_hash': canonical_hash({'papers': papers, 'chunks': chunks, 'state': state}),
            'dense_hash': canonical_hash(dense), 'sparse_hash': canonical_hash(sparse),
            'files': files}


def seal(store):
    """DB triggers prevent normal app writes to published PG authority."""
    statement = '''CREATE FUNCTION reject_sealed_rag_write() RETURNS trigger
      LANGUAGE plpgsql AS $$ BEGIN RAISE EXCEPTION 'sealed RAG release: build a fresh workspace'; END $$;'''
    statement += '\n'.join(f'CREATE TRIGGER sealed_release BEFORE INSERT OR UPDATE OR DELETE OR TRUNCATE '
                           f'ON {table} FOR EACH STATEMENT EXECUTE FUNCTION reject_sealed_rag_write();'
                           for table in SEALED_TABLES)
    store.database.migrate('v05-sealed-rag-release-v1', statement)


@dataclass(frozen=True)
class ReleaseSnapshot:
    registry: object
    release_id: str
    generation: int
    manifest_hash: str
    manifest_json: str

    @property
    def manifest(self):
        return json.loads(self.manifest_json)

    def verify(self, store=None):
        m = self.manifest
        if canonical_hash(m) != self.manifest_hash:
            raise ReleaseError('Manifest hash mismatch')
        for name, digest in m['code'].items():
            if file_hash(ROOT / name) != digest:
                raise ReleaseError('Running retrieval code differs from frozen release')
        for key in ('embedding_adapter', 'reranker_adapter'):
            adapter = m['models'].get(key)
            if adapter and file_hash(adapter['path']) != adapter['sha256']:
                raise ReleaseError('Model adapter hash mismatch')
        for name, digest in m['provenance'].get('files', {}).items():
            if file_hash(name) != digest:
                raise ReleaseError('Frozen embedding/input provenance hash mismatch')
        # Check markers before opening a store so modified bindings cannot route reads elsewhere.
        for name, digest in m['inventory']['files'].items():
            if file_hash(name) != digest:
                raise ReleaseError('Source/artifact hash mismatch: ' + Path(name).name)
        store = store or SealedResearchStore(m['workspace'])
        if store.root.resolve() != Path(m['workspace']).resolve():
            raise ReleaseError('Reader store workspace differs from pinned release')
        actual = inventory(store, m['inventory']['corpus_version'], m['inventory']['embedding_key'])
        if actual != m['inventory']:
            raise ReleaseError('Sealed release PG/dense/sparse integrity mismatch')
        with store.database.connect(read_only=True) as db:
            sealed = db.execute('''SELECT count(*) FROM pg_trigger t JOIN pg_class c ON c.oid=t.tgrelid
                JOIN pg_namespace n ON n.oid=c.relnamespace
                WHERE n.nspname=? AND t.tgname='sealed_release' AND t.tgenabled='O' ''',
                (store.database.schema,)).fetchone()[0]
        if sealed != len(SEALED_TABLES):
            raise ReleaseError('Published PG seal missing or disabled')
        return store

    def retriever(self, *, embedding=None, reranker=None, **kwargs):
        from .rag_retrieval import PublishedRAGRetriever
        store = self.verify()
        return PublishedRAGRetriever(store, corpus_version=self.manifest['inventory']['corpus_version'],
                                     embedding=embedding, reranker=reranker,
                                     release_snapshot=self, **kwargs)


class ReleaseRegistry:
    def __init__(self, root):
        self.root = isolated_path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.database = PostgresDatabase(self.root, kind='release')
        self.database.migrate('rag-release-registry-v1', SCHEMA)

    def register(self, release_id, store, *, corpus_version, embedding_key, models, provenance):
        _name(release_id); isolated_path(store.root)
        if Path(store.root).resolve() == self.root:
            raise ReleaseError('Registry and corpus workspace must differ')
        # A release per immutable workspace. Training artifacts must be ordinary hashed files.
        if not models.get('embedding_revision') or not models.get('reranker_revision'):
            raise ReleaseError('Pinned embedding/reranker revisions required')
        if embedding_key != models.get('embedding_model', '') + '@' + models['embedding_revision']:
            raise ReleaseError('Model manifest differs from dense embedding key')
        for model in ('embedding_adapter', 'reranker_adapter'):
            adapter = models.get(model)
            if adapter and file_hash(adapter['path']) != adapter['sha256']:
                raise ReleaseError('Model adapter hash mismatch')
        inv = inventory(store, corpus_version, embedding_key)
        code = {str((Path('src/researchclaw/knowledge') / n).as_posix()):
                file_hash(ROOT / 'src/researchclaw/knowledge' / n) for n in CODE_FILES}
        for name in ('postgres.py', 'storage.py', 'qdrant_store.py', 'models.py'):
            path = Path('src/researchclaw/core') / name
            code[path.as_posix()] = file_hash(ROOT / path)
        manifest = {'format': 1, 'release_id': release_id, 'workspace': str(store.root.resolve()),
                    'inventory': inv, 'models': models, 'provenance': provenance, 'code': code}
        encoded = json.dumps(manifest, ensure_ascii=False, sort_keys=True)
        # Seal first: a crash here leaves a safe orphan, never an active partial release.
        seal(store)
        if inventory(store, corpus_version, embedding_key) != inv:
            raise ReleaseError('Content changed while sealing')
        with self.database.connect() as db:
            db.execute('INSERT INTO rag_releases(release_id,manifest_json,manifest_hash) VALUES(?,?,?)',
                       (release_id, encoded, canonical_hash(manifest)))
        write_json(self.root / 'manifests' / (release_id + '.json'), manifest)
        return manifest

    def resolve(self, channel):
        _name(channel)
        with self.database.connect(read_only=True) as db:
            row = db.execute('''SELECT r.*,c.generation FROM rag_channels c JOIN rag_releases r
                ON r.release_id=c.release_id WHERE c.channel=?''', (channel,)).fetchone()
        if not row:
            raise ReleaseError('No published release for channel')
        snapshot = ReleaseSnapshot(self, row['release_id'], row['generation'], row['manifest_hash'], row['manifest_json'])
        snapshot.verify()
        return snapshot

    def activate(self, channel, release_id, *, expected_generation, reason='publish', before_commit=None):
        _name(channel); _name(release_id)
        with self.database.connect() as db:
            db.execute('INSERT INTO rag_channels(channel) VALUES(?) ON CONFLICT DO NOTHING', (channel,))
            current = db.execute('SELECT * FROM rag_channels WHERE channel=? FOR UPDATE', (channel,)).fetchone()
            if current['generation'] != expected_generation:
                raise ReleaseError('Stale publication generation; refresh before writing')
            row = db.execute('SELECT * FROM rag_releases WHERE release_id=?', (release_id,)).fetchone()
            if not row:
                raise ReleaseError('Incomplete/unregistered release cannot be published')
            snapshot = ReleaseSnapshot(self, release_id, expected_generation + 1, row['manifest_hash'], row['manifest_json'])
            snapshot.verify()
            db.execute('UPDATE rag_channels SET release_id=?,generation=? WHERE channel=?',
                       (release_id, expected_generation + 1, channel))
            db.execute('''INSERT INTO rag_release_events(channel,previous_release,release_id,generation,reason)
                       VALUES(?,?,?,?,?)''', (channel, current['release_id'], release_id, expected_generation + 1, reason))
            if before_commit:
                before_commit()  # Fault-injection hook: an exception rolls back pointer AND event.
        return snapshot

    def rollback(self, channel, release_id, *, expected_generation):
        with self.database.connect(read_only=True) as db:
            seen = db.execute('SELECT 1 FROM rag_release_events WHERE channel=? AND release_id=?',
                              (channel, release_id)).fetchone()
        if not seen:
            raise ReleaseError('Rollback target has never been published on this channel')
        return self.activate(channel, release_id, expected_generation=expected_generation, reason='rollback')

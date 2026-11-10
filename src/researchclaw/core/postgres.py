"""PostgreSQL connections, workspace isolation and versioned migrations.

Only the public store's parameter marker is adapted. SQL dialect operations
(FTS, DDL, upserts, transactions and catalog queries) are PostgreSQL-native.
"""
from __future__ import annotations

from contextlib import contextmanager, closing
import hashlib
import os
from pathlib import Path
import re

import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo

from researchclaw.core.config import PROJECT_ROOT, load_project_env

TEXT_PREFIX = '__researchclaw_text_v1__:'


def encode_text(value):
    # PostgreSQL TEXT rejects NUL; old PDF parser output can contain it.
    # Preserve those strings reversibly instead of silently deleting evidence.
    if isinstance(value, str) and ('\x00' in value or value.startswith(TEXT_PREFIX)):
        return TEXT_PREFIX + value.replace('\\', '\\\\').replace('\x00', '\\0')
    return value


def decode_value(value):
    if isinstance(value, str) and value.startswith(TEXT_PREFIX):
        return re.sub(r'\\([\\0])', lambda match: '\x00' if match[1] == '0' else '\\', value[len(TEXT_PREFIX):])
    if isinstance(value, dict):
        return {key: decode_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode_value(item) for item in value]
    return value


class Row(dict):
    """Mapping with positional access/iteration for the existing store API."""
    def __getitem__(self, key):
        return tuple(self.values())[key] if isinstance(key, (int, slice)) else super().__getitem__(key)

    def __iter__(self):
        return iter(self.values())


def row_factory(cursor):
    columns = [column.name for column in cursor.description] if cursor.description else []
    return lambda values: Row(zip(columns, map(decode_value, values)))


def connection_info():
    load_project_env()
    configured = os.getenv('RESEARCHCLAW_POSTGRES_DSN')
    if configured:
        return configured
    # The existing local Compose secret is reused, never copied or logged.
    from dotenv import dotenv_values
    password = dotenv_values(PROJECT_ROOT / 'infra/.env').get('POSTGRES_PASSWORD')
    if not password:
        raise RuntimeError('Set RESEARCHCLAW_POSTGRES_DSN or configure the existing infra/.env password')
    # Old .env.web files contain credentials/port 15432 from the retired stack.
    # They must not silently override the already-accepted infra/compose.yaml.
    # Use the one explicit DSN above for any deliberate non-local deployment.
    return make_conninfo(host='127.0.0.1', port='5432', dbname='researchclaw',
                         user='researchclaw', password=password)


def placeholders(statement):
    """Translate unquoted ? parameters; preserve literal question marks/percent."""
    result, quote, i = [], None, 0
    while i < len(statement):
        char = statement[i]
        if char in {"'", '"'}:
            if quote == char and i + 1 < len(statement) and statement[i + 1] == char:
                result.extend([char, char])
                i += 2
                continue
            quote = None if quote == char else char if quote is None else quote
        result.append('%s' if char == '?' and quote is None else '%%' if char == '%' else char)
        i += 1
    return ''.join(result)


class Connection:
    def __init__(self, raw):
        self.raw = raw

    def execute(self, statement, parameters=None):
        if parameters is None:
            return self.raw.execute(statement)
        return self.raw.execute(placeholders(statement), [encode_text(value) for value in parameters])

    def executemany(self, statement, parameters):
        cursor = self.raw.cursor()
        cursor.executemany(placeholders(statement), ([encode_text(value) for value in row] for row in parameters))
        return cursor

    def executescript(self, statement):
        return self.raw.execute(statement, prepare=False)


class PostgresDatabase:
    def __init__(self, workspace, *, kind='research', identity=None):
        self.root = Path(workspace).resolve()
        from .config import workspace_identity
        digest = hashlib.sha256((identity or workspace_identity(self.root)).encode()).hexdigest()[:24]
        self.schema = f'rc_{kind}_{digest}'
        if not re.fullmatch(r'rc_[a-z]+_[0-9a-f]{24}', self.schema):
            raise ValueError('Invalid workspace schema')

    @contextmanager
    def connect(self, *, autocommit=False, read_only=False):
        try:
            raw = psycopg.connect(connection_info(), connect_timeout=5, autocommit=autocommit,
                                   row_factory=row_factory)
        except psycopg.Error:
            raise RuntimeError('PostgreSQL unavailable or credentials invalid; SQLite fallback is disabled') from None
        with raw:
            raw.execute(sql.SQL('SET search_path TO {}, public').format(sql.Identifier(self.schema)))
            if read_only:
                raw.execute('SET TRANSACTION READ ONLY')
            yield Connection(raw)

    def migrate(self, version, statement):
        checksum = hashlib.sha256(statement.encode()).hexdigest()
        with self.connect() as db:
            db.execute('SELECT pg_advisory_xact_lock(hashtext(?))', (self.schema,))
            db.raw.execute(sql.SQL('CREATE SCHEMA IF NOT EXISTS {}').format(sql.Identifier(self.schema)))
            db.execute('CREATE TABLE IF NOT EXISTS schema_migrations (version TEXT PRIMARY KEY, checksum TEXT NOT NULL, applied_at TIMESTAMPTZ NOT NULL DEFAULT now())')
            previous = db.execute('SELECT checksum FROM schema_migrations WHERE version=?', (version,)).fetchone()
            if previous:
                if previous['checksum'] != checksum:
                    raise RuntimeError('Applied migration checksum changed; create a new migration version')
                return
            db.executescript(statement)
            db.execute('INSERT INTO schema_migrations(version,checksum) VALUES(?,?)', (version, checksum))


def preflight():
    """Probe both selected services without creating schemas or collections."""
    try:
        with psycopg.connect(connection_info(), connect_timeout=5) as db:
            db.execute('SELECT 1')
    except psycopg.Error:
        raise RuntimeError('PostgreSQL preflight failed; SQLite fallback is disabled') from None
    from qdrant_client import QdrantClient
    try:
        with closing(QdrantClient(url=os.getenv('RESEARCHCLAW_QDRANT_URL', 'http://127.0.0.1:6333'), timeout=5)) as client:
            client.get_collections()
    except Exception:
        raise RuntimeError('Qdrant preflight failed; SQLite fallback is disabled') from None

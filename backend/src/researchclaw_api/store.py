import json
from contextlib import contextmanager
from pathlib import Path
from researchclaw.core.postgres import PostgresDatabase, encode_text
from .conversation import ConversationStore, empty_state

WEB_SCHEMA = '''
CREATE TABLE sessions (
 id TEXT PRIMARY KEY, owner TEXT NOT NULL, project TEXT NOT NULL,
 messages TEXT NOT NULL DEFAULT '[]', title TEXT NOT NULL DEFAULT '新对话',
 memory TEXT NOT NULL DEFAULT '', created DOUBLE PRECISION NOT NULL DEFAULT 0);
CREATE TABLE runs (
 id TEXT PRIMARY KEY, session TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
 owner TEXT NOT NULL, status TEXT NOT NULL, answer TEXT, error TEXT,
 created_at TIMESTAMPTZ NOT NULL DEFAULT now());
CREATE UNIQUE INDEX one_active_run_per_session ON runs(session) WHERE status='running';
CREATE TABLE upload_names (
 owner TEXT, project TEXT, document TEXT, name TEXT, PRIMARY KEY(owner,project,document));
CREATE TABLE resource_origins (
 owner TEXT, project TEXT, kind TEXT, resource TEXT, session TEXT,
 PRIMARY KEY(owner,project,kind,resource));
CREATE TABLE paper_domains (
 owner TEXT, project TEXT, paper TEXT, domain TEXT, PRIMARY KEY(owner,project,paper));
CREATE TABLE web_jobs (
 id TEXT PRIMARY KEY, owner TEXT, project TEXT, state TEXT,
 created_at TIMESTAMPTZ NOT NULL DEFAULT now());
'''


class WebStore(ConversationStore):
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / 'web-runtime-binding.json'
        self.database = PostgresDatabase(self.root, kind='web')
        self.database.migrate('001-web', WEB_SCHEMA)
        self.migrate_conversations()
        from researchclaw.core.artifacts import write_json
        write_json(self.path, {'backend': 'postgresql', 'schema': self.database.schema})
        with self.connect() as db:
            # An interrupted model request must never be replayed automatically.
            db.execute("UPDATE runs SET status='interrupted', error='Backend restarted; inspect artifacts before retrying' WHERE status='running'")

    @contextmanager
    def connect(self):
        with self.database.connect() as db:
            yield db

    def session(self, identity, owner, project):
        with self.connect() as db:
            self._owned(db, identity, owner, project)
            return [json.loads(r['payload']) for r in db.execute('SELECT payload FROM conversation_messages WHERE session=? ORDER BY ordinal', (identity,))]

    def sessions(self, owner, project):
        with self.connect() as db:
            return [dict(r) for r in db.execute('SELECT id,project,title,memory,created,archived,parent_session FROM sessions WHERE owner=? AND project=? ORDER BY created DESC,id DESC', (owner, project))]

    def new_session(self, owner, project, title='新对话', messages=None):
        import time
        import uuid
        identity = uuid.uuid4().hex
        with self.connect() as db:
            db.execute('INSERT INTO sessions(id,owner,project,title,messages,created) VALUES(?,?,?,?,?,?)',
                       (identity,owner,project,title,'[]',time.time()))
            db.execute('INSERT INTO session_states(session,payload) VALUES(?,?)',(identity,json.dumps(empty_state())))
            self._sync_messages(db,identity,messages or [])
        return identity

    def claim(self, owner, project, kind, resource, session):
        if session:
            self.session(session, owner, project)
        with self.connect() as db:
            db.execute('INSERT INTO resource_origins VALUES(?,?,?,?,?) ON CONFLICT DO NOTHING',
                       (owner,project,kind,resource,session))

    def origins(self, owner, project, kind):
        with self.connect() as db:
            return {r['resource']:r['session'] for r in db.execute(
                'SELECT resource,session FROM resource_origins WHERE owner=? AND project=? AND kind=?', (owner,project,kind))}

    def edit_session(self, identity, owner, project, title=None, memory=None, delete=False, archived=None):
        with self.connect() as db:
            self._owned(db,identity,owner,project,True)
            if db.execute("SELECT 1 FROM runs WHERE session=? AND status='running'", (identity,)).fetchone():
                raise ValueError('对话正在运行，请完成后再修改或删除')
            if delete:
                self._delete_session(db,identity)
                # Origins deliberately remain. Deleting a conversation never deletes knowledge.
            else:
                if title is not None:
                    db.execute('UPDATE sessions SET title=? WHERE id=?', (title,identity))
                if memory is not None:
                    db.execute('UPDATE sessions SET memory=? WHERE id=?', (memory,identity))
                    db.execute("DELETE FROM conversation_memories WHERE source_session=? AND key='legacy-note'",(identity,))
                    if memory.strip():
                        import uuid, time
                        db.execute('INSERT INTO conversation_memories(id,owner,project,source_session,kind,key,content,status,method,created) VALUES(?,?,?,?,?,?,?,?,?,?)',
                                   (uuid.uuid4().hex,owner,project,identity,'constraint','legacy-note',memory,'active','explicit-user',time.time()))
                if archived is not None:
                    db.execute('UPDATE sessions SET archived=? WHERE id=?',(archived,identity))

    def upload_name(self, owner, project, document, name):
        with self.connect() as db:
            db.execute('INSERT INTO upload_names VALUES(?,?,?,?) ON CONFLICT(owner,project,document) DO UPDATE SET name=excluded.name', (owner,project,document,name))

    def create_run(self, identity, session_id, owner, project, existing, current_message=None):
        with self.connect() as db:
            if not existing:
                db.execute('INSERT INTO sessions (id,owner,project) VALUES (?,?,?)', (session_id, owner, project))
                db.execute('INSERT INTO session_states(session,payload) VALUES(?,?)',(session_id,json.dumps(empty_state())))
            if not db.execute('SELECT id FROM sessions WHERE id=? AND owner=? AND project=? FOR UPDATE',
                              (session_id, owner, project)).fetchone():
                raise KeyError(session_id)
            if db.execute("SELECT 1 FROM runs WHERE session=? AND status='running'", (session_id,)).fetchone():
                raise ValueError('Session has an active run')
            db.execute('INSERT INTO runs (id,session,owner,status) VALUES (?,?,?,?)', (identity, session_id, owner, 'running'))
            if current_message is not None:
                prior=[json.loads(r['payload']) for r in db.execute('SELECT payload FROM conversation_messages WHERE session=? ORDER BY ordinal',(session_id,))]
                self._sync_messages(db,session_id,prior+[{'role':'user','content':current_message}])

    def finish(self, run_id, session_id, messages, answer=None, error=None):
        with self.connect() as db:
            row=db.execute("SELECT runs.owner,sessions.project FROM runs JOIN sessions ON sessions.id=runs.session WHERE runs.id=? AND runs.session=? AND runs.status='running' FOR UPDATE OF sessions",(run_id,session_id)).fetchone()
            if row is None:
                raise ValueError('Run is not active')
            self._sync_messages(db,session_id,messages)
            db.execute('UPDATE runs SET status=?,answer=?,error=? WHERE id=?',
                       ('failed' if error else 'completed', answer, error, run_id))

    def claim_recovery(self, identity, owner, project):
        """Serialize explicit recovery against admission of another session turn."""
        with self.connect() as db:
            row=db.execute('SELECT * FROM runs WHERE id=? AND owner=?',(identity,owner)).fetchone()
            if row is None:
                raise KeyError(identity)
            self._owned(db,row['session'],owner,project,True)
            latest=db.execute('SELECT id FROM runs WHERE session=? ORDER BY created_at DESC,id DESC LIMIT 1',
                              (row['session'],)).fetchone()
            if latest['id']!=identity or row['status'] not in {'failed','interrupted'}:
                raise ValueError('Only the latest failed/interrupted turn can be recovered')
            if db.execute("SELECT 1 FROM runs WHERE session=? AND status='running'",(row['session'],)).fetchone():
                raise ValueError('Session has an active run')
            db.execute("UPDATE runs SET status='running',error=NULL WHERE id=?",(identity,))

    def run(self, identity, owner):
        with self.connect() as db:
            row = db.execute('SELECT * FROM runs WHERE id=? AND owner=?', (identity, owner)).fetchone()
        if row is None:
            raise KeyError(identity)
        result = dict(row)
        if result.get('created_at') is not None:
            result['created_at'] = result['created_at'].isoformat()
        return result

    def import_legacy(self, source):
        """Explicit read-only history import; never replay running requests."""
        import sqlite3
        from contextlib import closing
        from psycopg import sql
        source = Path(source).resolve()
        wal = Path(str(source) + '-wal')
        uri = source.as_uri() + '?mode=ro' + ('&immutable=1' if not wal.exists() or not wal.stat().st_size else '')
        imported = {}
        with closing(sqlite3.connect(uri, uri=True)) as old:
            old.row_factory = sqlite3.Row
            tables = {row[0] for row in old.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            with self.connect() as db:
                for table in ('sessions', 'runs', 'upload_names', 'resource_origins', 'paper_domains', 'web_jobs'):
                    if table not in tables:
                        continue
                    columns = {row[0] for row in db.execute('SELECT column_name FROM information_schema.columns WHERE table_schema=? AND table_name=?', (self.database.schema, table))}
                    count = 0
                    for original in old.execute('SELECT * FROM ' + table):
                        row = {key: original[key] for key in original.keys() if key in columns}
                        if table == 'runs' and row.get('status') == 'running':
                            row.update(status='interrupted', error='Historical request imported; no automatic replay')
                        if table == 'web_jobs':
                            state = json.loads(row['state'])
                            if state.get('status') in {'queued', 'running'}:
                                state.update(status='interrupted', error='Historical job imported; explicit resume required')
                            row['state'] = json.dumps(state, ensure_ascii=False)
                        statement = sql.SQL('INSERT INTO {} ({}) VALUES ({}) ON CONFLICT DO NOTHING').format(
                            sql.Identifier(table), sql.SQL(',').join(map(sql.Identifier, row)),
                            sql.SQL(',').join(sql.Placeholder() for _ in row))
                        count += db.raw.execute(statement, [encode_text(value) for value in row.values()]).rowcount
                    imported[table] = count
        self.migrate_conversations()
        return imported

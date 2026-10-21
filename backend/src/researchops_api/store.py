import json
from contextlib import contextmanager
from pathlib import Path
import sqlite3


class WebStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / 'web_state.db'
        with self.connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL, project TEXT NOT NULL,
                    messages TEXT NOT NULL DEFAULT '[]');
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY, session TEXT NOT NULL, owner TEXT NOT NULL,
                    status TEXT NOT NULL, answer TEXT, error TEXT);
            ''')
            # An interrupted model request must never be replayed automatically.
            db.execute("UPDATE runs SET status='interrupted', error='Backend restarted; inspect artifacts before retrying' WHERE status='running'")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def session(self, identity, owner, project):
        with self.connect() as db:
            row = db.execute('SELECT * FROM sessions WHERE id=? AND owner=? AND project=?', (identity, owner, project)).fetchone()
        if row is None:
            raise KeyError(identity)
        return json.loads(row['messages'])

    def create_run(self, identity, session_id, owner, project, existing):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if not existing:
                db.execute('INSERT INTO sessions (id,owner,project) VALUES (?,?,?)', (session_id, owner, project))
            if db.execute("SELECT 1 FROM runs WHERE session=? AND status='running'", (session_id,)).fetchone():
                raise ValueError('Session has an active run')
            db.execute('INSERT INTO runs (id,session,owner,status) VALUES (?,?,?,?)', (identity, session_id, owner, 'running'))

    def finish(self, run_id, session_id, messages, answer=None, error=None):
        with self.connect() as db:
            db.execute('UPDATE sessions SET messages=? WHERE id=?', (json.dumps(messages, ensure_ascii=False), session_id))
            db.execute('UPDATE runs SET status=?,answer=?,error=? WHERE id=?',
                       ('failed' if error else 'completed', answer, error, run_id))

    def run(self, identity, owner):
        with self.connect() as db:
            row = db.execute('SELECT * FROM runs WHERE id=? AND owner=?', (identity, owner)).fetchone()
        if row is None:
            raise KeyError(identity)
        return dict(row)

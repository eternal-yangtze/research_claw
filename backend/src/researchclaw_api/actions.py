"""PG-backed, one-use, session/version/payload-bound action confirmations."""
import hashlib
import json
import secrets
import time
import uuid
import re
from pathlib import Path

from researchclaw.core.artifacts import canonical_hash
from researchclaw.harness.runtime import _validate_arguments

SCHEMA = """
CREATE TABLE agent_actions (
 id TEXT PRIMARY KEY, session TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
 owner TEXT NOT NULL, project TEXT NOT NULL, action TEXT NOT NULL, arguments TEXT NOT NULL,
 reason TEXT NOT NULL, state_version INTEGER NOT NULL, payload_hash TEXT NOT NULL,
 status TEXT NOT NULL DEFAULT 'pending', token_hash TEXT, expires DOUBLE PRECISION,
 result TEXT, created DOUBLE PRECISION NOT NULL);
"""
ALLOWED_ACTIONS = {"download_pdf", "ingest_documents", "implement_experiment"}


class ActionStore:
    def __init__(self, store):
        self.store = store
        store.database.migrate("011-agent-actions", SCHEMA)
        store.database.migrate("012-agent-action-inputs", "ALTER TABLE agent_actions ADD COLUMN frozen_inputs TEXT NOT NULL DEFAULT '{}';")

    def inputs(self, owner, project, action, arguments):
        root=(self.store.root/'users'/owner/'projects'/project).resolve()
        from researchclaw.core.artifacts import file_hash
        if action=='implement_experiment':
            identity=arguments.get('idea_id','')
            if not re.fullmatch(r'[a-zA-Z0-9_-]{1,100}',identity):
                raise ValueError('Invalid prepared Idea ID')
            prep=root/'artifacts'/'experiments'/f'preparation_{identity}.json'
            if not prep.is_file():
                raise ValueError('Coding requires an existing prepared, frozen experiment contract; propose baselines first')
            paths=[prep]
            if arguments.get('study_result'):
                result=Path(arguments['study_result']).resolve()
                if not result.is_relative_to(root) or not result.is_file():
                    raise ValueError('Study result outside current project or missing')
                paths.append(result)
            return {str(p.relative_to(root)):file_hash(p) for p in paths}
        if action=='ingest_documents':
            source=Path(arguments.get('path','')).resolve()
            if not source.is_relative_to(root) or not source.exists():
                raise ValueError('Ingest only uploaded documents in this project')
            paths=[source] if source.is_file() else sorted(p for p in source.rglob('*') if p.is_file())
            if len(paths)>2000:
                raise ValueError('At most 2000 files per ingestion confirmation')
            return {str(p.relative_to(root)):file_hash(p) for p in paths}
        return {}

    def propose(self, session, owner, project, action, arguments, reason, registry):
        if action not in ALLOWED_ACTIONS:
            raise ValueError("Action not exposed by this research agent")
        _validate_arguments(registry.get(action).input_schema, arguments)
        version = self.store.state(session, owner, project)["state_version"]
        inputs=self.inputs(owner,project,action,arguments)
        identity = uuid.uuid4().hex
        fingerprint = canonical_hash([session, owner, project, action, arguments, version, inputs])
        with self.store.connect() as db:
            self.store._owned(db, session, owner, project)
            db.execute("INSERT INTO agent_actions(id,session,owner,project,action,arguments,reason,state_version,payload_hash,created) VALUES(?,?,?,?,?,?,?,?,?,?)",
                       (identity, session, owner, project, action, json.dumps(arguments), reason[:2000], version, fingerprint, time.time()))
            db.execute('UPDATE agent_actions SET frozen_inputs=? WHERE id=?',(json.dumps(inputs),identity))
        return {"action_id": identity, "status": "pending_user_confirmation", "action": action,
                "arguments": arguments, "state_version": version,
                "instruction": "No execution occurred. 请用户在页面待确认操作卡中点击‘查看并决定’，核对后确认或拒绝。不要要求用户手工调用 API。"}

    def _row(self, db, identity, owner, project):
        row = db.execute("SELECT * FROM agent_actions WHERE id=? AND owner=? AND project=? FOR UPDATE", (identity, owner, project)).fetchone()
        if not row:
            raise KeyError("Unknown action")
        self.store._owned(db, row["session"], owner, project)
        return dict(row)

    def prepare(self, identity, owner, project):
        token = secrets.token_urlsafe(32)
        with self.store.connect() as db:
            row = self._row(db, identity, owner, project)
            if row["status"] != "pending":
                raise ValueError("Action no longer pending")
            expires = time.time() + 300
            db.execute("UPDATE agent_actions SET token_hash=?,expires=? WHERE id=?", (hashlib.sha256(token.encode()).hexdigest(), expires, identity))
        return {"id": identity, "session_id": row["session"], "action": row["action"],
                "arguments": json.loads(row["arguments"]), "reason": row["reason"],
                "state_version": row["state_version"], "payload_hash": row["payload_hash"],
                "confirmation_token": token, "expires": expires, "status": "pending"}

    def consume(self, identity, owner, project, token):
        with self.store.connect() as db:
            row = self._row(db, identity, owner, project)
            state = db.execute("SELECT version FROM session_states WHERE session=? FOR UPDATE", (row["session"],)).fetchone()
            if row["status"] != "pending" or not row["token_hash"] or (row["expires"] or 0) < time.time() or not secrets.compare_digest(row["token_hash"], hashlib.sha256(token.encode()).hexdigest()):
                raise ValueError("Expired, replayed or invalid confirmation")
            if state["version"] != row["state_version"]:
                raise ValueError("Research state changed: create a new proposal")
            arguments = json.loads(row["arguments"])
            inputs=self.inputs(owner,project,row['action'],arguments)
            if inputs!=json.loads(row['frozen_inputs']) or canonical_hash([row["session"], owner, project, row["action"], arguments, row["state_version"], inputs]) != row["payload_hash"]:
                raise ValueError("Action payload changed")
            if db.execute("SELECT 1 FROM runs WHERE session=? AND status='running'", (row["session"],)).fetchone():
                raise ValueError("Session still running; retry after completion")
            db.execute("UPDATE agent_actions SET status='dispatching',token_hash=NULL WHERE id=?", (identity,))
        return {**row, "arguments": arguments}

    def finish(self, identity, result):
        with self.store.connect() as db:
            db.execute("UPDATE agent_actions SET status=?,result=? WHERE id=?", (result["status"], json.dumps(result, ensure_ascii=False, default=str), identity))

    def reject(self, identity, owner, project):
        with self.store.connect() as db:
            row = self._row(db, identity, owner, project)
            if row['status'] != 'pending':
                raise ValueError('Action no longer pending')
            db.execute("UPDATE agent_actions SET status='rejected',token_hash=NULL WHERE id=?", (identity,))

    def list(self, session, owner, project):
        with self.store.connect() as db:
            self.store._owned(db, session, owner, project)
            return [dict(r) for r in db.execute("SELECT id,action,arguments,reason,status,state_version,result FROM agent_actions WHERE session=? AND owner=? AND project=? ORDER BY created", (session, owner, project))]

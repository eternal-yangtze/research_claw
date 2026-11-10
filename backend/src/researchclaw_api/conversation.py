"""PostgreSQL-backed conversation lifecycle; no scientific facts in user memory."""
import hashlib
import json
import re
import time
import uuid


CONVERSATION_SCHEMA = '''
ALTER TABLE sessions ADD COLUMN archived BOOLEAN NOT NULL DEFAULT false;
ALTER TABLE sessions ADD COLUMN parent_session TEXT REFERENCES sessions(id) ON DELETE SET NULL;
CREATE TABLE conversation_messages (
 id TEXT PRIMARY KEY, session TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
 ordinal INTEGER NOT NULL, payload TEXT NOT NULL, source_session TEXT REFERENCES sessions(id) ON DELETE CASCADE,
 created_at TIMESTAMPTZ NOT NULL DEFAULT now(), UNIQUE(session,ordinal));
CREATE TABLE session_states (
 session TEXT PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
 version INTEGER NOT NULL DEFAULT 0, payload TEXT NOT NULL DEFAULT '{}');
CREATE TABLE conversation_memories (
 id TEXT PRIMARY KEY, owner TEXT NOT NULL, project TEXT NOT NULL,
 source_session TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
 source_message TEXT REFERENCES conversation_messages(id) ON DELETE CASCADE,
 kind TEXT NOT NULL, key TEXT NOT NULL, content TEXT NOT NULL,
 status TEXT NOT NULL CHECK(status IN ('proposed','active','superseded','deleted')),
 method TEXT NOT NULL, version INTEGER NOT NULL DEFAULT 1,
 supersedes TEXT, expires DOUBLE PRECISION, created DOUBLE PRECISION NOT NULL);
CREATE INDEX memory_scope ON conversation_memories(owner,project,status);
CREATE TABLE context_snapshots (
 id TEXT PRIMARY KEY, session TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
 version INTEGER NOT NULL, payload TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT now());
CREATE TABLE conversation_confirmations (
 digest TEXT PRIMARY KEY, session TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
 owner TEXT NOT NULL, project TEXT NOT NULL, fingerprint TEXT NOT NULL,
 expires DOUBLE PRECISION NOT NULL);
'''


def empty_state():
    return dict(goal='', acceptance_criteria=[], completed_steps=[], decisions=[], failed_attempts=[],
                evidence_refs=[], open_questions=[], pending_actions=[], confirmation_pending=None)


class StateConflict(ValueError):
    pass


class ConversationStore:
    def _owned(self, db, identity, owner, project, lock=False):
        row = db.execute('SELECT * FROM sessions WHERE id=? AND owner=? AND project=?' +
                         (' FOR UPDATE' if lock else ''), (identity, owner, project)).fetchone()
        if row is None:
            raise KeyError(identity)
        return row

    def _sync_messages(self, db, identity, messages):
        old = list(db.execute('SELECT id,payload FROM conversation_messages WHERE session=? ORDER BY ordinal', (identity,)))
        if len(messages) < len(old) or any(json.loads(r['payload']) != m for r, m in zip(old, messages)):
            raise StateConflict('Conversation history is append-only; reload before saving')
        for ordinal, message in enumerate(messages[len(old):], len(old)):
            from researchclaw.harness.context import TokenCounter
            db.execute('INSERT INTO conversation_messages(id,session,ordinal,payload,token_count) VALUES(?,?,?,?,?)',
                       (uuid.uuid4().hex, identity, ordinal, json.dumps(message, ensure_ascii=False),TokenCounter().messages([message])))
        # The legacy field is emptied after migration; rows are authoritative.
        db.execute("UPDATE sessions SET messages='[]' WHERE id=?", (identity,))

    def migrate_conversations(self):
        self.database.migrate('002-conversation-memory', CONVERSATION_SCHEMA)
        self.database.migrate('003-memory-dependencies','''
CREATE TABLE conversation_dependencies (
 session TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
 source TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
 PRIMARY KEY(session,source), CHECK(session<>source));
''')
        self.database.migrate('004-message-token-count','ALTER TABLE conversation_messages ADD COLUMN token_count INTEGER;')
        with self.connect() as db:
            for row in db.execute('SELECT id,messages,memory,owner,project FROM sessions FOR UPDATE').fetchall():
                messages = json.loads(row['messages'])
                if messages:
                    self._sync_messages(db, row['id'], messages)
                db.execute('INSERT INTO session_states(session,payload) VALUES(?,?) ON CONFLICT DO NOTHING',
                           (row['id'], json.dumps(empty_state())))
                if row['memory'] and not db.execute("SELECT 1 FROM conversation_memories WHERE source_session=? AND key='legacy-note'", (row['id'],)).fetchone():
                    db.execute('INSERT INTO conversation_memories(id,owner,project,source_session,kind,key,content,status,method,created) VALUES(?,?,?,?,?,?,?,?,?,?)',
                               (uuid.uuid4().hex,row['owner'],row['project'],row['id'],'constraint','legacy-note',row['memory'],'active','legacy-user-note',time.time()))

    def message_records(self, identity, owner, project):
        with self.connect() as db:
            self._owned(db, identity, owner, project)
            return [dict(id=r['id'], ordinal=r['ordinal'], source_session=r['source_session'],token_count=r['token_count'],
                         **json.loads(r['payload'])) for r in db.execute(
                             'SELECT * FROM conversation_messages WHERE session=? ORDER BY ordinal', (identity,))]

    def state(self, identity, owner, project):
        with self.connect() as db:
            self._owned(db, identity, owner, project)
            row = db.execute('SELECT * FROM session_states WHERE session=?', (identity,)).fetchone()
            return {**empty_state(), **json.loads(row['payload']), 'state_version': row['version']}

    def update_state(self, identity, owner, project, expected_version, patch):
        allowed = set(empty_state())
        if set(patch) - allowed:
            raise ValueError('Unknown state fields')
        if len(json.dumps(patch)) > 48000:
            raise ValueError('State too large; use artifact references')
        for key, value in patch.items():
            if key == 'goal' and not isinstance(value, str):
                raise ValueError('goal must be text')
            if key not in {'goal','confirmation_pending'} and not isinstance(value, list):
                raise ValueError('State list field expected')
        with self.connect() as db:
            self._owned(db, identity, owner, project, True)
            row = db.execute('SELECT * FROM session_states WHERE session=?', (identity,)).fetchone()
            if row['version'] != expected_version:
                raise StateConflict('State version changed; reload')
            state = {**empty_state(), **json.loads(row['payload']), **patch}
            if 'goal' in patch and patch['goal'] != json.loads(row['payload']).get('goal', ''):
                # Old plans do not silently remain valid after changing the research goal.
                for field in ('decisions','completed_steps','evidence_refs','pending_actions'):
                    if field not in patch:
                        state[field] = []
                state['confirmation_pending'] = None
            db.execute('UPDATE session_states SET version=version+1,payload=? WHERE session=?',
                       (json.dumps(state, ensure_ascii=False),identity))
        return {**state,'state_version':expected_version+1}

    def read_message(self,identity,owner,project,message_id):
        with self.connect() as db:
            self._owned(db,identity,owner,project)
            row=db.execute('SELECT id,payload,ordinal FROM conversation_messages WHERE id=? AND session=?',(message_id,identity)).fetchone()
            if not row:
                raise KeyError(message_id)
            return dict(id=row['id'],ordinal=row['ordinal'],**json.loads(row['payload']))

    def read_recalled_message(self,identity,owner,project,message_id,source_session=None):
        source_session=source_session or identity
        with self.connect() as db:
            for sid in sorted({identity,source_session}):
                self._owned(db,sid,owner,project,True)
            if source_session!=identity and not db.execute(
                'SELECT 1 FROM conversation_dependencies WHERE session=? AND source=?',
                (identity,source_session)).fetchone():
                raise ValueError('Search and register the source session before reading its messages')
            row=db.execute('SELECT id,payload,ordinal FROM conversation_messages WHERE id=? AND session=?',
                           (message_id,source_session)).fetchone()
            if not row:
                raise KeyError(message_id)
            return dict(id=row['id'],ordinal=row['ordinal'],source_session=source_session,**json.loads(row['payload']))

    def save_memory(self, identity, owner, project, *, content, key, kind='preference',
                    source_message=None, explicit=False, expires=None, session_only=False):
        if not content.strip() or len(content) > 8000 or not key.strip() or len(key) > 160:
            raise ValueError('Memory requires bounded content and key')
        if kind not in {'profile','preference','project-decision','constraint','episodic'}:
            raise ValueError('Unsupported memory kind')
        with self.connect() as db:
            self._owned(db,identity,owner,project,True)
            db.execute('SELECT pg_advisory_xact_lock(hashtext(?))',(f'memory:{owner}:{project}:{key}',))
            if source_message:
                row=db.execute('SELECT payload FROM conversation_messages WHERE id=? AND session=?', (source_message,identity)).fetchone()
                if not row or json.loads(row['payload']).get('role') != 'user':
                    raise ValueError('Memory must cite an owned user message')
            scope_clause=' AND source_session=?' if session_only else ''
            scope_params=(owner,project,key,identity) if session_only else (owner,project,key)
            previous=db.execute("SELECT id,version FROM conversation_memories WHERE owner=? AND project=? AND key=? AND status='active'"+scope_clause+" ORDER BY version DESC", scope_params).fetchone()
            if explicit:
                db.execute("UPDATE conversation_memories SET status='superseded' WHERE owner=? AND project=? AND key=? AND status='active'"+scope_clause, scope_params)
            mid=uuid.uuid4().hex
            db.execute('INSERT INTO conversation_memories(id,owner,project,source_session,source_message,kind,key,content,status,method,version,supersedes,expires,created) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                       (mid,owner,project,identity,source_message,kind,key,content,'active' if explicit else 'proposed',
                        ('session-explicit' if explicit else 'session-proposed') if session_only else ('explicit-user' if explicit else 'proposed'), (previous['version']+1) if previous else 1,
                        previous['id'] if previous and explicit else None, expires,time.time()))
            return mid

    def memories(self, owner, project, query='', include_proposed=False, limit=8, source_session=None):
        with self.connect() as db:
            if source_session is not None:
                self._owned(db,source_session,owner,project)
            rows=[dict(r) for r in db.execute('SELECT m.* FROM conversation_memories m JOIN sessions s ON m.source_session=s.id WHERE m.owner=? AND m.project=? AND s.owner=? AND s.project=? AND (m.expires IS NULL OR m.expires>?) AND (m.status=\'active\' OR (? AND m.status=\'proposed\')) ORDER BY m.created DESC,m.id',
                                              (owner,project,owner,project,time.time(),include_proposed))]
        if source_session is not None:
            rows=[r for r in rows if r['source_session']==source_session]
        def terms(text):
            return set(re.findall(r'[a-z0-9_-]+|[\u4e00-\u9fff]',text.lower()))
        q=terms(query)
        for row in rows:
            row['relevance']=len(q & terms(row['key']+' '+row['content']))
        if query:
            rows=[r for r in rows if r['relevance'] or r['kind'] in {'profile','constraint'}]
        rows.sort(key=lambda r:(-r['relevance'],-r['created']))
        return rows[:limit]

    def memory_action(self, mid, owner, project, action):
        with self.connect() as db:
            row=db.execute('SELECT * FROM conversation_memories WHERE id=? AND owner=? AND project=? FOR UPDATE', (mid,owner,project)).fetchone()
            if not row:
                raise KeyError(mid)
            self._owned(db,row['source_session'],owner,project,True)
            if action == 'delete':
                db.execute('DELETE FROM conversation_memories WHERE id=?',(mid,))
                if row['key']=='legacy-note':
                    # Otherwise a restart migration could resurrect the old note.
                    db.execute("UPDATE sessions SET memory='' WHERE id=?",(row['source_session'],))
            elif action == 'confirm' and row['status']=='proposed':
                db.execute('SELECT pg_advisory_xact_lock(hashtext(?))',(f"memory:{owner}:{project}:{row['key']}",))
                session_only=row['method']=='session-proposed'
                scope_clause=' AND source_session=?' if session_only else ''
                scope_params=(owner,project,row['key'],row['source_session']) if session_only else (owner,project,row['key'])
                old=db.execute("SELECT id,version FROM conversation_memories WHERE owner=? AND project=? AND key=? AND status='active'"+scope_clause+" ORDER BY version DESC",scope_params).fetchone()
                db.execute("UPDATE conversation_memories SET status='superseded' WHERE owner=? AND project=? AND key=? AND status='active'"+scope_clause,scope_params)
                db.execute("UPDATE conversation_memories SET status='active',method=?,version=?,supersedes=? WHERE id=?",('session-confirmed' if session_only else 'user-confirmed',(old['version']+1) if old else 1,old['id'] if old else None,mid))
            else:
                raise ValueError('Invalid memory transition')

    def branch_session(self, identity, owner, project, title):
        child=self.new_session(owner,project,title)
        try:
            with self.connect() as db:
                self._owned(db,identity,owner,project,True)
                db.execute('UPDATE sessions SET parent_session=? WHERE id=?',(identity,child))
                for ordinal,row in enumerate(db.execute('SELECT * FROM conversation_messages WHERE session=? ORDER BY ordinal',(identity,)).fetchall()):
                    db.execute('INSERT INTO conversation_messages(id,session,ordinal,payload,source_session) VALUES(?,?,?,?,?)',
                               (uuid.uuid4().hex,child,ordinal,row['payload'],row['source_session'] or identity))
                # Keep state independent but track its provenance in the parent edge.
                row=db.execute('SELECT payload FROM session_states WHERE session=?',(identity,)).fetchone()
                db.execute('UPDATE session_states SET payload=? WHERE session=?',(row['payload'],child))
        except Exception:
            self.edit_session(child,owner,project,delete=True)
            raise
        return child

    def _delete_fingerprint(self,db,identity,owner,project):
        self._owned(db,identity,owner,project,True)
        rows=[dict(r) for r in db.execute('SELECT id FROM conversation_messages WHERE session=? OR source_session=? ORDER BY id',(identity,identity))]
        memories=[dict(r) for r in db.execute('SELECT id,version,status FROM conversation_memories WHERE source_session=? ORDER BY id',(identity,))]
        children=self._dependent_sessions(db,identity)
        state=db.execute('SELECT version FROM session_states WHERE session=?',(identity,)).fetchone()
        jobs=[dict(id=r['id'],status=json.loads(r['state']).get('status')) for r in db.execute(
            'SELECT id,state FROM web_jobs WHERE owner=? AND project=? ORDER BY id',(owner,project))
              if json.loads(r['state']).get('session_id') in {identity,*children}]
        dependencies=[]
        for child in children:
            dependencies.append(dict(id=child,state=dict(db.execute('SELECT version,payload FROM session_states WHERE session=?',(child,)).fetchone()),
                                     messages=[r['id'] for r in db.execute('SELECT id FROM conversation_messages WHERE session=? ORDER BY id',(child,))],
                                     memories=[dict(r) for r in db.execute('SELECT id,version,status FROM conversation_memories WHERE source_session=? ORDER BY id',(child,))]))
        value={'messages':rows,'memories':memories,'branches':children,'dependents':dependencies,'jobs':jobs,'state_version':state['version']}
        return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest(),value

    def prepare_delete(self,identity,owner,project):
        import secrets
        token=secrets.token_urlsafe(32)
        with self.connect() as db:
            fingerprint,preview=self._delete_fingerprint(db,identity,owner,project)
            db.execute('INSERT INTO conversation_confirmations VALUES(?,?,?,?,?,?)',
                       (hashlib.sha256(token.encode()).hexdigest(),identity,owner,project,fingerprint,time.time()+300))
        return dict(token=token,expires_in=300,knowledge_retained=True,impact=preview)

    def confirm_delete(self,identity,owner,project,token):
        with self.connect() as db:
            fingerprint,_=self._delete_fingerprint(db,identity,owner,project)
            row=db.execute('SELECT * FROM conversation_confirmations WHERE digest=? AND session=? AND owner=? AND project=? FOR UPDATE',
                           (hashlib.sha256(token.encode()).hexdigest(),identity,owner,project)).fetchone()
            if not row or row['expires']<time.time() or row['fingerprint']!=fingerprint:
                raise StateConflict('Confirmation expired or target changed; prepare again')
            self._delete_session(db,identity)

    def _delete_session(self,db,identity):
        if db.execute("SELECT 1 FROM runs WHERE session=? AND status='running'",(identity,)).fetchone():
            raise StateConflict('Session has an active run')
        # Descendant state/snapshots may contain summaries of the deleted source.
        children=self._dependent_sessions(db,identity)
        for target in [identity,*children]:
            row=db.execute('SELECT owner,project FROM sessions WHERE id=? FOR UPDATE',(target,)).fetchone()
            jobs=db.execute('SELECT state FROM web_jobs WHERE owner=? AND project=?',(row['owner'],row['project'])).fetchall()
            if any((j:=json.loads(r['state'])).get('session_id')==target and j.get('status') in {'running','queued'} for r in jobs):
                raise StateConflict('A dependent conversation has a background job')
            if db.execute("SELECT 1 FROM runs WHERE session=? AND status='running'",(target,)).fetchone():
                raise StateConflict('A dependent conversation has an active run')
        # Remove private conversation artifacts as well as PG rows. PDFs and knowledge stay.
        self._purge_conversation_artifacts(db,[identity,*children])
        for child in children:
            if db.execute("SELECT 1 FROM runs WHERE session=? AND status='running'",(child,)).fetchone():
                raise StateConflict('A dependent branch has an active run')
            # Strict deletion: clear derived branch conversation/state, keep empty shell.
            db.execute('DELETE FROM conversation_messages WHERE session=?',(child,))
            db.execute('DELETE FROM conversation_memories WHERE source_session=?',(child,))
            db.execute('DELETE FROM context_snapshots WHERE session=?',(child,))
            db.execute('DELETE FROM runs WHERE session=?',(child,))
            db.execute('UPDATE session_states SET version=version+1,payload=? WHERE session=?',(json.dumps(empty_state()),child))
            db.execute("UPDATE sessions SET memory='',messages='[]' WHERE id=?",(child,))
        db.execute('DELETE FROM sessions WHERE id=?',(identity,))

    def _dependent_sessions(self,db,identity):
        edges=[(r['id'],r['parent_session']) for r in db.execute('SELECT id,parent_session FROM sessions WHERE parent_session IS NOT NULL')]
        edges += [(r['session'],r['source']) for r in db.execute('SELECT session,source FROM conversation_dependencies')]
        seen={identity}
        while True:
            next_ids={child for child,parent in edges if parent in seen}-seen
            if not next_ids:
                return sorted(seen-{identity})
            seen.update(next_ids)

    def _purge_conversation_artifacts(self,db,identities):
        import shutil
        for identity in identities:
            scope=db.execute('SELECT owner,project FROM sessions WHERE id=?',(identity,)).fetchone()
            from .accounts import USER_ID_PATTERN
            if (not re.fullmatch(USER_ID_PATTERN, scope['owner']) or scope['owner'] in {'.', '..'}
                    or not re.fullmatch(r'[a-zA-Z0-9_-]{1,64}', scope['project'])):
                raise ValueError('Unsafe artifact scope')
            root=(self.root/'users'/scope['owner']/'projects'/scope['project']).resolve()
            if not root.is_relative_to(self.root):
                raise ValueError('Artifact scope escapes workspace')
            # New Agent checkpoints contain session-derived plans and evidence views.
            # Delete them with the conversation, without touching shared paper rows.
            from researchclaw.core.postgres import PostgresDatabase
            marker=root/'knowledge-storage.json'
            if marker.is_file():
                project_db=PostgresDatabase(root,identity=json.loads(marker.read_text(encoding='utf8'))['collection'])
                with project_db.connect() as agent_db:
                    if agent_db.execute("SELECT to_regclass('agent_checkpoints')").fetchone()[0]:
                        threads=[r['run_id'] for r in agent_db.execute('SELECT run_id FROM agent_checkpoints WHERE session_id=?',(identity,))]
                        for table in ('checkpoint_writes','checkpoint_blobs','checkpoints'):
                            if agent_db.execute('SELECT to_regclass(?)',(table,)).fetchone()[0]:
                                for thread in threads:
                                    agent_db.execute('DELETE FROM '+table+' WHERE thread_id=?',(thread,))
                        agent_db.execute('DELETE FROM agent_checkpoints WHERE session_id=?',(identity,))
            for row in db.execute('SELECT id FROM runs WHERE session=?',(identity,)):
                if not re.fullmatch(r'[a-f0-9]{32}',row['id']):
                    continue  # Historical non-Web run IDs have no generated private directory.
                for group in ('runs','model_calls','artifacts/skills'):
                    path=(root/group/row['id']).resolve()
                    if not path.is_relative_to(root) or path==root:
                        raise ValueError('Artifact path escapes scope')
                    if path.is_dir():
                        shutil.rmtree(path)
            for job in db.execute('SELECT id,state FROM web_jobs WHERE owner=? AND project=?',(scope['owner'],scope['project'])).fetchall():
                if json.loads(job['state']).get('session_id')!=identity:
                    continue
                if not re.fullmatch(r'[a-f0-9]{32}',job['id']):
                    raise ValueError('Unsafe private job identifier')
                path=(root/'web_jobs'/job['id']).resolve()
                if not path.is_relative_to(root) or path==root:
                    raise ValueError('Job artifacts escape scope')
                if path.is_dir():
                    shutil.rmtree(path)
                db.execute('DELETE FROM web_jobs WHERE id=?',(job['id'],))

    def recall_for_session(self,identity,owner,project,query):
        memories=self.memories(owner,project,query)
        with self.connect() as db:
            self._owned(db,identity,owner,project,True)
            for m in memories:
                if m['source_session']!=identity:
                    db.execute('INSERT INTO conversation_dependencies(session,source) VALUES(?,?) ON CONFLICT DO NOTHING',(identity,m['source_session']))
        return memories

    def search_conversation_history(self,identity,owner,project,query,limit=6,scope='session'):
        """Recall historical evidence, not automatically confirmed profile facts.

        Cross-session reads register deletion dependencies in the same transaction
        as source locks. Bounded scanning is reported instead of claiming complete
        recall. Paper text/embeddings and Qdrant collections are unaffected.
        """
        from researchclaw.harness.history_retrieval import retrieve_history
        if scope not in {'session','project'} or not query.strip() or len(query)>2000 or not 1<=limit<=10:
            raise ValueError('Invalid history query, limit or scope')
        with self.connect() as db:
            self._owned(db,identity,owner,project)
            condition=' AND s.id=?' if scope=='session' else ''
            params=(owner,project,identity) if scope=='session' else (owner,project)
            rows=db.execute('SELECT m.id,m.session,m.ordinal,m.payload,m.created_at FROM conversation_messages m '
                'JOIN sessions s ON s.id=m.session WHERE s.owner=? AND s.project=?'+condition+
                ' ORDER BY m.created_at DESC,m.session,m.ordinal DESC LIMIT 4001',params).fetchall()
            truncated=len(rows)>4000
            rows=list(reversed(rows[:4000]))
            records=[]
            # Exclude the latest user request in the consuming session. It is a
            # query, not a historical source that answers itself.
            current=db.execute('SELECT id,payload FROM conversation_messages WHERE session=? ORDER BY ordinal DESC LIMIT 1',(identity,)).fetchone()
            for row in rows:
                if current and row['id']==current['id'] and json.loads(current['payload']).get('role')=='user':
                    continue
                payload=json.loads(row['payload'])
                records.append(dict(id=row['id'],source_session=row['session'],ordinal=len(records),
                                    date=str(row['created_at']),role=payload.get('role'),content=payload.get('content')))
            matches=retrieve_history(records,query,limit=limit)
            sources=sorted({identity}|{m['source_session'] for m in matches})
            for source in sources:
                self._owned(db,source,owner,project,True)
            for match in matches:
                if not db.execute('SELECT id FROM conversation_messages WHERE id=? AND session=?',
                                  (match['source_message'],match['source_session'])).fetchone():
                    raise StateConflict('History changed or was deleted; search again')
                if match['source_session']!=identity:
                    db.execute('INSERT INTO conversation_dependencies(session,source) VALUES(?,?) ON CONFLICT DO NOTHING',
                               (identity,match['source_session']))
        return dict(matches=matches,scope=scope,method='lexical-BM25',scanned_messages=len(records),truncated=truncated,
                    warning='Historical data, not current instructions or confirmed profile. Check user corrections. No match does not prove absence.')

    def save_context_snapshot(self,identity,owner,project,stats):
        with self.connect() as db:
            self._owned(db,identity,owner,project)
            db.execute('INSERT INTO context_snapshots(id,session,version,payload) VALUES(?,?,?,?)',
                       (uuid.uuid4().hex,identity,stats.get('state_version',0),json.dumps(stats,ensure_ascii=False)))
            for mid in stats.get('memory_ids',[]):
                row=db.execute("SELECT m.source_session FROM conversation_memories m JOIN sessions s ON s.id=m.source_session WHERE m.id=? AND m.owner=? AND m.project=? AND m.status='active' AND (m.expires IS NULL OR m.expires>?) FOR KEY SHARE OF m,s",(mid,owner,project,time.time())).fetchone()
                if not row:
                    raise StateConflict('Recalled memory changed or was deleted; rebuild context')
                if row['source_session']!=identity:
                    db.execute('INSERT INTO conversation_dependencies(session,source) VALUES(?,?) ON CONFLICT DO NOTHING',(identity,row['source_session']))

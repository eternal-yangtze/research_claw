"""Reversible test overlays. Never mutate a baseline corpus or delete its vectors."""
import hashlib
import json
import os
import secrets
import time
import uuid

from fastapi import Depends, HTTPException
from pydantic import BaseModel, Field

from researchclaw.core.artifacts import canonical_hash, file_hash, write_json


class VersionRequest(BaseModel):
    project_id: str = 'default'
    session_id: str = Field(pattern=r'^[a-f0-9]{32}$')
    confirmation_token: str | None = None


def read_version(root):
    marker = root / 'paper-test-version.json'
    return json.loads(marker.read_text(encoding='utf8')) if marker.is_file() else None


def inventory(root):
    return {p.name: file_hash(p) for p in sorted((root / 'uploads').glob('*')) if p.is_file()}


def install_versions(api, project, user):
    store = api.state.store
    store.database.migrate('015-paper-test-versions', '''CREATE TABLE paper_test_versions (
        id TEXT PRIMARY KEY, owner TEXT NOT NULL, project TEXT NOT NULL, state TEXT NOT NULL);''')

    def locked(db, root, owner, body):
        store.session(body.session_id, owner, body.project_id)
        db.execute('SELECT pg_advisory_xact_lock(?)', (int(hashlib.sha256(str(root).encode()).hexdigest()[:15], 16),))
        jobs = db.execute('SELECT state FROM web_jobs WHERE owner=? AND project=?', (owner, body.project_id))
        if any(json.loads(r['state']).get('status') in {'queued', 'running'} for r in jobs):
            raise ValueError('请等待本项目构建任务结束后再切换测试版本')
        if db.execute("SELECT 1 FROM runs r JOIN sessions s ON s.id=r.session WHERE s.owner=? AND s.project=? AND r.status='running'", (owner, body.project_id)).fetchone():
            raise ValueError('请等待本项目 Agent 回答结束后再切换测试版本')

    @api.post('/api/v1/library/test-version/start')
    def start(body: VersionRequest, owner=Depends(user)):
        root = project(owner, body.project_id)
        try:
            with store.connect() as db:
                locked(db, root, owner, body)
                old = read_version(root)
                if old and old.get('status') == 'active':
                    raise ValueError('已有活动测试版本，请先撤销该版本')
                state = {'id': uuid.uuid4().hex, 'status': 'active', 'created': time.time(),
                    'baseline_uploads': inventory(root), 'session_id': body.session_id}
                db.execute('INSERT INTO paper_test_versions VALUES(?,?,?,?)',
                    (state['id'], owner, body.project_id, json.dumps(state)))
                db.execute("UPDATE agent_actions SET status='rejected',token_hash=NULL WHERE owner=? AND project=? AND status='pending' AND action='ingest_documents'", (owner, body.project_id))
                write_json(root / 'paper-test-version.json', state)
            return {'id': state['id'], 'status': 'active', 'note': '后续入库进入独立测试增量索引；原库保留，撤销需再次确认。'}
        except KeyError:
            raise HTTPException(404, 'Unknown session')
        except ValueError as error:
            raise HTTPException(409, str(error))

    @api.post('/api/v1/library/test-version/prepare-rollback')
    def prepare(body: VersionRequest, owner=Depends(user)):
        root = project(owner, body.project_id)
        try:
            with store.connect() as db:
                locked(db, root, owner, body)
                state = read_version(root)
                if not state or state.get('status') != 'active':
                    raise ValueError('没有活动测试版本')
                token = secrets.token_urlsafe(32)
                current = inventory(root)
                state.update(token_hash=hashlib.sha256(token.encode()).hexdigest(), expires=time.time() + 300,
                    rollback_inventory=canonical_hash(current), rollback_session=body.session_id)
                db.execute('UPDATE paper_test_versions SET state=? WHERE id=?', (json.dumps(state), state['id']))
                # Tokens remain server side; the retrieval marker has no credentials.
            return {'id': state['id'], 'confirmation_token': token,
                'files_to_archive': [n for n in current if n not in state['baseline_uploads']],
                'note': '撤销后测试索引退出检索，新上传文件移入归档；原论文库、原上传及会话记录保留。不可用此入口删除原库。'}
        except KeyError:
            raise HTTPException(404, 'Unknown session')
        except ValueError as error:
            raise HTTPException(409, str(error))

    @api.post('/api/v1/library/test-version/rollback')
    def rollback(body: VersionRequest, owner=Depends(user)):
        root = project(owner, body.project_id)
        try:
            with store.connect() as db:
                locked(db, root, owner, body)
                active = read_version(root)
                if not active or active.get('status') != 'active':
                    raise ValueError('没有活动测试版本或已经撤销')
                row = db.execute('SELECT state FROM paper_test_versions WHERE id=? AND owner=? AND project=? FOR UPDATE',
                    (active['id'], owner, body.project_id)).fetchone()
                state = json.loads(row['state']) if row else {}
                if (state.get('expires', 0) < time.time() or state.get('rollback_session') != body.session_id
                        or not secrets.compare_digest(state.get('token_hash', ''), hashlib.sha256((body.confirmation_token or '').encode()).hexdigest())):
                    raise ValueError('撤销确认无效、过期或已使用，请重新预览')
                current = inventory(root)
                if canonical_hash(current) != state['rollback_inventory']:
                    raise ValueError('上传文件已变化，请重新预览撤销')
                folder = (root / 'pt' / state['id'][:16] / 'uploads').resolve()
                if not folder.is_relative_to(root.resolve()):
                    raise ValueError('Archive outside project')
                folder.mkdir(parents=True, exist_ok=True)
                archived = []
                for name in current:
                    if name in state['baseline_uploads']:
                        continue
                    source = (root / 'uploads' / name).resolve()
                    if not source.is_relative_to((root / 'uploads').resolve()):
                        raise ValueError('Invalid upload archive path')
                    destination = folder / name
                    # Windows long archive paths can exceed MAX_PATH even for valid IDs.
                    if os.name == 'nt':
                        os.replace('\\\\?\\' + str(source), '\\\\?\\' + str(destination))
                    else:
                        source.replace(destination)
                    archived.append(name)
                state.update(status='rolled_back', rolled_back=time.time(), archived_uploads=archived, token_hash=None)
                write_json(root / 'paper-test-version.json', {k: v for k, v in state.items() if k not in {'rollback_inventory', 'expires'}})
                db.execute('UPDATE paper_test_versions SET state=? WHERE id=?', (json.dumps(state), state['id']))
                # Invalidate pre-existing ingestion proposals: they target the former inventory/version.
                db.execute("UPDATE agent_actions SET status='rejected',token_hash=NULL WHERE owner=? AND project=? AND status='pending' AND action='ingest_documents'", (owner, body.project_id))
            return {'status': 'rolled_back', 'id': state['id'], 'archived_uploads': archived,
                'note': '测试索引已退出检索；文件和索引归档可供核对，未物理清空 PostgreSQL/Qdrant。'}
        except KeyError:
            raise HTTPException(404, 'Unknown session')
        except ValueError as error:
            raise HTTPException(409, str(error))

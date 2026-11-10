"""Chat paper panel and durable offline publication over the measured RAG builder."""
import hashlib
import json
import time
import uuid
from pathlib import Path

from fastapi import Depends, HTTPException
from pydantic import BaseModel, Field

from researchclaw.core.config import PROJECT_ROOT


def public_paper_url(url):
    """Public scholarly sites, including PMLR's current official asset repository."""
    import re
    from urllib.parse import urlparse
    parsed = urlparse(url)
    try:
        if parsed.scheme != 'https' or parsed.port not in {None, 443} or parsed.username:
            return False
    except ValueError:
        return False
    sites = {'arxiv.org', 'export.arxiv.org', 'aclanthology.org', 'openaccess.thecvf.com',
        'proceedings.mlr.press', 'jmlr.org', 'www.jmlr.org', 'zenodo.org'}
    official_asset = (parsed.hostname == 'raw.githubusercontent.com' and bool(re.fullmatch(
        r'/mlresearch/v\d+/(?:main|master)/assets/[a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+\.pdf', parsed.path)))
    return parsed.hostname in sites or official_asset


class IngestProposal(BaseModel):
    project_id: str = 'default'
    session_id: str = Field(pattern=r'^[a-f0-9]{32}$')


def progress_view(state):
    """File work and batch counts, never a simulated timer; ETA is explicitly an estimate."""
    result = dict(state)
    elapsed = max(0., time.time() - state['started'])
    total = state.get('total_files', 0)
    done = state.get('completed_files', 0)
    phase = state.get('phase', 'queued')
    fraction = done / max(1, total)
    if phase == 'embed' and total:
        fraction += .75 * state.get('vectors_done', 0) / max(1, state.get('vectors_total', 0)) / total
    result['parse_progress'] = round(min(100, 100 * (done + (phase == 'embed')) / max(1, total)), 1)
    result['build_progress'] = round(min(85, 85 * fraction), 1)
    if phase in {'dense_sync', 'bm25', 'published'}:
        result['parse_progress'] = 100
        result['build_progress'] = {'dense_sync': 88, 'bm25': 94, 'published': 99}[phase]
    if state['status'] == 'completed':
        result['parse_progress'] = result['build_progress'] = 100
    result['elapsed_seconds'] = round(state.get('elapsed_seconds', elapsed), 1)
    result['eta_seconds'] = (round(elapsed * (1 - fraction) / fraction)
        if state['status'] == 'running' and 0 < fraction < 1 and done >= 1 else None)
    result['eta_basis'] = '按本批已完成文件平均耗时估算，未包含未知模型加载和最终索引发布时间'
    return result


class PaperJobs:
    def __init__(self, store, project, pool):
        self.store, self.project, self.pool = store, project, pool

    def snapshot(self, owner, project_id, session_id):
        root = self.project(owner, project_id)
        self.store.session(session_id, owner, project_id)
        documents = []
        origins = self.store.origins(owner, project_id, 'document')
        names = {}
        with self.store.connect() as db:
            names = {r['document']: r['name'] for r in db.execute(
                'SELECT document,name FROM upload_names WHERE owner=? AND project=?', (owner, project_id))}
            jobs = [json.loads(r['state']) for r in db.execute(
                'SELECT state FROM web_jobs WHERE owner=? AND project=? ORDER BY created_at DESC LIMIT 40', (owner, project_id))]
        from researchclaw.application.paper_library import library_paths
        stores = []
        papers = []
        for path in library_paths(root):
            manifest = path / 'rag-search-index.json'
            if not manifest.is_file():
                continue
            from researchclaw.core.qdrant_store import QdrantResearchStore
            store = QdrantResearchStore(path, recover=False)
            with store.database.connect(read_only=True) as db:
                rows = [dict(r) for r in db.execute('SELECT paper_id,title,content_hash,quality_status,file_path FROM papers')]
            for row in rows:
                papers.append({k: row[k] for k in ('paper_id', 'title', 'content_hash', 'quality_status')} |
                    {'scope': 'project' if path.is_relative_to(root) else 'shared'})
            info = json.loads(manifest.read_text(encoding='utf8'))
            stores.append({'scope': 'project' if path.is_relative_to(root) else 'shared',
                'papers': len(rows), 'children': info['document_count'], 'version': info['corpus_version']})
        known = {r['content_hash'] for r in papers}
        for path in sorted((root / 'uploads').glob('*')):
            if not path.is_file():
                continue
            key = 'uploads/' + path.name
            digest = path.stem
            if len(digest) != 64:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
            documents.append({'document': key, 'name': names.get(key, path.name), 'bytes': path.stat().st_size,
                'indexed': digest in known, 'session_id': origins.get(key), 'supported': path.suffix.lower() == '.pdf'})
        from .paper_versions import read_version
        version = read_version(root)
        for job in jobs:
            if job.get('test_version'):
                job['publication_active'] = bool(version and version['status'] == 'active'
                    and version['id'][:16] in job.get('target', ''))
        return {'documents': documents, 'papers': papers, 'corpora': stores,
            'test_version': {k: version[k] for k in ('id', 'status', 'created')} if version else None,
            'paper_count': len({r['content_hash'] for r in papers}), 'capacity': 2000,
            'jobs': [progress_view(j) if j.get('kind') == 'rag-publication' else j for j in jobs]}

    def start(self, owner, project_id, session_id, frozen_inputs):
        root = self.project(owner, project_id)
        self.store.session(session_id, owner, project_id)
        paths = []
        from researchclaw.knowledge.rag_pipeline import sha256
        for relative, digest in frozen_inputs.items():
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or path.suffix.lower() != '.pdf' or sha256(path) != digest:
                raise ValueError('只接受已确认、哈希未变化的项目内 PDF')
            paths.append(path)
        if not paths:
            raise ValueError('没有可构建的 PDF')
        snapshot = self.snapshot(owner, project_id, session_id)
        shared_hashes = {p['content_hash'] for p in snapshot['papers'] if p['scope'] == 'shared'}
        from researchclaw.application.paper_library import active_test_library
        if active_test_library(root) and (root / 'paper_library' / 'rag-search-index.json').is_file():
            from researchclaw.core.qdrant_store import QdrantResearchStore
            with QdrantResearchStore(root / 'paper_library', recover=False).database.connect(read_only=True) as db:
                shared_hashes |= {r[0] for r in db.execute('SELECT content_hash FROM papers')}
        paths = [p for p in paths if sha256(p) not in shared_hashes]
        if not paths:
            raise ValueError('确认的论文均已存在于原知识库，无需重复构建')
        hashes = {p['content_hash'] for p in snapshot['papers']} | set(frozen_inputs.values())
        if (root / 'paper_library' / 'runtime-binding.json').is_file():
            from researchclaw.core.qdrant_store import QdrantResearchStore
            with QdrantResearchStore(root / 'paper_library', recover=False).database.connect(read_only=True) as db:
                hashes |= {r[0] for r in db.execute('SELECT content_hash FROM papers')}
        if len(hashes) > 2000:
            raise ValueError('论文总数超过 2000 篇硬上限')
        identity = uuid.uuid4().hex
        from researchclaw.application.paper_library import active_test_library
        target = active_test_library(root) or root / 'paper_library'
        state = {'id': identity, 'kind': 'rag-publication', 'action': 'incremental-build',
            'target': str(target), 'test_version': target != root / 'paper_library',
            'session_id': session_id, 'status': 'queued', 'phase': 'queued', 'stage': '等待离线构建',
            'started': time.time(), 'completed_files': 0, 'total_files': len(paths)}
        with self.store.connect() as db:
            # Serialize admission across requests/processes, not merely one Python lock.
            db.execute('SELECT pg_advisory_xact_lock(?)', (int(hashlib.sha256(str(root).encode()).hexdigest()[:15], 16),))
            jobs = db.execute('SELECT state FROM web_jobs WHERE owner=? AND project=?', (owner, project_id))
            if any(json.loads(r['state']).get('status') in {'queued', 'running'} for r in jobs):
                raise ValueError('本项目已有后台任务，请等待完成')
            db.execute('INSERT INTO web_jobs(id,owner,project,state) VALUES(?,?,?,?)',
                (identity, owner, project_id, json.dumps(state, ensure_ascii=False)))
        self.pool.submit(self.execute, owner, project_id, paths, state)
        return progress_view(state)

    def execute(self, owner, project_id, paths, state):
        root = self.project(owner, project_id)
        target = Path(state['target'])
        report_path = root / 'web_jobs' / state['id'] / 'build-report.json'
        def persist():
            with self.store.connect() as db:
                db.execute('UPDATE web_jobs SET state=? WHERE id=?', (json.dumps(state, ensure_ascii=False), state['id']))
        labels = {'inventory': '检查内容哈希与可复用缓存', 'parse': '清洗、解析论文与父子分块',
            'embed': '仅为新增切片计算向量', 'file_done': '文件处理完成',
            'dense_sync': '增量同步 Qdrant 向量', 'bm25': '更新 BM25 统计与发布检索索引', 'published': '索引已发布'}
        def progress(phase, **values):
            state.update(phase=phase, stage=labels[phase], **values)
            persist()
        try:
            state['status'] = 'running'; persist()
            manifest = target / 'rag-search-index.json'
            info = json.loads(manifest.read_text(encoding='utf8')) if manifest.is_file() else {}
            model = info.get('embedding_key', 'BAAI/bge-small-en-v1.5@5c38ec7c405ec4b44b94cc5a9bb96e735b38267a').split('@', 1)
            from researchclaw.knowledge.rag_pipeline import build_corpus, PAGE_CHUNK_POLICY
            build_corpus(target, [('user-papers', p) for p in paths], corpus_version=info.get('corpus_version', 'web-papers-v1'),
                model_name=model[0] or None, model_revision=model[1] if len(model) == 2 else None,
                chunk_policy=PAGE_CHUNK_POLICY, report_path=report_path, progress=progress)
            from researchclaw.core.qdrant_store import QdrantResearchStore
            from researchclaw.knowledge.research_rag import build_structure_cache
            build_structure_cache(QdrantResearchStore(target, recover=False))
            report = json.loads(report_path.read_text(encoding='utf8'))
            if not report['papers_indexed']:
                raise ValueError('未获得可用文本；扫描件或损坏论文需先 OCR/修复，本次未新增可检索论文')
            state.update(status='completed', stage='完成：新论文现在可在对话中检索',
                summary={k: report[k] for k in ('papers_encoded', 'papers_reused', 'vectors_encoded',
                    'vectors_reused', 'dense_points_written', 'quality_counts', 'children')},
                rejected_papers=[r for r in report['papers'] if r.get('published') is False])
        except Exception as error:
            import logging
            logging.exception('Paper publication failed: %s', state['id'])
            state.update(status='failed', stage='构建失败；修复后重新确认可复用已有缓存', error=str(error))
        state['elapsed_seconds'] = round(time.time() - state['started'], 2)
        persist()


def install_library(api, project, user, pool, actions):
    service = PaperJobs(api.state.store, project, pool)
    api.state.paper_jobs = service
    from .paper_versions import install_versions
    install_versions(api, project, user)

    @api.get('/api/v1/library/paper')
    def paper(paper_id: str, project_id: str = 'default', owner=Depends(user)):
        from researchclaw.application.paper_library import PaperLibrary
        from researchclaw.application.workflow import ResearchClawApp
        from fastapi.responses import FileResponse
        library = PaperLibrary(ResearchClawApp(project(owner, project_id)))
        for store in library.stores:
            row = store.get_paper(paper_id)
            if row and row.get('file_path'):
                path = Path(row['file_path'])
                if path.is_file() and path.suffix.lower() == '.pdf':
                    return FileResponse(path, media_type='application/pdf', content_disposition_type='inline')
        raise HTTPException(404, '原始 PDF 不可用')

    @api.get('/api/v1/library')
    def snapshot(session_id: str, project_id: str = 'default', owner=Depends(user)):
        try:
            return service.snapshot(owner, project_id, session_id)
        except KeyError:
            raise HTTPException(404, 'Unknown session')

    @api.post('/api/v1/library/propose-ingest')
    def propose(body: IngestProposal, owner=Depends(user)):
        from researchclaw.application.workflow import ResearchClawApp
        root = project(owner, body.project_id)
        try:
            snapshot = service.snapshot(owner, body.project_id, body.session_id)
            pending = [d for d in snapshot['documents'] if not d['indexed'] and d['supported']]
            if not pending:
                raise ValueError('没有待入库 PDF；已入库的同哈希论文会自动去重')
            # Bind the exact upload inventory. New uploads invalidate this proposal.
            return actions.propose(body.session_id, owner, body.project_id, 'ingest_documents',
                {'path': str(root / 'uploads')}, '确认将待处理 PDF 增量加入本项目知识库；历史论文仅复用，不重新编码。',
                ResearchClawApp(root).registry)
        except KeyError:
            raise HTTPException(404, 'Unknown session')
        except ValueError as error:
            raise HTTPException(409, str(error))

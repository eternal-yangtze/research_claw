"""Authenticated project workbench; jobs invoke the same CLI as the terminal."""
import json
import hashlib
import logging
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from fastapi import Depends, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from researchclaw.application.workflow import ResearchClawApp
from researchclaw.core.config import PROJECT_ROOT, resolve_source_path


def classify_domain(text):
    """Transparent provisional labels; users can override them explicitly."""
    text=text.casefold()
    groups=[('计算机视觉 / 参数高效适配',('lora','visual prompt','视觉提示','低秩','parameter efficient')),
            ('计算机视觉',('image','vision','图像','视觉','classification')),
            ('自然语言处理 / RAG',('retrieval','language model','rag','检索','语言模型')),
            ('机器学习 / 优化',('learning','optimization','训练','优化'))]
    return next((domain for domain,words in groups if any(w in text for w in words)), '待分类')


def scalar_metrics(value, prefix='', depth=0):
    if depth > 6 or not isinstance(value, dict):
        return {}
    result = {}
    for key, item in value.items():
        name = prefix + str(key)
        if isinstance(item, (int, float, bool)) or item is None:
            result[name] = item
        elif isinstance(item, dict):
            result.update(scalar_metrics(item, name + '.', depth + 1))
        elif isinstance(item, list):
            for index, entry in enumerate(item[:50]):
                if isinstance(entry, dict):
                    label = entry.get('name', entry.get('id', index))
                    result.update(scalar_metrics(entry, f'{name}[{label}].', depth + 1))
    return result


class Action(BaseModel):
    project_id: str = 'default'
    action: str
    text: str = Field(default='', max_length=12000)
    document: str = ''
    paper_ids: list[str] = Field(default_factory=list)
    cascade: bool = False
    confirm: bool = False
    artifact: str = ''
    token_cap: int = Field(default=60000, ge=1000, le=600000)
    session_id: str | None = Field(default=None, pattern=r'^[a-f0-9]{32}$')
    domain: str = Field(default='', max_length=80)


def install_workbench(api, project, user, pool):
    locks = {}
    guard = threading.Lock()
    with api.state.store.connect() as db:
        for row in db.execute('SELECT id,state FROM web_jobs').fetchall():
            value = json.loads(row['state'])
            if value.get('status') in {'queued', 'running'}:
                value.update(status='interrupted', error='Backend restarted. Inspect child process, logs and checkpoint before explicit resume; no automatic model replay.')
                db.execute('UPDATE web_jobs SET state=? WHERE id=?', (json.dumps(value), row['id']))

    def lock(root):
        with guard:
            return locks.setdefault(str(root), threading.RLock())

    def files(root):
        # Only serve documents inside this authenticated project, never arbitrary paths.
        return {p.relative_to(root).as_posix(): p for folder in ('uploads', 'artifacts/papers')
                for p in (root / folder).rglob('*') if p.is_file() and p.suffix.lower() in {'.pdf', '.md', '.txt'}}

    def sync(research, progress=None):
        rows = [dict(r) for r in research.store.all_chunks()]
        vectors = {}
        for offset in range(0, len(rows), 64):
            if progress:
                progress('向量化 / 首次加载模型', offset, len(rows))
            vectors.update(research.retriever._vectors(rows[offset:offset+64]))
        backend = research.retriever.vector_backend
        if progress:
            progress('同步向量库', len(rows), len(rows))
        if hasattr(backend, 'sync'):
            backend.sync(rows, vectors)
        research.store.require_index_ready()
        return {'chunks': len(rows), 'vectors': len(vectors), 'backend': backend.name}

    @api.post('/api/v1/workbench/dataset-upload', status_code=201)
    async def dataset_upload(request: Request, dataset_id: str, filename: str, project_id: str = 'default', owner=Depends(user)):
        from researchclaw.experiments.datasets import REGISTRY
        if dataset_id not in REGISTRY or filename not in {f['filename'] for f in REGISTRY[dataset_id]['files']}:
            raise HTTPException(422, 'Upload original official filenames for a supported dataset')
        root = project(owner, project_id) / 'dataset_uploads' / dataset_id
        root.mkdir(parents=True, exist_ok=True)
        temporary = root / (uuid.uuid4().hex + '.part')
        size = 0
        try:
            with temporary.open('wb') as output:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > 1024 * 1024 * 1024:
                        raise HTTPException(413, 'Archive exceeds 1 GiB; use CLI for larger archives')
                    output.write(chunk)
            if not size:
                raise HTTPException(422, 'Empty archive')
            temporary.replace(root / filename)
        finally:
            temporary.unlink(missing_ok=True)
        return {'status': 'staged', 'path': str(root / filename), 'bytes': size, 'indexed': False}

    @api.get('/api/v1/workbench')
    def snapshot(project_id: str = 'default', session_id: str = '', owner=Depends(user)):
        root = project(owner, project_id)
        if session_id:
            try:
                api.state.store.session(session_id,owner,project_id)
            except KeyError:
                raise HTTPException(404,'Unknown session')
        research = ResearchClawApp(root)
        with research.store.connect() as db:
            papers = [dict(r) for r in db.execute('SELECT paper_id,title,file_path,quality_status FROM papers')]
        for paper in papers:
            paper['file_path'] = resolve_source_path(paper['file_path'])
        indexed = {str(Path(p['file_path']).resolve()): p for p in papers if p['file_path']}
        document_origins = api.state.store.origins(owner,project_id,'document')
        paper_origins = api.state.store.origins(owner,project_id,'paper')
        documents = []
        with api.state.store.connect() as db:
            names = {row['document']:row['name'] for row in db.execute('SELECT document,name FROM upload_names WHERE owner=? AND project=?', (owner,project_id))}
        for key, path in files(root).items():
            try:
                size = path.stat().st_size
            except FileNotFoundError:
                continue  # A concurrent clear/delete may have moved the file.
            paper = indexed.get(str(path.resolve()))
            documents.append({'document': key, 'name': names.get(key, path.name), 'path': str(path), 'bytes': size,
                              'indexed': bool(paper and research.store.paper_chunks(paper['paper_id'])), 'paper': paper,
                              'session_id':document_origins.get(key), 'can_delete':bool(session_id and document_origins.get(key)==session_id)})
        jobs = []
        with api.state.store.connect() as db:
            job_rows = db.execute('SELECT state FROM web_jobs WHERE owner=? AND project=? ORDER BY created_at DESC,id DESC LIMIT 30', (owner, project_id)).fetchall()
        for row in job_rows:
            try:
                job = json.loads(row['state'])
                if job['status'] == 'running':
                    job['elapsed_seconds'] = round(time.time() - job['started'], 2)
                log = root / 'web_jobs' / job['id'] / 'output.log'
                job['output'] = log.read_text(encoding='utf-8', errors='replace')[-16000:] if log.exists() else ''
                jobs.append(job)
            except (ValueError, OSError):
                continue
        artifacts, measured, coding_reports = [], [], []
        for folder in ('runs', 'studies', 'study_runs', 'coding', 'datasets', 'model_calls', 'artifacts', 'vector_audit'):
            for path in (root / folder).rglob('*.json*'):
                if len(artifacts) >= 300:
                    break
                artifacts.append({'path': str(path.relative_to(root)), 'bytes': path.stat().st_size})
                if path.suffix == '.json' and path.stat().st_size < 2_000_000:
                    try:
                        payload = json.loads(path.read_text(encoding='utf-8'))
                        values = scalar_metrics(payload)
                        if values:
                            measured.append({'source': str(path.relative_to(root)), 'values': values})
                    except (ValueError, OSError):
                        pass
        for job in jobs:
            try:
                values = scalar_metrics(json.loads(job['output']))
                if values:
                    measured.append({'source': 'web_jobs/' + job['id'] + '/output.log', 'values': values})
            except ValueError:
                pass
        # Final coding reports and graph commits must not disappear when the
        # general artifact preview reaches its display limit.
        for path in (root / 'coding').glob('*/report.json'):
            try:
                payload = json.loads(path.read_text(encoding='utf-8'))
                coding_reports.append(payload)
                source = str(path.relative_to(root))
                if not any(m['source'] == source for m in measured):
                    measured.append({'source':source, 'values':scalar_metrics(payload)})
            except (ValueError, OSError):
                pass
        dataset_files = [{'path': str(p.relative_to(root)), 'bytes': p.stat().st_size}
                         for p in (root / 'dataset_uploads').rglob('*') if p.is_file()]
        dataset_states = []
        for path in (root / 'datasets').glob('*/state.json'):
            try:
                value = json.loads(path.read_text(encoding='utf-8'))
                verification = value.get('verification',{})
                dataset_states.append({'dataset_id':value.get('dataset_id'), 'status':value.get('status'),
                    'source_format':value.get('source_format'), 'classes':verification.get('class_count'),
                    'split_counts':verification.get('split_counts'), 'manifest_path':verification.get('manifest_path'),
                    'root':verification.get('root'), 'reuse':value.get('web_reuse'), 'state':str(path.relative_to(root))})
            except (ValueError, OSError):
                continue
        nodes = []
        for name, role in [('retrieve','证据检索'), ('planner','方案 Agent'), ('compile_gate','机械门禁'), ('critic','评审 Agent'), ('final_review','终审 Agent'), ('decide','宿主决策'), ('submit','正式提交')]:
            commits = [p for base in (root / 'study_runs',root / 'studies')
                       for p in base.rglob(name + '_*.json') if p.parent.name == 'node_commits']
            nodes.append({'id':name, 'role':role, 'commits':len(commits), 'status':'有完成记录' if commits else '未观察到完成记录'})
        coding_summary = {'executed_reports':len(coding_reports),
                          'existing_source_rechecks':sum(r.get('execution_kind')=='recheck_existing_source' for r in coding_reports),
                          'generation_reports':sum(r.get('execution_kind')!='recheck_existing_source' for r in coding_reports),
                          'pass_rate':sum(r.get('status') == 'coding_verified' for r in coding_reports)/len(coding_reports) if coding_reports else None,
                          'total_tokens':sum(r.get('total_tokens', 0) for r in coding_reports) if coding_reports else None,
                          'wall_seconds':sum(r.get('wall_seconds', 0) for r in coding_reports) if coding_reports else None,
                          'research_success':None}
        for paper in papers:
            paper['chunks'] = len(research.store.paper_chunks(paper['paper_id']))
            # Search candidates legitimately have no downloaded local file yet.
            source = Path(paper['file_path']) if paper['file_path'] else None
            paper['source_exists'] = source.is_file() if source is not None else False
            paper['name'] = names.get('uploads/' + source.name, paper['title']) if source is not None else paper['title']
            paper['session_id'] = paper_origins.get(paper['paper_id'])
            paper['can_delete'] = bool(session_id and paper['session_id']==session_id)
            with api.state.store.connect() as db:
                category = db.execute('SELECT domain FROM paper_domains WHERE owner=? AND project=? AND paper=?',(owner,project_id,paper['paper_id'])).fetchone()
            paper['domain'] = category['domain'] if category else classify_domain(paper['title']+' '+ ' '.join(r['text'][:300] for r in research.store.paper_chunks(paper['paper_id'])[:3]))
        return {'workspace': str(root), 'knowledge': papers, 'documents': documents, 'counts': research.store.counts(), 'jobs': jobs, 'dataset_files': dataset_files, 'dataset_states':dataset_states,
                'artifacts': artifacts, 'backend': research.retriever.vector_backend.name,
                'knowledge_backend': 'postgresql+qdrant',
                'index_operations': research.store.index_status(),
                'knowledge_collection': getattr(research.store,'collection',None),
                'runtime_database': 'PostgreSQL / ' + research.store.database.schema,
                'embedding': research.retriever.embedding.model_name,
                'agent_graph': ['retrieve → planner → compile_gate', 'compile_gate → critic → final_review',
                                'final_review → decide → revise / submit'],
                'agent_nodes':nodes, 'coding_summary':coding_summary,
                'measured_metrics': measured, 'quality_metrics': None}

    @api.get('/api/v1/workbench/file')
    def document(document: str, project_id: str = 'default', download: bool = False, owner=Depends(user)):
        root = project(owner, project_id)
        path = files(root).get(document)
        if path is None:
            raise HTTPException(404, 'Unknown project document')
        return FileResponse(path, filename=path.name, content_disposition_type='attachment' if download else 'inline')

    @api.get('/api/v1/workbench/artifact')
    def artifact(path: str, project_id: str = 'default', owner=Depends(user)):
        root = project(owner, project_id).resolve()
        target = (root / path).resolve()
        if not target.is_relative_to(root) or target.suffix not in {'.json', '.jsonl', '.log'} or not target.is_file():
            raise HTTPException(404, 'Unknown artifact')
        return {'path': path, 'content': target.read_text(encoding='utf-8', errors='replace')[-100000:]}

    @api.post('/api/v1/workbench/action')
    def action(body: Action, owner=Depends(user)):
        root = project(owner, body.project_id)
        research = ResearchClawApp(root)
        if body.session_id:
            try:
                api.state.store.session(body.session_id,owner,body.project_id)
            except KeyError:
                raise HTTPException(404,'Unknown session')
        if body.action == 'retry-index':
            if not body.confirm:
                raise HTTPException(422, 'Explicit retry confirmation required')
            with lock(root):
                research.store.flush_outbox()
                try:
                    research.store.require_index_ready()
                except RuntimeError as error:
                    raise HTTPException(503, str(error))
            return {'status': 'completed', 'operations': research.store.index_status()}
        if body.action == 'set-domain':
            if not body.domain.strip() or not body.paper_ids:
                raise HTTPException(422,'Choose papers and a domain')
            with research.store.connect() as db:
                valid={r[0] for r in db.execute('SELECT paper_id FROM papers')}
            if not set(body.paper_ids)<=valid:
                raise HTTPException(404,'Unknown paper')
            with api.state.store.connect() as db:
                for pid in body.paper_ids:
                    db.execute('INSERT INTO paper_domains VALUES(?,?,?,?) ON CONFLICT(owner,project,paper) DO UPDATE SET domain=excluded.domain',(owner,body.project_id,pid,body.domain.strip()))
            return {'status':'updated'}
        if body.action in {'delete-file','delete-knowledge','clear','remove'}:
            if not body.session_id:
                raise HTTPException(422,'必须选择当前对话；旧知识库不可通过清空接口删除')
            owned_papers={k for k,v in api.state.store.origins(owner,body.project_id,'paper').items() if v==body.session_id}
            owned_documents={k for k,v in api.state.store.origins(owner,body.project_id,'document').items() if v==body.session_id}
            if body.action=='remove':
                raise HTTPException(422,'Use scoped delete-knowledge with session_id')
            if body.action=='delete-file' and body.document not in owned_documents:
                raise HTTPException(403,'只允许删除当前对话新增的文件，历史论文受保护')
            if body.action=='delete-knowledge' and not set(body.paper_ids)<=owned_papers:
                raise HTTPException(403,'只允许删除当前对话新增的知识，历史知识受保护')
        if body.action in {'process', 'delete-file', 'delete-knowledge', 'clear'}:
            if not body.confirm:
                raise HTTPException(422, 'Explicit confirmation required')
            with api.state.store.connect() as db:
                active = [json.loads(r[0]) for r in db.execute('SELECT state FROM web_jobs WHERE owner=? AND project=?', (owner,body.project_id))]
            if any(j.get('status') in {'queued','running'} for j in active):
                raise HTTPException(409, '当前任务运行中，请完成后再处理或删除资料')
            identity = uuid.uuid4().hex
            folder = root / 'web_jobs' / identity
            folder.mkdir(parents=True)
            state = {'id':identity,'action':body.action,'session_id':body.session_id,'status':'queued','stage':'等待处理','started':time.time(),'completed_files':0,'total_files':len(files(root))}
            def persist():
                with api.state.store.connect() as db:
                    db.execute('INSERT INTO web_jobs(id,owner,project,state) VALUES(?,?,?,?) ON CONFLICT(id) DO UPDATE SET state=excluded.state',(identity,owner,body.project_id,json.dumps(state,ensure_ascii=False)))
                with (folder/'state.events.jsonl').open('a',encoding='utf-8') as journal:
                    journal.write(json.dumps(state,ensure_ascii=False)+'\n')
            persist()
            def process_documents():
                with lock(root):
                    try:
                        state.update(status='running',stage='检查文件'); persist()
                        if body.action == 'process':
                            for key,path in files(root).items():
                                state.update(stage='解析与切片',document=key); persist()
                                digest=hashlib.sha256(path.read_bytes()).hexdigest()
                                with research.store.connect() as db:
                                    row=db.execute('SELECT paper_id FROM papers WHERE file_path=? AND content_hash=?',(str(path),digest)).fetchone()
                                if not row or not research.store.paper_chunks(row[0]):
                                    with research.store.connect() as db:
                                        existing_ids={r[0] for r in db.execute('SELECT paper_id FROM papers')}
                                    result = research.ingestor.ingest_file(path)
                                    if result.paper_id not in existing_ids:
                                        origin=api.state.store.origins(owner,body.project_id,'document').get(key)
                                        api.state.store.claim(owner,body.project_id,'paper',result.paper_id,origin)
                                    if result.chunks == 0:
                                        raise ValueError('未提取到文本：扫描 PDF 需要设置 RESEARCHCLAW_PDF_PARSER=docling 后重试')
                                state['completed_files']+=1; persist()
                        else:
                            selected=files(root).get(body.document)
                            if body.action=='delete-file' and selected is None:
                                raise ValueError('文件不存在，请刷新列表')
                            with research.store.connect() as db:
                                ids=list(owned_papers) if body.action=='clear' else list(body.paper_ids)
                                if body.action=='delete-file':
                                    ids=[r[0] for r in db.execute('SELECT paper_id FROM papers WHERE file_path=?',(str(selected),)) if r[0] in owned_papers] if body.cascade else []
                                chunk_ids=[r[0] for pid in ids for r in db.execute('SELECT chunk_id FROM chunks WHERE paper_id=?',(pid,))]
                                state.update(stage='删除所选论文的向量（无需加载模型）'); persist()
                                backend=research.retriever.vector_backend
                                # PostgreSQL cascade + outbox owns index deletion.
                                # Never delete Qdrant before the SQL transaction commits.
                                for pid in ids:
                                    db.execute('DELETE FROM papers WHERE paper_id=?',(pid,))
                            if selected is not None and body.action=='delete-file':
                                trash=root/'trash'/identity
                                trash.mkdir(parents=True)
                                selected.replace(trash/selected.name)
                            if body.action=='clear':
                                trash=root/'trash'/identity
                                for key,path in files(root).items():
                                    if key in owned_documents:
                                        destination=trash/key
                                        destination.parent.mkdir(parents=True,exist_ok=True)
                                        path.replace(destination)
                        if body.action == 'process':
                            state.update(stage='加载 Embedding / 向量化并同步向量库'); persist()
                            def progress(stage, done, total):
                                state.update(stage=stage, vectors_done=done, vectors_total=total); persist()
                            state['index']=sync(research, progress)
                        else:
                            state['index']={'backend':backend.name,'deleted_chunks':len(chunk_ids),'embedding_required':False}
                        research.store.require_index_ready()
                        state.update(status='completed',stage='完成：知识库已同步')
                    except Exception as error:
                        state.update(status='failed',stage='处理失败：入库切片保留，可重试向量化' if body.action=='process' else '删除未完成，请检查错误后重试',error=str(error))
                    state['elapsed_seconds']=round(time.time()-state['started'],2); persist()
            pool.submit(process_documents)
            return dict(state)
        if body.action == 'reuse-datasets':
            from researchclaw.core.artifacts import file_hash
            ids = body.text.split()
            if not body.confirm or not ids or any(x not in {'cifar10','cifar100','dtd','pets','flowers102'} for x in ids):
                raise HTTPException(422, 'Confirm reuse and choose supported dataset IDs')
            reused = []
            with lock(root):
                for dataset_id in ids:
                    source = PROJECT_ROOT / 'datasets' / 'workspace_batch_vit_v2' / dataset_id / 'state.json'
                    if not source.is_file():
                        raise HTTPException(422, 'Local verified library unavailable; download or upload official archives')
                    payload = json.loads(source.read_text(encoding='utf-8'))
                    verification = payload.get('verification', {})
                    manifest = Path(verification.get('manifest_path',''))
                    if payload.get('status') != 'verified' or not manifest.is_file() or file_hash(manifest) != verification.get('manifest_sha256') or not Path(verification.get('root','')).is_dir():
                        raise HTTPException(422, 'Local dataset manifest/root failed reuse validation')
                    target = root / 'datasets' / dataset_id / 'state.json'
                    if target.exists():
                        raise HTTPException(409, 'Dataset state already exists; preserve it or choose a new project')
                    payload['web_reuse'] = {'source':str(source), 'mode':'reference_existing_verified_assets', 'manifest_rehashed':True, 'images_rehashed':False}
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
                    reused.append({'dataset_id':dataset_id,'state':str(target),'verification':verification})
            return {'status':'reused','datasets':reused,'note':'Existing images referenced; this is not a new dataset download or full image re-verification'}
        if body.action == 'index':
            if not body.confirm:
                raise HTTPException(422, 'Explicit confirmation required')
            with lock(root):
                if body.action == 'index':
                    results = []
                    for p in files(root).values():
                        digest = hashlib.sha256(p.read_bytes()).hexdigest()
                        with research.store.connect() as db:
                            row = db.execute('SELECT paper_id FROM papers WHERE file_path=? AND content_hash=?', (str(p), digest)).fetchone()
                        if row and research.store.paper_chunks(row[0]):
                            results.append({'paper_id': row[0], 'status': 'unchanged', 'path': str(p)})
                        else:
                            with research.store.connect() as db:
                                previous={r[0] for r in db.execute('SELECT paper_id FROM papers')}
                            result=research.ingestor.ingest_file(p)
                            if result.paper_id not in previous:
                                key=p.relative_to(root).as_posix()
                                api.state.store.claim(owner,body.project_id,'paper',result.paper_id,api.state.store.origins(owner,body.project_id,'document').get(key))
                            results.append(result.to_dict())
                    return {'results': results, 'index': sync(research)}
        allowed = {'search', 'ask', 'online-search', 'download-papers', 'research', 'plan-study', 'resume-study', 'datasets-acquire',
                   'dataset-import', 'prepare-experiment', 'coding-plan', 'code', 'resume', 'skills', 'status', 'tools',
                   'batch-research', 'silver-evaluate', 'reanalyze', 'audit', 'validate-formulas', 'recheck-example'}
        if body.action not in allowed:
            raise HTTPException(422, 'Unknown action')
        if body.action in {'research', 'plan-study', 'resume-study', 'coding-plan', 'code', 'resume', 'datasets-acquire', 'prepare-experiment', 'batch-research', 'silver-evaluate', 'reanalyze', 'recheck-example'} and not body.confirm:
            raise HTTPException(422, 'Confirm network/model or Docker execution')
        args = [body.action]
        study_key = uuid.uuid4().hex[:12]
        job_workspace = root
        if body.action in {'search', 'ask', 'online-search', 'download-papers', 'research', 'plan-study'}:
            if not body.text.strip():
                raise HTTPException(422, 'Input required')
            args += [body.text]
        if body.action == 'download-papers':
            # Interactive search avoids downloading/parsing the entire ACL catalog.
            # The CLI still exposes all seven sources explicitly.
            args += ['--sources', 'arxiv', 'openalex', 'openreview', 'dblp', 'crossref']
        if body.action == 'research':
            args += ['--max-downloads', '3', '--max-queries', '2', '--token-cap', str(body.token_cap)]
        if body.action == 'plan-study':
            study_source = root / 'study_sources' / study_key
            job_workspace = root / 'study_runs' / study_key
            args += ['--source', str(study_source), '--dataset-workspace', str(root), '--token-cap', str(body.token_cap)]
        if body.action == 'resume-study':
            target = (root / body.artifact).resolve()
            if not target.is_relative_to((root / 'study_runs').resolve()) or target.name != 'study_snapshot.json' or not target.is_file() or not body.text.strip():
                raise HTTPException(422, 'Select study_runs/<id>/study_snapshot.json and repeat the original Idea exactly')
            binding = json.loads(target.read_text(encoding='utf-8'))
            study_source = Path(binding['source_workspace']).resolve()
            if not study_source.is_relative_to(root.resolve()):
                raise HTTPException(422, 'Study source must belong to this project')
            job_workspace = target.parent
            args = ['plan-study', body.text, '--source', str(study_source), '--dataset-workspace', str(root),
                    '--token-cap', str(body.token_cap), '--retrieval-strategy', binding.get('retrieval_strategy','legacy'),
                    '--orchestration', binding.get('orchestration','langgraph')]
        if body.action in {'coding-plan', 'code', 'resume', 'prepare-experiment'}:
            if not body.text or not all(c.isalnum() or c in '_-' for c in body.text):
                raise HTTPException(422, 'Input must be an idea ID')
            args = ['code' if body.action == 'resume' else body.action, '--idea-id', body.text]
            if body.action == 'resume':
                args += ['--resume']
            if body.action in {'coding-plan', 'code', 'resume'}:
                if body.artifact and body.token_cap < 120000:
                    raise HTTPException(422, 'Accepted-study Coding requires token_cap >= 120000; set the budget input before execution')
                args += ['--token-cap', str(body.token_cap)]
                if body.artifact:
                    target = (root / body.artifact).resolve()
                    if not target.is_relative_to(root.resolve()) or not target.is_file() or target.suffix != '.json':
                        raise HTTPException(422, 'Select a JSON artifact inside this project')
                    args += ['--study-result', str(target)]
        if body.action == 'datasets-acquire':
            ids = body.text.split()
            if not ids or any(x not in {'cifar10', 'cifar100', 'dtd', 'pets', 'flowers102'} for x in ids):
                raise HTTPException(422, 'Choose supported dataset IDs')
            args += ids
        if body.action == 'dataset-import':
            if body.text not in {'cifar10', 'cifar100', 'dtd', 'pets', 'flowers102'}:
                raise HTTPException(422, 'Choose supported dataset ID')
            args += [body.text, '--path', str(root / 'dataset_uploads' / body.text)]
        if body.action in {'batch-research','silver-evaluate','reanalyze'}:
            if not body.text or not re.fullmatch(r'[a-zA-Z0-9_-]{1,128}',body.text):
                raise HTTPException(422,'Enter an existing idea/task ID')
            args += [body.text] if body.action == 'reanalyze' else ['--idea-id',body.text]
            if body.action == 'batch-research':
                args += ['--target-papers','6','--shortlist-size','3','--limit-per-query','3']
            if body.action == 'silver-evaluate':
                args += ['--max-papers','3','--workers','1']
        if body.action == 'validate-formulas':
            pdf = files(root).get(body.document)
            parser_json = (root / body.artifact).resolve()
            if not pdf or pdf.suffix != '.pdf' or not parser_json.is_relative_to(root.resolve()) or not parser_json.is_file() or parser_json.suffix != '.json':
                raise HTTPException(422,'Select a current-project PDF and its parser JSON artifact')
            args += ['--pdf',str(pdf),'--docling-json',str(parser_json)]
        if body.action == 'recheck-example':
            example = PROJECT_ROOT / 'self-revise/backend-revise/internal_experiments/runs/v03_p8/20261023_a/implementation_continued/coding/study_coding_a99bc09cbc6909af5978_continued'
            if not (example / 'inputs/coding_plan.json').is_file():
                raise HTTPException(422,'Local example unavailable; generate source from an accepted study first')
            args = ['coding-recheck','--plan',str(example / 'inputs/coding_plan.json'),'--project',str(example / 'project')]
        identity = uuid.uuid4().hex
        folder = root / 'web_jobs' / identity
        folder.mkdir(parents=True)
        state = {'id': identity, 'action': body.action, 'session_id':body.session_id, 'status': 'queued', 'started': time.time(), 'argv': args}
        def save():
            with api.state.store.connect() as db:
                db.execute('INSERT INTO web_jobs(id,owner,project,state) VALUES(?,?,?,?) ON CONFLICT(id) DO UPDATE SET state=excluded.state',
                           (identity, owner, body.project_id, json.dumps(state, ensure_ascii=False)))
            with (folder / 'state.events.jsonl').open('a', encoding='utf-8') as journal:
                journal.write(json.dumps(state, ensure_ascii=False) + '\n')
        save()
        def execute():
            with lock(root):
                try:
                    state['status'] = 'running'
                    save()
                    before_files=set(files(root))
                    with research.store.connect() as db:
                        before_papers={r[0] for r in db.execute('SELECT paper_id FROM papers')}
                    if body.action == 'plan-study':
                        study_source.mkdir(parents=True, exist_ok=True)
                        ResearchClawApp(study_source).store.copy_from(research.store)
                    with (folder / 'output.log').open('w', encoding='utf-8') as output:
                        process = subprocess.Popen([sys.executable, '-u', str(PROJECT_ROOT / 'researchclaw.py'),
                            '--workspace', str(job_workspace), *args], cwd=PROJECT_ROOT, stdout=output,
                            stderr=subprocess.STDOUT, shell=False)
                        state['process'] = process.pid
                        save()
                        try:
                            state['exit_code'] = process.wait(timeout=240 if body.action == 'download-papers' else None)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            state['exit_code'] = process.wait()
                            state['error'] = '在线查找/下载超过 240 秒，已终止。已下载文件保留；请检查来源日志、网络，使用简短英文关键词重试，或上传本地论文。'
                    state['status'] = 'completed' if state['exit_code'] == 0 else 'failed'
                    if body.action == 'download-papers':
                        log = (folder / 'output.log').read_text(encoding='utf-8', errors='replace')
                        decoder = json.JSONDecoder()
                        result = None
                        for offset, char in enumerate(log):
                            if char != '{':
                                continue
                            try:
                                value, _ = decoder.raw_decode(log[offset:])
                                if isinstance(value, dict) and 'downloaded' in value and 'indexed' in value:
                                    result = value
                            except ValueError:
                                pass
                        if result is not None:
                            state['download_summary'] = result
                            state['status'] = {'partial': 'partial', 'failed': 'failed'}.get(result['status'], 'completed')
                            if not result['downloaded']:
                                state['error'] = result['message']
                        elif set(files(root)) - before_files:
                            state['status'] = 'partial'
                            state['error'] = '部分文件已下载并保留；其余下载未完成，请确认左侧文件。'
                except Exception as error:
                    state.update(status='failed', error=str(error))
                if 'before_files' in locals():
                    for key in set(files(root))-before_files:
                        api.state.store.claim(owner,body.project_id,'document',key,body.session_id)
                    with research.store.connect() as db:
                        after_papers={r[0] for r in db.execute('SELECT paper_id FROM papers')}
                    for pid in after_papers-before_papers:
                        api.state.store.claim(owner,body.project_id,'paper',pid,body.session_id)
                state['elapsed_seconds'] = round(time.time() - state['started'], 2)
                save()
        future = pool.submit(execute)
        def report_failure(done):
            if done.exception():
                logging.error('Workbench job persistence failed: %s', identity, exc_info=(type(done.exception()), done.exception(), done.exception().__traceback__))
        future.add_done_callback(report_failure)
        return dict(state)

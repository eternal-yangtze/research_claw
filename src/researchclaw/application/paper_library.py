"""Read published paper indexes; online chat must never rebuild embeddings."""
import json
import os
from pathlib import Path
import threading
import uuid

from researchclaw.core.artifacts import write_json
from researchclaw.core.config import PROJECT_ROOT

_MODELS, _LOCK = {}, threading.Lock()


def library_paths(workspace):
    paths = [Path(workspace) / 'paper_library', Path(workspace)]
    active = active_test_library(workspace)
    if active:
        paths.insert(0, active)
    shared = os.getenv('RESEARCHCLAW_SHARED_CORPUS', str(PROJECT_ROOT / 'knowledge_base/rag_corpora/time_series_domain_release_v3_1'))
    if shared and shared != 'off':
        paths.append(Path(shared).resolve())
    return list(dict.fromkeys(p.resolve() for p in paths))


def active_test_library(workspace):
    """Only the explicitly active test overlay participates in retrieval."""
    root = Path(workspace).resolve()
    marker = root / 'paper-test-version.json'
    if not marker.is_file():
        return None
    state = json.loads(marker.read_text(encoding='utf8'))
    identity = state.get('id', '')
    import re
    if state.get('status') != 'active':
        return None
    if not re.fullmatch(r'[a-f0-9]{32}', identity):
        raise ValueError('Invalid paper test version')
    # Keep Windows vector-cache paths below MAX_PATH for ordinary workspaces.
    return root / 'pt' / identity[:16] / 'corpus'


class PaperLibrary:
    def __init__(self, app):
        self.app=app
        from researchclaw.core.qdrant_store import build_research_store
        self.stores = [app.store]
        for path in library_paths(app.workspace):
            if path != app.workspace and (path / 'rag-search-index.json').is_file():
                self.stores.append(build_research_store(path))

    def status(self):
        corpora = []
        for store in self.stores:
            manifest = store.root / 'rag-search-index.json'
            if not manifest.is_file():
                continue
            info = json.loads(manifest.read_text(encoding='utf8'))
            with store.database.connect(read_only=True) as db:
                count = db.execute('SELECT count(*) FROM papers').fetchone()[0]
            corpora.append({'scope': 'project' if store.root.is_relative_to(self.app.workspace) else 'shared',
                'papers': count, 'children': info['document_count'], 'version': info['corpus_version']})
        uploads = self.app.workspace / 'uploads'
        return {'corpora': corpora, 'staged_directory': str(uploads),
            'uploaded_files': [{'name': p.name, 'bytes': p.stat().st_size} for p in sorted(uploads.glob('*.pdf'))],
            'note': 'Counts and upload inventory only, not evidence of topical coverage. Search contents to assess coverage. Uploads are not necessarily indexed; use the paper panel to confirm.'}

    def locate(self,chunk_id):
        for store in self.stores:
            row=store.get_chunk(chunk_id)
            if row: return store,dict(row)
        raise KeyError('Unknown chunk in the current project/shared paper library')

    def get_chunk(self,chunk_id):
        _,row=self.locate(chunk_id)
        # Numeric embeddings/internal paths are not paper evidence and can
        # overwhelm context. Preserve exact text and stable provenance only.
        return {k:row.get(k) for k in ('chunk_id','paper_id','paper_title','section','page','text',
                                     'content_hash','node_type','parent_chunk_id','source_hash')}

    def search(self,query,limit=6,paper_id=None):
        if not isinstance(query,str) or not query.strip() or len(query)>3000 or not 1<=limit<=10:
            raise ValueError('Use a nonempty query <=3000 characters and limit 1..10')
        if paper_id is not None:
            import re
            if not re.fullmatch(r'[a-zA-Z0-9_-]{1,160}', paper_id):
                raise ValueError('Invalid paper ID')
        from researchclaw.knowledge.rag_retrieval import PublishedRAGRetriever
        from researchclaw.knowledge.retrieval import SentenceTransformerEmbedding,CrossEncoderReranker
        from researchclaw.knowledge.research_rag import ResearchRAG
        hits=[]
        for store in self.stores:
            manifest=store.root/'rag-search-index.json'
            if not manifest.is_file():
                continue  # Offline publication is explicit, never encode during chat.
            if paper_id and not store.get_paper(paper_id):
                continue
            info=json.loads(manifest.read_text(encoding='utf8'))
            key=info['embedding_key']
            with _LOCK:
                if key not in _MODELS:
                    name,revision=key.split('@',1)
                    _MODELS[key]=(SentenceTransformerEmbedding(name,revision=revision),
                        CrossEncoderReranker('cross-encoder/ms-marco-MiniLM-L6-v2',revision='233902d25c440f23af6f7d6e94d2946bac0bee0a'))
                embedding,reranker=_MODELS[key]
            retriever=PublishedRAGRetriever(store,corpus_version=info['corpus_version'],embedding=embedding,reranker=reranker)
            # The Worker performs semantic rewriting/expansion; this primitive is
            # the exact, measured anchored evidence route without nested LLM calls.
            rag=ResearchRAG(retriever,model=False,policy='anchored')
            plan={'original_query':query,'rewritten_query':query,'alternatives':[]}
            selected=(retriever.search(query,limit=limit,paper_ids=[paper_id],
                max_hits_per_paper=min(limit,4),rerank_windowed=True) if paper_id else
                rag.retrieve(plan,limit=limit,policy='anchored'))
            if getattr(retriever,'lexical_backend','')!='qdrant_bm25':
                raise RuntimeError('Publish Qdrant BM25 before chat; PostgreSQL recall fallback is disabled')
            for row in rag.contexts(selected,preserve_child=True):
                text=row['parent_text']
                row={k:v for k,v in row.items() if k not in {'parent_text','child_text'}}
                hits.append({**row,'text':text,'retrieval':'qdrant_dense+qdrant_bm25+rrf60+rerank+anchored',
                    'corpus_version':info['corpus_version']})
        # The project overlay must not hide a stronger match in the shared library.
        if len(self.stores) > 1 and hits:
            scores = reranker.score(query, [r['text'] for r in hits])
            hits = [r for r, _ in sorted(zip(hits, scores), key=lambda pair: -float(pair[1]))]
        unique = {}; paper_counts = {}
        for hit in hits:
            if hit['chunk_id'] not in unique and paper_counts.get(hit['paper_id'], 0) < (min(limit,4) if paper_id else 2):
                unique[hit['chunk_id']] = hit
                paper_counts[hit['paper_id']] = paper_counts.get(hit['paper_id'], 0) + 1
        return list(unique.values())[:limit]

    def references(self, paper_id):
        """Original bibliography text for expansion, never invented/citation-verified metadata."""
        import fitz
        import re
        for store in self.stores:
            row = store.get_paper(paper_id)
            if not row:
                continue
            path = Path(row.get('file_path') or '')
            if not path.is_file() or path.suffix.lower() != '.pdf':
                return {'paper_id': paper_id, 'available': False, 'reason': 'Original PDF unavailable'}
            with fitz.open(path) as doc:
                found = False; excerpts = []; remaining = 4500; truncated = False
                for index in range(max(0, len(doc) - 15), len(doc)):
                    text = doc[index].get_text()
                    heading = re.search(r'(?im)^\s*(?:\d+\.?\s+)?(?:references|bibliography)\s*$', text)
                    if heading:
                        found = True
                        text = text[heading.end():]
                    if found:
                        excerpt = text[:remaining]
                        import hashlib
                        identity = 'bib_' + hashlib.sha256(f'{paper_id}:{index + 1}:{excerpt}'.encode()).hexdigest()[:32]
                        excerpts.append({'chunk_id': identity, 'paper_id': paper_id, 'title': row['title'],
                            'section': 'References / bibliography', 'page': index + 1, 'text': excerpt,
                            'source_type': 'bibliography', 'scientific_findings_verified': False})
                        remaining -= len(excerpts[-1]['text'])
                    if not remaining:
                        truncated = True
                        break
            return {'paper_id': paper_id, 'title': row['title'], 'available': found,
                'excerpts': excerpts, 'truncated': truncated,
                'note': 'Bounded bibliography excerpt (<=4500 characters), not the complete reference list. Cite its returned bib_ chunk_id for bibliographic identification ONLY. Recommend only titles actually visible here; do not reread a large artifact to claim exhaustive coverage. Verify title/authors/DOI before recommending. References are leads, not evidence of their contents.'}
        raise ValueError('Paper ID not found in current project/shared library')

    def evidence(self,question,limit=5):
        hits=self.search(question,limit)
        value={'question':question,'citations':hits,'verified':bool(hits),
               'answer':'Source passages for grounded synthesis; no generated scientific conclusion.' if hits else
                        'Evidence insufficient. No published matching paper evidence; upload and publish offline or expand searches.',
               'evidence_policy':'anchored','retrieval':'Qdrant dense/BM25; PostgreSQL exact-source reads only'}
        path=self.app.artifact_root/'evidence'/(uuid.uuid4().hex+'.json')
        write_json(path,value)
        return {**value,'artifact_path':str(path)}

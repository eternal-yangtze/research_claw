"""Capture/verify library boundaries and versioned experiment inputs, read-only DB."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import re

ROOT=next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'self-revise/backend-revise/scripts')]
from research_rag import engine, WORKSPACE
from researchclaw.core.config import load_project_env
OUT=ROOT/'self-revise/backend-revise/rag-evidence/expanded-rag-v20261109'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def cleanup_tests():
    """Remove only this run's deterministic pytest schemas with synthetic papers."""
    from researchclaw.core.postgres import PostgresDatabase
    from psycopg import sql
    store=engine().store
    names=['test_qdrant_sync_filter_update_delete_and_alias',
           'test_incremental_revision_streams_old_vectors_and_only_uploads_new_chunks',
           'test_configured_vector_backend_failure_is_not_silently_local',
           'test_qdrant_incompatible_collection_is_audited_as_failed']
    removed=[]
    for directory in ['rag-expanded-tests-20261109','rag-expanded-final-tests-20261109']:
        base=(ROOT/'.maintenance'/directory).resolve()
        assert base.parent==ROOT/'.maintenance'
        for name in names:
            target=base/(re.sub(r'[\W]','_',name)[:30]+'0')
            schema=PostgresDatabase(target).schema
            assert schema!=store.database.schema
            with store.database.connect() as db:
                exists=db.execute('SELECT 1 FROM pg_namespace WHERE nspname=?',(schema,)).fetchone()
                if not exists:continue
                papers=db.raw.execute(sql.SQL('SELECT paper_id,title FROM {}.papers').format(sql.Identifier(schema))).fetchall()
                assert papers and all(row['paper_id'] in {'p1','p2','p3'} and row['title'] in {'Graph Paper','Vision Paper','Adapter Paper'} for row in papers)
                db.raw.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(schema)))
                removed.append({'schema':schema,'workspace':str(target),'synthetic_papers_verified':True})
    data={'removed_test_schemas':removed,'production_schema_untouched':store.database.schema,
          'pending_temporary_directory':str(ROOT/'self-revise/frontend-revise/local-runtime/rag-expanded-final-tests-20261109'),
          'reason':'native Remove-Item denied by Windows ACL; old first test directory removed successfully'}
    (OUT/'test-cleanup.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')
    print('Removed isolated synthetic pytest schemas: '+str(len(removed)))


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['before','after','cleanup-tests']);a=p.parse_args()
    if a.phase=='cleanup-tests':
        load_project_env();cleanup_tests();return
    load_project_env();r=engine();state=json.loads((WORKSPACE/'rag-search-index.json').read_text(encoding='utf8'))
    with r.store.database.connect(read_only=True) as db:
        papers=[dict(row) for row in db.execute('SELECT paper_id,content_hash FROM papers ORDER BY paper_id')]
        children=db.execute("SELECT count(*) FROM chunks WHERE node_type='child'").fetchone()[0]
        copies=db.execute('SELECT count(*) FROM chunk_embeddings').fetchone()[0]
    storage={'papers':len(papers),'paper_hash_digest':hashlib.sha256(json.dumps(papers,sort_keys=True).encode()).hexdigest(),
        'children':children,'pg_embedding_copies':copies,'dense_points':r.store.client.count(state['dense_collection'],exact=True).count,
        'sparse_points':r.store.client.count(state['sparse_collection'],exact=True).count,
        'index_manifest_sha256':sha(WORKSPACE/'rag-search-index.json'),'corpus_version':state['corpus_version'],
        'environment_sha256':sha(ROOT/'self-revise/环境.md')}
    data={'storage':storage,'files_sha256':{name:sha(ROOT/name) for name in [
        'src/researchclaw/knowledge/research_rag.py','src/researchclaw/knowledge/rag_retrieval.py',
        'src/researchclaw/knowledge/retrieval.py','src/researchclaw/knowledge/qdrant_lexical.py',
        'self-revise/backend-revise/scripts/evaluate_rag_expanded.py','self-revise/backend-revise/scripts/report_rag_expanded.py']}}
    if a.phase=='after':
        before=json.loads((OUT/'boundary-before.json').read_text(encoding='utf8'))
        data['storage_unchanged']=before['storage']==storage
        data['count_invariant']=storage['papers']==204 and children==storage['dense_points']==storage['sparse_points']==10955 and copies==0
        cleanup=json.loads((ROOT/'self-revise/cleanup-history/整理清单.json').read_text(encoding='utf8'))
        data['environment_unchanged_since_cleanup']=storage['environment_sha256']==cleanup['protected_before_sha256']['环境.md']
        assert data['storage_unchanged'] and data['count_invariant'] and data['environment_unchanged_since_cleanup']
        protected={name:value for name,value in cleanup['protected_before_sha256'].items()
                   if any(token in name for token in ['金标','gold','silver','REVIEW.md','H08-2-followup'])}
        data['protected_annotation_files_verified']=len(protected)
        relocation_path=ROOT/'self-revise/cleanup-history/文档整合清单20261109.json'
        relocations=json.loads(relocation_path.read_text(encoding='utf-8-sig')).get('moves', {}) if relocation_path.exists() else {}
        data['annotation_changes']=[name for name,value in protected.items()
                                   if sha(ROOT/'self-revise'/relocations.get(name, name))!=value]
        assert not data['annotation_changes']
        ragas=[json.loads(p.read_text(encoding='utf8')) for p in (OUT/'ragas').glob('*.json')]
        data['ragas_scored']=sum(r['status']=='scored' for r in ragas)
        assert data['ragas_scored']==len(ragas)==96
    (OUT/('boundary-'+a.phase+'.json')).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')
    if a.phase=='after':
        manifest={str(p.relative_to(OUT)).replace('\\','/'):{'sha256':sha(p),'bytes':p.stat().st_size}
                  for p in OUT.rglob('*') if p.is_file() and p.name!='evidence-manifest.json'}
        (OUT/'evidence-manifest.json').write_text(json.dumps({'files':manifest,'count':len(manifest),
            'scope':'expanded evaluation artifacts; original source PDFs and published indexes remain external and unchanged'},ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(storage,ensure_ascii=False))


if __name__=='__main__':main()

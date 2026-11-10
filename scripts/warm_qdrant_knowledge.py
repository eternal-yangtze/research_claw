"""Build a real neural index from existing Qdrant chunks, without reingesting files."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src')]
os.environ['RESEARCHCLAW_KNOWLEDGE_BACKEND']='qdrant'
os.environ['RESEARCHCLAW_VECTOR_BACKEND']='qdrant'
os.environ['RESEARCHCLAW_RETRIEVAL_BACKEND']='neural'
from researchclaw.application.workflow import ResearchClawApp


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('workspace')
    args=parser.parse_args()
    app=ResearchClawApp(args.workspace)
    before=app.store.counts()
    rows=[dict(row) for row in app.store.all_chunks()]
    vectors={}
    started=time.monotonic()
    for offset in range(0,len(rows),64):
        vectors.update(app.retriever._vectors(rows[offset:offset+64]))
        print(f'Encoded {min(offset+64,len(rows))}/{len(rows)}',flush=True)
    status=app.retriever.vector_backend.sync(rows,vectors)
    after=app.store.counts()
    assert before['papers']==after['papers'] and before['chunks']==after['chunks']
    result={'workspace':str(app.workspace),'status':status.__dict__,'seconds':round(time.monotonic()-started,2),
            'papers':after['papers'],'chunks':after['chunks'],'embedding_cache':after['chunk_embeddings'],
            'reingested_files':0}
    report=app.workspace/'knowledge-index-verification.json'
    report.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()

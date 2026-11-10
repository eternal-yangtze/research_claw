"""Build/reuse the current library's Qdrant BM25 index without dense rebuild."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from researchclaw.core.qdrant_store import QdrantResearchStore
from researchclaw.knowledge.qdrant_lexical import build_lexical_index

def main():
    store=QdrantResearchStore(ROOT/'knowledge_base/rag_corpora/time_series_domain_release_v3_1',recover=False)
    with store.database.connect(read_only=True) as db:
        state=dict(db.execute("SELECT corpus_version,embedding_key FROM vector_sync_state WHERE status='ready' ORDER BY updated_at DESC LIMIT 1").fetchone())
    result=build_lexical_index(store,**state)
    out=ROOT/'self-revise/backend-revise/rag-evidence/practical-retrieval-v1'
    name='lexical-index-noop.json' if result['status']=='reused' else 'lexical-index-build.json'
    (out/name).write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result))

if __name__=='__main__':main()

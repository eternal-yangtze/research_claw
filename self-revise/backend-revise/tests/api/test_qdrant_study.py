import json

from qdrant_client import QdrantClient

from researchclaw.application import study
from researchclaw.core.qdrant_store import build_research_store
from researchclaw.evaluation.precoding.gates import CRITERIA
from researchclaw.harness.runtime import ModelTurn


def test_study_creates_qdrant_corpus_snapshot_without_sqlite_knowledge(tmp_path,monkeypatch):
    client=QdrantClient(':memory:')
    monkeypatch.setattr('qdrant_client.QdrantClient',lambda *a,**k:client)
    monkeypatch.setenv('RESEARCHCLAW_KNOWLEDGE_BACKEND','qdrant')
    source=tmp_path/'source'
    build_research_store(source)
    class Retriever:
        def __init__(self,*args):
            pass
        def search(self,*args,**kwargs):
            return []
    class Model:
        model='boundary-test'
        calls=0
        def complete_bounded(self,messages,tools,max_tokens,timeout):
            self.calls+=1
            if 'You plan literature evidence retrieval' in messages[0]['content']:
                value={'retrieval_query':'time series detection','preserved_constraints':['time series']}
            elif self.calls==2:
                value={'status':'needs_adapter','task_family':'other','reason':'No time-series adapter'}
            else:
                value={'criteria':{key:'pass' for key in CRITERIA}}
            return ModelTurn(text=json.dumps(value),usage={'total_tokens':100})
    monkeypatch.setattr(study,'HybridRetriever',Retriever)
    output=tmp_path/'study'
    result=study.plan_study({'description':'Time series detection'},source=source,output=output,
                            model=Model(),retrieval_strategy='legacy',orchestration='native')
    assert result['status']=='blocked_before_coding'
    assert (output/'study_corpus/knowledge-storage.json').is_file()
    assert not (output/'study_corpus.db').exists()

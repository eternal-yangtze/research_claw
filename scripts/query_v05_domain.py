"""Query the new unified index on CUDA using the exact evaluated recipe."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--question',required=True);p.add_argument('--output-id',required=True)
    p.add_argument('--adapter-arm',choices=['base','hard-lora','distilled-lora'],default='base')
    p.add_argument('--paper-id',help='Optional user-specified paper scope; never infer from evaluation gold')
    p.add_argument('--training-run',default='fullcorpus-gpt-r2-s42');a=p.parse_args()
    if not all(re.fullmatch(r'[A-Za-z0-9_-]{1,80}',s) for s in [a.run,a.output_id]):raise ValueError('Invalid run/output ID')
    base=ROOT/'self-revisev0.5/workspace'/a.run;output=base/'queries'/(a.output_id+'.json')
    if output.exists():raise RuntimeError('Immutable query output ID')
    manifest=json.loads((base/'index/manifest.json').read_text());raw=(base/'corpus.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==manifest['corpus_sha256']
    for name,sha in manifest['files'].items():assert hashlib.sha256((base/'index'/name).read_bytes()).hexdigest()==sha
    docs=json.loads(raw);del raw
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    from researchclaw.knowledge.cuda_bm25 import CUDABM25
    from researchclaw.knowledge.qwen_reranker import QwenReranker
    from researchclaw.knowledge.gpu_client import QWEN_REVISION
    if not torch.cuda.is_available():raise RuntimeError('CUDA mandatory')
    embedding=SentenceTransformer(manifest['embedding_model'],revision=manifest['embedding_revision'],device='cuda',local_files_only=True)
    vector=embedding.encode(['Represent this sentence for searching relevant passages: '+a.question],normalize_embeddings=True,convert_to_tensor=True)[0]
    scores=(torch.tensor(np.load(base/'index/vectors.npy'),device='cuda')@vector).cpu().numpy()
    lexical=CUDABM25.load(base/'index').get_scores(re.findall(r'\w+',a.question.lower()))
    allowed=np.array([i for i,d in enumerate(docs) if not a.paper_id or d['paper_id']==a.paper_id],dtype=np.int64)
    if not len(allowed):raise ValueError('Requested paper ID is absent from the pinned corpus')
    fused={}
    for order in [allowed[np.argsort(-scores[allowed],kind='stable')[:80]],allowed[np.argsort(-lexical[allowed],kind='stable')[:80]]]:
        for rank,i in enumerate(order,1):fused[int(i)]=fused.get(int(i),0)+1/(60+rank)
    pool=sorted(fused,key=lambda i:(-fused[i],docs[i]['id']))[:40]
    reranker=QwenReranker(revision=QWEN_REVISION,batch_size=8,max_length=2048)
    adapter_sha256=None
    if a.adapter_arm!='base':
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',a.training_run):raise ValueError('Invalid training run')
        from peft import PeftModel
        adapter=ROOT/'self-revisev0.5/artifacts/p3'/a.training_run/a.adapter_arm
        adapter_sha256=hashlib.sha256((adapter/'adapter_model.safetensors').read_bytes()).hexdigest()
        reranker.model=PeftModel.from_pretrained(reranker.model,adapter).eval()
    ranks=reranker.score(a.question,[docs[i]['text'] for i in pool])
    order=sorted(range(len(pool)),key=lambda i:(-ranks[i],i))
    hits=[dict(docs[pool[i]],qwen_score=float(ranks[i]),rrf_score=fused[pool[i]]) for i in order[:10]]
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(dict(question=a.question,corpus_sha256=manifest['corpus_sha256'],paper_scope=a.paper_id,scope_source='explicit user input' if a.paper_id else 'global corpus',recipe='BGE80 + BM25_80 -> RRF60 top40 -> Qwen; isolated candidate, not a production activation',qwen_revision=QWEN_REVISION,adapter_arm=a.adapter_arm,adapter_sha256=adapter_sha256,hits=hits),ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(dict(status='completed',output=str(output),hits=len(hits))),flush=True)

if __name__=='__main__':main()

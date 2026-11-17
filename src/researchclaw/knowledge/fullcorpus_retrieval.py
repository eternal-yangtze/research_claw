"""Isolated, pinned CUDA index retrieval; no access to the production 204-paper release."""
import hashlib
import json
from pathlib import Path
import re

def rank_fusion_score(first, second):
    """Rank-only RRF: avoid dependence on changed adapter logit scale."""
    if len(first)!=len(second):raise ValueError('Rank arrays differ')
    result=[0.]*len(first)
    for values in (first,second):
        for rank,i in enumerate(sorted(range(len(values)),key=lambda i:(-values[i],i)),1):result[i]+=1/(60+rank)
    return result

class FullCorpusRetriever:
    def __init__(self,index,*,adapter=None,expected_adapter_identity=None):
        import numpy as np
        import torch
        from sentence_transformers import SentenceTransformer
        from .gpu_client import BGE_REVISION,QWEN_REVISION
        from .qwen_reranker import QwenReranker
        if not torch.cuda.is_available():raise RuntimeError('CUDA mandatory, no CPU neural fallback')
        self.index=Path(index);self.summary=json.loads((self.index/'summary.json').read_text())
        if self.summary['bge_revision']!=BGE_REVISION:raise ValueError('Embedding/index model mismatch')
        # Verify every persisted index component before using row IDs and vectors together.
        for filename,digest in self.summary['index_sha256'].items():
            if hashlib.sha256((self.index/filename).read_bytes()).hexdigest()!=digest:raise ValueError('Corrupt index: '+filename)
        self.docs=json.loads((self.index/'corpus.json').read_text());self.terms=lambda x:re.findall(r'\w+',x.lower())
        from .cuda_bm25 import CUDABM25
        lexical=json.loads((self.index/'cuda-bm25-summary.json').read_text())
        if lexical['status']!='passed' or lexical['corpus_sha256']!=self.summary['index_sha256']['corpus.json']:raise ValueError('CUDA lexical index/corpus mismatch')
        for filename,digest in lexical['files'].items():
            if hashlib.sha256((self.index/filename).read_bytes()).hexdigest()!=digest:raise ValueError('CUDA lexical cache hash mismatch')
        self.bm=CUDABM25.load(self.index)
        self.vectors=torch.tensor(np.load(self.index/'vectors.npy'),device='cuda')
        self.owners=torch.tensor(np.load(self.index/'owners.npy').astype('int64'),device='cuda')
        if len(self.vectors)!=len(self.owners) or int(self.owners.max())>=len(self.docs):raise ValueError('Broken index ownership')
        self.embed=SentenceTransformer('BAAI/bge-small-en-v1.5',revision=BGE_REVISION,device='cuda',local_files_only=True)
        self.reranker=QwenReranker(revision=QWEN_REVISION,batch_size=8,max_length=2048)
        self.adapter_identity=None
        if adapter:
            from peft import PeftModel
            from .trained_reranker import adapter_identity
            path=Path(adapter)
            identity=adapter_identity(path,root=self.index.parents[3])
            if expected_adapter_identity and identity['identity_sha256']!=expected_adapter_identity:raise ValueError('Selected adapter config/weight identity mismatch')
            self.adapter_identity=hashlib.sha256((path/'adapter_model.safetensors').read_bytes()).hexdigest()
            self.reranker.model=PeftModel.from_pretrained(self.reranker.model,path).eval()

    def search(self,question,*,limit=5,fusion='rank'):
        import numpy as np
        import torch
        if not question.strip() or not 1<=limit<=20:raise ValueError('Nonempty query and limit1..20 required')
        if fusion not in ('rank','none'):raise ValueError('Fixed rank fusion or pure reranking only')
        q=self.embed.encode(['Represent this sentence for searching relevant passages: '+question],normalize_embeddings=True,convert_to_tensor=True)
        dense=torch.full((len(self.docs),),-float('inf'),device='cuda')
        dense.scatter_reduce_(0,self.owners,self.vectors@q[0],reduce='amax',include_self=True)
        ds=dense.cpu().numpy();bs=self.bm.get_scores(self.terms(question));fused={}
        for order in [np.argsort(-bs,kind='stable')[:80],np.argsort(-ds,kind='stable')[:80]]:
            for rank,i in enumerate(order,1):fused[int(i)]=fused.get(int(i),0)+1/(60+rank)
        pool=sorted(fused,key=lambda i:(-fused[i],self.docs[i]['id']))[:40]
        scores=self.reranker.score(question,[self.docs[i]['text'] for i in pool])
        final=rank_fusion_score(scores,[fused[i] for i in pool]) if fusion=='rank' else scores
        order=sorted(range(len(pool)),key=lambda i:(-final[i],i))[:limit]
        return dict(question=question,scope='isolated full corpus; no paper prefilter',adapter_sha256=self.adapter_identity,
                    fusion=fusion,candidate_ids=[self.docs[i]['id'] for i in pool],
                    evidence=[dict(self.docs[pool[i]],rerank_margin=scores[i],final_rank_score=final[i]) for i in order])

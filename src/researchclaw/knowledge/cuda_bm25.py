"""GPU sparse BM25 with rank_bm25 Okapi weights; tokenization/statistics stay CPU."""
from collections import Counter
import json
from pathlib import Path

class CUDABM25:
    def __init__(self,matrix,vocabulary):
        import torch
        if not torch.cuda.is_available():raise RuntimeError('CUDA BM25 requires CUDA, no silent device substitution')
        self.vocabulary=vocabulary;self.documents=matrix.shape[0]
        self.matrix=torch.sparse_csr_tensor(torch.tensor(matrix.indptr,dtype=torch.int64,device='cuda'),torch.tensor(matrix.indices,dtype=torch.int64,device='cuda'),torch.tensor(matrix.data,dtype=torch.float64,device='cuda'),size=matrix.shape,device='cuda')

    @classmethod
    def build(cls,tokenized):
        import numpy as np
        from rank_bm25 import BM25Okapi
        from scipy.sparse import csr_matrix
        reference=BM25Okapi(tokenized,k1=1.5,b=.75)
        if not reference.avgdl:raise ValueError('Nonempty token corpus required')
        vocabulary={term:i for i,term in enumerate(reference.idf)};indptr=[0];indices=[];values=[]
        for length,frequencies in zip(reference.doc_len,reference.doc_freqs):
            factor=reference.k1*(1-reference.b+reference.b*length/reference.avgdl)
            for term,frequency in sorted(frequencies.items(),key=lambda item:vocabulary[item[0]]):
                indices.append(vocabulary[term]);values.append(reference.idf[term]*frequency*(reference.k1+1)/(frequency+factor))
            indptr.append(len(indices))
        matrix=csr_matrix((np.asarray(values,dtype=np.float64),np.asarray(indices,dtype=np.int64),np.asarray(indptr,dtype=np.int64)),shape=(len(tokenized),len(vocabulary)))
        return cls(matrix,vocabulary),matrix,reference

    def get_scores(self,query):
        import torch
        counts=Counter(term for term in query if term in self.vocabulary)
        vector=torch.zeros((len(self.vocabulary),1),dtype=torch.float64,device='cuda')
        if counts:
            ids=torch.tensor([self.vocabulary[t] for t in counts],device='cuda')
            vector[ids,0]=torch.tensor(list(counts.values()),dtype=torch.float64,device='cuda')
        return torch.sparse.mm(self.matrix,vector).flatten().cpu().numpy()

    @classmethod
    def load(cls,directory):
        from scipy.sparse import load_npz
        directory=Path(directory)
        return cls(load_npz(directory/'cuda-bm25.npz'),json.loads((directory/'cuda-bm25-vocabulary.json').read_text(encoding='utf8')))

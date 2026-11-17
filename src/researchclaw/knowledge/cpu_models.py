"""Explicit CPU neural models; GPU backends never silently fall back here."""
import os
import threading


class CPUEmbedding:
    def __init__(self, model_name, revision):
        from .gpu_client import BGE_REVISION
        self.model_name=model_name or os.getenv('RESEARCHCLAW_EMBEDDING_MODEL','BAAI/bge-small-en-v1.5')
        self.model_revision=revision or os.getenv('RESEARCHCLAW_EMBEDDING_REVISION',BGE_REVISION)
        self.batch_size=16
        self.device='cpu'
        self._model=None
        self._lock=threading.Lock()

    def _load(self):
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer
            torch.set_num_threads(max(1,int(os.getenv('RESEARCHCLAW_CPU_THREADS','4'))))
            self._model=SentenceTransformer(self.model_name,revision=self.model_revision,device='cpu',
                local_files_only=os.getenv('RESEARCHCLAW_CPU_LOCAL_ONLY','true').lower()=='true')
        return self._model

    def encode(self,texts):
        if not texts:return []
        with self._lock:
            model=self._load()
            if any(len(model.tokenizer.encode(t,add_special_tokens=True,truncation=False))>model.max_seq_length for t in texts):
                raise ValueError('CPU embedding input exceeds model limit; split without silent truncation')
            return [tuple(float(x) for x in v) for v in model.encode(list(texts),batch_size=self.batch_size,
                normalize_embeddings=True,show_progress_bar=False,convert_to_numpy=True)]


class CPUReranker:
    def __init__(self,model_name,revision):
        self.model_name=model_name
        self.model_revision=revision
        self.device='cpu'
        self._model=None
        self._lock=threading.Lock()

    def _load(self):
        if self._model is None:
            import torch
            from sentence_transformers import CrossEncoder
            torch.set_num_threads(max(1,int(os.getenv('RESEARCHCLAW_CPU_THREADS','4'))))
            self._model=CrossEncoder(self.model_name,revision=self.model_revision,device='cpu',
                local_files_only=os.getenv('RESEARCHCLAW_CPU_LOCAL_ONLY','true').lower()=='true')
        return self._model

    def score(self,query,documents):
        if not documents:return []
        with self._lock:
            model=self._load()
            tokenizer=model.tokenizer
            query_ids=tokenizer.encode(query,add_special_tokens=False)
            budget=512-len(query_ids)-4
            if budget<32:raise ValueError('Query exhausts CPU reranker token budget')
            # Score every lossless source window and aggregate by parent, rather
            # than discarding evidence after the first 512 tokens.
            from .token_safety import source_windows
            windows=[];owners=[]
            for i,text in enumerate(documents):
                for span in source_windows(text,tokenizer,maximum=budget):
                    windows.append((query,span['text']));owners.append(i)
            if not windows:return [0.0]*len(documents)
            scores=model.predict(windows,batch_size=16,show_progress_bar=False)
            result=[float('-inf')]*len(documents)
            for i,s in zip(owners,scores):result[i]=max(result[i],float(s))
            return [0.0 if s==float('-inf') else s for s in result]

    def score_windowed(self,query,documents):
        return self.score(query,documents)


class ReviewedCPUReranker(CPUReranker):
    """Explicit CPU neural ranking plus journaled API answerability review.

    Provider failures retain the CPU ranking. An uncertain dispatch stays in
    the journal and is never silently billed again.
    """
    def score(self,query,documents):
        baseline=super().score(query,documents)
        if not documents:return baseline
        from pathlib import Path
        from researchclaw.core.artifacts import canonical_hash,workspace_lock,write_json
        from researchclaw.harness.runtime import OpenAICompatibleModel
        from researchclaw.orchestration.study_calls import StudyCalls
        from .cpu_evidence_review import review,SYSTEM
        root=Path(os.getenv('RESEARCHCLAW_CPU_REVIEW_ROOT','.cpu-runtime/review'))
        key=canonical_hash([SYSTEM,query,list(documents)])
        # Serialize journal admission across both threads and worker processes.
        with self._lock:
            try:
                with workspace_lock(root/key[:24]):
                    calls=StudyCalls(root/key[:24],OpenAICompatibleModel(max_attempts=1,timeout=120,temperature=0,reasoning_effort='low'),200000,max_model_turns=1)
                    indices=review(query,list(documents),calls,'evidence')
                    order=indices+[i for i in sorted(range(len(documents)),key=lambda i:-baseline[i]) if i not in indices]
                    write_json(root/key[:24]/'result.json',dict(status='completed',ranking=order,request_sha256=key,api_review=True))
                    scores=[0.0]*len(documents)
                    for rank,i in enumerate(order):scores[i]=float(len(order)-rank)
                    return scores
            except Exception as error:
                write_json(root/(key[:24]+'-fallback.json'),dict(status='cpu_fallback',error=type(error).__name__,request_sha256=key))
                return baseline


def cpu_models(model_name,revision):
    from .gpu_client import MINILM_REVISION
    cls=ReviewedCPUReranker if os.getenv('RESEARCHCLAW_CPU_EVIDENCE_REVIEW','false').lower()=='true' else CPUReranker
    return CPUEmbedding(model_name,revision),cls('cross-encoder/ms-marco-MiniLM-L6-v2',MINILM_REVISION)

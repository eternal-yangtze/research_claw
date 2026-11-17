"""Single Web/CLI/evaluation retrieval model selection, explicit CUDA rollback."""
import os
from .gpu_client import GPUClient, RemoteEmbedding, RemoteReranker


def configuration_key():
    if os.getenv('RESEARCHCLAW_NEURAL_BACKEND')=='local_cpu':
        return ('local_cpu','minilm',(os.getenv('RESEARCHCLAW_CPU_THREADS','4'),os.getenv('RESEARCHCLAW_CPU_EVIDENCE_REVIEW','false'),os.getenv('RESEARCHCLAW_CPU_REVIEW_ROOT','.cpu-runtime/review')))
    return (os.getenv('RESEARCHCLAW_NEURAL_BACKEND','gpu_ssh'),
            os.getenv('RESEARCHCLAW_GPU_RERANKER','qwen'),os.getenv('RESEARCHCLAW_GPU_PORT','8766'))


def retrieval_models(model_name, revision, *, reranker_family=None):
    backend,family,_=configuration_key()
    if backend=='local_cpu':
        if reranker_family not in (None,'minilm'):
            raise ValueError('CPU endpoint uses explicit MiniLM; Qwen GPU remains on GPU endpoints')
        from .cpu_models import cpu_models
        return cpu_models(model_name,revision)
    family=reranker_family or family
    if family not in ('qwen','minilm'):
        raise ValueError('GPU reranker must be qwen or explicit minilm rollback')
    if backend=='gpu_ssh':
        client=GPUClient()
        return RemoteEmbedding(model_name,revision=revision,client=client),RemoteReranker(family=family,client=client)
    if backend=='local_cuda':
        import torch
        if not torch.cuda.is_available():raise RuntimeError('CUDA required; neural CPU fallback disabled')
        from .retrieval import SentenceTransformerEmbedding, CrossEncoderReranker
        from .qwen_reranker import QwenReranker
        from .gpu_client import QWEN_REVISION, MINILM_REVISION
        reranker=(QwenReranker(revision=QWEN_REVISION,batch_size=8) if family=='qwen' else
                  CrossEncoderReranker('cross-encoder/ms-marco-MiniLM-L6-v2',revision=MINILM_REVISION,device='cuda'))
        return SentenceTransformerEmbedding(model_name,revision=revision,device='cuda'),reranker
    raise ValueError('Neural backend must be gpu_ssh or local_cuda; CPU fallback disabled')

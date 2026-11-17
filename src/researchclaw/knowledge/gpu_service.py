"""Loopback-only, authenticated CUDA service with a bounded FIFO worker.

No conversations, documents, credentials or request bodies are persisted. The
trusted application backend holds the SSH identity; this is not a public API.
"""
import hmac
import json
import os
from pathlib import Path
import queue
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .gpu_client import BGE_REVISION, QWEN_REVISION, MINILM_REVISION


class RequestError(ValueError):
    pass


class CUDAModels:
    def __init__(self):
        import torch
        from sentence_transformers import SentenceTransformer
        from .qwen_reranker import QwenReranker
        if not torch.cuda.is_available():raise RuntimeError('CUDA required; CPU fallback disabled')
        torch.set_num_threads(4)
        self.torch=torch
        self.embedding=SentenceTransformer('BAAI/bge-small-en-v1.5',revision=BGE_REVISION,device='cuda',local_files_only=True)
        self.reranker=QwenReranker(revision=QWEN_REVISION,batch_size=8,max_length=2048)
        self.reranker.maximum_windows=256
        self.minilm=None
        self.adapter_reranker=None
        self.adapter_identity=None
        self.device=torch.cuda.get_device_name(0)
        self.calls=0
        self.embedding.encode(['GPU service warmup'],normalize_embeddings=True,show_progress_bar=False)
        self.reranker.score('warmup',['GPU inference readiness'])

    def execute(self, operation, body):
        self.torch.cuda.reset_peak_memory_stats()
        started=time.perf_counter()
        if operation=='encode':
            if body.get('model')!='BAAI/bge-small-en-v1.5' or body.get('revision')!=BGE_REVISION:
                raise RequestError('Embedding model/revision mismatch; rebuild or use a compatible release')
            texts=body.get('texts')
            validate_texts(texts,128)
            if any(len(self.embedding.tokenizer.encode(t,add_special_tokens=True,truncation=False))>512 for t in texts):
                raise RequestError('Embedding input exceeds 512 tokens; split explicitly, no silent truncation')
            vectors=self.embedding.encode(texts,batch_size=64,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
            value=dict(vectors=vectors.tolist())
        elif operation=='score':
            docs=body.get('documents');validate_texts(docs,128)
            query=body.get('query')
            if not isinstance(query,str) or not query.strip() or len(query)>4096:raise RequestError('Query must be 1..4096 characters')
            family=body.get('family','qwen')
            model='Qwen/Qwen3-Reranker-0.6B' if family=='qwen' else 'cross-encoder/ms-marco-MiniLM-L6-v2'
            revision=QWEN_REVISION if family=='qwen' else MINILM_REVISION
            if family not in ('qwen','minilm') or body.get('model')!=model or body.get('revision')!=revision:
                raise RequestError('Reranker model/revision mismatch')
            if family!='qwen' and body.get('adapter') is not None:
                raise RequestError('LoRA adapter requires the Qwen family')
            if family=='qwen':
                adapter=body.get('adapter')
                active=self.reranker
                if adapter is not None:
                    from .trained_reranker import adapter_identity
                    from .qwen_reranker import QwenReranker
                    root=Path(__file__).resolve().parents[3]/'self-revisev0.5'
                    if not isinstance(adapter,dict) or not isinstance(adapter.get('directory'),str):
                        raise RequestError('Explicit adapter identity required')
                    directory=(root/adapter['directory']).resolve()
                    try: identity=adapter_identity(directory,root=root)
                    except (ValueError,OSError): raise RequestError('Invalid adapter files') from None
                    if identity!=adapter: raise RequestError('Adapter content hash mismatch')
                    if self.adapter_identity!=identity:
                        from peft import PeftModel
                        # Single GPU worker serializes loading and scoring. Retain at most
                        # one candidate alongside base; never merge into base weights.
                        self.adapter_reranker=None
                        self.torch.cuda.empty_cache()
                        candidate=QwenReranker(revision=QWEN_REVISION,batch_size=4,max_length=2048)
                        candidate.maximum_windows=256
                        candidate.model=PeftModel.from_pretrained(candidate.model,directory).eval()
                        self.adapter_reranker=candidate;self.adapter_identity=identity
                    active=self.adapter_reranker
                scores=active.score(query,docs)
            else:
                from sentence_transformers import CrossEncoder
                from .token_safety import source_windows
                if self.minilm is None:
                    self.minilm=CrossEncoder(model,revision=revision,device='cuda',local_files_only=True)
                budget=512-len(self.minilm.tokenizer.encode(query,add_special_tokens=False))-4
                if budget<32:raise RequestError('Query exhausts MiniLM pair budget')
                windows=[];owners=[]
                for i,doc in enumerate(docs):
                    for span in source_windows(doc or ' ',self.minilm.tokenizer,maximum=budget):
                        windows.append((query,span['text']));owners.append(i)
                        if len(windows)>256:raise RequestError('Too many reranker windows')
                values=self.minilm.predict(windows,batch_size=32,show_progress_bar=False).tolist()
                scores=[float('-inf')]*len(docs)
                for owner,score in zip(owners,values):scores[owner]=max(scores[owner],float(score))
            value=dict(scores=scores)
        else:raise RequestError('Unknown operation')
        self.torch.cuda.synchronize();self.calls+=1
        value['diagnostics']=dict(device='cuda:0',gpu=self.device,seconds=time.perf_counter()-started,
                                  peak_allocated_bytes=self.torch.cuda.max_memory_allocated(),calls=self.calls,
                                  embedding_revision=BGE_REVISION,qwen_revision=QWEN_REVISION)
        if operation=='score' and body.get('family','qwen')=='qwen':
            value['diagnostics'].update(active.last_diagnostics)
            value['diagnostics']['adapter_sha256']=adapter['sha256'] if adapter is not None else None
        return value


def validate_texts(texts,limit):
    if not isinstance(texts,list) or not 1<=len(texts)<=limit or any(not isinstance(t,str) or len(t)>64000 for t in texts):
        raise RequestError('Expected bounded text list (1..128 items, <=64000 characters each)')
    if sum(len(t) for t in texts)>512000:raise RequestError('Text batch too large')


class InferenceQueue:
    def __init__(self, backend, *, capacity=8, wait_seconds=60):
        self.backend=backend;self.wait_seconds=wait_seconds
        self.queue=queue.Queue(maxsize=capacity)
        self.failed=False
        threading.Thread(target=self._work,daemon=True).start()

    def _work(self):
        while True:
            operation,body,deadline,event,box=self.queue.get()
            try:
                if self.failed:
                    box.update(status=503,result={'error':'GPU worker failed; controlled restart required'})
                elif time.monotonic()>deadline:
                    box.update(status=504,result={'error':'GPU queue deadline exceeded; request not executed'})
                else:
                    started=time.monotonic()
                    result=self.backend.execute(operation,body)
                    result['diagnostics']['queue_seconds']=max(0,started-(deadline-self.wait_seconds))
                    box.update(status=200,result=result)
            except (RequestError,ValueError):box.update(status=422,result={'error':'Invalid or over-budget model input'})
            except Exception:
                # Includes OOM: never retry with a different model/device/batch.
                self.failed=True
                box.update(status=503,result={'error':'CUDA inference failed; no CPU fallback or automatic replay'})
            finally:
                event.set();self.queue.task_done()

    def submit(self, operation, body):
        event=threading.Event();box={};deadline=time.monotonic()+self.wait_seconds
        try:self.queue.put_nowait((operation,body,deadline,event,box))
        except queue.Full:return 429,{'error':'GPU queue full; retry only as an explicit new request'}
        if not event.wait(self.wait_seconds+90):return 504,{'error':'GPU request deadline exceeded; result unknown, not replayed'}
        return box['status'],box['result']


def make_server(backend, token, *, port=8766, inference_queue=None):
    jobs=inference_queue or InferenceQueue(backend)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def reply(self,status,value):
            raw=json.dumps(value,allow_nan=False).encode('utf8')
            self.send_response(status);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(raw)));self.end_headers()
            try:self.wfile.write(raw)
            except (BrokenPipeError,ConnectionResetError):pass
        def do_POST(self):
            self.connection.settimeout(15)
            if not hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+token):
                self.reply(401,{'error':'Unauthorized'});return
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=4*1024*1024:raise ValueError()
                body=json.loads(self.rfile.read(size))
                if not isinstance(body,dict):raise ValueError()
            except Exception:
                self.reply(400,{'error':'Invalid JSON body or request limit'});return
            operation=self.path.strip('/')
            if operation=='health':
                self.reply(200,dict(status='failed' if jobs.failed else 'ready',device='cuda:0',gpu=backend.device,
                                    queue_depth=jobs.queue.qsize(),queue_capacity=jobs.queue.maxsize,
                                    embedding_revision=BGE_REVISION,qwen_revision=QWEN_REVISION,
                                    pid=os.getpid(),calls=backend.calls));return
            if operation not in ('score','encode'):
                self.reply(404,{'error':'Unknown operation'});return
            status,result=jobs.submit(operation,body);self.reply(status,result)
    class Server(ThreadingHTTPServer):
        daemon_threads=True
        request_queue_size=16
        def __init__(self,*args):
            self.slots=threading.BoundedSemaphore(16)
            super().__init__(*args)
        def process_request(self,request,address):
            if not self.slots.acquire(blocking=False):
                try:request.sendall(b'HTTP/1.0 503 Service Unavailable\r\nContent-Length: 0\r\n\r\n')
                finally:self.shutdown_request(request)
                return
            try:super().process_request(request,address)
            except Exception:self.slots.release();raise
        def process_request_thread(self,request,address):
            try:super().process_request_thread(request,address)
            finally:self.slots.release()
    server=Server(('127.0.0.1',port),Handler)
    server.inference_queue=jobs
    return server


def serve(port=8766):
    import fcntl
    root=Path(__file__).resolve().parents[3]/'self-revisev0.5'
    state=root/'workspace/gpu-online';state.mkdir(parents=True,exist_ok=True)
    os.chmod(state,0o700)
    lock=(state/'service.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state.joinpath('pid').write_text(str(os.getpid()))
    # Token never appears in process arguments, sync files, logs or artifacts.
    token=secrets.token_urlsafe(48)
    token_path=state/'token'
    fd=os.open(token_path,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
    with os.fdopen(fd,'w') as handle:handle.write(token)
    os.chmod(token_path,0o600)
    os.environ.update(HF_HOME=str(root/'model-cache'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                      XDG_CACHE_HOME=str(root/'cache'))
    backend=CUDAModels()
    server=make_server(backend,token,port=port)
    state.joinpath('pid').write_text(str(os.getpid()))
    print(json.dumps(dict(status='ready',port=port,gpu=backend.device,pid=os.getpid())),flush=True)
    server.serve_forever()

"""Private GPU inference over pinned SSH direct-tcpip, without a public port.

Inference requests are never replayed after a transport timeout. A later new
request may reconnect. Only the application's trusted backend reads SSH secrets.
"""
import http.client
import json
import math
import os
from pathlib import Path
import threading

ROOT = Path(__file__).resolve().parents[3]
QWEN_REVISION = 'e61197ed45024b0ed8a2d74b80b4d909f1255473'
BGE_REVISION = '5c38ec7c405ec4b44b94cc5a9bb96e735b38267a'
MINILM_REVISION = '233902d25c440f23af6f7d6e94d2946bac0bee0a'


class GPUServiceError(RuntimeError):
    pass


class ChannelSocket:
    """Match socket.makefile lifetime semantics for an SSH channel.

    HTTP/1.0 makes HTTPConnection close its socket as soon as headers arrive.
    Real socket file objects retain the fd; a raw Paramiko channel does not.
    Defer that close until the response file closes, otherwise only the first
    SSH packet can survive for larger responses.
    """
    def __init__(self,channel):
        self.channel=channel;self.references=0;self.closing=False
    def __getattr__(self,name):return getattr(self.channel,name)
    def close(self):
        self.closing=True
        if not self.references:self.channel.close()
    def makefile(self,*args):
        wrapped=self.channel.makefile(*args);self.references+=1;owner=self
        class Reader:
            closed=False
            def __getattr__(self,name):return getattr(wrapped,name)
            def close(self):
                if self.closed:return
                self.closed=True
                try:wrapped.close()
                finally:
                    owner.references-=1
                    if owner.closing and not owner.references:owner.channel.close()
        return Reader()


def read_response(response, maximum=64*1024*1024):
    """Paramiko file reads may return short chunks before HTTP body EOF."""
    chunks=[];size=0
    while True:
        chunk=response.read(min(65536,maximum-size+1))
        if not chunk:break
        chunks.append(chunk);size+=len(chunk)
        if size>maximum:raise GPUServiceError('GPU response exceeded limit')
    return b''.join(chunks)


def connect_gpu(config_path=None):
    import paramiko
    from dotenv import dotenv_values
    config = dotenv_values(config_path or ROOT / '.env.gpu.local', interpolate=False)
    setting = lambda key, default='': os.environ.get(key, config.get(key) or default)
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    pinned = ROOT / '.env.gpu.known_hosts'
    if pinned.exists(): client.load_host_keys(str(pinned))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    try:
        client.connect(hostname=setting('GPU_SSH_HOST'), port=int(setting('GPU_SSH_PORT', '22')),
                       username=setting('GPU_SSH_USER'), password=setting('GPU_SSH_PASSWORD') or None,
                       key_filename=setting('GPU_SSH_KEY_FILE') or None, allow_agent=False, look_for_keys=False,
                       timeout=15, banner_timeout=20, auth_timeout=20)
        client.get_transport().set_keepalive(30)
        return client
    except Exception as exc:
        client.close()
        raise GPUServiceError('GPU SSH connection failed; check pinned host identity and local configuration ('+type(exc).__name__+')') from None


class GPUClient:
    def __init__(self, *, timeout=None, config_path=None, port=None):
        self.timeout = float(timeout or os.getenv('RESEARCHCLAW_GPU_TIMEOUT', '120'))
        self.port = int(port or os.getenv('RESEARCHCLAW_GPU_PORT', '8766'))
        self.config_path = config_path
        self._ssh = None
        self._lock = threading.Lock()
        self.last_diagnostics = {}

    def close(self):
        with self._lock:
            if self._ssh: self._ssh.close()
            self._ssh = None

    def request(self, operation, payload=None):
        connection = channel = None
        try:
            with self._lock:
                if self._ssh is None or not self._ssh.get_transport().is_active():
                    if self._ssh: self._ssh.close()
                    self._ssh = connect_gpu(self.config_path)
                with self._ssh.open_sftp() as sftp:
                    root = sftp.normalize('.') + '/ResearchClaw/self-revisev0.5/workspace/gpu-online'
                    with sftp.open(root+'/token', 'r') as handle:
                        token = handle.read(256).decode('ascii').strip()
                channel = self._ssh.get_transport().open_channel('direct-tcpip', ('127.0.0.1', self.port), ('127.0.0.1', 0), timeout=15)
            channel.settimeout(self.timeout)
            connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=self.timeout)
            connection.sock = ChannelSocket(channel)
            body = json.dumps(payload or {}, ensure_ascii=False).encode('utf8')
            connection.request('POST', '/'+operation, body=body,
                               headers={'Authorization':'Bearer '+token,'Content-Type':'application/json','Connection':'close'})
            response = connection.getresponse()
            raw = read_response(response)
            try:
                result = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise GPUServiceError('Invalid GPU response: status='+str(response.status)+', bytes='+str(len(raw))+
                                      ', declared='+str(response.getheader('Content-Length'))+', position='+str(exc.pos)) from None
            if response.status != 200:
                raise GPUServiceError('GPU service '+str(response.status)+': '+str(result.get('error','request failed')))
            self.last_diagnostics = result.get('diagnostics', {})
            return result
        except GPUServiceError:
            raise
        except Exception as exc:
            raise GPUServiceError('GPU service unavailable or request timed out; request was not replayed; CPU fallback disabled ('+type(exc).__name__+')') from None
        finally:
            if connection: connection.close()
            if channel: channel.close()


class RemoteEmbedding:
    def __init__(self, model_name, *, revision, client=None):
        self.model_name, self.model_revision = model_name, revision
        self.client = client or GPUClient()
        self.device = 'cuda:remote'
        self.batch_size = 32

    def encode(self, texts):
        if not texts:return []
        vectors=[]
        for start in range(0,len(texts),128):
            batch=list(texts[start:start+128])
            result=self.client.request('encode',dict(texts=batch,model=self.model_name,revision=self.model_revision))
            values=result['vectors']
            if len(values)!=len(batch) or any(len(v)!=384 or not all(math.isfinite(x) for x in v) for v in values):
                raise GPUServiceError('GPU embedding response shape/value mismatch')
            vectors.extend(tuple(v) for v in values)
        return vectors


class RemoteReranker:
    def __init__(self, *, family='qwen', client=None):
        self.family=family
        self.model_name='Qwen/Qwen3-Reranker-0.6B' if family=='qwen' else 'cross-encoder/ms-marco-MiniLM-L6-v2'
        self.model_revision=QWEN_REVISION if family=='qwen' else MINILM_REVISION
        self.device='cuda:remote'
        self.client=client or GPUClient()

    def score(self, query, documents):
        if not documents:return []
        result=self.client.request('score',dict(query=query,documents=list(documents),family=self.family,
                                                model=self.model_name,revision=self.model_revision))
        values=result['scores']
        if len(values)!=len(documents) or not all(math.isfinite(x) for x in values):
            raise GPUServiceError('GPU reranking response shape/value mismatch')
        self.last_diagnostics=result.get('diagnostics',{})
        return values

    score_windowed=score

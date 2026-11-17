"""Explicit hashed LoRA identities; never substitute a candidate for base implicitly."""
import hashlib
import json
from pathlib import Path
from .gpu_client import RemoteReranker, GPUServiceError


def adapter_identity(directory, *, root):
    path = Path(directory).resolve(); root = Path(root).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Adapter must stay within isolated v0.5 root')
    files = {}
    for name in ('adapter_config.json', 'adapter_model.safetensors'):
        item = path/name
        if not item.is_file() or item.is_symlink():
            raise ValueError('Ordinary adapter files required')
        files[name] = hashlib.sha256(item.read_bytes()).hexdigest()
    config = json.loads((path/'adapter_config.json').read_text(encoding='utf8'))
    if config.get('base_model_name_or_path') != 'Qwen/Qwen3-Reranker-0.6B' or config.get('peft_type') != 'LORA':
        raise ValueError('Only Qwen3-Reranker-0.6B LoRA supported')
    digest = hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
    return dict(directory=path.relative_to(root).as_posix(),files=files,
                sha256=files['adapter_model.safetensors'],identity_sha256=digest)


class AdapterRemoteReranker(RemoteReranker):
    def __init__(self, *, adapter, client=None):
        super().__init__(client=client)
        if set(adapter.get('files',{})) != {'adapter_config.json','adapter_model.safetensors'}:
            raise ValueError('Adapter config and weights hashes required')
        self.adapter = adapter
        self.adapter_sha256 = adapter['sha256']

    def score(self, query, documents):
        if not documents: return []
        result=self.client.request('score',dict(query=query,documents=list(documents),family=self.family,
            model=self.model_name,revision=self.model_revision,adapter=self.adapter))
        import math
        values=result['scores']; self.last_diagnostics=result.get('diagnostics',{})
        if self.last_diagnostics.get('adapter_sha256') != self.adapter_sha256:
            raise GPUServiceError('GPU did not attest the requested adapter')
        if len(values)!=len(documents) or not all(math.isfinite(x) for x in values):
            raise GPUServiceError('Invalid trained reranker scores')
        return values

    score_windowed=score

"""Installed model snapshots must not depend on a remote metadata probe."""
import sys
from types import SimpleNamespace

from researchclaw.knowledge.retrieval import SentenceTransformerEmbedding


def test_complete_embedding_snapshot_loaded_by_path(tmp_path, monkeypatch):
    for name in ['modules.json', 'config.json', 'model.safetensors', 'tokenizer_config.json']:
        (tmp_path/name).touch()
    monkeypatch.setenv('RESEARCHCLAW_EMBEDDING_LOCAL_ONLY', '0')
    monkeypatch.setattr('huggingface_hub.snapshot_download', lambda *a, **k: str(tmp_path))
    def constructor(path, **kwargs):
        assert path == str(tmp_path)
        assert kwargs['local_files_only'] is True
        return SimpleNamespace(encode=lambda *a, **k: [[1.0, 0.0]])
    monkeypatch.setitem(sys.modules, 'sentence_transformers', SimpleNamespace(SentenceTransformer=constructor))
    assert SentenceTransformerEmbedding().encode(['real input']) == [(1.0, 0.0)]

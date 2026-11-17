"""Immutable evaluation inputs and content-keyed request replay records."""
import hashlib
import json
from pathlib import Path


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def freeze_json(path, value):
    """Create once, or reject a changed durable input instead of reusing results."""
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    try:
        with path.open('x',encoding='utf8') as handle:
            json.dump(value,handle,ensure_ascii=False,indent=2)
    except FileExistsError:
        if fingerprint(json.loads(path.read_text(encoding='utf8')))!=fingerprint(value):
            raise ValueError('Frozen evaluation input changed: '+str(path))
    return fingerprint(value)


def record_request(directory, *, messages, model, provider_host, settings):
    """Store exact model-visible messages, excluding API headers and credentials."""
    request={'messages':messages,'model':model,'provider_host':provider_host,'settings':settings}
    identity=fingerprint(request)
    freeze_json(Path(directory)/(identity+'.json'),request)
    return identity

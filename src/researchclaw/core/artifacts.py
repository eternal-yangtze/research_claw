"""Shared artifact hashes, atomic JSON writes and nonblocking workspace locks."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any


_snapshot_publish_lock = threading.Lock()


def progress(message: str) -> None:
    """A detached console must never invalidate a saved scientific result."""
    try:
        print(message, flush=True)
    except (OSError, ValueError):
        # Windows can invalidate the pipe when a launcher disconnects. Durable
        # JSON artifacts, rather than console delivery, determine task success.
        pass


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path: str | Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Separate writers must never share a temporary pathname. Flush before
    # publication so checkpoints survive a process crash after replacement.
    descriptor, name = tempfile.mkstemp(prefix='.rc-', suffix='.tmp', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor,'w',encoding='utf-8') as stream:
            json.dump(value,stream,ensure_ascii=False,indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        # Windows readers can briefly deny delete-sharing. Keep the previous
        # complete snapshot visible and bound retries to less than one second.
        with _snapshot_publish_lock:
            for attempt in range(10):
                try:
                    os.replace(temporary,path)
                    break
                except PermissionError as error:
                    if os.name != 'nt' or error.winerror not in {5,32,33} or attempt == 9:
                        raise
                    time.sleep(.01*(attempt+1))
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def workspace_lock(directory: Path):
    """OS lock is released automatically when a cancelled process exits."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".preparation.lock").open("a+b") as stream:
        stream.seek(0, 2)
        if not stream.tell():
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)

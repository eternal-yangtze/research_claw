from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_project_env(*, override: bool = False) -> Path | None:
    """Load the project .env without logging or returning secret values.

    Adapted from TaleClaw's MIT-licensed runtime/env_loader.py.
    """

    candidates = (Path.cwd() / ".env", PROJECT_ROOT / ".env")
    for env_path in dict.fromkeys(path.resolve() for path in candidates):
        if not env_path.is_file():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, raw_value = line.partition("=")
            key = key.strip()
            if not key or (not override and key in os.environ):
                continue
            os.environ[key] = _clean_value(raw_value.strip())
        return env_path
    return None


def _clean_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value

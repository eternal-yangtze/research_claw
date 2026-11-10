from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def workspace_identity(path) -> str:
    """Keep persisted namespaces stable across the documented release archive move."""
    resolved = Path(path).resolve()
    archive = PROJECT_ROOT / 'self-revise/backend-revise/internal_experiments'
    if resolved.is_relative_to(archive):
        return str(PROJECT_ROOT / 'internal_experiments' / resolved.relative_to(archive))
    return str(resolved)


def load_project_env(*, override: bool = False) -> Path | None:
    """Load the project .env without logging or returning secret values."""

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


def resolve_source_path(value: str | None) -> str | None:
    """Map archived absolute paths at read time; retain stored provenance unchanged."""
    if not value or Path(value).is_file():
        return value
    normalized = value.replace('\\', '/')
    if normalized.startswith('/app/'):
        relative = Path(normalized[len('/app/'):])
        if '..' not in relative.parts:
            for root in (PROJECT_ROOT, PROJECT_ROOT / 'self-revise/backend-revise/internal_experiments/assets'):
                candidate = root / relative
                if candidate.is_file():
                    return str(candidate.resolve())
    for old_root in ('researchops_mvp', 'ResearchClaw'):
        marker = '/' + old_root + '/'
        position = normalized.casefold().find(marker.casefold())
        if position < 0:
            continue
        relative = Path(normalized[position + len(marker):])
        if '..' in relative.parts or relative.is_absolute():
            return value
        if relative.parts and relative.parts[0] == 'internal_experiments':
            archived = PROJECT_ROOT / 'self-revise/backend-revise' / relative
            if archived.is_file():
                return str(archived.resolve())
        for root in (PROJECT_ROOT, PROJECT_ROOT / 'self-revise/backend-revise/internal_experiments/assets',
                     PROJECT_ROOT / 'knowledge_base', PROJECT_ROOT / 'papers',
                     PROJECT_ROOT / 'self-revise/backend-revise/internal_experiments/assets/knowledge_base'):
            candidate = root / relative
            if candidate.is_file():
                return str(candidate.resolve())
    return value

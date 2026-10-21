"""Separate local paper/dataset assets while retaining workspace-relative APIs."""
from __future__ import annotations

import os
from pathlib import Path

from researchops.core.config import PROJECT_ROOT


def ensure_workspace_assets(workspace: Path, *, project_root: Path = PROJECT_ROOT) -> None:
    """Provision links only for this project's canonical knowledge_base workspaces.

    Existing physical directories are never moved at runtime. Historical directories
    are handled by the explicit maintenance migration. External/test workspaces keep
    their original self-contained layout.
    """
    root = project_root.resolve()
    workspace = workspace.resolve()
    try:
        relative = workspace.relative_to(root / "knowledge_base")
    except ValueError:
        return
    if not relative.parts:
        raise ValueError("Choose a named workspace inside knowledge_base")
    for link_relative, target_relative in (
        ("artifacts/papers", Path("papers") / relative / "artifacts/papers"),
        ("artifacts/sources", Path("papers") / relative / "artifacts/sources"),
        ("datasets", Path("datasets") / relative),
    ):
        link = workspace / link_relative
        if link.exists() or link.is_symlink():
            continue
        target = root / target_relative
        # Refuse a category symlink escaping the intended local project.
        if not target.resolve().is_relative_to(root):
            raise ValueError("Asset destination escapes project")
        target.mkdir(parents=True, exist_ok=True)
        link.parent.mkdir(parents=True, exist_ok=True)
        try:
            if os.name == "nt":
                import _winapi
                _winapi.CreateJunction(str(target), str(link))
            else:
                link.symlink_to(target, target_is_directory=True)
        except FileExistsError:
            # Another initializer may have created the same link concurrently.
            if link.resolve() != target.resolve():
                raise

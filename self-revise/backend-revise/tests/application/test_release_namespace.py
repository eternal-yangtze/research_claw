import hashlib
from pathlib import Path

from researchclaw.core.config import PROJECT_ROOT, workspace_identity
from researchclaw.core.postgres import PostgresDatabase
from researchclaw.knowledge.vector_backend import _project_id


def test_archived_web_workspace_keeps_persisted_namespace():
    relative = Path('assets/web_workspace/users/example/projects/default')
    moved = PROJECT_ROOT / 'self-revise/backend-revise/internal_experiments' / relative
    old = PROJECT_ROOT / 'internal_experiments' / relative
    assert workspace_identity(moved) == str(old)
    digest = hashlib.sha256(str(old).encode()).hexdigest()[:24]
    assert PostgresDatabase(moved).schema == 'rc_research_' + digest
    assert _project_id(moved) == 'project_' + hashlib.sha256(str(old).encode()).hexdigest()[:16]


def test_new_unrelated_workspace_identity_is_not_rewritten(tmp_path):
    assert workspace_identity(tmp_path) == str(tmp_path.resolve())

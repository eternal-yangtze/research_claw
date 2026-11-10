"""One-shot, scoped documentation cleanup; never touches library or source PDFs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
BASE = ROOT / "self-revise"
MANIFEST = BASE / "cleanup-history/整理清单.json"
OBSOLETE_DOCS = [
    "1. v20261101后续建议.md", "11-文件夹与核心文件说明.md",
    "12-历史测试性能存档.md", "DEVELOPMENT.md", "prompt1.md",
    "v20261107RAG重构.md", "v20261109RAG_step2.md",
]
OBSOLETE_DATA = [
    "rag-evidence/practical-retrieval-v1/silver-status.json",
    "rag-evidence/final-rag-v20261109/ragas-ledger/budget/ledger.json",
    "rag-evidence/final-rag-v20261109/ragas-ledger/budget/calls/786daab1408a4d43a45aac0d90becdff.request.json",
    "rag-evidence/final-rag-v20261109/ragas-ledger/model_observations/88319c81db75e915118378582155dbe747876f7970f3b99871de89078219c1cb.json",
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run() -> None:
    if MANIFEST.exists():
        raise SystemExit("Already consolidated; inspect manifest rather than overwriting evidence")
    targets = [BASE / name for name in OBSOLETE_DOCS + OBSOLETE_DATA]
    for path in targets:
        path.resolve().relative_to(BASE.resolve())
        if not path.is_file():
            raise RuntimeError(f"Expected exact file missing: {path}")
    protected = {}
    excluded = []
    for path in BASE.rglob("*"):
        if "pytest-temp2" in path.parts:
            continue
        if path.is_file() and path not in targets:
            try:
                protected[path.relative_to(BASE).as_posix()] = digest(path)
            except PermissionError:
                excluded.append(path.relative_to(BASE).as_posix())
    deleted = [{"path": p.relative_to(BASE).as_posix(), "bytes": p.stat().st_size,
                "sha256": digest(p), "reason": "merged_into_current_plan" if p.name in OBSOLETE_DOCS
                else "redundant_completion_status_or_initial_sandbox_denied_ledger"}
               for p in targets]
    # Mechanical link updates only; never rewrite immutable environment or evidence JSON.
    updated = []
    for path in BASE.rglob("*.md"):
        if path in targets or path.name == "环境.md" or "pytest-temp2" in path.parts:
            continue
        original = path.read_text(encoding="utf-8")
        revised = original
        for old in OBSOLETE_DOCS:
            revised = revised.replace(f"]({old})", "](项目进展与后续计划.md)")
            revised = revised.replace(f"](<{old}>)", "](项目进展与后续计划.md)")
        if revised != original:
            path.write_text(revised, encoding="utf-8")
            key = path.relative_to(BASE).as_posix()
            updated.append({"path": key, "before": protected[key], "after": digest(path)})
    for path in targets:
        path.unlink()
    changed = {item["path"]: item["after"] for item in updated}
    failures = []
    for relative, before in protected.items():
        path = BASE / relative
        if not path.is_file() or digest(path) != changed.get(relative, before):
            failures.append(relative)
    payload = {
        "date": "2026-11-09", "scope": "self-revise only; no source paper/database/backup changes",
        "deleted": deleted, "deleted_count": len(deleted),
        "deleted_bytes": sum(item["bytes"] for item in deleted),
        "recovery": "permanently unlinked; SHA256 inventory is not a content backup; superseded plans consolidated",
        "protected_before_sha256": protected, "mechanical_link_updates": updated,
        "protected_verification_failures": failures,
        "environment_unchanged": digest(BASE / "环境.md") == protected["环境.md"],
        "annotation_files_unchanged": all(
            digest(BASE / name) == value for name, value in protected.items()
            if any(token in name for token in ("金标", "gold", "silver", "REVIEW.md", "H08-2-followup"))
        ),
        "unreadable_files": excluded,
        "pending_temporary_directories": [{
            "path": "rag-evidence/final-rag-v20261109/pytest-temp2",
            "reason": "previous Windows ACL removal attempts denied; not retried or bypassed",
        }],
        "retention": "Final raw results, live RAGAS ledger including provider errors, comparison and ablation data, labels and images, TEI/schema/typed reproduction inputs, environment and migration evidence retained",
        "previous_cleanup_manifest": "rag-evidence/final-rag-v20261109/cleanup-manifest.json",
    }
    MANIFEST.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in (
        "deleted_count", "deleted_bytes", "environment_unchanged", "annotation_files_unchanged",
        "protected_verification_failures")}, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    run()

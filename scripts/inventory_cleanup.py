"""Read-only asset inventory, metric extraction, and post-migration verification."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import stat
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAINTENANCE_PATHS = (
    "src/researchops/application/cli.py", "src/researchops/application/workflow.py",
    "src/researchops/application/study.py", "src/researchops/experiments/datasets.py",
    "src/researchops/papers/acquisition.py", "src/researchops/harness/skills/loader.py",
    "src/researchops/coding/sandbox.py", "src/researchops/README.md", "config/benchmarks/README.md",
)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def inventory():
    if (ROOT / ".maintenance" / "protected_before.json").exists():
        raise SystemExit("Baseline exists; preserve it and explicitly archive it before creating a new baseline")
    manifest = []
    skipped_links = []
    roots = [ROOT / ".env", ROOT / ".env.example", ROOT / "src", ROOT / "config", ROOT / "skills"]
    roots += sorted(p for p in ROOT.glob("workspace*") if p.is_dir() and not p.is_junction())
    roots += [ROOT / name for name in ("papers", "datasets", "knowledge_base") if (ROOT / name).is_dir()]
    for base in roots:
        paths = [base] if base.is_file() else sorted(base.rglob("*"))
        for path in paths:
            if getattr(path.lstat(), "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                skipped_links.append(path.relative_to(ROOT).as_posix())
                continue
            if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            relative = path.relative_to(ROOT).as_posix()
            info = path.stat()
            # Native dataset manifests already contain checksums. For a same-volume
            # directory move, file ID + exact size/mtime verifies unchanged identity
            # without opening hundreds of thousands of dataset images again.
            metadata_only = "/datasets/" in relative or path.suffix in {".pt", ".pth", ".ckpt", ".parquet"}
            unreadable = None
            try:
                checksum = None if metadata_only else digest(path)
            except OSError as error:
                # Preserve pre-existing access restrictions; the directory move
                # still permits verification of file identity and metadata.
                checksum, unreadable = None, str(error)
            manifest.append({"path": relative, "size": info.st_size,
                             "mtime_ns": info.st_mtime_ns, "file_id": info.st_ino,
                             "sha256": checksum, "preexisting_read_error": unreadable})
        print(json.dumps({"inventoried": base.name, "files_so_far": len(manifest)}), flush=True)
    save(ROOT / ".maintenance" / "protected_before.json", manifest)
    save(ROOT / ".maintenance" / "historical_links.json", skipped_links)
    print(json.dumps({"protected_files": len(manifest), "bytes": sum(x["size"] for x in manifest)}))


def metrics():
    catalog_path = ROOT / "evaluation" / "results" / "historical_metric_catalog.json"
    if catalog_path.exists() and not (ROOT / "TODO2.md").exists():
        print("Historical catalog retained; original narrative documents were consolidated")
        return
    documents = [ROOT / "README.md", ROOT / "route2.md", ROOT / "TODO2.md", *sorted((ROOT / "docs").glob("*.md"))]
    catalog = []
    for path in documents:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
        tables, numeric_prose, heading = [], [], ""
        index = 0
        while index < len(lines):
            line = lines[index]
            if line.startswith("#"):
                heading = line.lstrip("# ")
            if line.startswith("|"):
                block = []
                while index < len(lines) and lines[index].startswith("|"):
                    block.append(lines[index])
                    index += 1
                if any(re.search(r"\d", x) for x in block):
                    tables.append({"section": heading, "markdown": "\n".join(block)})
                continue
            if re.search(r"\d", line) and not line.startswith(("```", ".\\", "python ", "#")):
                numeric_prose.append({"section": heading, "text": line})
            index += 1
        if tables or numeric_prose:
            catalog.append({"original_document": path.relative_to(ROOT).as_posix(),
                            "sha256": digest(path), "tables": tables, "numeric_context": numeric_prose})
    save(ROOT / "evaluation" / "results" / "historical_metric_catalog.json",
         {"measurement_date": "2026-10-20", "meaning": "Historical stage-specific measurements; do not pool different judges or test sets",
          "documents": catalog})
    for path in sorted((ROOT / "docs").glob("*.json")):
        if "METRICS" in path.name:
            save(ROOT / "evaluation" / "results" / path.name.lower(), json.loads(path.read_text(encoding="utf-8")))
    public = {}
    for name in ("workspace_precoding_v2", "workspace_precoding_v41_revalidation_live",
                 "workspace_rag_incremental_v1_live", "workspace_rag_incremental_v2_planned",
                 "workspace_rag_cpu_v31", "workspace_rag_fusion_v1", "workspace_skills_v1",
                 "workspace_skills_composition_v1", "workspace_skills_study_v2"):
        for file in ("report.json", "audit.json"):
            path = ROOT / name / file
            if path.exists():
                data = path.read_text(encoding="utf-8")
                # Remove workstation-specific prefixes in public copies only.
                data = data.replace(json.dumps(str(ROOT))[1:-1], "<PROJECT>").replace(ROOT.as_posix(), "<PROJECT>")
                public[f"{name}/{file}"] = json.loads(data)
    save(ROOT / "evaluation" / "results" / "measured_reports.json", public)
    print(json.dumps({"documents": len(catalog), "tables": sum(len(x["tables"]) for x in catalog),
                      "reports": len(public)}))


def record_edits():
    before = {r["path"]: r for r in json.loads((ROOT / ".maintenance/protected_before.json").read_text(encoding="utf-8"))}
    edits = []
    for name in MAINTENANCE_PATHS:
        if name not in before:
            continue
        path = ROOT / name
        after = digest(path) if path.is_file() else None
        if after != before[name]["sha256"]:
            edits.append({"path": name, "before_sha256": before[name]["sha256"], "after_sha256": after,
                          "reason": "Declared path/layout compatibility update or consolidated narrative document"})
    save(ROOT / ".maintenance/intentional_path_edits.json", edits)
    print(json.dumps({"declared_path_edits": len(edits)}))


def verify():
    rows = json.loads((ROOT / ".maintenance" / "protected_before.json").read_text(encoding="utf-8"))
    missing, changed, intentional = [], [], []
    edits_path = ROOT / ".maintenance/intentional_path_edits.json"
    edits = {r["path"]: r for r in json.loads(edits_path.read_text(encoding="utf-8"))} if edits_path.exists() else {}
    for row in rows:
        path = ROOT / row["path"]
        edit = edits.get(row["path"])
        if edit and row["path"] in MAINTENANCE_PATHS and edit["before_sha256"] == row["sha256"]:
            after = digest(path) if path.is_file() else None
            if after == edit["after_sha256"]:
                intentional.append(row["path"])
                continue
        if not path.is_file():
            missing.append(row["path"])
        elif path.stat().st_size != row["size"] or (
            digest(path) != row["sha256"] if row["sha256"] else
            (path.stat().st_mtime_ns != row["mtime_ns"] or path.stat().st_ino != row["file_id"])
        ):
            changed.append(row["path"])
    result = {"protected_files": len(rows), "missing": missing, "changed": changed,
              "declared_path_edits": intentional,
              "sha256_files": sum(bool(r["sha256"]) for r in rows),
              "file_identity_checks": sum(not r["sha256"] for r in rows),
              "all_protected_checks_passed": not missing and not changed}
    save(ROOT / ".maintenance" / "protected_after.json", result)
    print(json.dumps(result, ensure_ascii=False))
    if missing or changed:
        raise SystemExit(1)


def database_paths():
    checked, unavailable = 0, []
    for db in sorted((ROOT / "knowledge_base").rglob("*.db")):
        try:
            con = sqlite3.connect(db.as_uri() + "?mode=ro", uri=True)
            try:
                tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                if "papers" not in tables:
                    continue
                columns = {r[1] for r in con.execute("PRAGMA table_info(papers)")}
                for column in ("file_path", "source_archive_path"):
                    if column not in columns:
                        continue
                    for value, expected in con.execute(f"SELECT {column}, content_hash FROM papers WHERE {column} IS NOT NULL"):
                        if not value:
                            continue
                        checked += 1
                        path = Path(value)
                        if not path.is_absolute():
                            path = ROOT / path
                        if not path.is_file():
                            unavailable.append({"database": db.relative_to(ROOT).as_posix(), "path": value})
            finally:
                con.close()
        except sqlite3.DatabaseError as error:
            unavailable.append({"database": db.relative_to(ROOT).as_posix(), "error": str(error)})
    result = {"checked_references": checked, "unavailable": unavailable,
              "note": "Missing files may be historical pre-existing references; protected manifest is the byte preservation check"}
    save(ROOT / ".maintenance" / "database_path_check.json", result)
    print(json.dumps({"checked_references": checked, "unavailable_count": len(unavailable)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("inventory", "metrics", "verify", "database-paths", "record-edits"))
    args = parser.parse_args()
    {"inventory": inventory, "metrics": metrics, "verify": verify, "database-paths": database_paths, "record-edits": record_edits}[args.stage]()

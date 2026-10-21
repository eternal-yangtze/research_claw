from pathlib import Path

from researchops.core.layout import ensure_workspace_assets


def test_central_assets_are_reachable_and_initialization_is_repeatable(tmp_path):
    workspace = tmp_path / "knowledge_base" / "new_idea"
    ensure_workspace_assets(workspace, project_root=tmp_path)
    paper = workspace / "artifacts" / "papers" / "sample.pdf"
    paper.write_bytes(b"local-test-pdf")
    dataset = workspace / "datasets" / "sample.txt"
    dataset.write_text("sample", encoding="utf-8")
    ensure_workspace_assets(workspace, project_root=tmp_path)
    assert (tmp_path / "papers/new_idea/artifacts/papers/sample.pdf").read_bytes() == b"local-test-pdf"
    assert (tmp_path / "datasets/new_idea/sample.txt").read_text(encoding="utf-8") == "sample"


def test_external_workspace_and_existing_assets_are_preserved(tmp_path):
    external = tmp_path / "independent"
    ensure_workspace_assets(external, project_root=tmp_path)
    assert not external.exists()
    workspace = tmp_path / "knowledge_base/old_idea"
    paper = workspace / "artifacts/papers/existing.pdf"
    paper.parent.mkdir(parents=True)
    paper.write_bytes(b"preserve")
    ensure_workspace_assets(workspace, project_root=tmp_path)
    assert paper.read_bytes() == b"preserve"
    assert not (tmp_path / "papers/old_idea/artifacts/papers").exists()


def test_study_resume_accepts_canonical_path_after_directory_relocation(tmp_path, monkeypatch):
    import json
    import sqlite3
    import researchops.application.study as study

    old_source = tmp_path / "workspace_source"
    old_source.mkdir()
    output = tmp_path / "study_output"
    output.mkdir()
    sqlite3.connect(output / "study_corpus.db").close()
    (output / "study_snapshot.json").write_text(json.dumps({"source_workspace": str(old_source)}), encoding="utf-8")
    moved_source = tmp_path / "knowledge_base/workspace_source"
    moved_source.parent.mkdir()
    old_source.rename(moved_source)
    if __import__("os").name == "nt":
        import _winapi
        _winapi.CreateJunction(str(moved_source), str(old_source))
    else:
        old_source.symlink_to(moved_source, target_is_directory=True)
    # Stop immediately after the pin check, before any model or database work.
    class PinPassed(Exception):
        pass
    def stop(*args, **kwargs):
        raise PinPassed
    monkeypatch.setattr(study, "ResearchStore", stop)
    import pytest
    with pytest.raises(PinPassed):
        study.plan_study("test", source=moved_source, output=output, token_cap=1000)

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _sha256(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def test_v03_manifest_freezes_thirty_existing_cases_without_overlap():
    manifest = _read("config/benchmarks/v03_regression_manifest.json")
    rag = _read("config/benchmarks/rag_incremental_v1.json")
    skills = _read("config/benchmarks/skills_v1.json")
    precoding = _read("config/benchmarks/precoding_heldout_v2.json")
    available = {
        "rag": {query["id"] for idea in rag["ideas"] for query in idea["queries"]},
        "skills": {case["id"] for case in skills["cases"]},
        "precoding": {case["case_id"] for case in precoding["case_expectations"]},
    }
    selected = []
    for track in manifest["tracks"]:
        development = set(track["development_ids"])
        regression = set(track["regression_ids"])
        assert development.isdisjoint(regression)
        assert development | regression <= available[track["name"]]
        assert _sha256(track["source"]) == track["source_sha256"]
        selected.extend(development | regression)
    assert len(selected) == len(set(selected)) == manifest["total_cases"] == 30
    assert manifest["counts"] == {"development": 15, "regression": 15}

import json

import pytest

from researchops.datasets import canonical_hash, file_hash, write_json
from researchops.experiment_preparation import ExperimentPreparation, baseline_evidence
from researchops.runtime import ModelTurn


class RecommendationModel:
    model = "dataset-test-model"

    def __init__(self, bad_citation=False):
        self.calls = 0
        self.bad_citation = bad_citation

    def complete(self, messages, tools, max_tokens):
        self.calls += 1
        return ModelTurn(text=json.dumps({
            "task_family": "image_classification",
            "recommendations": [
                {"dataset_id": dataset_id, "priority": index, "role": "primary",
                 "reason": "Complementary low-data classification benchmark.",
                 "citations": ["made-up" if self.bad_citation else "node1"]}
                for index, dataset_id in enumerate(("flowers102", "dtd"))
            ],
            "baseline_citations": ["node1", "node2"], "manual_recommendations": [], "limitations": [],
        }), usage={"total_tokens": 150})


def setup_preparation(tmp_path, monkeypatch, *, silver_pass=True):
    from researchops import experiment_preparation
    idea = {"idea_id": "test", "title": "ViT adaptation"}
    batch_path = tmp_path / "artifacts" / "batch_research_test.json"
    write_json(batch_path, {"idea_id": "test", "acquisitions": [], "synthesis": {"final_synthesis": {"decision": "proceed"}}})
    write_json(tmp_path / "artifacts" / "evaluation" / "silver_report_test.json", {
        "idea_id": "test", "source_batch_report_path": str(batch_path),
        "quality_gate": {"status": "pass" if silver_pass else "fail"}, "silver_ready": silver_pass,
    })
    evidence = [{"chunk_id": "node1", "text": "Linear probing, full fine-tuning, LoRA, VPT and bottleneck adapters."},
                {"chunk_id": "node2", "text": "Flowers102 and DTD evaluate small-data adaptation."}]
    monkeypatch.setattr(experiment_preparation, "evidence_for_datasets", lambda *args: evidence)
    monkeypatch.setattr(experiment_preparation, "hardware_profile", lambda: {"cuda_available": False})
    states = []
    for dataset_id in ("flowers102", "dtd"):
        directory = tmp_path / dataset_id
        directory.mkdir()
        manifest = directory / "samples.jsonl"
        manifest.write_text("sample manifests")
        subset = directory / "train.json"
        write_json(subset, ["train:1"])
        states.append({"dataset_id": dataset_id, "status": "verified", "files": [{"bytes": 100}], "verification": {
            "dataset_id": dataset_id, "name": dataset_id, "root": str(directory), "class_count": 2,
            "sample_count": 10, "split_counts": {"train": 6, "val": 2, "test": 2},
            "manifest_path": str(manifest), "manifest_sha256": file_hash(manifest),
            "train_fractions": {"1.0": {"path": str(subset), "sha256": file_hash(subset), "count": 1}},
            "exclusions": {}, "split_seed": 42, "official_test_preserved": True,
            "checks": {"every_sample_readable": True}, "source_split_counts": {"train": 6, "val": 2, "test": 2},
        }})
    model = RecommendationModel()
    pipeline = ExperimentPreparation(workspace=tmp_path, store=None, model=model)
    monkeypatch.setattr(pipeline.datasets, "acquire_many", lambda *args, **kwargs: states)
    return pipeline, model, idea


def test_full_preparation_freezes_stable_contract_and_reuses_model_cache(tmp_path, monkeypatch):
    pipeline, model, idea = setup_preparation(tmp_path, monkeypatch)
    first = pipeline.run(idea)
    second = pipeline.run(idea)
    assert first["status"] == second["status"] == "ready_for_coding"
    assert all(first["quality_gates"].values())
    assert first["frozen_spec"] == second["frozen_spec"]
    assert first["llm_usage_this_run"]["total_tokens"] == 150
    assert second["llm_usage_this_run"]["total_tokens"] == 0
    assert first["required_user_actions_before_coding"] == []
    assert model.calls == 1


def test_silver_failure_blocks_freeze_even_with_verified_data(tmp_path, monkeypatch):
    pipeline, model, idea = setup_preparation(tmp_path, monkeypatch, silver_pass=False)
    result = pipeline.run(idea)
    assert result["status"] == "blocked_before_coding"
    assert result["frozen_spec"] is None
    assert result["quality_gates"]["two_verified_primary_datasets"] is True
    assert not list(pipeline.output.glob("frozen_*.json"))


def test_recommendation_cannot_cite_nonexistent_paper_nodes(tmp_path):
    pipeline = ExperimentPreparation(workspace=tmp_path, store=None, model=RecommendationModel(bad_citation=True))
    with pytest.raises(ValueError, match="must cite supplied"):
        pipeline.recommend({"idea_id": "test"}, {}, [{"chunk_id": "node1"}, {"chunk_id": "node2"}])


def test_missing_method_family_is_visible_in_baseline_gate():
    supported = baseline_evidence([{"chunk_id": "node1", "text": "LoRA and VPT"}])
    assert supported["lora"] and supported["vpt_shallow"]
    assert not supported["linear_probe"]
    assert not supported["full_finetune"]

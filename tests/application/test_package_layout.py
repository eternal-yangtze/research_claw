"""Refactoring must preserve persisted-code imports and configuration lookup."""
import importlib
from pathlib import Path


def test_legacy_imports_resolve_to_the_same_implementation_objects():
    aliases = {
        "runtime": "harness.runtime", "config": "core.config", "models": "core.models", "storage": "core.storage",
        "literature": "papers.connectors", "acquisition": "papers.acquisition", "ingestion": "papers.ingestion",
        "paper_analysis": "papers.analysis", "retrieval": "knowledge.retrieval", "compression": "knowledge.compression",
        "novelty": "knowledge.novelty", "research": "pipelines.research", "batch_research": "pipelines.batch",
        "silver_evaluation": "evaluation.silver", "datasets": "experiments.datasets",
        "experiment_preparation": "experiments.preparation", "experiment_support": "experiments.support",
        "coding_plan": "coding.plan", "coding_sandbox": "coding.sandbox", "workflow": "application.workflow",
        "cli": "application.cli",
    }
    for old, new in aliases.items():
        assert importlib.import_module('researchops.' + old) is importlib.import_module('researchops.' + new)


def test_relocated_configuration_and_template_paths_keep_original_contract():
    from researchops.core.config import PROJECT_ROOT
    from researchops.core.artifacts import file_hash, workspace_lock
    from researchops.experiments.datasets import dataset_lock
    from researchops.coding import agent
    assert (PROJECT_ROOT / 'researchops.py').is_file()
    assert PROJECT_ROOT == Path(__file__).resolve().parents[2]
    assert dataset_lock is workspace_lock
    template = Path(agent.__file__).parent / 'templates' / 'vit_peft'
    assert all((template / name).is_file() for name in ('models.py', 'train.py', 'metrics.py', 'run_experiment.py', 'README.md', 'tests/test_experiment.py'))
    assert file_hash(template / 'models.py')

"""External verifier, mounted read-only outside the generated project in Docker."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import math
import sys
import subprocess
import traceback
import runpy
import types
import io
from contextlib import redirect_stdout, redirect_stderr
from unittest.mock import patch
from pathlib import Path

import numpy as np
import torch

from researchops.experiments.support import seed_everything, spec_hash


def source_fingerprint(model):
    return hashlib.sha256(b"".join(value.detach().cpu().numpy().tobytes()
                                  for _, value in sorted(model.state_dict().items()))).hexdigest()


def independent_metrics(logits, labels, classes):
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    pred = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    correct = pred == labels
    f1s = []
    for label in range(classes):
        tp = ((pred == label) & (labels == label)).sum()
        denominator = (pred == label).sum() + (labels == label).sum()
        f1s.append(2 * tp / denominator if denominator else 0.0)
    ece = 0.0
    indices = np.minimum((confidence * 15).astype(int), 14)
    for index in range(15):
        mask = indices == index
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return {"accuracy": float(correct.mean()), "macro_f1": float(np.mean(f1s)), "ece": float(ece)}


def run_verification(spec, output, group="all"):
    torch.set_num_threads(2)
    models = importlib.import_module("models") if group in {"all", "models", "training"} else None
    metrics_module = importlib.import_module("metrics") if group in {"all", "metrics"} else None
    train = importlib.import_module("train") if group in {"all", "training"} else None
    checks, measurements = [], {}

    def check(name, action):
        owner = ("models" if name.startswith("model_") or name == "initialization_identity" else
                 "metrics" if name == "independent_metrics" else
                 "training" if name.startswith("real_data_resume_") or name in {"checkpoint_configuration_guard", "full_training_control_flow"} else
                 "dispatch" if name in {"experiment_cli", "frozen_job_dispatch"} else "tests")
        if group != "all" and owner != group:
            return
        captured_out, captured_err = io.StringIO(), io.StringIO()
        try:
            with redirect_stdout(captured_out), redirect_stderr(captured_err):
                detail = action()
            checks.append({"name": name, "passed": True, "detail": detail})
        except Exception:
            error = traceback.format_exc()[-4000:]
            if captured_out.getvalue():
                error += "\nCaptured stdout:\n" + captured_out.getvalue()[-1500:]
            if captured_err.getvalue():
                error += "\nCaptured stderr:\n" + captured_err.getvalue()[-2500:]
            checks.append({"name": name, "passed": False, "error": error[-8000:]})

    outputs = {}
    x = torch.randn(2, 3, 32, 32)
    y = torch.tensor([0, 1])
    for method in spec["methods"]:
        def model_check(method=method):
            seed_everything(42)
            net = models.build_model(copy.deepcopy(spec), method, 7, smoke=True)
            assert net.adaptation_method == method
            net.eval()
            logits = net(x)
            assert tuple(logits.shape) == (2, 7) and torch.isfinite(logits).all()
            outputs[method] = logits.detach().clone()
            named = dict(net.named_parameters())
            trainable = {name: value for name, value in named.items() if value.requires_grad}
            assert trainable
            if method != "full_finetune":
                allowed = {"linear_probe": ("head",), "lora": ("head", "lora_"),
                           "vpt_shallow": ("head", "prompt"), "bottleneck_adapter": ("head", "adapter")}[method]
                assert all(any(marker in name for marker in allowed) for name in trainable), list(trainable)
                if method != "linear_probe":
                    marker = {"lora": "lora_", "vpt_shallow": "prompt", "bottleneck_adapter": "adapter"}[method]
                    assert any(marker in name for name in trainable)
                if method == "lora":
                    # Standard Q/V LoRA owns independent rank-r A/B matrices;
                    # tying the two updates silently changes the frozen method.
                    rank = int(spec["methods"]["lora"]["rank"])
                    expected = 2 * 2 * 48 * rank * 2  # Q,V; A,B; dim; rank; blocks
                    actual = sum(p.numel() for n, p in trainable.items() if "lora_" in n)
                    assert actual == expected, ("Frozen LoRA requires independent rank-r Q and V updates in both blocks; "
                        f"expected {expected} adaptation parameters, got {actual}. Do not share Q/V A or B matrices.")
            before = {name: param.detach().clone() for name, param in named.items()}
            optimizer = torch.optim.AdamW(trainable.values(), lr=0.01)
            loss = torch.nn.functional.cross_entropy(net(x), y)
            loss.backward()
            assert torch.isfinite(loss)
            assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in trainable.values())
            if method in {"lora", "vpt_shallow", "bottleneck_adapter"}:
                marker = {"lora": "lora_", "vpt_shallow": "prompt", "bottleneck_adapter": "adapter"}[method]
                assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in trainable.items() if marker in n), "Adaptation parameters must participate in forward/backward"
            optimizer.step()
            assert all(torch.equal(before[name], param) for name, param in named.items() if not param.requires_grad)
            assert any(not torch.equal(before[name], param) for name, param in trainable.items())
            return {"trainable_parameters": sum(p.numel() for p in trainable.values()), "finite_loss": float(loss)}
        check("model_" + method, model_check)

    def identity():
        for method in ("lora", "bottleneck_adapter"):
            assert torch.allclose(outputs[method], outputs["linear_probe"], atol=2e-6, rtol=2e-5), method
        return "Zero-init LoRA/adapter preserve original logits"
    check("initialization_identity", identity)

    def metrics_check():
        logits = torch.tensor([[8., 0., 0.], [0., 8., 0.], [8., 0., 0.], [0., 0., 8.]])
        labels = torch.tensor([0, 1, 1, 2])
        expected = independent_metrics(logits.numpy(), labels.numpy(), 3)
        actual = metrics_module.classification_metrics(logits, labels, 3)
        for key in expected:
            assert abs(float(actual[key]) - expected[key]) < 1e-6, (key, actual, expected)
        return actual
    check("independent_metrics", metrics_check)

    for dataset in spec["datasets"]:
        for method in spec["methods"]:
            def data_check(dataset=dataset, method=method):
                job = {"job_id": f"smoke_{dataset['dataset_id']}_{method}", "stage": "validation_tuning",
                       "dataset_id": dataset["dataset_id"], "method": method, "fraction": 0.1,
                       "seed": 42, "lr": 0.001, "evaluate_test": False}
                target = output / job["job_id"]
                first = train.run_job(copy.deepcopy(spec), job, str(target), smoke=True, max_steps=2, resume=False)
                assert first["mode"] == "smoke" and first["test_evaluated"] is False
                assert first["global_step"] == 2
                checkpoint = torch.load(target / "checkpoint.pt", map_location="cpu", weights_only=False)
                assert checkpoint["global_step"] == 2 and checkpoint["optimizer"]["state"]
                before_hash = hashlib.sha256((target / "checkpoint.pt").read_bytes()).hexdigest()
                second = train.run_job(copy.deepcopy(spec), job, str(target), smoke=True, max_steps=3, resume=True)
                assert second["global_step"] == 3 and second["mode"] == "smoke" and not second["test_evaluated"]
                assert second["spec_hash"] == spec_hash(spec) and second["job_id"] == job["job_id"]
                checkpoint = torch.load(target / "checkpoint.pt", map_location="cpu", weights_only=False)
                assert checkpoint["global_step"] == 3 and checkpoint["job_id"] == job["job_id"]
                assert hashlib.sha256((target / "checkpoint.pt").read_bytes()).hexdigest() != before_hash
                logits, labels = np.load(target / "val_logits.npy"), np.load(target / "val_labels.npy")
                assert len(labels) > 0 and logits.shape == (len(labels), dataset["class_count"])
                expected = independent_metrics(logits, labels, dataset["class_count"])
                for key, value in expected.items():
                    assert math.isfinite(float(second["metrics"][key])) and abs(float(second["metrics"][key]) - value) < 1e-5
                assert first["trainable_parameters"] > 0
                return {"dataset": dataset["dataset_id"], "method": method, "resumed_step": 3,
                        "validation_samples": len(labels), "validation_metrics": expected,
                        "research_result": False}
            check(f"real_data_resume_{dataset['dataset_id']}_{method}", data_check)

    # Ensure the same checkpoint cannot silently resume another job/config.
    def mismatch():
        from researchops.experiments.support import load_checkpoint
        seed_everything(42)
        net = models.build_model(spec, "linear_probe", 7, smoke=True)
        optimizer = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad])
        any_checkpoint = next(output.rglob("checkpoint.pt"))
        try:
            load_checkpoint(any_checkpoint, net, optimizer, None, spec_hash="wrong", job_id="wrong")
        except ValueError:
            return "Mismatched configuration rejected before state restoration"
        raise AssertionError("Accepted wrong job/config")
    check("checkpoint_configuration_guard", mismatch)

    def cli_check():
        result = subprocess.run([sys.executable, "run_experiment.py", "--help"],
            capture_output=True, text=True, timeout=20)
        assert result.returncode == 0, result.stderr
        assert "--spec" in result.stdout and "--resume" in result.stdout and "--smoke" in result.stdout
        return "Experiment CLI accepts frozen specification, smoke and resume flags"
    check("experiment_cli", cli_check)

    def frozen_dispatch():
        """Check all frozen jobs without training a pretrained model."""
        calls = []
        stub = types.ModuleType("train")
        def fake_job(current_spec, job, output_dir, **kwargs):
            assert isinstance(job["lr"], (int, float)) and job["lr"] > 0, "Resolve LR from validation tuning before dispatch"
            assert not (job["stage"] == "validation_tuning" and job.get("evaluate_test"))
            if job["stage"] == "ablation":
                for key in ("rank", "alpha", "dropout"):
                    if key in job:
                        assert current_spec["methods"]["lora"][key] == job[key], "Apply ablation to copied spec"
            calls.append(copy.deepcopy(job))
            target = Path(output_dir)
            target.mkdir(parents=True, exist_ok=True)
            result = {"job_id": job["job_id"], "spec_hash": spec_hash(current_spec), "mode": "full",
                "global_step": 2, "test_evaluated": bool(job.get("evaluate_test")), "lr": job["lr"],
                "trainable_parameters": 10, "training_seconds": 1.0, "peak_memory": None,
                "metrics": {"accuracy": 0.5, "macro_f1": 0.4, "ece": 0.1},
                "test_metrics": {"accuracy": 0.4, "macro_f1": 0.3, "ece": 0.1}}
            (target / "result.json").write_text(json.dumps(result), encoding="utf-8")
            return result
        stub.run_job = fake_job
        dispatch_dir = output / "dispatch"
        spec_file = output / "dispatch_spec.json"
        spec_file.write_text(json.dumps(spec), encoding="utf-8")
        with patch.dict(sys.modules, {"train": stub}), patch.object(sys, "argv", ["run_experiment.py",
            "--spec", str(spec_file), "--output", str(dispatch_dir)]):
            try:
                runpy.run_path("run_experiment.py", run_name="__main__")
            except SystemExit as error:
                assert error.code in (None, 0)
        assert len(calls) == len(spec["jobs"]), (len(calls), len(spec["jobs"]))
        assert [j["job_id"] for j in calls] == [j["job_id"] for j in spec["jobs"]]
        return {"jobs_dispatched_with_mock_trainer": len(calls), "research_training_executed": False}
    check("frozen_job_dispatch", frozen_dispatch)

    def full_training_control():
        """Exercise full-mode control flow with synthetic data/tiny model only."""
        tiny = copy.deepcopy(spec)
        tiny["training"]["max_epochs"] = 2
        tiny["training"]["warmup_epochs"] = 1
        tiny["training"]["workers"] = 0
        tiny["training"]["batch_size"] = 2
        accessed = []
        class Data(torch.utils.data.Dataset):
            def __init__(self, dataset_spec, split, **kwargs):
                accessed.append((split, kwargs.get("allow_test", False)))
            def __len__(self): return 4
            def __getitem__(self, index): return torch.full((3, 32, 32), index / 4), index % 2
        original_build = models.build_model
        def tiny_model(current, method, classes, **kwargs):
            return original_build(current, method, classes, smoke=True)
        job = {"job_id": "full_control", "stage": "main", "dataset_id": spec["datasets"][0]["dataset_id"],
            "method": "linear_probe", "fraction": 0.1, "seed": 42, "lr": 0.001, "evaluate_test": True}
        with patch.object(train, "ManifestDataset", Data), patch.object(train, "build_model", tiny_model):
            result = train.run_job(tiny, job, str(output / "full_control"), smoke=False)
        assert result["mode"] == "full"
        assert result["test_evaluated"] and result.get("test_metrics"), "Full main jobs must evaluate test with best validation checkpoint"
        assert ("test", True) in accessed, "Test access must be explicit and stage-controlled"
        assert result["global_step"] <= 4, "Respect max_epochs from frozen spec"
        return {"full_control_tested": True, "synthetic_fixture": True, "research_result": False}
    check("full_training_control_flow", full_training_control)

    def generated_tests():
        result = subprocess.run([sys.executable, "-m", "pytest", "-q", "--tb=short", "-p", "no:cacheprovider",
            "--basetemp=" + str(output / "pytest"), "tests"], capture_output=True, text=True, timeout=45)
        assert result.returncode == 0, result.stdout[-6000:] + result.stderr[-2000:]
        return result.stdout[-2000:]
    check("generated_unit_tests", generated_tests)
    measurements["checks_passed"] = sum(item["passed"] for item in checks)
    measurements["checks_total"] = len(checks)
    return {"passed": bool(checks) and all(item["passed"] for item in checks), "checks": checks, "group": group,
            "measurements": measurements, "mode": "engineering_smoke", "research_success": None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--group", choices=("all", "models", "metrics", "training", "dispatch", "tests"), default="all")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(args.project).resolve()))
    try:
        result = run_verification(json.loads(Path(args.spec).read_text(encoding="utf-8")), output, args.group)
    except Exception:
        result = {"passed": False, "checks": [{"name": "import_or_setup", "passed": False,
                  "error": traceback.format_exc()[-6000:]}]}
    (output / "verifier.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()

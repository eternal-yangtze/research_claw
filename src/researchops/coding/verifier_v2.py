"""Independent arm-aware checks for accepted v2 study code, run inside Docker.

Smoke verifies real train/validation images and checkpoint continuation. Dispatch
uses a fake trainer explicitly; neither stage is a scientific accuracy result.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import runpy
import subprocess
import sys
import traceback
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from researchops.coding.verifier import independent_metrics
from researchops.experiments.support import spec_hash, seed_everything


def run_verification(spec, output, group="all"):
    torch.set_num_threads(2)
    models = importlib.import_module("models") if group in {"all", "models", "training"} else None
    train = importlib.import_module("train") if group in {"all", "training"} else None
    checks = []

    def check(owner, name, action):
        if group not in {"all", owner}:
            return
        try:
            detail = action()
            checks.append({"name": name, "passed": True, "detail": detail})
        except Exception:
            checks.append({"name": name, "passed": False, "error": traceback.format_exc()[-7000:]})

    x, y = torch.randn(2, 3, 32, 32), torch.tensor([0, 1])
    for arm_id, arm in spec["methods"].items():
        def model_check(arm_id=arm_id, arm=arm):
            seed_everything(42)
            net = models.build_model(copy.deepcopy(spec), arm_id, 7, smoke=True)
            assert net.adaptation_method == arm["method"] and net.arm_id == arm_id
            logits = net(x)
            assert logits.shape == (2, 7) and torch.isfinite(logits).all()
            named = dict(net.named_parameters())
            learned = {n: p for n, p in named.items() if p.requires_grad}
            assert learned
            if arm["method"] != "full_finetune":
                marker = {"lora": "lora_", "bottleneck_adapter": "adapter", "vpt_shallow": "prompt",
                          "linear_probe": "head"}[arm["method"]]
                assert all("head" in n or marker in n for n in learned)
                assert any(marker in n for n in learned)
            if arm["method"] == "lora":
                actual = sum(p.numel() for n, p in learned.items() if "lora_" in n)
                expected = 2 * 2 * 48 * arm["rank"] * 2
                assert actual == expected, ("independent Q/V rank and both tiny blocks", actual, expected)
                # Compare the initialized adapted network to the same base with updates zeroed.
                net.eval()
                with torch.no_grad():
                    before_logits = net(x).clone()
                    for n, p in named.items():
                        if "lora_" in n:
                            p.zero_()
                    assert torch.allclose(before_logits, net(x), atol=2e-6, rtol=2e-5)
                # Rebuild so zeroing A for the identity check does not kill gradients.
                seed_everything(42)
                net = models.build_model(copy.deepcopy(spec), arm_id, 7, smoke=True)
                named = dict(net.named_parameters())
                learned = {n: p for n, p in named.items() if p.requires_grad}
            frozen = {n: p.detach().clone() for n, p in named.items() if not p.requires_grad}
            opt = torch.optim.AdamW(learned.values(), lr=0.01)
            loss = torch.nn.functional.cross_entropy(net(x), y)
            loss.backward()
            assert torch.isfinite(loss)
            if arm["method"] != "linear_probe":
                marker = {"lora": "lora_", "bottleneck_adapter": "adapter", "vpt_shallow": "prompt",
                          "full_finetune": ""}[arm["method"]]
                assert any(marker in n and p.grad is not None and p.grad.abs().sum() > 0 for n, p in learned.items())
            opt.step()
            assert all(torch.equal(p, frozen[n]) for n, p in named.items() if n in frozen)
            return {"arm_id": arm_id, "trainable_parameters": sum(p.numel() for p in learned.values()),
                    "finite_loss": float(loss.detach())}
        check("models", "model_" + arm_id, model_check)

    def metrics_check():
        module = importlib.import_module("metrics")
        logits = torch.tensor([[8., 0., 0.], [0., 8., 0.], [8., 0., 0.], [0., 0., 8.]])
        labels = torch.tensor([0, 1, 1, 2])
        expected = independent_metrics(logits.numpy(), labels.numpy(), 3)
        actual = module.classification_metrics(logits, labels, 3)
        assert all(abs(float(actual[k]) - v) < 1e-6 for k, v in expected.items())
        return actual
    check("metrics", "independent_metrics", metrics_check)

    for data in spec["datasets"]:
        for arm_id, arm in spec["methods"].items():
            def training_check(data=data, arm_id=arm_id, arm=arm):
                job = next(copy.deepcopy(j) for j in spec["jobs"] if j["dataset_id"] == data["dataset_id"]
                           and j["arm_id"] == arm_id and j["stage"] == "validation_tuning")
                target = output / ("smoke_" + data["dataset_id"] + "_" + arm_id)
                accesses = []
                from researchops.experiments.support import ManifestDataset
                def watched(ds, split, **kwargs):
                    accesses.append(split)
                    assert split != "test", "smoke cannot access test"
                    return ManifestDataset(ds, split, **kwargs)
                with patch.object(train, "ManifestDataset", watched):
                    first = train.run_job(copy.deepcopy(spec), job, str(target), smoke=True, max_steps=2)
                    first_hash = hashlib.sha256((target / "checkpoint.pt").read_bytes()).hexdigest()
                    second = train.run_job(copy.deepcopy(spec), job, str(target), smoke=True, max_steps=3, resume=True)
                assert {"train", "val"} <= set(accesses)
                assert first["global_step"] == 2 and second["global_step"] == 3
                assert second["mode"] == "smoke" and not second["test_evaluated"]
                assert second["spec_hash"] == spec_hash(spec) and second["job_id"] == job["job_id"]
                assert second["arm_id"] == arm_id
                cp = torch.load(target / "checkpoint.pt", map_location="cpu", weights_only=False)
                assert cp["global_step"] == 3 and cp["optimizer"]["state"] and cp["rng"]
                assert first_hash != hashlib.sha256((target / "checkpoint.pt").read_bytes()).hexdigest()
                logits, labels = np.load(target / "val_logits.npy"), np.load(target / "val_labels.npy")
                assert logits.shape == (len(labels), data["class_count"]) and len(labels) > 0
                expected = independent_metrics(logits, labels, data["class_count"])
                assert all(math.isfinite(float(second["metrics"][k])) and abs(second["metrics"][k]-v) < 1e-5 for k,v in expected.items())
                wrong = {**job, "job_id": "wrong-job"}
                try:
                    train.run_job(copy.deepcopy(spec), wrong, str(target), smoke=True, max_steps=4, resume=True)
                except ValueError:
                    pass
                else:
                    raise AssertionError("wrong-job checkpoint accepted")
                # Evaluation-only main uses selected weights and never constructs an optimizer.
                main = next(copy.deepcopy(j) for j in spec["jobs"] if j["dataset_id"] == data["dataset_id"]
                            and j["arm_id"] == arm_id and j["stage"] == "main")
                main.update(lr=job["lr"], selected_checkpoint=str(target / "checkpoint.pt"))
                with patch.object(torch.optim, "AdamW", side_effect=AssertionError("main must not retrain")), \
                     patch.object(train, "ManifestDataset", watched):
                    evaluated = train.run_job(copy.deepcopy(spec), main, str(target / "main"), smoke=True)
                assert not evaluated["test_evaluated"] and evaluated["mode"] == "smoke"
                assert evaluated["global_step"] == 3 and evaluated["arm_id"] == arm_id
                return {"real_images": True, "resume_step": 3, "main_no_retraining": True,
                        "validation_samples": len(labels), "research_result": False}
            check("training", f"real_data_resume_{data['dataset_id']}_{arm_id}", training_check)

    def cli_check():
        result = subprocess.run([sys.executable, "run_experiment.py", "--help"], capture_output=True, text=True, timeout=20)
        assert result.returncode == 0 and all(s in result.stdout for s in ("--spec", "--smoke", "--resume")), result.stderr
        return "CLI compatible"
    check("dispatch", "experiment_cli", cli_check)

    def dispatch_check():
        calls, chosen = [], {}
        stub = types.ModuleType("train")
        def fake(current, job, output_dir, **kwargs):
            assert current == spec, "v2 input contract must not be mutated"
            assert isinstance(job["lr"], (int, float)) and job["lr"] > 0
            key = tuple(job[k] for k in ("dataset_id", "arm_id", "fraction", "seed"))
            target = Path(output_dir)
            if job["stage"] == "validation_tuning":
                assert not job["evaluate_test"]
                # Different arms deliberately prefer different LR. Ties prefer smaller LR.
                arm_index = list(spec["methods"]).index(job["arm_id"])
                accuracy = float(job["lr"]) if arm_index % 2 else 0.5
                candidate = (accuracy, -job["lr"], str(target / "checkpoint.pt"))
                chosen[key] = max(chosen.get(key, candidate), candidate)
            else:
                assert job["operation"] == "evaluate_selected_checkpoint_no_retraining"
                assert job["selected_checkpoint"] == chosen[key][2], (job, chosen[key])
                assert job["lr"] == -chosen[key][1]
                accuracy = 0.4
            calls.append(copy.deepcopy(job))
            target.mkdir(parents=True, exist_ok=True)
            value = {"job_id": job["job_id"], "spec_hash": spec_hash(current), "arm_id": job["arm_id"],
                     "mode": "full", "global_step": 2, "test_evaluated": job["evaluate_test"],
                     "metrics": {"accuracy": accuracy, "macro_f1": 0.4, "ece": 0.1},
                     "test_metrics": {"accuracy": 0.4}, "trainable_parameters": 10}
            (target / "result.json").write_text(json.dumps(value), encoding="utf-8")
            return value
        stub.run_job = fake
        path, target = output / "dispatch_spec.json", output / "dispatch"
        path.write_text(json.dumps(spec), encoding="utf-8")
        with patch.dict(sys.modules, {"train": stub}), patch.object(sys, "argv", ["run_experiment.py", "--spec", str(path), "--output", str(target)]):
            try:
                runpy.run_path("run_experiment.py", run_name="__main__")
            except SystemExit as e:
                assert e.code in (0, None)
        assert [j["job_id"] for j in calls] == [j["job_id"] for j in spec["jobs"]]
        return {"mock_trainer_dispatches": len(calls), "arm_isolation_and_lr_ties": True, "research_training_executed": False}
    check("dispatch", "frozen_job_dispatch", dispatch_check)

    def tests_check():
        result = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
            "--basetemp=" + str(output / "pytest"), "tests"], capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stdout[-5000:] + result.stderr[-2000:]
        return result.stdout[-1500:]
    check("tests", "generated_unit_tests", tests_check)
    return {"passed": bool(checks) and all(c["passed"] for c in checks), "checks": checks, "group": group,
        "mode": "engineering_smoke", "research_success": None,
        "measurements": {"checks_passed": sum(c["passed"] for c in checks), "checks_total": len(checks)}}

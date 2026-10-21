"""Docker-only execution of generated code, with immutable data/harness mounts."""
from __future__ import annotations

import json
import subprocess
import uuid
from pathlib import Path

from researchops.core.artifacts import file_hash, write_json

IMAGE = "researchops-coding:cpu-v1"


def project_execution_spec(spec, dataset_root: Path):
    value = json.loads(json.dumps(spec))
    root = dataset_root.resolve()
    for dataset in value["datasets"]:
        for key in ("root", "manifest_path"):
            dataset[key] = "/data/" + Path(dataset[key]).resolve().relative_to(root).as_posix()
        for fraction in dataset["train_fractions"].values():
            fraction["path"] = "/data/" + Path(fraction["path"]).resolve().relative_to(root).as_posix()
    value["execution_projection"] = {"purpose": "path projection only; frozen research design unchanged",
                                     "mode": "engineering_smoke", "research_result": False}
    return value


class CodingSandbox:
    def __init__(self, workspace, project, run_dir, *, timeout=180, image=IMAGE):
        self.workspace, self.project, self.run_dir = map(lambda p: Path(p).resolve(), (workspace, project, run_dir))
        self.timeout, self.image = timeout, image

    def verify(self, round_number, *, group="all"):
        if group not in {"all", "models", "metrics", "training", "dispatch", "tests"}:
            raise ValueError("Unknown verifier group")
        output = self.run_dir / f"verification_{round_number}"
        output.mkdir(parents=True, exist_ok=True)
        name = "researchops-verify-" + uuid.uuid4().hex[:12]
        harness = Path(__file__).resolve().parents[2]
        inputs = self.run_dir / "inputs"
        try:
            inspected = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", self.image],
                capture_output=True, text=True, timeout=20, check=True)
            image_id = inspected.stdout.strip()
            if not image_id.startswith("sha256:"):
                raise ValueError("Docker image ID unavailable")
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            report = {"passed": False, "checks": [{"name": "sandbox_image", "passed": False, "error": str(error)}]}
            write_json(output / "verification_report.json", report)
            return report
        command = ["docker", "run", "--rm", "--name", name, "--network", "none", "--read-only",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--pids-limit", "128",
            "--cpus", "2", "--memory", "3g", "--user", "65534:65534",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=256m", "--shm-size", "128m",
            "--mount", f"type=bind,source={self.project},target=/work,readonly",
            "--mount", f"type=bind,source={(self.workspace / 'datasets').resolve()},target=/data,readonly",
            "--mount", f"type=bind,source={harness},target=/harness,readonly",
            "--mount", f"type=bind,source={inputs},target=/inputs,readonly",
            "--mount", f"type=bind,source={output},target=/results",
            "--env", "PYTHONPATH=/harness", image_id,
            "python", "-m", "researchops.coding.verifier", "--spec", "/inputs/execution_spec.json",
            "--project", "/work", "--output", "/results", "--group", group]
        try:
            result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                    timeout=self.timeout, shell=False)
            (output / "stdout.log").write_text(result.stdout, encoding="utf-8")
            (output / "stderr.log").write_text(result.stderr, encoding="utf-8")
            target = output / "verifier.json"
            if target.is_file():
                report = json.loads(target.read_text(encoding="utf-8"))
            else:
                report = {"passed": False, "checks": [{"name": "sandbox_execution", "passed": False,
                    "error": result.stderr[-4000:]}]}
            report.update(returncode=result.returncode, report_path=str(target),
                          sandbox={"network": "none", "data_read_only": True, "project_read_only": True,
                                   "image": self.image, "image_id": image_id, "cpus": 2, "memory": "3g"})
            report["passed"] = bool(report["passed"] and result.returncode == 0)
        except subprocess.TimeoutExpired:
            # Only this application's explicitly named ephemeral container is removed.
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=20)
            report = {"passed": False, "checks": [{"name": "wall_timeout", "passed": False,
                      "error": f"Sandbox exceeded {self.timeout}s; own container stopped"}]}
        except (OSError, subprocess.SubprocessError) as error:
            report = {"passed": False, "checks": [{"name": "sandbox_unavailable", "passed": False,
                       "error": f"{type(error).__name__}: {error}"}]}
        write_json(output / "verification_report.json", report)
        return report


def stage_inputs(workspace, run_dir, plan):
    root = Path(workspace).resolve() / "datasets"
    for artifact in plan.get("source_artifacts", []):
        if file_hash(artifact["path"]) != artifact["sha256"]:
            raise ValueError("Coding plan evidence artifact changed")
    source = Path(plan["frozen_spec"]["path"])
    if file_hash(source) != plan["frozen_spec"]["sha256"]:
        raise ValueError("Frozen contract changed")
    spec = json.loads(source.read_text(encoding="utf-8"))
    for dataset in spec["datasets"]:
        if file_hash(dataset["manifest_path"]) != dataset["manifest_sha256"]:
            raise ValueError("Dataset manifest changed")
        for fraction in dataset["train_fractions"].values():
            if file_hash(fraction["path"]) != fraction["sha256"]:
                raise ValueError("Subset manifest changed")
    target = Path(run_dir) / "inputs"
    write_json(target / "execution_spec.json", project_execution_spec(spec, root))
    write_json(target / "coding_plan.json", plan)
    return target

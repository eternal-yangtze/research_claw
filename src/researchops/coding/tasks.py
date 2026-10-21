"""Compile the frozen API into bounded, independently verifiable coding tasks.

Curie-inspired staged planning and RD-Agent-inspired per-task feedback. This is
an implementation for ResearchOps' contracts, not a vendored upstream runtime.
"""
from __future__ import annotations


def compile_tasks(api):
    task_limits = {"metrics": 20000, "models": 45000, "training": 80000,
                   "dispatch": 60000, "tests": 40000, "documentation": 20000}
    layout = [
        ("metrics", "metrics.py", [], "metrics", 3000),
        ("models", "models.py", [], "models", 7000),
        ("training", "train.py", ["metrics", "models"], "training", 8000),
        ("dispatch", "run_experiment.py", ["training"], "dispatch", 6000),
        ("tests", "tests/test_experiment.py", ["training", "dispatch"], "tests", 4000),
        ("documentation", "README.md", ["tests"], None, 2500),
    ]
    return [{"id": name, "outputs": [path], "depends_on": parents,
             "interfaces": api[path], "verification_group": group,
             "acceptance": "External behavioral checks" if group else "Nonempty documented smoke/full/resume commands",
             "max_attempts": 3, "max_output_tokens": tokens, "max_total_tokens": task_limits[name],
             "max_context_bytes": 42000}
            for name, path, parents, group, tokens in layout]


def validate_tasks(tasks, allowed):
    names = [t["id"] for t in tasks]
    if not tasks or len(names) != len(set(names)):
        raise ValueError("Execution tasks require unique IDs")
    seen, outputs = set(), []
    for task in tasks:
        if not set(task["depends_on"]) <= seen:
            raise ValueError("Execution tasks must be topologically ordered")
        if len(task["outputs"]) != 1 or task["outputs"][0] not in allowed:
            raise ValueError("Each execution task owns exactly one allowed output")
        if task["max_attempts"] < 1 or task["max_total_tokens"] <= task["max_output_tokens"]:
            raise ValueError("Invalid task budget")
        if task["verification_group"] not in {None, "models", "metrics", "training", "dispatch", "tests"}:
            raise ValueError("Unknown verification group")
        seen.add(task["id"])
        outputs.extend(task["outputs"])
    if len(outputs) != len(set(outputs)) or set(outputs) != set(allowed):
        raise ValueError("Execution tasks must own every allowed output exactly once")

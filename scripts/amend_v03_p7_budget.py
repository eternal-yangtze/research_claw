"""Audited P7 allowance amendment before ANY judge or regression calls.

The task explicitly permits up to 5x budgets. Keep initial manifests/code and
every generation debit; only raise the root cap and freeze evaluator memoization.
"""
from __future__ import annotations

import argparse
import ast
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from researchops.core.artifacts import file_hash, workspace_lock, write_json
from researchops.core.config import load_project_env
from researchops.harness.budget import BudgetedModel
from researchops.harness.runtime import OpenAICompatibleModel


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    load_project_env()
    with workspace_lock(root):
        if (root / "budget_amendment.json").exists():
            raise ValueError("Amendment already applied; use prepare to freeze the new configuration")
        if (root / "evaluation_budget/ledger.json").exists() or any((root / "answers").glob("R0[456]*/*.json")):
            raise ValueError("Amendment must precede judge or regression observations")
        previous, frozen = read(root / "config.json"), read(root / "freeze.json")
        snapshot = root / "source_snapshot_initial/evaluate_v03_ragas.py"
        current = Path(__file__).with_name("evaluate_v03_ragas.py")
        def normalized_functions(path):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            # Only normalize the explicitly amended StudyCalls admission caps.
            # Prompt construction, retrieval, validators and loop counts remain compared.
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "StudyCalls":
                    node.args[2] = ast.Constant(value="AUDITED_TOKEN_CAP")
                    for kw in node.keywords:
                        if kw.arg == "max_model_turns":
                            kw.value = ast.Constant(value="AUDITED_DISPATCH_CAP")
            return {n.name: ast.dump(n) for n in tree.body if isinstance(n, ast.FunctionDef)}
        old_funcs, new_funcs = normalized_functions(snapshot), normalized_functions(current)
        unchanged = all(old_funcs[n] == new_funcs[n] for n in ("generate", "queries", "split_ids", "provider"))
        if not unchanged:
            raise ValueError("Generator logic changed; this operational amendment cannot authorize scientific changes")
        if file_hash(snapshot) != frozen["code"]["scripts\\evaluate_v03_ragas.py"]:
            raise ValueError("Initial executable snapshot does not match its frozen hash")
        model = OpenAICompatibleModel(max_attempts=1, stream=True, reasoning_effort="low", temperature=0.01)
        budget = BudgetedModel(model, root / "generation_budget", 500000, resume=True, max_model_turns=144)
        before = budget.summary()
        budget.increase_token_limit(1000000, reason="User authorized moderate increases <=5x; observed provider counts exceed character estimates. Preserve all existing calls and unknown usage.")
        for scope_id in list(budget.state["scopes"]):
            budget.increase_scope_limits(scope_id, token_cap=200000, max_model_turns=8,
                reason="Same user-authorized amendment; retain all prior case calls; one bounded recovery may follow an admission failure")
        amendment = {"status": "applied_before_judge_and_regression", "original_generation_cap": 500000,
            "new_generation_cap": 1000000, "original_evaluator_cap": 1500000,
            "new_evaluator_cap": 6000000, "generation_multiplier": 2, "evaluation_multiplier": 4,
            "authorization": "User: do not leave functionality incomplete due to budget; preferably <=5x original",
            "generator_functions_AST_unchanged": unchanged,
            "old_executable_sha256": file_hash(snapshot), "new_executable_sha256": file_hash(current),
            "evaluation_change": "Memoize Context Precision/Recall only when all their required input fields are identical; retain per-row scores and provenance",
            "before": before, "after": budget.summary(), "initial_config": previous, "initial_freeze": frozen}
        (root / "config.json").rename(root / "config.initial.json")
        (root / "freeze.json").rename(root / "freeze.initial.json")
        write_json(root / "budget_amendment.json", amendment)
        print(json.dumps(amendment, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

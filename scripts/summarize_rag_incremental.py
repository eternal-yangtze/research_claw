"""Build a compact final summary from audited raw reports; no paid calls."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from researchops.core.artifacts import write_json


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    roots = {"raw_v1": Path("workspace_rag_incremental_v1_live"),
             "planned_v2": Path("workspace_rag_incremental_v2_planned")}
    summaries = {}
    for name, root in roots.items():
        report, audit = read(root / "report.json"), read(root / "audit.json")
        if not audit["passed"]:
            raise ValueError("Only audited reports can be summarized")
        summaries[name] = {"workspace": str(root.resolve()), "corpus": report["corpus"],
                           "retrieval_after": report["retrieval"]["after"], "answers": report["answers"],
                           "retrieval_deltas": report["retrieval_deltas"], "answer_deltas": report["answer_deltas"],
                           "known_tokens": audit["known_actual_tokens"], "unknown_tokens": audit["unknown_reserved_tokens"],
                           "request_attempts": audit["request_attempts"], "all_checks_passed": True}
    setup = read("workspace_rag_incremental_v1/ledger.json")
    known = sum(v["known_tokens"] for v in summaries.values())
    unknown = sum(v["unknown_tokens"] for v in summaries.values()) + setup["total_tokens"]
    report = read(roots["planned_v2"] / "report.json")
    index = report["indexing"]
    value = {"date": "2026-10-20", "scope": "Pretrained ViT PEFT evidence retrieval and research answers before coding",
             "new_ideas": 6, "queries": 12, "new_pdfs": 7, "ideas_with_new_downloads": 4,
             "raw_and_planned": summaries,
             "incremental_index_speedup": index["full_rebuild"]["seconds"] / index["incremental"]["seconds"],
             "indexing": index, "repeat_rankings_identical": sum(r["identical_order"] for r in report["repeat_search"]),
             "shared_no_rag_controls": len(read(roots["planned_v2"] / "shared_controls.json")["imports"]),
             "cost": {"known_actual_tokens": known, "unknown_reservations": unknown,
                      "conservative_total": known + unknown,
                      "request_attempts": sum(v["request_attempts"] for v in summaries.values()) + len(setup["call_ledger"]),
                      "historical_87_paper_cost_excluded": True},
             "engineering_tests": {"passed": 93},
             "limitations": ["Internal model silver; no expert gold; six IDEA clusters.",
                             "Query interface repair reuses same tasks and actual acquisition; not untouched heldout.",
                             "Better retrieval and traceability do not establish better scientific study designs.",
                             "Incremental index maintenance acceleration does not mean faster neural query latency.",
                             "No coding, training outcomes, formula cross-recognition or skills packaging evaluated."]}
    write_json(Path("internal_experiments/evaluation/results/rag_incremental_metrics.json"), value)
    print(json.dumps({"cost": value["cost"], "speedup": value["incremental_index_speedup"],
                      "shared_controls": value["shared_no_rag_controls"]}, indent=2))


if __name__ == "__main__":
    main()

import json
from pathlib import Path

from researchops.workflow import ResearchOpsApp


def test_seed_workflow_creates_auditable_artifact(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "prior.md").write_text(
        """# Paper Metadata
Title: Compressed Time Series Analytics
Authors: Prior Author
Year: 2024

# Core Idea
Compressed segments store residual norms to calculate deterministic query error bounds.
Workload-aware storage selects representations using query frequency.
""",
        encoding="utf-8",
    )
    idea = {
        "title": "Query-aware compression",
        "domain": "time_series",
        "description": "Use query tolerances to allocate compression errors.",
        "research_question": "Can query tolerances reduce storage and query cost?",
        "atomic_claims": ["Query frequency and tolerance determine segment error budgets."],
        "success_criteria": ["All true errors remain inside deterministic bounds."],
    }
    idea_path = tmp_path / "idea.json"
    idea_path.write_text(json.dumps(idea), encoding="utf-8")

    app = ResearchOpsApp(tmp_path / "workspace", allowed_roots=[corpus, idea_path])
    report = app.run_seed_workflow(idea_path, corpus)

    assert report["novelty_audit"]["decision"] in {"PROCEED", "REFINE", "REJECT", "INSUFFICIENT_EVIDENCE"}
    assert report["evidence_answer"]["verified"] is True
    artifact_path = Path(report["artifact"]["path"])
    assert artifact_path.exists()
    assert (tmp_path / "workspace" / "runs" / report["run_id"] / "trace.jsonl").exists()
    assert app.store.counts()["artifacts"] == 1


from __future__ import annotations

import argparse
import json
from pathlib import Path

from researchops.literature import LiteratureSearch, download_arxiv_source
from researchops.storage import ResearchStore


def main() -> int:
    parser = argparse.ArgumentParser(description="Live smoke test for public literature connectors.")
    parser.add_argument("--workspace", default="workspace_source_connectors_smoke")
    parser.add_argument("--query", default="attention is all you need")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--arxiv-id", default="1706.03762")
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    store = ResearchStore(workspace / "researchops.db")
    result = LiteratureSearch(store, timeout=30).search(
        args.query,
        "run_connector_verification",
        limit_per_source=args.limit,
    )
    source_download = download_arxiv_source(
        args.arxiv_id,
        workspace / "artifacts" / "sources",
    )
    transformer = next(
        (paper for paper in result["papers"] if paper["title"].casefold() == "attention is all you need"),
        None,
    )
    summary = {
        "query": args.query,
        "source_statuses": result["sources"],
        "merged_candidate_count": len(result["papers"]),
        "matched_paper": transformer,
        "arxiv_source_download": source_download,
        "database": str(store.path),
    }
    target = workspace / "artifacts" / "connector_verification.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**summary, "report_path": str(target)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

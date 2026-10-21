"""Prepare an independent CPU index; preserve all previous benchmark workspaces."""
import os
import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from researchops.core.artifacts import write_json, file_hash
from researchops.core.storage import ResearchStore
from researchops.knowledge.evidence_retrieval import EvidenceRetriever

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("workspace_rag_incremental_v1_live/passages.db"))
    parser.add_argument("--output", type=Path, default=Path("workspace_rag_cpu_v3"))
    args = parser.parse_args()
    os.environ["RESEARCHOPS_EMBEDDING_LOCAL_ONLY"] = "true"
    os.environ["RESEARCHOPS_RERANKER_LOCAL_ONLY"] = "true"
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    torch.set_num_threads(4)
    source = args.source.resolve()
    root = args.output.resolve()
    if root == source.parent or source.parent in root.parents:
        raise ValueError("Use an independent output workspace")
    root.mkdir(parents=True, exist_ok=True)
    target = root / "passages.db"
    if not target.exists():
        with sqlite3.connect(f"{source.as_uri()}?mode=ro", uri=True) as src:
            with sqlite3.connect(target) as dst:
                src.backup(dst)
        write_json(root / "index_source.json", {"source": str(source), "sha256_at_copy": file_hash(source)})
    engine = EvidenceRetriever(ResearchStore(target))
    result = engine._refresh()
    write_json(root / "initial_indexing.json", result)
    print(result, flush=True)

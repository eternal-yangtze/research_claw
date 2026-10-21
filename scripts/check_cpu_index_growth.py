"""Actual staged index additions, with old-vector identity checks and no new labels."""
import json
import os
import sqlite3
import sys
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from researchops.core.artifacts import write_json, canonical_hash
from researchops.core.models import Chunk
from researchops.core.storage import ResearchStore
from researchops.knowledge.passages import paper_from_row
from researchops.knowledge.evidence_retrieval import EvidenceRetriever, VERSION


def add_paper(source, target, pid):
    target.upsert_paper(paper_from_row(source.get_paper(pid)))
    chunks = []
    for row in source.paper_chunks(pid):
        raw = dict(row)
        value = {f.name: raw[f.name] for f in fields(Chunk) if f.name in raw}
        value.update(bbox=tuple(json.loads(raw["bbox_json"])) if raw["bbox_json"] else None,
                     provenance=json.loads(raw["provenance_json"] or "{}"),
                     vector=tuple(json.loads(raw["vector_json"] or "[]")))
        chunks.append(Chunk(**value))
    target.replace_chunks(pid, chunks)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    os.environ["RESEARCHOPS_EMBEDDING_LOCAL_ONLY"] = "true"
    os.environ["RESEARCHOPS_RERANKER_LOCAL_ONLY"] = "true"
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    torch.set_num_threads(4)
    root = Path("workspace_rag_cpu_growth_v1")
    root.mkdir(exist_ok=False)
    source = ResearchStore(Path("workspace_rag_cpu_v31/passages.db"))
    initial = set(json.loads(Path("workspace_rag_incremental_v1_live/snapshot.json").read_text(encoding="utf-8"))["initial_paper_ids"])
    new = sorted({r["paper_id"] for r in source.all_chunks()} - initial, key=lambda pid: source.get_paper(pid)["content_hash"])
    target = ResearchStore(root / "passages.db")
    for pid in sorted(initial):
        add_paper(source, target, pid)
    # Reuse only the old 87-paper checkpoint; new-paper embeddings are deliberately not copied.
    with sqlite3.connect(target.path) as connection:
        connection.execute("ATTACH DATABASE ? AS prior", (str(source.path.resolve()),))
        connection.execute("INSERT INTO chunk_embeddings SELECT e.* FROM prior.chunk_embeddings e JOIN chunks c ON c.chunk_id=e.chunk_id WHERE e.model_revision LIKE ?", (VERSION + ":%",))
    engine = EvidenceRetriever(target)
    namespace = f"{VERSION}:{engine.embedding_key}"
    before = {cid: canonical_hash(record["vector_json"]) for cid, record in target.chunk_embeddings(engine.embedding.model_name, namespace).items()}
    stages = [{"papers": 87, **engine._refresh()}]
    for lower, upper in [(0, 2), (2, 5), (5, 7)]:
        for pid in new[lower:upper]:
            add_paper(source, target, pid)
        result = {"papers": 87 + upper, **engine._refresh()}
        write_json(root / f"stage_{87+upper}.json", result)
        stages.append(result)
        print(result, flush=True)
    after = target.chunk_embeddings(engine.embedding.model_name, namespace)
    identical = all(cid in after and canonical_hash(after[cid]["vector_json"]) == digest for cid, digest in before.items())
    report = {"stages": stages, "original_vectors": len(before), "old_vectors_identical": identical,
              "new_passages_encoded": sum(s["encoded_passages"] for s in stages[1:]),
              "initial_checkpoint_reused": "old87 from workspace_rag_cpu_v31; not a cold index run",
              "new_paper_order": new}
    write_json(root / "report.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)

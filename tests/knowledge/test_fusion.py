from researchops.knowledge.fusion_retrieval import fuse_candidates
from researchops.knowledge.retrieval import SearchHit
from researchops.experiments.fusion_design import build_fusion_packet, improve_fusion_answer


def hit(pid):
    return SearchHit(f"c{pid}", pid, f"Paper {pid}", "Method", "Original mechanism evidence and experimental scope. " * 30,
                     1, 1, 1, 1, 1, "page1", "test", "test")


def test_fusion_preserves_anchor_and_checks_quotes_without_labels():
    class Calls:
        def call(self, role, system, payload, ceiling):
            assert "qrels" not in payload and "reference" not in payload
            return {"papers": [{"paper_id": p["paper_id"], "utility": 3,
                                "chunk_id": p["passages"][0]["chunk_id"],
                                "quote": p["passages"][0]["text"][:60] if p["paper_id"] != "bad" else "Invented quote " * 4}
                               for p in payload["papers"]]}
    selected, diagnostic = fuse_candidates("mechanism", "question", [hit("old")], [hit("new"), hit("bad")], Calls())
    assert selected[0].paper_id == "old"
    assert len({h.paper_id for h in selected}) == 3
    assert len(diagnostic["rejected_assessments"]) == 1
    assert not diagnostic["fallback"]


def test_invalid_rerank_falls_back_and_exposes_failure():
    class Calls:
        def call(self, *args):
            return {"papers": []}
    selected, diagnostic = fuse_candidates("query", "question", [hit("old")], [hit("new")], Calls())
    assert selected[0].paper_id == "old"
    assert diagnostic["fallback"] and len(diagnostic["errors"]) == 2


def test_expansion_memory_rejects_changed_text_and_future_papers(monkeypatch):
    from researchops.knowledge import fusion_retrieval as module
    class Store:
        def get_chunk(self, chunk_id):
            pid = chunk_id[1:]
            return {"paper_id": pid, "text": "corrected" if pid == "stale" else hit(pid).text}
    class CPU:
        embedding = reranker = None
        def __init__(self, *args, **kwargs):
            pass
        def search(self, *args, **kwargs):
            return [hit("active")]
    class Calls:
        def call(self, role, system, payload, ceiling):
            assert [p["paper_id"] for p in payload["papers"]] == ["active"]
            return {"papers": [{"paper_id": "active", "utility": 2, "chunk_id": "cactive",
                                "quote": hit("active").text[:50]}]}
    monkeypatch.setattr(module, "EvidenceRetriever", CPU)
    monkeypatch.setattr(module, "HybridRetriever", CPU)
    retriever = module.FusionRetriever(Store(), Calls())
    selected = retriever.search("query", paper_ids={"active", "stale"},
                                previous_hits=[hit("stale"), hit("future")])
    assert {h.paper_id for h in selected} == {"active"}


def test_fusion_packet_keeps_original_mechanism_context_under_ceiling():
    class Calls:
        def call(self, *args):
            return {"cards": [], "gaps": []}
    selected = [hit(str(i)) for i in range(20)]
    packet = build_fusion_packet("question", {}, selected, Calls())
    assert sum(len(p["text"]) for p in packet["passages"]) <= 10000
    assert packet["passages"][0]["text"] == selected[0].text[:1100]


def test_invalid_citation_is_repaired_even_when_model_says_no_revision():
    class Calls:
        def call(self, role, system, payload, ceiling):
            if role == "question_coverage_contract_v1":
                return {"requirements": [{"id": "C1"}, {"id": "C2"}]}
            if role == "operational_study_blueprint_v1":
                return {"hypothesis": "test"}
            if role.startswith("coverage_evidence_check"):
                return {"needs_revision": False, "coverage": [{"id": "C1", "covered": True}, {"id": "C2", "covered": True}],
                        "facts": [{"index": 0, "supported": True}], "design_issues": []}
            return {"facts": [{"claim": "mechanism", "citations": ["missing" if role == "coverage_balanced_answer_v1" else "c1"]}],
                    "design": "controlled proposal", "limitations": []}
    result = improve_fusion_answer("question", {}, {"passages": [{"chunk_id": "c1", "text": "source"}]}, Calls(), "answer")
    assert result["mechanical_checks"][0]["invalid_fact_citations"] == [0]
    assert result["final_invalid_citation_facts"] == []

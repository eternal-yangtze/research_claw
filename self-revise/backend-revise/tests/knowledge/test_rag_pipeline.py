from pathlib import Path

import pytest

from researchclaw.knowledge.rag_pipeline import (
    _clean_lines, quality_grade, CleanedPage, _parent_child_chunks, repair_mojibake,
)
from researchclaw.knowledge.rag_retrieval import PublishedRAGRetriever
from researchclaw.application.cli import _apply_source_exclusions, build_parser


def test_cleaning_quality_and_parent_child_are_reproducible():
    lines = ["Journal Header", "1", "A bounded-error method improves compression.",
             "Journal Header", "2", "The guarantee holds for every segment.",
             "Journal Header"]
    cleaned = _clean_lines(lines)
    assert "Journal Header" not in cleaned
    assert "1" not in cleaned and "2" not in cleaned
    pages = [CleanedPage(1, " ".join(cleaned), {"text_chars": 220, "replacement_chars": 0,
                                                "ocr_required": False, "page_quality": "A"})]
    grade, detail = quality_grade(pages)
    assert grade == "A" and detail["pages"] == 1
    first = _parent_child_chunks("paper", pages, grade, "time-series")
    second = _parent_child_chunks("paper", pages, grade, "time-series")
    assert first == second
    assert [node.node_type for node in first].count("parent") == 1
    children = [node for node in first if node.node_type == "child"]
    assert children and all(node.parent_chunk_id for node in children)
    assert all(node.provenance["domain"] == "time-series" for node in first)


def test_quality_does_not_treat_ocr_flag_as_automatic_rejection():
    pages = [CleanedPage(1, "recognised text", {"text_chars": 100, "replacement_chars": 0,
                                                "ocr_required": True, "page_quality": "B"}),
             CleanedPage(2, "reliable text", {"text_chars": 300, "replacement_chars": 0,
                                              "ocr_required": False, "page_quality": "A"}),
             CleanedPage(3, "reliable text", {"text_chars": 300, "replacement_chars": 0,
                                              "ocr_required": False, "page_quality": "A"}),
             CleanedPage(4, "reliable text", {"text_chars": 300, "replacement_chars": 0,
                                              "ocr_required": False, "page_quality": "A"})]
    grade, _ = quality_grade(pages)
    assert grade == "B"


def test_repairs_common_mojibake_and_bounds_parent_context():
    assert repair_mojibake("MuÃ±iz â€“ method") == "Muñiz – method"
    assert repair_mojibake("Abstractâ€”method in Xiâ€™an Δ") == "Abstract—method in Xi’an Δ"
    retriever = object.__new__(PublishedRAGRetriever)
    retriever.parent_word_budget = 100
    parent = " ".join([f"before{i}" for i in range(150)] + ["matched", "evidence"]
                      + [f"after{i}" for i in range(150)])
    focused = retriever._focused_parent(parent, "matched evidence")
    assert len(focused.split()) == 100
    assert "matched evidence" in focused


def test_rag_build_accepts_audited_exact_path_exclusions():
    args = build_parser().parse_args([
        "rag-build", "--source", "time-series=input", "--exclude-path", "irrelevant.pdf",
    ])
    assert args.exclude_path == ["irrelevant.pdf"]


def test_rag_build_accepts_explicit_base_revision_for_incremental_reuse():
    args = build_parser().parse_args([
        "rag-build", "--source", "time-series=input",
        "--corpus-version", "v2", "--base-corpus-version", "v1",
    ])
    assert args.corpus_version == "v2"
    assert args.base_corpus_version == "v1"


def test_rag_build_exclusion_also_blocks_renamed_identical_copy(tmp_path):
    reviewed = tmp_path / "reviewed-out.pdf"
    renamed = tmp_path / "renamed-copy.pdf"
    allowed = tmp_path / "allowed.pdf"
    reviewed.write_bytes(b"off-topic paper")
    renamed.write_bytes(reviewed.read_bytes())
    allowed.write_bytes(b"time-series paper")

    accepted, excluded = _apply_source_exclusions(
        [("time-series", reviewed), ("time-series", renamed), ("time-series", allowed)],
        {reviewed},
    )

    assert accepted == [("time-series", allowed)]
    assert set(excluded) == {str(reviewed), str(renamed)}


def test_published_retriever_defaults_to_cross_paper_diversity():
    retriever = object.__new__(PublishedRAGRetriever)
    retriever.max_hits_per_paper = 1
    assert retriever.max_hits_per_paper == 1


def test_coverage_retrieval_keeps_two_pages_and_original_query():
    from types import SimpleNamespace
    from researchclaw.knowledge.research_rag import ResearchRAG
    from researchclaw.knowledge.rag_retrieval import RAGHit
    def hit(i, paper):
        return RAGHit(str(i), 'parent'+str(i), paper, paper, i, 'evidence', 'context', 1, 1, .03, 1., 'A')
    class Retriever:
        reranker = SimpleNamespace(score=lambda query, docs: [0.] * len(docs))
        def __init__(self): self.queries=[]
        def search(self, query, **kwargs):
            self.queries.append(query)
            return [hit(1,'a'), hit(2,'a'), hit(3,'a'), hit(4,'b')]
    retriever=Retriever()
    rag=ResearchRAG(retriever,model=object())
    hits=rag.retrieve({'rewritten_query':'English query','original_query':'原问题','alternatives':[]},
                      limit=4,policy='coverage')
    assert retriever.queries == ['English query','原问题']
    assert [(h.paper_id,h.page) for h in hits] == [('a',1),('a',2),('b',4)]


def test_coverage_context_never_drops_the_retrieved_child():
    from researchclaw.knowledge.research_rag import ResearchRAG
    from researchclaw.knowledge.rag_retrieval import RAGHit
    rag=ResearchRAG(object(),model=object())
    hit=RAGHit('c','p','paper','title',3,'necessary error condition','truncated unrelated window',1,1,.03,1.,'A')
    row=rag.contexts([hit],structured=False,preserve_child=True)[0]
    assert 'necessary error condition' in row['parent_text']
    assert row['child_restored']


def test_windowed_reranker_scores_evidence_beyond_first_model_window():
    from types import SimpleNamespace
    from researchclaw.knowledge.retrieval import CrossEncoderReranker
    reranker=object.__new__(CrossEncoderReranker)
    tokenizer=SimpleNamespace(encode=lambda text, **kwargs: text.split(),
                              decode=lambda tokens, **kwargs: ' '.join(tokens))
    reranker._model=SimpleNamespace(tokenizer=tokenizer)
    reranker.score=lambda query, documents: [float('critical-condition' in d) for d in documents]
    long=' '.join(['filler']*600+['critical-condition'])
    assert reranker.score_windowed('condition', [long,'unrelated']) == [1.,0.]

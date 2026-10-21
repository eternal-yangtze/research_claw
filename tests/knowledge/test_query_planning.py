import pytest

from researchops.knowledge.query_planning import plan_retrieval_query


def test_query_planning_retains_original_question_and_has_no_retrieval_labels():
    class Calls:
        def call(self, role, system, payload, output_tokens):
            assert set(payload) == {"question", "idea"}
            assert "rank=4" in payload["question"]
            return {"retrieval_query": "low rank vision transformer adaptation rank four",
                    "preserved_constraints": ["rank=4", "validation split only"]}
    question = "保持rank=4，使用验证集选参，寻找视觉适配证据"
    result = plan_retrieval_query(question, "vision adaptation", Calls())
    assert result["original_question"] == question
    assert result["preserved_constraints"] == ["rank=4", "validation split only"]


def test_query_planning_fails_on_missing_or_unbounded_query():
    class Calls:
        def call(self, *args):
            return {"retrieval_query": "x" * 900, "preserved_constraints": []}
    with pytest.raises(ValueError):
        plan_retrieval_query("question", "idea", Calls())

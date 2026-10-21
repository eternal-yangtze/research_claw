---
name: evidence-retrieve
description: "Retrieve traceable passages and answer a research question from the EXISTING knowledge base, with exact chunk citations and formula quality flags. Use for 查已有知识库、RAG检索、引用取证、文献中的机制问题. Do not download papers or design a full experiment; route those goals to literature-acquire or study-plan."
---

# Retrieve evidence

1. Preserve the user's actual question and named mechanisms. Read the current schema from load_skill.
2. Invoke `evidence-retrieve` with the loaded hash and `arguments={"question": <question>, "limit": <1..10>}`. Omit limit to use the host default.
3. Use returned excerpts and stable citations. Distinguish extractive relevance from proof of a scientific claim; do not infer superiority from a retrieved method description.
4. Report the evidence artifact and source locations. When no admissible evidence exists, say evidence is insufficient; do not invent citations or automatically start acquisition.
5. Preserve formula policy. Withheld formula nodes must not support facts; source-recovered formulas remain inferred silver evidence requiring original-page review.

Output contract: `result.output` contains the extractive `answer`, `citations`, `verified`, `unsupported_claims`, and `evidence_artifact`. Verification checks locators and source excerpts, not independent semantic correctness. This skill uses the mature local hybrid retrieval path; optional fusion planning belongs to study-plan.

from __future__ import annotations

import json
from pathlib import Path

import pytest

from researchops.core.artifacts import file_hash
from researchops.core.models import Paper
from researchops.core.storage import ResearchStore
from researchops.papers.ingestion import DocumentIngestor, IngestResult
from researchops.papers.validation.automatic import automatic_formula_check, formula_evidence_policy
from researchops.papers.validation.formulas import compare_latex
from researchops.papers.validation.macros import collect_macros, expand_macros
from researchops.papers.validation.pipeline import decide_formula
from researchops.papers.validation.sources import source_formulas


def registry(text):
    return collect_macros([{"member": "main.tex", "text": text}])


def references(text):
    return source_formulas([{"member": "main.tex", "text": text, "content_sha256": "test"}])


def test_nested_positional_optional_macros_and_operators():
    reg = registry(r"\newcommand{\vect}[1]{\mathbf{#1}}"
                   r"\newcommand{\inner}[2][x]{\vect{#1}^{#2}}"
                   r"\def\pair#1#2{\langle #1,#2\rangle}"
                   r"\DeclareMathOperator{\MyOp}{rank}")
    result = expand_macros(r"\pair{\inner{2}}{\inner[y]{3}}+\MyOp(A)", reg)
    assert result["complete"]
    assert compare_latex(result["latex"], r"\langle\mathbf{x}^2,\mathbf{y}^3\rangle+\operatorname{rank}(A)")["normalized_exact_match"]
    refs = references(r"\newcommand{\v}[1]{\mathbf{#1}}\[\v{x}_i=1\]")
    assert decide_formula(r"\mathbf{x}_i=1", refs, {"status": "matched_version"})["status"] == "accepted"
    # Source definitions must not be applied to arbitrary OCR commands.
    assert decide_formula(r"\v{x}_i=1", refs, {"status": "matched_version"})["status"] != "accepted"


@pytest.mark.parametrize("declarations,formula", [
    (r"\def\a{\b}\def\b{\a}", r"\a"),
    (r"\newcommand{\a}{\input{secret}}", r"\a"),
    (r"\def\a{x}\def\a{y}", r"\a"),
    (r"{\def\a{x}}", r"\a"),
    (r"\begingroup\def\a{x}\endgroup", r"\a"),
    (r"\iftrue\def\a{x}\fi", r"\a"),
    (r"\edef\a{x}", r"\a"),
    (r"\def\a#1,#2{#1+#2}", r"\a{x}{y}"),
])
def test_unsafe_ambiguous_dynamic_and_recursive_macros_are_unresolved(declarations, formula):
    result = expand_macros(formula, registry(declarations))
    assert not result["complete"]
    assert result["errors"]


def test_expansion_limits_and_control_word_boundaries():
    reg = registry(r"\def\a{\alpha}")
    assert compare_latex(expand_macros(r"\a x", reg)["latex"], r"\alpha x")["normalized_exact_match"]
    assert not expand_macros(r"\a", reg, max_steps=0)["complete"]
    assert not expand_macros("x" * 30, reg, max_chars=10)["complete"]


def test_vectors_keep_semantics_and_source_recovery_is_separate():
    refs = references(r"\[\Delta\mathbf{W}=\mathbf{a}_i\mathbf{b}_i^\top\]")
    recovered = decide_formula(r"\Delta W=a_i b_i^\top", refs, {"status": "matched_version"})
    assert recovered["status"] == "source_recovered"
    assert r"\mathbf" in recovered["selected_latex"]
    assert not recovered["safe_as_sole_reproduction_evidence"]
    assert not compare_latex(r"\mathbf{x}", "x")["normalized_exact_match"]
    assert compare_latex("𝐖", r"\mathbf{W}")["normalized_exact_match"]
    assert compare_latex(r"\bm{x}", r"\boldsymbol{x}")["normalized_exact_match"]
    assert compare_latex("x⃗", r"\vec{x}")["normalized_exact_match"]
    assert not compare_latex(r"\vec{x}", r"\mathbf{x}")["normalized_exact_match"]
    assert decide_formula(r"\Delta W=a^i b_i^\top", refs, {"status": "matched_version"})["status"] != "source_recovered"
    assert decide_formula(r"\Delta W=a_i b_i^\top", refs, {"status": "unverified"})["status"] == "unverified"
    ambiguous = references(r"\[\mathbf{x}=y\]\[x=\mathbf{y}\]")
    assert decide_formula("x=y", ambiguous, {"status": "matched_version"})["status"] != "source_recovered"


def real_artifacts(tmp_path):
    fitz = pytest.importorskip("pymupdf")
    pdf = tmp_path / "paper.pdf"
    with fitz.open() as document:
        page = document.new_page(width=300, height=200)
        page.insert_text((20, 20), "arXiv:2203.12119v2")
        page.insert_text((20, 70), "x = 1")
        document.save(pdf)
    source = tmp_path / "main.tex"
    source.write_text(r"\newcommand{\v}[1]{\mathbf{#1}}\[\v{x}=1\]", encoding="utf-8")
    receipt = source.with_name(source.name + ".receipt.json")
    receipt.write_text(json.dumps({"url": "https://export.arxiv.org/e-print/2203.12119v2",
            "arxiv_id": "2203.12119", "source_version": "v2", "sha256": file_hash(source)}), encoding="utf-8")
    parser = tmp_path / "parser.json"
    payload = {"pdf_sha256": file_hash(pdf), "parser": "docling", "document": {"texts": [
        {"label": "formula", "self_ref": "#/texts/0", "text": "x=1", "prov": [
            {"page_no": 1, "bbox": {"l": 15, "t": 50, "r": 100, "b": 80, "coord_origin": "TOPLEFT"}}]}]}}
    parser.write_text(json.dumps(payload), encoding="utf-8")
    return pdf, source, parser, payload


def test_automatic_cache_checks_receipt_parser_and_crop_integrity(tmp_path):
    pdf, source, parser, payload = real_artifacts(tmp_path)
    first = automatic_formula_check(pdf, parser, tmp_path / "checks", source_package=source)
    assert first["summary"]["source_recovered"] == 1
    assert automatic_formula_check(pdf, parser, tmp_path / "checks", source_package=source)["cache_hit"]
    report = json.loads(Path(first["report_path"]).read_text(encoding="utf-8"))
    crop = Path(report["regions"][0]["locations"][0]["crop_path"])
    crop.write_bytes(b"broken")
    assert not automatic_formula_check(pdf, parser, tmp_path / "checks", source_package=source)["cache_hit"]
    adapter = next((tmp_path / "checks").glob("*/parser_input.json"))
    adapter.write_text("{}", encoding="utf-8")
    assert not automatic_formula_check(pdf, parser, tmp_path / "checks", source_package=source)["cache_hit"]
    payload["pdf_sha256"] = "wrong"
    parser.write_text(json.dumps(payload), encoding="utf-8")
    assert automatic_formula_check(pdf, parser, tmp_path / "checks", source_package=source)["status"] == "validation_failed"


def test_standard_ingestion_attaches_formula_report_and_passages_preserve_policy(tmp_path, monkeypatch):
    from researchops.core.models import Chunk
    from researchops.knowledge.passages import materialize_passages
    pdf, source, parser, payload = real_artifacts(tmp_path)
    store = ResearchStore(tmp_path / "raw.db")
    ingestor = DocumentIngestor(store)
    seed = Paper("p", "Formula paper", file_path=str(pdf), content_hash=file_hash(pdf), source_archive_path=str(source))
    def cached_conversion(path, paper):
        store.upsert_paper(paper)
        nodes = [Chunk("c", "p", "formula", "x=1", node_type="formula", provenance={"self_ref": "#/texts/0"}),
                 Chunk("p2", "p", "text", "Unrelated method context")]
        store.replace_chunks("p", nodes)
        sidecar = ingestor._write_pdf_sidecar("p", pdf, "docling", file_hash(pdf), nodes, payload["document"])
        return IngestResult("p", "Formula paper", 2, "docling", "review_required", str(pdf), sidecar)
    monkeypatch.setenv("RESEARCHOPS_PDF_PARSER", "docling")
    monkeypatch.setattr(ingestor, "_ingest_pdf_docling", cached_conversion)
    result = ingestor.ingest_file(pdf, seed)
    assert result.formula_validation["summary"]["source_recovered"] == 1
    assert formula_evidence_policy(store.get_chunk("c"))["eligible_for_planning"]
    target = ResearchStore(tmp_path / "passages.db")
    materialize_passages(store, target, ["p"])
    formula = next(r for r in target.all_chunks() if r["node_type"] == "formula")
    assert formula_evidence_policy(formula)["status"] == "source_recovered"
    assert formula_evidence_policy(formula)["eligible_for_planning"]
    Path(result.formula_validation["report_path"]).write_text("{}", encoding="utf-8")
    assert not formula_evidence_policy(formula)["eligible_for_planning"]


def test_reproduction_gate_rejects_formula_only_or_withheld_facts(tmp_path):
    from researchops.core.models import Chunk
    from researchops.papers.validation.automatic import formula_reproduction_gate
    store = ResearchStore(tmp_path / "raw.db")
    store.upsert_paper(Paper("p", "P"))
    report = tmp_path / "report.json"
    report.write_text("{}", encoding="utf-8")
    store.replace_chunks("p", [Chunk("f", "p", "formula", "x=1", node_type="formula",
        provenance={"formula_validation": {"status": "accepted", "report_path": str(report), "report_sha256": file_hash(report)}}),
        Chunk("t", "p", "method", "prose"), Chunk("u", "p", "formula", "y=?", node_type="formula")])
    evidence = [{"chunk_id": "f"}, {"chunk_id": "t"}]
    assert not formula_reproduction_gate({"baseline_citations": ["f"]}, evidence, store)["passed"]
    assert formula_reproduction_gate({"baseline_citations": ["f", "t"]}, evidence, store)["passed"]
    assert not formula_reproduction_gate({"baseline_citations": ["f", "t"], "evidence_claims": [
        {"kind": "paper_fact", "citations": ["f"]}]}, evidence, store)["passed"]
    assert not formula_reproduction_gate({"baseline_citations": ["u", "t"]}, evidence, store)["passed"]


def test_region_cap_defers_formulas_without_claiming_absence(tmp_path):
    from researchops.papers.validation.pipeline import validate_formulas
    pdf, source, parser, payload = real_artifacts(tmp_path)
    payload["document"]["texts"] *= 3
    parser.write_text(json.dumps(payload), encoding="utf-8")
    result = validate_formulas(pdf, tmp_path / "runs", docling_json=parser, max_regions=1)
    assert result["summary"]["deferred_regions"] == 2
    assert result["summary"]["detected_formula_regions"] == 3
    assert result["summary"]["recognition_accuracy"] is None

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest

from researchops.core.artifacts import file_hash
from researchops.papers.validation.formulas import compare_latex, normalize_latex, strip_comments
from researchops.papers.validation.pipeline import decide_formula, top_left_bbox, validate_formulas
from researchops.papers.validation.sources import read_tex_package, source_formulas, source_identity


def test_formula_normalization_preserves_scientific_differences() -> None:
    assert compare_latex(r"\left( x_i + \dfrac{a}{b}\right)", r"(x_{i}+\frac{a}{b})")["normalized_exact_match"]
    assert compare_latex(r"W _ { T } ^ { ( \tau ) }", r"W^{(\tau)}_T")["normalized_exact_match"]
    for left, right in [(r"x_i", r"x^i"), (r"a-b", r"a+b"),
                        (r"\alpha y", r"\alphay"), (r"\text{a b}", r"\text{ab}")]:
        assert not compare_latex(left, right)["normalized_exact_match"]
    assert not compare_latex(r"\frac{a}{b", r"\frac{a}{b}")["left_structure"]["balanced"]
    assert strip_comments("x% comment\ny") == "x\ny"
    assert normalize_latex(r"x\%") != normalize_latex("x")


def test_static_source_reader_rejects_unsafe_and_oversized_members(tmp_path: Path) -> None:
    path = tmp_path / "source.tar"
    for name, kind in [("../main.tex", None), ("C:/main.tex", None), ("main.tex", tarfile.SYMTYPE)]:
        with tarfile.open(path, "w") as archive:
            member = tarfile.TarInfo(name)
            member.size = 1
            if kind:
                member.type = kind
                member.linkname = "other.tex"
            archive.addfile(member, io.BytesIO(b"x"))
        with pytest.raises(ValueError, match="unsafe"):
            read_tex_package(path)
    path = tmp_path / "main.tex"
    path.write_text("x" * 100, encoding="utf-8")
    with pytest.raises(ValueError, match="limit"):
        read_tex_package(path, max_member_bytes=30)


def test_source_extraction_and_no_false_acceptance(tmp_path: Path) -> None:
    path = tmp_path / "main.tex"
    path.write_text(r"\newcommand{\mymacro}{x}\begin{equation}x_i=1\label{eq:x}\end{equation}"
                    r"\[\mymacro^2\]", encoding="utf-8")
    refs = source_formulas(read_tex_package(path))
    assert len(refs) == 2
    assert refs[0]["labels"] == ["eq:x"]
    matched = {"status": "matched_version"}
    assert decide_formula("x_{i}=1", refs, matched)["status"] == "accepted"
    assert decide_formula("x_{i}=1", refs, {"status": "unverified"})["status"] == "unverified"
    assert decide_formula(r"\mymacro^2", refs, matched)["status"] == "unverified"
    assert decide_formula("x_{i}=2", refs, matched)["status"] == "disputed"
    assert not decide_formula("x_{i}=1", refs, matched)["safe_as_sole_reproduction_evidence"]


def test_identity_requires_pdf_watermark_version_and_receipt_hash(tmp_path: Path) -> None:
    path = tmp_path / "main.tex"
    path.write_text("x", encoding="utf-8")
    receipt = {"url": "https://export.arxiv.org/e-print/2203.12119v2", "arxiv_id": "2203.12119",
               "source_version": "v2", "sha256": file_hash(path)}
    assert source_identity("arXiv:2203.12119v2", path, receipt)["status"] == "matched_version"
    for text, proof in [("arXiv:2203.12119v1", receipt), ("no watermark", receipt),
                        ("arXiv:2203.12119v2", {**receipt, "sha256": "bad"})]:
        assert source_identity(text, path, proof)["status"] == "unverified"


def test_bottom_left_coordinates_and_unknown_origin() -> None:
    assert top_left_bbox({"l": 10, "t": 90, "r": 30, "b": 60, "coord_origin": "BOTTOMLEFT"}, 100, 100) == [10, 10, 30, 40]
    with pytest.raises(ValueError, match="origin"):
        top_left_bbox({"l": 10, "t": 10, "r": 30, "b": 40}, 100, 100)


def test_pipeline_crops_original_and_preserves_runs(tmp_path: Path) -> None:
    fitz = pytest.importorskip("pymupdf")
    pdf = tmp_path / "paper.pdf"
    with fitz.open() as document:
        page = document.new_page(width=200, height=200)
        page.insert_text((20, 40), "x = 1")
        document.save(pdf)
    parser = tmp_path / "parser.json"
    payload = {"pdf_sha256": file_hash(pdf), "document": {"texts": [{"self_ref": "#/texts/0", "label": "formula",
               "text": "x=1", "prov": [{"page_no": 1, "bbox": {"l": 10, "t": 180, "r": 100, "b": 150, "coord_origin": "BOTTOMLEFT"}}]}]}}
    parser.write_text(json.dumps(payload), encoding="utf-8")
    first = validate_formulas(pdf, tmp_path / "runs", docling_json=parser)
    second = validate_formulas(pdf, tmp_path / "runs", docling_json=parser)
    assert first["report_path"] != second["report_path"]
    report = json.loads(Path(first["report_path"]).read_text(encoding="utf-8"))
    assert report["summary"]["regions_with_crops"] == 1
    assert report["summary"]["recognition_accuracy"] is None
    assert report["regions"][0]["decision"]["status"] == "unverified"
    assert Path(report["regions"][0]["locations"][0]["crop_path"]).is_file()
    payload["pdf_sha256"] = "wrong"
    parser.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        validate_formulas(pdf, tmp_path / "runs", docling_json=parser)

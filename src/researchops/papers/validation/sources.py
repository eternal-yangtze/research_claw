"""Read bounded allowlisted TeX statically. Never extract or compile packages."""
from __future__ import annotations

import gzip
import hashlib
import re
import tarfile
from pathlib import Path, PurePosixPath

from researchops.core.artifacts import canonical_hash, file_hash
from researchops.papers.connectors import normalize_arxiv_id
from .formulas import normalize_latex, strip_comments
from .macros import collect_macros, expand_macros


ENVIRONMENT = re.compile(r"\\begin\{(equation\*?|align\*?|gather\*?|multline\*?|flalign\*?|eqnarray\*?)\}(.*?)\\end\{\1\}|\\\[(.*?)\\\]|\$\$(.*?)\$\$", re.S)


def read_tex_package(path: Path, *, max_total_bytes: int = 20_000_000,
                     max_member_bytes: int = 2_000_000, max_members: int = 4000) -> list[dict]:
    if path.stat().st_size > 100_000_000:
        raise ValueError("source package exceeds compressed size limit")
    files = []
    total = 0
    archive_bytes = 0

    def add(name: str, data: bytes) -> None:
        nonlocal total
        total += len(data)
        if len(data) > max_member_bytes or total > max_total_bytes:
            raise ValueError("source package exceeds bounded text limit")
        files.append({"member": name, "text": data.decode("utf-8", errors="replace"),
                      "content_sha256": hashlib.sha256(data).hexdigest()})

    try:
        archive = tarfile.open(path, mode="r:*")
    except tarfile.ReadError:
        if path.name.endswith(".gz"):
            with gzip.open(path, "rb") as stream:
                add(path.name, stream.read(max_member_bytes + 1))
        elif path.suffix.lower() == ".tex":
            with path.open("rb") as stream:
                add(path.name, stream.read(max_member_bytes + 1))
        else:
            raise ValueError("unsupported static source format")
        return files
    with archive:
        for ordinal, member in enumerate(archive):
            if ordinal >= max_members:
                raise ValueError("source package member limit exceeded")
            name = member.name.replace("\\", "/")
            parts = PurePosixPath(name)
            if parts.is_absolute() or ".." in parts.parts or ":" in name or member.issym() or member.islnk():
                raise ValueError("unsafe source archive member")
            archive_bytes += member.size
            if member.size < 0 or archive_bytes > 100_000_000:
                raise ValueError("uncompressed source archive size limit exceeded")
            if not member.isfile() or parts.suffix.lower() not in {".tex", ".ltx", ".sty", ".cls"}:
                continue
            if member.size > max_member_bytes:
                raise ValueError("source member exceeds text limit")
            with archive.extractfile(member) as stream:
                add(name, stream.read(max_member_bytes + 1))
    return files


def source_formulas(files: list[dict]) -> list[dict]:
    registry = collect_macros(files)
    formulas = []
    for item in files:
        if Path(item["member"]).suffix.lower() in {".sty", ".cls"}:
            continue  # inspect macro declarations, not equations in package code
        text = strip_comments(item["text"])
        for ordinal, match in enumerate(ENVIRONMENT.finditer(text)):
            if len(formulas) >= 1500:
                raise ValueError("static source formula limit exceeded")
            expression = next(g for g in match.groups()[1:] if g is not None)
            if len(expression) > 20000:
                continue  # oversized source blocks cannot corroborate a candidate
            expansion = expand_macros(expression, registry)
            labels = re.findall(r"\\label\{([^{}]+)\}", expression)
            formulas.append({"member": item["member"], "ordinal": ordinal,
                             "line": text.count("\n", 0, match.start()) + 1,
                             "environment": match[1] or "display_delimiter",
                             "latex": expression.strip(), "labels": labels,
                             "expanded_latex": expansion["latex"], "macro_expansion": expansion,
                             "normalized": normalize_latex(expansion["latex"]),
                             "unexpanded_macros": expansion.get("unexpanded_macros", ["input_limit"]),
                             "member_sha256": item["content_sha256"],
                             "candidate_sha256": canonical_hash(expression.strip())})
    return formulas


def source_identity(pdf_first_page: str, source: Path, receipt: dict | None) -> dict:
    # Read the PDF's own arXiv watermark instead of trusting an external filename.
    pdf_ids = re.findall(r"arXiv:\s*((?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})v\d+)", pdf_first_page, re.I)
    pdf_ids = sorted(set(pdf_ids))
    proof = receipt or {}
    source_id, source_version = normalize_arxiv_id(proof.get("url"))
    record_id, record_version = normalize_arxiv_id(str(proof.get("arxiv_id", "")) + str(proof.get("source_version") or ""))
    digest_matches = proof.get("sha256") == file_hash(source)
    expected = source_id + source_version if source_id and source_version else None
    matched = (len(pdf_ids) == 1 and pdf_ids[0] == expected and digest_matches
               and source_id == record_id and source_version == record_version)
    return {"status": "matched_version" if matched else "unverified",
            "pdf_watermark_ids": pdf_ids, "source_requested_id": expected,
            "source_hash_matches_receipt": digest_matches,
            "basis": "PDF arXiv watermark + versioned source URL + artifact hash",
            "reference_type": "source_derived_silver", "human_gold": False}

"""Temporary silver primary reference with an explicit reserved gold interface."""
from __future__ import annotations

import json
from pathlib import Path

from researchops.core.artifacts import file_hash


def load_reference(*, silver: Path, gold: Path | None = None, source="silver") -> tuple[dict, dict]:
    if source not in {"silver", "gold"}:
        raise ValueError("reference source must be silver or gold")
    target = silver if source == "silver" else gold
    if target is None:
        raise ValueError("Gold reference interface is reserved; no gold supplied")
    return json.loads(target.read_text(encoding="utf-8")), {
        "source": source, "role": "temporary_primary_reference" if source == "silver" else "primary_reference",
        "path": str(target.resolve()), "sha256": file_hash(target), "gold_interface": str(gold) if gold else None,
        "human_gold": source == "gold"}

"""CLI for immutable formula validation artifacts (no old DB/index mutation)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from researchops.papers.validation import validate_formulas


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--docling-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-package")
    parser.add_argument("--source-receipt")
    args = parser.parse_args()
    result = validate_formulas(args.pdf, args.output, docling_json=args.docling_json,
                               source_package=args.source_package, source_receipt=args.source_receipt)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

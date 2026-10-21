"""Locally re-recognize bounded disputed regions; preserve every candidate."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from researchops.papers.validation.recognition import refine_disputed_regions

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-regions", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    print(json.dumps(refine_disputed_regions([Path(p) for p in args.report], Path(args.output),
                                            max_regions=args.max_regions, max_new_tokens=args.max_new_tokens),
                     ensure_ascii=False, indent=2))

"""Fetch one explicit arXiv source version and preserve its receipt."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()) / "src"))
from researchclaw.papers.connectors import download_arxiv_source

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--arxiv-id", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(download_arxiv_source(args.arxiv_id, args.output), ensure_ascii=False, indent=2))

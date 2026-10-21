#!/usr/bin/env python3
"""Verify that ResearchOps can reach the LLM configured in its .env file."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from researchops.config import load_project_env  # noqa: E402
from researchops.runtime import OpenAICompatibleModel  # noqa: E402


SUCCESS_TOKEN = "RESEARCHOPS_CONNECTION_OK"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test the LLM connection configured in researchops_mvp/.env."
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Timeout in seconds for each HTTP attempt (default: 30).",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Maximum HTTP attempts for retryable errors (default: 3).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print one JSON result instead of human-readable output.",
    )
    return parser.parse_args()


def safe_endpoint(value: str) -> str:
    """Return an endpoint suitable for logs, without credentials or query data."""

    parsed = urlsplit(value)
    host = parsed.hostname or ""
    if parsed.port:
        host = f"{host}:{parsed.port}"
    return urlunsplit((parsed.scheme, host, parsed.path, "", ""))


def emit(result: dict[str, object], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(result, ensure_ascii=False))
        return

    labels = {
        "status": "Status",
        "env_file": "Env file",
        "endpoint": "Endpoint",
        "model": "Model",
        "latency_ms": "Latency",
        "response": "Response",
        "error": "Error",
    }
    for key in ("status", "env_file", "endpoint", "model", "latency_ms", "response", "error"):
        if key not in result:
            continue
        value = result[key]
        if key == "latency_ms":
            value = f"{value} ms"
        print(f"{labels[key]}: {value}")


def main() -> int:
    args = parse_args()
    if args.timeout <= 0:
        print("Error: --timeout must be greater than zero.", file=sys.stderr)
        return 2
    if args.max_attempts < 1:
        print("Error: --max-attempts must be at least one.", file=sys.stderr)
        return 2

    expected_env = (PROJECT_ROOT / ".env").resolve()
    if not expected_env.is_file():
        emit(
            {"status": "CONFIG_ERROR", "error": f"Missing env file: {expected_env}"},
            as_json=args.json,
        )
        return 2

    # load_project_env checks the current directory first. Pinning it to the
    # project root ensures this test never picks up an unrelated parent .env.
    os.chdir(PROJECT_ROOT)
    loaded_env = load_project_env(override=True)
    if loaded_env is None or loaded_env.resolve() != expected_env:
        emit(
            {"status": "CONFIG_ERROR", "error": "Failed to load the project .env file."},
            as_json=args.json,
        )
        return 2

    try:
        model = OpenAICompatibleModel(timeout=args.timeout, max_attempts=args.max_attempts)
    except (TypeError, ValueError) as error:
        emit(
            {
                "status": "CONFIG_ERROR",
                "env_file": str(expected_env),
                "error": str(error),
            },
            as_json=args.json,
        )
        return 2

    result: dict[str, object] = {
        "status": "CONNECTING",
        "env_file": str(expected_env),
        "endpoint": safe_endpoint(model.base_url),
        "model": model.model,
    }
    if not args.json:
        emit(result, as_json=False)
        print("Sending a minimal Chat Completions request...")

    started = time.perf_counter()
    try:
        turn = model.complete(
            [
                {
                    "role": "system",
                    "content": f"Reply with exactly {SUCCESS_TOKEN} and nothing else.",
                },
                {"role": "user", "content": "Connection test."},
            ],
            [],
            32,
        )
    except Exception as error:  # The CLI converts provider failures into a stable exit code.
        result["status"] = "FAILED"
        result["latency_ms"] = round((time.perf_counter() - started) * 1000)
        result["error"] = f"{type(error).__name__}: {error}"
        emit(result, as_json=args.json)
        return 1

    response = turn.text.strip()
    result["latency_ms"] = round((time.perf_counter() - started) * 1000)
    result["response"] = response
    if SUCCESS_TOKEN not in response:
        result["status"] = "UNEXPECTED_RESPONSE"
        result["error"] = "The endpoint responded, but did not return the requested token."
        emit(result, as_json=args.json)
        return 3

    result["status"] = "SUCCESS"
    emit(result, as_json=args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

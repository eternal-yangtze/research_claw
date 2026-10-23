"""Summarize real P5/P6 evidence and protocol fault tests without paid calls."""
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from importlib.metadata import version
from pathlib import Path

from researchops.core.artifacts import file_hash, write_json


def inspect_wire(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    messages = [json.loads(row["raw"]) for row in rows]
    methods = [m["method"] for m in messages if "method" in m]
    responses = {m["id"] for m in messages if "id" in m and ("result" in m or "error" in m)}
    requests = {m["id"] for m in messages if "id" in m and "method" in m}
    protocols = [m["result"]["protocolVersion"] for m in messages
                 if isinstance(m.get("result"), dict) and "protocolVersion" in m["result"]]
    tools = [t for m in messages for t in (m.get("result") or {}).get("tools", [])]
    errors = [m for m in messages if "error" in m or (m.get("result") or {}).get("isError")]
    return {"path": str(path.resolve()), "sha256": file_hash(path), "messages": len(messages),
            "methods": methods, "protocol_versions": protocols, "request_ids": sorted(requests, key=str),
            "unanswered_ids": sorted(requests-responses, key=str), "error_responses": len(errors), "tools": tools}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coding-report", type=Path, required=True)
    parser.add_argument("--paper-report", type=Path, required=True)
    parser.add_argument("--pytest-root", type=Path, required=True)
    parser.add_argument("--junit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    coding = json.loads(args.coding_report.read_text(encoding="utf-8"))
    paper = json.loads(args.paper_report.read_text(encoding="utf-8"))
    junit = ET.parse(args.junit).getroot()
    suites = list(junit.iter("testsuite"))
    stats = {key: sum(int(s.get(key, 0)) for s in suites) for key in ("tests", "failures", "errors", "skipped")}
    stats["passed"] = stats["tests"] - stats["failures"] - stats["errors"] - stats["skipped"]
    live_wires = [inspect_wire(Path(r["audit_path"])) for r in paper["micro_agent"]["mcp_reads"]]
    test_wires = [inspect_wire(p) for p in args.pytest_root.rglob("*wire.jsonl")]
    checks = {"coding_real_passed": coding["passed"], "paper_real_passed": paper["passed"],
        "regression_passed": stats["failures"] == stats["errors"] == 0,
        "real_mcp_complete_requests": bool(live_wires) and all(not w["unanswered_ids"] for w in live_wires),
        "actual_handshake": all("initialize" in w["methods"] and "notifications/initialized" in w["methods"] for w in live_wires),
        "actual_catalog_and_call": all("tools/list" in w["methods"] and "tools/call" in w["methods"] for w in live_wires),
        "fault_cancellation_recorded": any("notifications/cancelled" in w["methods"] for w in test_wires),
        "fault_errors_recorded": any(w["error_responses"] >= 4 for w in test_wires),
        "coding_all_attempts_under_5x": coding["aggregate_charged_tokens"] <= coding["aggregate_token_cap"]}
    prior = []
    for p in coding.get("prior_runs", []):
        path = Path(p) / "report.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        prior.append({"path": str(path.resolve()), "sha256": file_hash(path),
                      "budget": report["result"]["harness"]["budget"], "reason": report["result"]["stop_reason"]})
    report = {"passed": all(checks.values()), "checks": checks, "python": sys.version,
        "mcp_sdk_version": version("mcp"), "regression": stats,
        "coding": {"path": str(args.coding_report.resolve()), "sha256": file_hash(args.coding_report),
                   "status": coding["result"]["status"], "budget": coding["result"]["harness"]["budget"],
                   "prior_failures": prior, "aggregate_charged_tokens": coding["aggregate_charged_tokens"]},
        "paper": {"path": str(args.paper_report.resolve()), "sha256": file_hash(args.paper_report),
                  "budget": paper["budget"], "source_sha256": paper["source_sha256"]},
        "live_wire_audits": live_wires, "fault_wire_audits": test_wires,
        "readiness": "P7 may start: export real contexts/answers next. RAGAS scores and joint P8 chain are not yet implemented."}
    write_json(args.output / "report.json", report)
    print(json.dumps({"passed": report["passed"], "checks": checks, "regression": stats,
                      "live_wire_sessions": len(live_wires), "fault_wire_sessions": len(test_wires)}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

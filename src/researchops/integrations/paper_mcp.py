"""Read-only, startup-bound MCP paper service and official-SDK stdio client.

Wire capture wraps the actual server UTF-8 streams, before parsing and after
writing. No transcript is synthesized from client method names or exceptions.
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import sqlite3
import sys
import time
import uuid
from contextlib import asynccontextmanager
from importlib.metadata import version
from pathlib import Path

from researchops.core.artifacts import canonical_hash, file_hash, write_json

TOOL_SCHEMA = {"type": "object", "properties": {"chunk_id": {"type": "string", "minLength": 1}},
               "required": ["chunk_id"], "additionalProperties": False}


class PaperScope:
    """One project-local database and one eligible paper snapshot, pinned at startup."""

    def __init__(self, workspace, database, paper_id, chunk_ids, *, snapshot=None):
        self.workspace, self.database = Path(workspace).resolve(), Path(database).resolve()
        if not self.database.is_relative_to(self.workspace) or not self.database.is_file():
            raise PermissionError("MCP database must be inside the bound workspace")
        if not paper_id or not chunk_ids or any(not isinstance(c, str) or not c for c in chunk_ids):
            raise ValueError("A paper and nonempty eligible chunk IDs are required")
        self.paper_id, self.chunk_ids = paper_id, frozenset(chunk_ids)
        self.project_id = canonical_hash(str(self.workspace))[:24]
        self.snapshot = snapshot
        # Startup validates every allowlisted ID too. The handler checks again.
        self.hashes = {c: canonical_hash(self._read(c)) for c in self.chunk_ids}
        if snapshot is not None and snapshot != canonical_hash(self.hashes):
            raise ValueError("MCP paper snapshot changed")

    def _read(self, chunk_id):
        with sqlite3.connect(f"{self.database.as_uri()}?mode=ro", uri=True) as db:
            db.row_factory = sqlite3.Row
            row = db.execute("SELECT * FROM chunks WHERE chunk_id=?", (chunk_id,)).fetchone()
        if row is None:
            raise KeyError("chunk_not_found")
        if row["paper_id"] != self.paper_id or chunk_id not in self.chunk_ids:
            raise PermissionError("chunk_outside_bound_paper_scope")
        return dict(row)

    def read(self, chunk_id):
        row = self._read(chunk_id)
        if canonical_hash(row) != self.hashes.get(chunk_id):
            raise ValueError("chunk_snapshot_changed")
        return {"paper_id": self.paper_id, "chunk_id": chunk_id, "section": row["section"],
                "text": row["text"], "page": row["page"], "full_text": True,
                "project_id": self.project_id, "transport": "mcp_stdio"}

    def binding(self):
        return {"workspace": str(self.workspace), "database": str(self.database),
                "paper_id": self.paper_id, "chunk_ids": sorted(self.chunk_ids),
                "snapshot": canonical_hash(self.hashes)}


class WireStream:
    def __init__(self, stream, audit, direction):
        self.stream, self.audit, self.direction = stream, Path(audit), direction

    def __aiter__(self):
        return self

    async def __anext__(self):
        line = await self.stream.readline()
        if not line:
            raise StopAsyncIteration
        self.record(line)
        return line

    def record(self, raw):
        with self.audit.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"time": time.time(), "direction": self.direction,
                                     "raw": raw}, ensure_ascii=False) + "\n")

    async def write(self, raw):
        result = await self.stream.write(raw)
        await self.stream.flush()
        self.record(raw)
        return result

    async def flush(self):
        await self.stream.flush()


async def serve(binding_path: Path, audit: Path):
    import anyio
    import mcp_types as types
    from mcp.server import Server
    from mcp.server.stdio import stdio_server

    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    scope = PaperScope(**binding)
    audit.parent.mkdir(parents=True, exist_ok=True)

    async def list_tools(ctx, params):
        return types.ListToolsResult(tools=[types.Tool(name="get_paper_chunk",
            description="Read one eligible chunk from the startup-bound paper snapshot.",
            input_schema=TOOL_SCHEMA, annotations=types.ToolAnnotations(read_only_hint=True,
                destructive_hint=False, idempotent_hint=True, open_world_hint=False))])

    async def call_tool(ctx, params):
        args = params.arguments or {}
        try:
            if params.name != "get_paper_chunk":
                raise ValueError("unknown_tool")
            if set(args) != {"chunk_id"} or not isinstance(args["chunk_id"], str) or not args["chunk_id"]:
                raise ValueError("invalid_arguments: expected only a nonempty chunk_id string")
            result = scope.read(args["chunk_id"])
            return types.CallToolResult(content=[types.TextContent(type="text", text=json.dumps(result, ensure_ascii=False))])
        except (ValueError, KeyError, PermissionError) as error:
            return types.CallToolResult(is_error=True, content=[types.TextContent(type="text",
                text=json.dumps({"error": type(error).__name__, "reason": str(error)}))])

    server = Server("researchops-paper-reader", version="0.3.0", on_list_tools=list_tools, on_call_tool=call_tool)
    # Explicit streams retain raw boundary capture. Business code never prints to stdout.
    stdin = anyio.wrap_file(io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8"))
    stdout = anyio.wrap_file(io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", write_through=True))
    async with stdio_server(WireStream(stdin, audit, "client_to_server"),
                            WireStream(stdout, audit, "server_to_client")) as (read, write):
        await server.run(read, write, server.create_initialization_options())


@asynccontextmanager
async def connect(binding_path, audit_path, *, timeout=10, command=None, args=None):
    """Use SDK transport lifecycle, including bounded subprocess tree cleanup."""
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    params = StdioServerParameters(command=command or sys.executable,
        args=args if args is not None else ["-B", "-m", "researchops.integrations.paper_mcp",
              "--binding", str(binding_path), "--audit", str(audit_path)], env={"PYTHONUTF8": "1"},
        cwd=str(Path(binding_path).resolve().parent))
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write, read_timeout_seconds=timeout) as session:
            initialized = await session.initialize()
            yield session, initialized


class PaperMCPClient:
    """Synchronous executor adapter; each read owns a bounded SDK subprocess."""

    def __init__(self, scope: PaperScope, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.binding = self.root / "binding.json"
        write_json(self.binding, scope.binding())

    def read(self, chunk_id, *, timeout=10, cancellation=None):
        if cancellation is not None and cancellation.is_set():
            raise TimeoutError("MCP read cancelled before dispatch")
        audit = self.root / (uuid.uuid4().hex + ".wire.jsonl")

        async def run():
            import anyio
            result = {}

            async def request():
                async with connect(self.binding, audit, timeout=timeout) as (session, initialized):
                    tools = await session.list_tools()
                    response = await session.call_tool("get_paper_chunk", {"chunk_id": chunk_id})
                    if response.is_error:
                        raise RuntimeError(response.content[0].text)
                    result.update(json.loads(response.content[0].text))
                    result["mcp"] = {"audit_path": str(audit), "protocol_version": initialized.protocol_version,
                                     "sdk_version": version("mcp"), "tools": [t.name for t in tools.tools]}

            with anyio.fail_after(timeout):
                async with anyio.create_task_group() as tg:
                    async def worker():
                        await request()
                        tg.cancel_scope.cancel()

                    async def watch():
                        while True:
                            if cancellation is not None and cancellation.is_set():
                                raise TimeoutError("MCP read cancelled")
                            await anyio.sleep(0.025)

                    tg.start_soon(worker)
                    tg.start_soon(watch)
            return result

        return asyncio.run(run())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(serve(args.binding, args.audit))


if __name__ == "__main__":
    main()

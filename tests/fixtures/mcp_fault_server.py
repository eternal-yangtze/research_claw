"""Protocol-only fault fixture. Never registered in the business tool catalog."""
import asyncio
import io
import os
import sys
from pathlib import Path

import anyio
import mcp_types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server
from researchops.integrations.paper_mcp import WireStream


async def main():
    audit = Path(sys.argv[1])
    audit.with_suffix(".pid").write_text(str(os.getpid()), encoding="utf-8")
    async def call(ctx, params):
        if params.name == "exit":
            os._exit(17)
        await anyio.sleep(30)
        return types.CallToolResult(content=[types.TextContent(type="text", text="late")])
    server = Server("fault-fixture", on_call_tool=call)
    stdin = anyio.wrap_file(io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8"))
    stdout = anyio.wrap_file(io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", write_through=True))
    async with stdio_server(WireStream(stdin, audit, "client_to_server"),
                            WireStream(stdout, audit, "server_to_client")) as (read, write):
        await server.run(read, write, server.create_initialization_options())


asyncio.run(main())

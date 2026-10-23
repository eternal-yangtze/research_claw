import asyncio
import json
import os
import sys
import time
from pathlib import Path

import pytest

from researchops.core.artifacts import write_json
from researchops.core.models import Paper, Chunk
from researchops.core.storage import ResearchStore
from researchops.integrations.paper_mcp import PaperScope, PaperMCPClient, connect


def fixture_scope(tmp_path):
    store = ResearchStore(tmp_path / "corpus.db")
    for p in ("bound", "foreign"):
        store.upsert_paper(Paper(p, p))
        store.replace_chunks(p, [Chunk(p + "_chunk", p, "Methods", "Original source text " + p)])
    return store, PaperScope(tmp_path, store.path, "bound", ["bound_chunk"])


def test_real_sdk_handshake_and_wire_errors(tmp_path):
    _, scope = fixture_scope(tmp_path)
    binding, audit = tmp_path / "binding.json", tmp_path / "wire.jsonl"
    write_json(binding, scope.binding())

    async def run():
        async with connect(binding, audit) as (session, initialized):
            assert initialized.protocol_version
            tools = await session.list_tools()
            assert [t.name for t in tools.tools] == ["get_paper_chunk"]
            ok = await session.call_tool("get_paper_chunk", {"chunk_id": "bound_chunk"})
            assert not ok.is_error and json.loads(ok.content[0].text)["text"] == "Original source text bound"
            for args in ({"chunk_id": "missing"}, {"chunk_id": "foreign_chunk"},
                         {"chunk_id": 1}, {"chunk_id": "bound_chunk", "path": "C:/secret"}):
                result = await session.call_tool("get_paper_chunk", args)
                assert result.is_error
    asyncio.run(run())
    messages = [json.loads(json.loads(l)["raw"]) for l in audit.read_text(encoding="utf-8").splitlines()]
    methods = [m.get("method") for m in messages]
    assert all(m in methods for m in ("initialize", "notifications/initialized", "tools/list", "tools/call"))
    ids = {m["id"] for m in messages if "method" in m and "id" in m}
    assert ids <= {m["id"] for m in messages if "result" in m or "error" in m}


def test_scope_pinning_and_executor_client(tmp_path):
    store, scope = fixture_scope(tmp_path)
    with pytest.raises(PermissionError):
        PaperScope(tmp_path / "other", store.path, "bound", ["bound_chunk"])
    with pytest.raises(PermissionError):
        PaperScope(tmp_path, store.path, "bound", ["foreign_chunk"])
    client = PaperMCPClient(scope, tmp_path / "mcp")
    result = client.read("bound_chunk")
    assert result["transport"] == "mcp_stdio" and Path(result["mcp"]["audit_path"]).is_file()
    store.replace_chunks("bound", [Chunk("bound_chunk", "bound", "Methods", "changed")])
    with pytest.raises(Exception):
        client.read("bound_chunk")


def process_alive(pid):
    if sys.platform == "win32":
        import ctypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x00100000, 0, pid)
        if not handle:
            return False
        try:
            return kernel.WaitForSingleObject(handle, 0) == 258
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


@pytest.mark.parametrize("fault", ["timeout", "cancel", "exit"])
def test_real_sdk_timeout_cancel_exit_reaps_child(tmp_path, fault):
    import anyio
    audit = tmp_path / "fault.wire.jsonl"
    helper = Path(__file__).resolve().parents[1] / "fixtures" / "mcp_fault_server.py"
    started = time.monotonic()

    async def run():
        async with connect(tmp_path / "unused.json", audit, timeout=5,
                           command=sys.executable, args=[str(helper), str(audit)]) as (session, _):
            if fault == "cancel":
                with anyio.move_on_after(0.1) as scope:
                    await session.call_tool("slow", {})
                assert scope.cancelled_caught
                # Let the SDK flush its actual cancelled notification.
                await anyio.sleep(0.1)
            else:
                await session.call_tool("exit" if fault == "exit" else "slow", {}, read_timeout_seconds=0.1)
    if fault == "cancel":
        asyncio.run(run())
    else:
        with pytest.raises(Exception):
            asyncio.run(run())
    assert time.monotonic() - started < 10
    pid = int(audit.with_suffix(".pid").read_text(encoding="utf-8"))
    assert not process_alive(pid)
    messages = [json.loads(json.loads(l)["raw"]) for l in audit.read_text(encoding="utf-8").splitlines()]
    if fault in {"cancel", "timeout"}:
        assert any(m.get("method") == "notifications/cancelled" for m in messages)

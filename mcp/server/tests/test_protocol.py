from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import pytest


@pytest.fixture
def anyio_backend():
    return "asyncio"


@asynccontextmanager
async def client(tmp_path, **environment):
    env = dict(os.environ)
    env.pop("APPIMAGE", None)
    env.update({"TFT_MCP_AUDIT_PATH": str(tmp_path / "audit.jsonl"),
                "TFT_MCP_NATIVE_LOG_DIR": str(tmp_path / "native")})
    env.update(environment)
    command = os.environ.get("TFT_MCP_TEST_COMMAND")
    if command:
        env.pop("PYTHONPATH", None)
        args = []
    else:
        command = sys.executable
        args = ["-m", "tft_mcp"]
        env["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
    parameters = StdioServerParameters(command=command, args=args, env=env, cwd=str(tmp_path))
    async with stdio_client(parameters) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


@pytest.mark.anyio
async def test_production_discovery_and_idle(tmp_path):
    async with client(tmp_path) as session:
        tools = (await session.list_tools()).tools
        assert len(tools) == 25
        assert {tool.name for tool in tools} == {
            "start_game", "get_game_status", "close_game", "end_turn",
            "search_champions", "get_champion", "search_traits", "get_trait",
            "get_trait_champions", "search_items", "get_item", "get_board",
            "get_bench", "get_shop", "get_items", "get_economy", "get_traits",
            "get_round", "get_players", "buy_unit", "sell_unit", "refresh_shop",
            "buy_xp", "move_unit", "equip_item",
        }
        status = await session.call_tool("get_game_status", {})
        assert not status.isError
        assert status.structuredContent == {"state": "idle", "game_id": None,
            "controlled_player_id": None, "round": None, "planning_budget": None, "outcome": None}


@pytest.mark.anyio
async def test_strict_inputs_lifecycle_and_restart(tmp_path):
    async with client(tmp_path) as session:
        for name, arguments in [
            ("start_game", {}), ("start_game", {"seed": 1.5}),
            ("start_game", {"seed": True}), ("start_game", {"seed": False}),
            ("start_game", {"seed": 1.0}), ("start_game", {"seed": -1}),
            ("start_game", {"seed": 2147483648}), ("start_game", {"seed": "1"}),
            ("start_game", {"seed": 1, "extra": 2}),
            ("get_game_status", {"extra": 1}), ("close_game", {"extra": 1}), ("end_turn", {"extra": 1}),
        ]:
            error = await session.call_tool(name, arguments)
            assert error.isError, (name, arguments)
            assert set(error.structuredContent) == {"code", "message", "details"}
            assert error.structuredContent["code"] == "invalid_input"
        started = await session.call_tool("start_game", {"seed": 2147483647})
        assert not started.isError
        assert started.structuredContent["state"] == "running"
        active_id = started.structuredContent["game_id"]
        duplicate = await session.call_tool("start_game", {"seed": 0})
        assert duplicate.isError
        assert duplicate.structuredContent["code"] == "game_active"
        assert (await session.call_tool("get_game_status", {})).structuredContent == started.structuredContent
        closed = await session.call_tool("close_game", {})
        assert closed.structuredContent["closed_game_id"] == active_id
        assert closed.structuredContent["outcome"]["reason"] == "closed_incomplete"
        assert (await session.call_tool("close_game", {})).structuredContent["closed_game_id"] is None
        restarted = await session.call_tool("start_game", {"seed": 0})
        assert not restarted.isError
        assert restarted.structuredContent["game_id"] != active_id
    async with client(tmp_path) as session:
        assert (await session.call_tool("get_game_status", {})).structuredContent["state"] == "idle"
        assert not (await session.call_tool("start_game", {"seed": 0})).isError


@pytest.mark.anyio
async def test_bootstrap_reexec_records_fixed_hash_and_reproducibility(tmp_path):
    import subprocess

    expected = subprocess.check_output([sys.executable, "-c", 'print(hash("tft-mcp"))'],
                                      env={**os.environ, "PYTHONHASHSEED": "0"}, text=True).strip()
    for hash_seed in ["random", "42", "0"]:
        async with client(tmp_path, PYTHONHASHSEED=hash_seed) as session:
            assert not (await session.call_tool("start_game", {"seed": 123})).isError
    records = [json.loads(line) for line in (tmp_path / "audit.jsonl").read_text().splitlines()]
    starts = [record for record in records if record["event"] == "game_started"]
    assert len(starts) == 3
    for server_id in {record["server_id"] for record in records}:
        server_records = [record for record in records if record["server_id"] == server_id]
        sequences = [record["sequence"] for record in server_records]
        assert sequences == sorted(set(sequences))
        events = [record["event"] for record in server_records]
        assert events.index("tool_request") < events.index("game_started") < events.index("tool_result")
    for record in starts:
        assert record["hash_seed"] == "0"
        assert str(record["hash_probe"]) == expected
        assert record["seed"] == record["baseline_seed"] == 123
        assert record["baseline"] == "Simulator.generators.default_agent.Default_Agent(False)"
        from Simulator.simulators.tft_simulator import TFT_Simulator
        assert record["simulator"]["environment_name"] == TFT_Simulator.metadata["name"]
        assert record["simulator"]["revision"]
        assert len(record["simulator"]["source_sha256"]) == 64
        assert record["configuration"]["num_players"] == 8
        assert record["runtime"]["python"]
        assert set(record["runtime"]["dependencies"]) == {"numpy", "PettingZoo", "gymnasium", "mcp"}
        assert all(record["runtime"]["dependencies"].values())


@pytest.mark.anyio
async def test_audit_and_native_failure_leave_idle(tmp_path):
    async with client(tmp_path, TFT_MCP_AUDIT_PATH="/proc/tft-mcp-audit.jsonl") as session:
        error = await session.call_tool("start_game", {"seed": 0})
        assert error.isError
        assert error.structuredContent["code"] == "log_unavailable"
        rejected = await session.call_tool("end_turn", {"extra": 1})
        assert rejected.isError
        assert rejected.structuredContent["code"] == "log_unavailable"
    async with client(tmp_path, TFT_MCP_NATIVE_LOG_DIR="/proc/tft-mcp-native") as session:
        error = await session.call_tool("start_game", {"seed": 0})
        assert error.isError
        assert error.structuredContent["code"] == "log_unavailable"
        status = await session.call_tool("get_game_status", {})
        assert status.structuredContent["state"] == "idle"


@pytest.mark.anyio
async def test_failed_close_keeps_game_active(tmp_path):
    async with client(tmp_path) as session:
        started = await session.call_tool("start_game", {"seed": 0})
        audit = tmp_path / "audit.jsonl"
        saved = audit.read_bytes()
        audit.unlink()
        audit.mkdir()
        failed = await session.call_tool("close_game", {})
        assert failed.isError
        assert failed.structuredContent["code"] == "log_unavailable"
        audit.rmdir()
        audit.write_bytes(saved)
        assert (await session.call_tool("get_game_status", {})).structuredContent == started.structuredContent
        assert not (await session.call_tool("close_game", {})).isError


def test_protocol_stdout_is_json_and_malformed_envelopes_are_native_errors(tmp_path):
    import select
    import subprocess

    environment = dict(os.environ, TFT_MCP_AUDIT_PATH=str(tmp_path / "audit.jsonl"))
    environment.pop("APPIMAGE", None)
    command = os.environ.get("TFT_MCP_TEST_COMMAND")
    if command:
        environment.pop("PYTHONPATH", None)
        arguments = [command]
    else:
        environment["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
        arguments = [sys.executable, "-m", "tft_mcp"]
    with (tmp_path / "stderr.log").open("w") as diagnostics:
        process = subprocess.Popen(arguments, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=diagnostics, env=environment, cwd=tmp_path, text=True)
        def request(payload):
            process.stdin.write(json.dumps(payload) + "\n")
            process.stdin.flush()
            assert select.select([process.stdout], [], [], 15)[0], "No protocol response"
            return json.loads(process.stdout.readline())
        try:
            initialized = request({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
                "protocolVersion": "2024-11-05", "capabilities": {},
                "clientInfo": {"name": "stdout-acceptance", "version": "1"}}})
            assert initialized["id"] == 1
            process.stdin.write('{"jsonrpc":"2.0","method":"notifications/initialized"}\n')
            process.stdin.flush()
            malformed = request({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
                "name": "start_game", "arguments": []}})
            assert malformed["error"]["code"] == -32602
            started = request({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {
                "name": "start_game", "arguments": {"seed": 0}}})
            assert started["result"]["structuredContent"]["state"] == "running"
            closed = request({"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {
                "name": "close_game", "arguments": {}}})
            assert closed["result"]["structuredContent"]["status"]["state"] == "idle"
        finally:
            process.stdin.close()
            process.wait(timeout=15)
            process.stdout.close()
        assert process.returncode == 0


@pytest.mark.anyio
async def test_seed_zero_full_lobby_read_independent_replay(tmp_path):
    accepted = []
    for with_reads in (False, True):
        run_path = tmp_path / str(with_reads)
        run_path.mkdir()
        async with client(run_path) as session:
            assert not (await session.call_tool('start_game', {'seed': 0})).isError
            results = []
            for _ in range(30):
                if with_reads:
                    for _ in range(2):
                        status = await session.call_tool('get_game_status', {})
                        assert not status.isError
                result = await session.call_tool('end_turn', {})
                assert not result.isError, result.structuredContent
                data = {key: value for key, value in result.structuredContent.items() if key != 'game_id'}
                results.append(data)
                if data['state'] == 'terminal':
                    break
            assert results[-1]['outcome'] == {'controlled_placement': 8, 'lobby_complete': True, 'reason': 'lobby_complete'}
            assert results[-1]['planning_budget'] is None
            rejected = await session.call_tool('end_turn', {})
            assert rejected.isError and rejected.structuredContent['code'] == 'game_terminal'
            rejected = await session.call_tool('start_game', {'seed': 1})
            assert rejected.isError and rejected.structuredContent['code'] == 'game_active'
            receipt = await session.call_tool('close_game', {})
            assert receipt.structuredContent['outcome'] == results[-1]['outcome']
            assert (await session.call_tool('start_game', {'seed': 0})).structuredContent['round'] == 1
        events = [json.loads(line) for line in (run_path / 'audit.jsonl').read_text().splitlines()]
        progress = [{key: event[key] for key in ('player_id', 'round', 'kind', 'action', 'placements')}
                    for event in events if event['event'] == 'progression']
        completed = [event for event in events if event['event'] == 'lobby_complete']
        assert len(completed) == 1
        assert len(completed[0]['placements']) == 8
        accepted.append((results, progress))
    assert accepted[0] == accepted[1]

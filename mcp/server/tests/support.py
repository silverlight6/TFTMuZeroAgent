"""Real simulator fixtures, SDK clients and shared transaction assertions."""

from contextlib import asynccontextmanager
from copy import deepcopy
import os
from pathlib import Path
import pickle
import random
import sys

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.memory import create_client_server_memory_streams
import numpy as np

from tft_mcp.session import GameSession


INSPECTION_TOOLS = ('get_board', 'get_bench', 'get_shop', 'get_items', 'get_economy', 'get_traits', 'get_round')


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


def started_session(tmp_path):
    session = GameSession(tmp_path / 'audit.jsonl', tmp_path / 'native')
    session.start_game(0)
    return session



def session_fixture(tmp_path):
    session = started_session(tmp_path)
    session.game.player_manager.player_states['player_0'].gold = 100
    return session


def install_units(session, bench=(), board=(), offer='garen', chosen=False):
    from Simulator.battle.champion import champion
    player = session.game.player_manager.player_states['player_0']
    with session.simulator_scope():
        player.bench = [None] * 9
        player.board = [[None] * 4 for _ in range(7)]
        player.triple_catalog = []
        player.num_units_in_play = 0
        player.thieves_gloves_loc = []
        for kind, units in [('bench', bench), ('board', board)]:
            for position, name, stars, items in units:
                unit = champion(name, stars=stars, itemlist=items)
                entry = next((e for e in player.triple_catalog if e['name'] == name and e['level'] == stars), None)
                if entry:
                    entry['num'] += 1
                else:
                    player.triple_catalog.append({'name': name, 'level': stars, 'num': 1})
                if kind == 'bench':
                    player.bench[position] = unit
                    unit.bench_loc = position
                    if items and items[0] == 'thieves_gloves':
                        player.thieves_gloves_loc.append([position, -1])
                else:
                    x, y = position
                    player.board[x][y] = unit
                    unit.x, unit.y = x, y
                    player.num_units_in_play += 1
                    if items and items[0] == 'thieves_gloves':
                        player.thieves_gloves_loc.append([x, y])
        unit = champion(offer, chosen=chosen)
        player.shop[0] = f'{offer}_{chosen}_c' if chosen else offer
        player.shop_champions[0] = unit
    return player


def board(x, y):
    return {'kind': 'board', 'x': x, 'y': y}


def bench(slot):
    return {'kind': 'bench', 'slot': slot}


def normalized_graph(game):
    """Retain the graph, normalizing only native wall-clock fields and Player log clocks."""
    from Simulator.battle.champion import champion
    from Simulator.game.player import Player
    graph = deepcopy(game)
    seen = set()
    def normalize(value):
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, (Player, champion)):
            value.start_time = 0
        if isinstance(value, Player):
            lines = []
            for line in value.log:
                assert line[:8] == f"{value.player_num:<8}"
                float(line[8:28])  # The unchanged Player.print clock field.
                lines.append(line[:8] + f"{0:<20}" + line[28:])
            value.log[:] = lines
        if hasattr(value, "__dict__"):
            normalize(vars(value))
        elif isinstance(value, dict):
            for item in value.values():
                normalize(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                normalize(item)
    normalize(graph)
    return pickle.dumps(graph)


def gameplay(session):
    from tft_mcp.session import freeze_player
    return {'status': {key: value for key, value in session.get_game_status().items() if key != 'game_id'},
            'players': {key: freeze_player(player, session.game.game_round.current_round)
                        for key, player in session.game.player_manager.player_states.items() if player is not None},
            'placements': session.placements, 'snapshot': session.terminal_snapshot,
            'episode_rng': repr(session.game.rng.py.getstate()),
            'baseline_rng': repr(session.baseline_rng)}


def gameplay_with_numpy(session):
    return gameplay(session), deepcopy(session.game.rng.np.bit_generator.state)


@asynccontextmanager
async def memory_client(session, monkeypatch):
    """Exercise the unchanged transport with a preconfigured real GameSession."""
    import tft_mcp.transport as transport

    with monkeypatch.context() as patch:
        patch.setattr(transport, 'GameSession', lambda *args: session)
        async with create_client_server_memory_streams() as (client_streams, server_streams):
            @asynccontextmanager
            async def streams():
                yield server_streams
            patch.setattr(transport, 'stdio_server', streams)
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(transport.serve)
                try:
                    async with ClientSession(*client_streams) as sdk:
                        await sdk.initialize()
                        yield sdk
                finally:
                    tasks.cancel_scope.cancel()


def simulator_bindings():
    from Simulator.battle import champion, origin_class
    return champion.log, champion.test_multiple, origin_class.game_compositions, origin_class.game_comp_tiers


def capture_committed_state(session):
    return {
        'identities': (session.game, session.baselines, session.module_state, session.baseline_rng, session.python_rng),
        'graph': pickle.dumps(session.game),
        'gameplay': gameplay_with_numpy(session),
        'native_dir': session.native_dir,
        'logs': (session.audit_path.read_bytes(), (session.native_dir / 'log.txt').read_bytes()),
        'process_rng': (random.getstate(), pickle.dumps(np.random.get_state())),
        'bindings': simulator_bindings(),
    }


def assert_committed_state_unchanged(session, before):
    current = capture_committed_state(session)
    for key in ('identities', 'bindings'):
        assert all(actual is expected for actual, expected in zip(current[key], before[key], strict=True))
    for key in ('graph', 'gameplay', 'native_dir', 'logs', 'process_rng'):
        assert current[key] == before[key], key
    assert_native_aliases(session.game)


def assert_native_aliases(game):
    assert game.rng is game.combat_ctx.rng
    assert game.pool_obj is game.player_manager.pool_obj
    assert game.game_round.PLAYERS is game.player_manager.player_states
    for key, player in game.player_manager.player_states.items():
        if player:
            assert game.player_manager.observation_states[key].player is player
            assert game.player_manager.action_handlers[key].player is player
            assert player.pool_obj is game.pool_obj

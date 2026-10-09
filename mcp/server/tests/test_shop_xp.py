import pytest

from tft_mcp.session import SessionError
from test_buy_sell import session_fixture
from test_progression import gameplay as progression_gameplay


def gameplay(session):
    from copy import deepcopy
    return progression_gameplay(session), deepcopy(session.game.rng.np.bit_generator.state)


def test_refresh_spends_native_cost_and_returns_detached_actual_shop(tmp_path):
    session = session_fixture(tmp_path)
    player = session.game.player_manager.player_states['player_0']
    player.refresh_cost = 3
    receipt = session.refresh_shop()
    assert receipt == {'gold_spent': 3, 'slots': session.get_shop()['slots'], 'status': session.get_game_status()}
    assert session.get_economy()['gold'] == 97
    assert receipt['status']['round'] == 1
    assert receipt['status']['planning_budget']['remaining'] == 13
    assert session.game.agent_selection == 'player_0'
    receipt['slots'][0]['unit']['items'].append('bf_sword')
    assert session.get_shop()['slots'][0]['unit']['items'] == []


@pytest.mark.parametrize('level,xp,cost,capacity,expected', [
    (3, 0, 4, 6, (3, 4, 6)),
    (1, 0, 4, 1, (3, 0, 3)),
    (1, 0, 7, 4, (3, 3, 6)),
    (8, 79, 4, 10, (9, 0, 11)),
])
def test_xp_uses_recursive_native_progression_and_preserves_bonus_capacity(tmp_path, level, xp, cost, capacity, expected):
    session = session_fixture(tmp_path)
    player = session.game.player_manager.player_states['player_0']
    player.level, player.exp, player.exp_cost, player.max_units = level, xp, cost, capacity
    shop = session.get_shop()
    receipt = session.buy_xp()
    assert (receipt['level'], receipt['xp'], receipt['unit_capacity']) == expected
    assert receipt['gold_spent'] == cost
    assert receipt['xp_before'] == xp and receipt['level_before'] == level
    assert session.get_economy()['gold'] == 100 - cost
    assert session.get_shop() == shop
    assert receipt['status']['planning_budget']['remaining'] == 13


@pytest.mark.anyio
async def test_production_shop_xp_discovery_strict_requests_and_earned_gold(tmp_path):
    from test_protocol import client
    from tft_mcp.transport import EMPTY_INPUT, SHOP_PROPERTIES, STATUS_SCHEMA
    async with client(tmp_path) as sdk:
        tools = {tool.name: tool for tool in (await sdk.list_tools()).tools}
        for name in ('refresh_shop', 'buy_xp'):
            assert tools[name].inputSchema == EMPTY_INPUT
            assert tools[name].outputSchema['additionalProperties'] is False
            assert tools[name].outputSchema['properties']['status'] == STATUS_SCHEMA
            assert (await sdk.call_tool(name)).structuredContent['code'] == 'no_game'
            assert (await sdk.call_tool(name, {'player_id': 'player_0'})).structuredContent['code'] == 'invalid_input'
        assert tools['refresh_shop'].outputSchema['properties']['slots'] == SHOP_PROPERTIES['slots']
        assert set(tools['buy_xp'].outputSchema['required']) == {'gold_spent', 'xp_before', 'xp', 'level_before', 'level', 'unit_capacity', 'status'}
        await sdk.call_tool('start_game', {'seed': 0})
        for _ in range(3):
            assert not (await sdk.call_tool('end_turn')).isError
        before = (await sdk.call_tool('get_economy')).structuredContent
        refreshed = await sdk.call_tool('refresh_shop')
        assert not refreshed.isError, refreshed.structuredContent
        assert refreshed.structuredContent['slots'] == (await sdk.call_tool('get_shop')).structuredContent['slots']
        xp = await sdk.call_tool('buy_xp')
        assert not xp.isError, xp.structuredContent
        receipt = xp.structuredContent
        assert receipt['level'] > receipt['level_before']
        after = (await sdk.call_tool('get_economy')).structuredContent
        assert before['gold'] - after['gold'] == receipt['gold_spent'] + refreshed.structuredContent['gold_spent']
        assert after['round'] == before['round']
        assert after['planning_budget']['remaining'] == 12
        await sdk.call_tool('close_game')


@pytest.mark.parametrize('failure', ['native_action', 'observation', 'baseline', 'corrupt', 'noop', 'postcondition', 'native', 'audit'])
@pytest.mark.parametrize('action', ['refresh_shop', 'buy_xp'])
def test_failed_action_discards_aggregate_rng_logs_and_retries(tmp_path, monkeypatch, failure, action):
    import os
    import random
    import numpy as np
    from Simulator.battle import champion, origin_class
    from Simulator.simulators.tft_simulator import TFT_Simulator
    import tft_mcp.session as module
    session = session_fixture(tmp_path / 'actual')
    reference = session_fixture(tmp_path / 'reference')
    method = lambda current: getattr(current, action)()
    original_game = session.game
    original_policies = session.baselines
    original_modules = session.module_state
    original_rng = session.baseline_rng
    before = gameplay(session)
    logs = (session.audit_path.read_bytes(), (session.native_dir / 'log.txt').read_bytes())
    process_rng = random.getstate(), np.random.get_state()
    bindings = champion.log, champion.test_multiple, origin_class.game_compositions, origin_class.game_comp_tiers
    with monkeypatch.context() as patch:
        original_step = TFT_Simulator.step
        def fail_step(game, native_action):
            selected = game.agent_selection
            result = original_step(game, native_action)
            if failure == 'observation' and selected == 'player_0':
                raise RuntimeError('after real native action and observation update')
            if failure == 'baseline' and selected != 'player_0':
                raise RuntimeError('after real baseline action')
            if failure == 'corrupt' and selected == 'player_0':
                player = game.player_manager.player_states['player_0']
                player.gold += 1
            if failure == 'native' and selected == 'player_0':
                with open('log.txt', 'a') as stream:
                    stream.write('partial candidate native log')
                raise OSError('native log write failed')
            return result
        if failure in {'native_action', 'noop'}:
            from Simulator.game.player import Player
            name = 'refresh_shop_action' if action == 'refresh_shop' else 'buy_exp_action'
            original_action = getattr(Player, name)
            def fail_native(player, *args):
                if failure == 'noop':
                    return False
                original_action(player, *args)
                raise RuntimeError('after native mutation before observation update')
            patch.setattr(Player, name, fail_native)
        elif failure in {'observation', 'baseline', 'corrupt', 'native'}:
            # Failure injection exercises rollback, never substitutes a successful simulator.
            patch.setattr(TFT_Simulator, 'step', fail_step)
        elif failure == 'postcondition':
            def fail_receipt(*args):
                raise RuntimeError('postcondition inspection failed after native progression')
            patch.setattr(module, 'check_action_status', fail_receipt)
        else:
            def fail_replace(*args):
                raise OSError('audit commit failed')
            patch.setattr(os, 'replace', fail_replace)
        with pytest.raises(SessionError) as error:
            method(session)
        assert error.value.code == ('log_unavailable' if failure in {'native', 'audit'} else 'internal_error')
    assert session.game is original_game
    assert session.baselines is original_policies
    assert session.module_state is original_modules
    assert session.baseline_rng is original_rng
    assert gameplay(session) == before
    assert logs == (session.audit_path.read_bytes(), (session.native_dir / 'log.txt').read_bytes())
    assert random.getstate() == process_rng[0]
    assert np.array_equal(np.random.get_state()[1], process_rng[1][1])
    assert np.random.get_state()[2:] == process_rng[1][2:]
    assert all(actual is expected for actual, expected in zip((champion.log, champion.test_multiple, origin_class.game_compositions, origin_class.game_comp_tiers), bindings))
    method(session)
    method(reference)
    assert gameplay(session) == gameplay(reference)
    game = session.game
    assert game.rng is game.combat_ctx.rng
    assert game.pool_obj is game.player_manager.pool_obj
    assert game.game_round.PLAYERS is game.player_manager.player_states
    for key, player in game.player_manager.player_states.items():
        if player:
            assert game.player_manager.observation_states[key].player is player
            assert game.player_manager.action_handlers[key].player is player
            assert player.pool_obj is game.pool_obj



@pytest.mark.parametrize('action', ['refresh_shop', 'buy_xp'])
def test_rejections_preserve_gameplay_and_follow_validation_order(tmp_path, action):
    session = session_fixture(tmp_path)
    method = getattr(session, action)
    player = session.game.player_manager.player_states['player_0']
    player.gold = 0
    before = gameplay(session)
    with pytest.raises(SessionError) as error:
        method()
    assert error.value.code == 'insufficient_gold'
    assert {key: error.value.details[key] for key in ('resource', 'required', 'available')} == {'resource': 'gold', 'required': getattr(player, 'refresh_cost' if action == 'refresh_shop' else 'exp_cost'), 'available': 0}
    assert gameplay(session) == before
    if action == 'buy_xp':
        player.level = player.max_level
        before = gameplay(session)
        with pytest.raises(SessionError) as error:
            method()
        assert error.value.code == 'level_cap'
        assert {key: error.value.details[key] for key in ('level', 'max_level')} == {'level': player.level, 'max_level': player.max_level}
        assert gameplay(session) == before
    for _ in range(14):
        session.controlled_action([0, 0, 0])
    before = gameplay(session)
    with pytest.raises(SessionError) as error:
        method()
    assert error.value.code == 'budget_exhausted'
    assert gameplay(session) == before
    session.state = 'terminal'
    with pytest.raises(SessionError) as error:
        method()
    assert error.value.code == 'game_terminal'
    with pytest.raises(SessionError) as error:
        method(player_id='player_0')
    assert error.value.code == 'invalid_input'


@pytest.mark.anyio
async def test_sdk_memory_real_session_cap_success_and_atomic_rejection(tmp_path, monkeypatch):
    from contextlib import asynccontextmanager
    import anyio
    from mcp import ClientSession
    from mcp.shared.memory import create_client_server_memory_streams
    import tft_mcp.transport as transport
    session = session_fixture(tmp_path)
    player = session.game.player_manager.player_states['player_0']
    player.level = player.max_level - 1
    player.exp = player.level_costs[player.level] - 1
    player.max_units += 2
    monkeypatch.setattr(transport, 'GameSession', lambda *args: session)
    async with create_client_server_memory_streams() as (client_streams, server_streams):
        @asynccontextmanager
        async def streams():
            yield server_streams
        monkeypatch.setattr(transport, 'stdio_server', streams)
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(transport.serve)
            async with ClientSession(*client_streams) as sdk:
                await sdk.initialize()
                result = await sdk.call_tool('buy_xp')
                assert not result.isError
                receipt = result.structuredContent
                assert receipt['level'] == player.max_level and receipt['xp'] == 0
                assert receipt['unit_capacity'] == 4
                before = gameplay(session)
                budget = session.get_game_status()
                error = await sdk.call_tool('buy_xp')
                assert error.isError and error.structuredContent['code'] == 'level_cap'
                assert error.structuredContent['details']['max_level'] == player.max_level
                assert gameplay(session) == before and session.get_game_status() == budget
                await sdk.call_tool('close_game')
            tasks.cancel_scope.cancel()


@pytest.mark.anyio
async def test_production_refresh_replay_ignores_extra_reads_and_rejections(tmp_path):
    import json
    from test_protocol import client
    journeys = []
    for extra in (False, True):
        path = tmp_path / str(extra)
        path.mkdir()
        async with client(path) as sdk:
            await sdk.call_tool('start_game', {'seed': 0})
            for _ in range(3):
                await sdk.call_tool('end_turn')
            receipts = []
            for action in ('refresh_shop', 'buy_xp', 'refresh_shop'):
                if extra:
                    for name in ('get_shop', 'get_items', 'search_items', 'get_economy', 'get_board', 'search_champions', 'get_bench', 'get_traits', 'get_round'):
                        assert not (await sdk.call_tool(name)).isError
                    assert (await sdk.call_tool(action, {'extra': True})).isError
                result = await sdk.call_tool(action)
                assert not result.isError, result.structuredContent
                data = result.structuredContent
                data['status'].pop('game_id')
                receipts.append(data)
            state = []
            for name in ('get_shop', 'get_economy', 'get_board', 'get_bench', 'get_items'):
                data = (await sdk.call_tool(name)).structuredContent
                data.pop('game_id')
                state.append(data)
            await sdk.call_tool('close_game')
        events = [json.loads(line) for line in (path / 'audit.jsonl').read_text().splitlines()]
        progress = [{key: event[key] for key in ('player_id', 'round', 'kind', 'action', 'placements')}
                    for event in events if event['event'] == 'progression']
        journeys.append((receipts, state, progress))
    assert journeys[0] == journeys[1]


@pytest.mark.parametrize('corruption', ['shop_names', 'shop_units', 'xp', 'capacity', 'level', 'cap_residual'])
def test_corrupt_native_postconditions_discard_all_candidate_effects(tmp_path, monkeypatch, corruption):
    from Simulator.game.player import Player
    session = session_fixture(tmp_path)
    player = session.game.player_manager.player_states['player_0']
    refresh = corruption.startswith('shop_')
    if corruption == 'cap_residual':
        player.level = player.max_level - 1
        player.exp = player.level_costs[player.level] - 1
    name = 'refresh_shop_action' if refresh else 'buy_exp_action'
    native = getattr(Player, name)
    def corrupt(current):
        shop, units = current.shop, current.shop_champions
        result = native(current)
        if corruption == 'shop_names':
            current.shop = shop
        elif corruption == 'shop_units':
            current.shop_champions = units
        elif corruption in {'xp', 'cap_residual'}:
            current.exp += 1
        elif corruption == 'capacity':
            current.max_units += 1
        else:
            current.level = current.max_level + 1
        return result
    before = gameplay(session)
    monkeypatch.setattr(Player, name, corrupt)
    with pytest.raises(SessionError) as error:
        getattr(session, 'refresh_shop' if refresh else 'buy_xp')()
    assert error.value.code == 'internal_error'
    assert gameplay(session) == before


def test_native_thresholds_determine_progression_without_adapter_constants(tmp_path):
    session = session_fixture(tmp_path)
    player = session.game.player_manager.player_states['player_0']
    player.level_costs = list(player.level_costs)
    player.level_costs[1:4] = [1, 1, 9]
    player.exp_cost = 5
    receipt = session.buy_xp()
    assert (receipt['level'], receipt['xp'], receipt['unit_capacity'], receipt['gold_spent']) == (3, 3, 3, 5)


@pytest.mark.anyio
async def test_production_shop_xp_audit_failure_recovers_without_gameplay_change(tmp_path):
    from test_protocol import client
    async with client(tmp_path) as sdk:
        await sdk.call_tool('start_game', {'seed': 0})
        for _ in range(3):
            await sdk.call_tool('end_turn')
        for name in ('refresh_shop', 'buy_xp'):
            before = [(await sdk.call_tool(read)).structuredContent for read in ('get_game_status', 'get_economy', 'get_shop')]
            audit = tmp_path / 'audit.jsonl'
            saved = audit.read_bytes()
            audit.unlink()
            audit.mkdir()
            rejected = await sdk.call_tool(name)
            assert rejected.isError and rejected.structuredContent['code'] == 'log_unavailable'
            audit.rmdir()
            audit.write_bytes(saved)
            assert [(await sdk.call_tool(read)).structuredContent for read in ('get_game_status', 'get_economy', 'get_shop')] == before
            assert not (await sdk.call_tool(name)).isError
        await sdk.call_tool('close_game')

import pytest

from support import install_units, session_fixture, board, bench, normalized_graph
from tft_mcp.session import SessionError


def test_bench_entry_uses_board_orientation_and_one_action(tmp_path):
    session = session_fixture(tmp_path)
    install_units(session, bench=[(2, 'garen', 1, ['bf_sword'])])
    receipt = session.move_unit(source=bench(2), target=board(6, 3))
    assert receipt['source'] == bench(2) and receipt['target'] == board(6, 3)
    assert [c['location'] for c in receipt['unit_changes']] == [board(6, 3), bench(2)]
    assert session.get_board()['slots'][27]['unit']['champion'] == 'garen'
    assert session.get_bench()['slots'][2]['unit'] is None
    assert receipt['status']['planning_budget']['remaining'] == 13
    assert receipt['status']['round'] == 1


@pytest.mark.parametrize('source,target,code', [
    (bench(2), bench(2), 'unsupported_action'),
    (bench(2), bench(3), 'unsupported_action'),
    (board(0, 0), bench(2), 'empty_slot'),
    (bench(2), board(0, 0), 'capacity_exceeded'),
])
def test_restrictions_are_atomic(tmp_path, source, target, code):
    from support import gameplay_with_numpy as gameplay
    session = session_fixture(tmp_path)
    player = install_units(session, bench=[(2, 'garen', 1, [])])
    player.max_units = 0
    before = gameplay(session)
    with pytest.raises(SessionError) as error:
        session.move_unit(source=source, target=target)
    assert error.value.code == code
    assert gameplay(session) == before


@pytest.mark.parametrize('source,target,bench_units,board_units,expected', [
    (board(6, 3), board(0, 0), [], [((6, 3), 'garen', 1, [])], [(board(0, 0), 'garen')]),
    (board(6, 3), board(0, 0), [], [((6, 3), 'garen', 1, []), ((0, 0), 'fiora', 1, [])], [(board(0, 0), 'garen'), (board(6, 3), 'fiora')]),
    (bench(5), board(0, 0), [(5, 'garen', 1, [])], [((0, 0), 'fiora', 1, [])], [(board(0, 0), 'garen'), (bench(0), 'fiora')]),
    (board(0, 0), bench(5), [], [((0, 0), 'garen', 1, [])], [(bench(5), 'garen')]),
    (board(0, 0), bench(0), [(0, 'fiora', 1, [])], [((0, 0), 'garen', 1, [])], [(bench(0), 'garen'), (board(0, 0), 'fiora')]),
])
def test_real_native_movement_families(tmp_path, source, target, bench_units, board_units, expected):
    session = session_fixture(tmp_path)
    from copy import deepcopy
    import numpy as np
    from tft_mcp.session import freeze_player, location_flat
    player = install_units(session, bench_units, board_units)
    manager = session.game.player_manager
    with session.simulator_scope():
        player.update_team_tiers()
        observation_class = type(manager.observation_states['player_0'])
        action_class = type(manager.action_handlers['player_0'])
        manager.observation_states['player_0'] = observation_class(player)
        manager.action_handlers['player_0'] = action_class(player)
        native = deepcopy(player)
        native.move_champ_action(location_flat(source), location_flat(target))
        native.actions_remaining -= 1
    session.move_unit(source=source, target=target)
    from tft_mcp.session import location_unit
    player = session.game.player_manager.player_states['player_0']
    for location, name in expected:
        assert location_unit(player, location).name == name
    assert freeze_player(player, 1) == freeze_player(native, 1)
    manager = session.game.player_manager
    with session.simulator_scope():
        actual_observation = manager.observation_states['player_0'].fetch_player_observation()
        expected_observation = observation_class(native).fetch_player_observation()
        for key in actual_observation:
            np.testing.assert_array_equal(actual_observation[key], expected_observation[key])
        np.testing.assert_array_equal(manager.action_handlers['player_0'].fetch_action_mask(), action_class(native).fetch_action_mask())


def test_directed_occupied_bench_rejects_earlier_vacancy(tmp_path):
    session = session_fixture(tmp_path)
    install_units(session, [(5, 'fiora', 1, [])], [((0, 0), 'garen', 1, [])])
    with pytest.raises(SessionError) as error:
        session.move_unit(source=board(0, 0), target=bench(5))
    assert error.value.code == 'unsupported_action'


def test_identical_visible_board_swap_consumes_action_without_delta(tmp_path):
    session = session_fixture(tmp_path)
    install_units(session, board=[((0, 0), 'garen', 1, []), ((6, 3), 'garen', 1, [])])
    receipt = session.move_unit(source=board(6, 3), target=board(0, 0))
    assert receipt['unit_changes'] == []
    assert receipt['status']['planning_budget']['remaining'] == 13


@pytest.mark.anyio
async def test_production_sdk_movement_discovery_and_natural_journey(tmp_path):
    from support import client
    from tft_mcp.transport import LOCATION_SCHEMA, UNIT_CHANGE_SCHEMA, STATUS_SCHEMA
    async with client(tmp_path) as sdk:
        tools = {tool.name: tool for tool in (await sdk.list_tools()).tools}
        assert tools['move_unit'].inputSchema['additionalProperties'] is False
        assert tools['move_unit'].inputSchema['properties'] == {'source': LOCATION_SCHEMA, 'target': LOCATION_SCHEMA}
        assert tools['move_unit'].outputSchema['properties']['unit_changes']['items'] == UNIT_CHANGE_SCHEMA
        assert tools['move_unit'].outputSchema['properties']['status'] == STATUS_SCHEMA
        assert set(tools['move_unit'].inputSchema['required']) == {'source', 'target'}
        assert set(tools['move_unit'].outputSchema['required']) == {'source', 'target', 'unit_changes', 'status'}
        assert (await sdk.call_tool('move_unit', {'source': bench(0), 'target': board(0, 0)})).structuredContent['code'] == 'no_game'
        await sdk.call_tool('start_game', {'seed': 0})
        await sdk.call_tool('end_turn')
        offers = (await sdk.call_tool('get_shop')).structuredContent['slots']
        gold = (await sdk.call_tool('get_economy')).structuredContent['gold']
        slot = next(o['slot'] for o in offers if o['purchase_cost'] <= gold)
        bought = await sdk.call_tool('buy_unit', {'shop_slot': slot})
        assert not bought.isError
        origin = next(c['location'] for c in bought.structuredContent['unit_changes'] if c['after'])
        for source, target in [(origin, board(6, 3)), (board(6, 3), board(0, 0)), (board(0, 0), board(5, 2)), (board(5, 2), bench(8))]:
            moved = await sdk.call_tool('move_unit', {'source': source, 'target': target})
            assert not moved.isError, moved.structuredContent
            assert moved.structuredContent['source'] == source and moved.structuredContent['target'] == target
            assert moved.structuredContent['status']['round'] == 2
        assert (await sdk.call_tool('move_unit', {'source': bench(8), 'target': bench(8)})).structuredContent['code'] == 'unsupported_action'
        assert (await sdk.call_tool('move_unit', {'source': board(0, 0), 'target': bench(8)})).structuredContent['code'] == 'empty_slot'
        await sdk.call_tool('close_game')


@pytest.mark.parametrize('incoming,outgoing,slot', [(True, False, 5), (False, True, 5), (True, True, 5), (True, True, 0), (False, True, 0)])
def test_native_glove_swap_tracking_with_actual_first_vacancy(tmp_path, incoming, outgoing, slot):
    session = session_fixture(tmp_path)
    gloves = ['thieves_gloves', 'bf_sword', 'chain_vest']
    install_units(session, [(slot, 'garen', 1, gloves if incoming else [])], [((0, 0), 'fiora', 1, gloves if outgoing else [])])
    receipt = session.move_unit(source=bench(slot), target=board(0, 0))
    assert [c['location'] for c in receipt['unit_changes']] == [board(0, 0), bench(0)] + ([bench(slot)] if slot else [])
    player = session.game.player_manager.player_states['player_0']
    assert sorted(player.thieves_gloves_loc) == sorted(([[0, 0]] if incoming else []) + ([[0, -1]] if outgoing else []))


def test_full_bench_swap_and_full_regular_board_capacity(tmp_path):
    session = session_fixture(tmp_path)
    names = ['garen', 'fiora', 'nami', 'vayne', 'nidalee', 'diana', 'lissandra', 'wukong', 'maokai']
    player = install_units(session, [(i, name, 1, []) for i, name in enumerate(names)], [((0, 0), 'twistedfate', 1, [])])
    player.max_units = 1
    session.move_unit(source=board(0, 0), target=bench(8))
    player = session.game.player_manager.player_states['player_0']
    assert player.bench[8].name == 'twistedfate'
    assert player.board[0][0].name == 'maokai' and player.num_units_in_play == 1


def test_azir_native_entry_reposition_guard_linkage_and_removal(tmp_path):
    session = session_fixture(tmp_path)
    install_units(session, [(0, 'azir', 1, [])])
    receipt = session.move_unit(source=bench(0), target=board(3, 2))
    player = session.game.player_manager.player_states['player_0']
    azir = player.board[3][2]
    assert azir.overlord and len(azir.sandguard_overlord_coordinates) == 2
    assert player.num_units_in_play == 1
    assert len([c for c in receipt['unit_changes'] if c['after'] and c['after']['champion'] == 'sandguard']) == 2
    guard = list(azir.sandguard_overlord_coordinates[0])
    session.move_unit(source=board(*guard), target=board(6, 3))
    assert [6, 3] in session.game.player_manager.player_states['player_0'].board[3][2].sandguard_overlord_coordinates
    session.move_unit(source=board(3, 2), target=board(0, 0))
    player = session.game.player_manager.player_states['player_0']
    links = list(player.board[0][0].sandguard_overlord_coordinates)
    assert session.game.player_manager.action_handlers['player_0'].move_sell_board_mask[0][0][36] == 0
    removed = session.move_unit(source=board(0, 0), target=bench(8))
    player = session.game.player_manager.player_states['player_0']
    assert player.bench[8].overlord is False
    assert player.bench[8].sandguard_overlord_coordinates == links
    assert player.num_units_in_play == 0
    assert len([c for c in removed['unit_changes'] if c['before'] and c['before']['champion'] == 'sandguard']) == 2


@pytest.mark.parametrize('special', ['sandguard', 'garen'])
def test_board_dummy_positions_but_cannot_leave_or_be_displaced(tmp_path, special):
    session = session_fixture(tmp_path)
    player = install_units(session, [(0, 'fiora', 1, [])], [((0, 0), special, 1, [])])
    player.board[0][0].target_dummy = True
    player.num_units_in_play = 0
    session.move_unit(source=board(0, 0), target=board(6, 3))
    for source, target in [(board(6, 3), bench(0)), (board(6, 3), bench(8)), (bench(0), board(6, 3))]:
        with pytest.raises(SessionError) as error:
            session.move_unit(source=source, target=target)
        assert error.value.code == 'unsupported_action'


def test_azir_guard_capacity_accounts_for_displacement_and_outgoing_guard_removal(tmp_path):
    session = session_fixture(tmp_path)
    player = install_units(session, [(0, 'azir', 1, [])], [((x, y), 'garen', 1, []) for x in range(7) for y in range(4)])
    player.max_units = 28
    with pytest.raises(SessionError) as error:
        session.move_unit(source=bench(0), target=board(0, 0))
    assert error.value.code == 'capacity_exceeded'
    assert error.value.details['available'] == 0
    # Actual native entry then a second Azir replaces it on a crowded board.
    session = session_fixture(tmp_path / 'replacement')
    player = install_units(session, [(0, 'azir', 1, []), (1, 'azir', 1, [])])
    session.move_unit(source=bench(0), target=board(3, 2))
    player = session.game.player_manager.player_states['player_0']
    from Simulator.battle.champion import champion
    with session.simulator_scope():
        for x in range(7):
            for y in range(4):
                if player.board[x][y] is None:
                    player.board[x][y] = champion('garen', target_dummy=True)
    session.move_unit(source=bench(1), target=board(3, 2))
    player = session.game.player_manager.player_states['player_0']
    assert player.board[3][2].overlord and not player.bench[0].overlord
    assert player.num_units_in_play == 1


@pytest.mark.parametrize('failure', ['native_action', 'observation', 'mask', 'baseline', 'noop', 'corrupt', 'receipt', 'postcondition', 'native', 'audit'])
def test_movement_failure_rolls_back_full_graph_rng_logs_and_fresh_retry(tmp_path, monkeypatch, failure):
    from support import capture_committed_state, assert_committed_state_unchanged, assert_native_aliases
    import os
    from Simulator.game.player import Player
    from Simulator.simulators.tft_simulator import TFT_Simulator
    import tft_mcp.session as module
    from support import gameplay_with_numpy as gameplay
    sessions = [session_fixture(tmp_path / name) for name in ('actual', 'reference')]
    for session in sessions:
        install_units(session, [(0, 'garen', 1, ['bf_sword'])])
    session, reference = sessions
    before = capture_committed_state(session)
    with monkeypatch.context() as patch:
        native = Player.move_champ_action
        def fail_native(player, *args):
            if failure == 'noop':
                return False
            result = native(player, *args)
            raise RuntimeError('after actual native movement')
        step = TFT_Simulator.step
        def fail_step(game, action):
            selected = game.agent_selection
            result = step(game, action)
            if (failure == 'observation' and selected == 'player_0') or (failure == 'baseline' and selected != 'player_0'):
                raise RuntimeError('after actual simulator step')
            if failure == 'corrupt' and selected == 'player_0':
                game.player_manager.player_states['player_0'].gold += 1
            if failure == 'native' and selected == 'player_0':
                with open('log.txt', 'a') as stream:
                    stream.write('partial candidate movement log')
                raise OSError('native file fault')
            return result
        def fail_after(*args):
            raise RuntimeError('movement receipt or postcondition fault')
        if failure in {'native_action', 'noop'}:
            patch.setattr(Player, 'move_champ_action', fail_native)
        elif failure == 'mask':
            from Simulator.encoding.token.action import ActionToken
            original = ActionToken.update_action_mask
            def fail_mask(handler, action):
                original(handler, action)
                raise RuntimeError('after actual native action-mask update')
            patch.setattr(ActionToken, 'update_action_mask', fail_mask)
        elif failure in {'observation', 'baseline', 'corrupt', 'native'}:
            patch.setattr(TFT_Simulator, 'step', fail_step)
        elif failure in {'receipt', 'postcondition'}:
            patch.setattr(module, 'unit_changes' if failure == 'receipt' else 'check_action_status', fail_after)
        else:
            def fail_replace(*args):
                raise OSError('audit replacement fault')
            patch.setattr(os, 'replace', fail_replace)
        with pytest.raises(SessionError) as error:
            session.move_unit(source=bench(0), target=board(6, 3))
        assert error.value.code == ('log_unavailable' if failure in {'native', 'audit'} else 'internal_error')
    assert_committed_state_unchanged(session, before)
    session.move_unit(source=bench(0), target=board(6, 3))
    reference.move_unit(source=bench(0), target=board(6, 3))
    assert gameplay(session) == gameplay(reference)
    assert normalized_graph(session.game) == normalized_graph(reference.game)
    assert_native_aliases(session.game)


@pytest.mark.parametrize('corruption', ['identity_noop', 'equipment', 'catalog', 'count', 'shop', 'inventory', 'gloves', 'coordinates'])
def test_corrupt_movement_results_cannot_commit(tmp_path, monkeypatch, corruption):
    import pickle
    from Simulator.game.player import Player
    session = session_fixture(tmp_path)
    install_units(session, board=[((0, 0), 'garen', 1, []), ((6, 3), 'garen', 1, [])])
    native = Player.move_champ_action
    def corrupt(player, *args):
        if corruption == 'identity_noop':
            return True
        result = native(player, *args)
        if corruption == 'equipment':
            player.board[0][0].items.append('bf_sword')
        elif corruption == 'catalog':
            player.triple_catalog.clear()
        elif corruption == 'count':
            player.num_units_in_play += 1
        elif corruption == 'shop':
            player.shop[0] = None
        elif corruption == 'inventory':
            player.item_bench[0] = 'bf_sword'
        elif corruption == 'gloves':
            player.thieves_gloves_loc.append([0, 0])
        else:
            player.board[0][0].x = 5
        return result
    before = pickle.dumps(session.game)
    monkeypatch.setattr(Player, 'move_champ_action', corrupt)
    with pytest.raises(SessionError) as error:
        session.move_unit(source=board(6, 3), target=board(0, 0))
    assert error.value.code == 'internal_error'
    assert pickle.dumps(session.game) == before


@pytest.mark.anyio
async def test_production_all_family_movement_tape_replays_with_extra_reads_and_rejection(tmp_path):
    import json
    from support import client
    journeys = []
    tape = [(bench(0), board(6, 3)), (bench(1), board(0, 0)),
            (board(6, 3), board(0, 0)), (board(0, 0), board(6, 3)),
            (board(6, 3), bench(0)), (bench(0), board(0, 0)),
            (board(0, 0), bench(0)), (board(0, 0), board(6, 3)), (board(6, 3), bench(8))]
    for extra in (False, True):
        path = tmp_path / str(extra)
        path.mkdir()
        async with client(path) as sdk:
            await sdk.call_tool('start_game', {'seed': 0})
            for _ in range(3):
                await sdk.call_tool('end_turn')
            offers = (await sdk.call_tool('get_shop')).structuredContent['slots']
            chosen = []
            for offer in offers:
                if offer['unit'] and offer['unit']['champion'] not in [o['unit']['champion'] for o in chosen]:
                    chosen.append(offer)
                if len(chosen) == 2:
                    break
            assert len(chosen) == 2
            for offer in chosen:
                bought = await sdk.call_tool('buy_unit', {'shop_slot': offer['slot']})
                assert not bought.isError, bought.structuredContent
            receipts = []
            for source, target in tape:
                if extra:
                    for name in ('get_board', 'get_traits', 'get_items', 'get_bench', 'get_shop', 'get_economy', 'get_round'):
                        assert not (await sdk.call_tool(name)).isError
                    rejected = await sdk.call_tool('move_unit', {'source': source, 'target': source})
                    assert rejected.isError and rejected.structuredContent['code'] == 'unsupported_action'
                result = await sdk.call_tool('move_unit', {'source': source, 'target': target})
                assert not result.isError, result.structuredContent
                data = result.structuredContent
                data['status'].pop('game_id')
                receipts.append(data)
            state = []
            for name in ('get_board', 'get_traits', 'get_bench', 'get_items', 'get_economy', 'get_shop'):
                data = (await sdk.call_tool(name)).structuredContent
                data.pop('game_id')
                state.append(data)
            assert receipts[-1]['status']['planning_budget']['remaining'] == 3
            await sdk.call_tool('close_game')
        events = [json.loads(line) for line in (path / 'audit.jsonl').read_text().splitlines()]
        progress = [{key: event[key] for key in ('player_id', 'round', 'kind', 'action', 'placements')}
                    for event in events if event['event'] == 'progression']
        assert any(e['event'] == 'tool_result' and e['tool'] == 'move_unit' for e in events)
        journeys.append((receipts, state, progress))
    assert journeys[0] == journeys[1]


@pytest.mark.anyio
async def test_production_strict_movement_inputs_preserve_state(tmp_path):
    from support import client
    invalid_locations = [None, {}, {'kind': 'board', 'x': 0}, {'kind': 'board', 'x': 0, 'y': 0, 'slot': 0},
                         {'kind': 'bench', 'slot': True}, {'kind': 'bench', 'slot': 1.0}, {'kind': 'bench', 'slot': None},
                         {'kind': 'bench', 'slot': -1}, {'kind': 'bench', 'slot': 9}, board(7, 0), board(0, 4),
                         board(True, 0), board(0, 1.0), {'kind': 'opponent', 'slot': 0}]
    async with client(tmp_path) as sdk:
        await sdk.call_tool('start_game', {'seed': 0})
        before = (await sdk.call_tool('get_game_status')).structuredContent
        for field in ('source', 'target'):
            for location in invalid_locations:
                arguments = {'source': bench(0), 'target': board(0, 0), field: location}
                error = await sdk.call_tool('move_unit', arguments)
                assert error.isError and error.structuredContent['code'] == 'invalid_input'
        for arguments in ({}, {'source': bench(0)}, {'source': bench(0), 'target': board(0, 0), 'player_id': 'player_0'}):
            assert (await sdk.call_tool('move_unit', arguments)).structuredContent['code'] == 'invalid_input'
        assert (await sdk.call_tool('get_game_status')).structuredContent == before
        await sdk.call_tool('close_game')


@pytest.mark.parametrize('corruption', ['coordinates', 'missing_guard', 'overlord', 'old_links'])
def test_corrupt_azir_movement_rolls_back(tmp_path, monkeypatch, corruption):
    import pickle
    from Simulator.game.player import Player
    session = session_fixture(tmp_path)
    install_units(session, [(0, 'azir', 1, [])])
    if corruption == 'old_links':
        session.move_unit(source=bench(0), target=board(3, 2))
    native = Player.move_champ_action
    def corrupt(player, *args):
        result = native(player, *args)
        if corruption == 'old_links':
            player.bench[0].sandguard_overlord_coordinates = []
        else:
            azir = player.board[3][2]
            if corruption == 'coordinates':
                azir.sandguard_overlord_coordinates[0] = [7, 0]
            elif corruption == 'missing_guard':
                x, y = azir.sandguard_overlord_coordinates[0]
                player.board[x][y] = None
            else:
                azir.overlord = False
        return result
    before = pickle.dumps(session.game)
    monkeypatch.setattr(Player, 'move_champ_action', corrupt)
    with pytest.raises(SessionError) as error:
        session.move_unit(source=board(3, 2) if corruption == 'old_links' else bench(0),
                          target=bench(0) if corruption == 'old_links' else board(3, 2))
    assert error.value.code == 'internal_error'
    assert pickle.dumps(session.game) == before


@pytest.mark.anyio
async def test_sdk_memory_rare_real_azir_fixture_and_atomic_guard_rejection(tmp_path, monkeypatch):
    from support import memory_client
    from support import gameplay_with_numpy as gameplay
    session = session_fixture(tmp_path)
    install_units(session, [(0, 'azir', 1, [])])
    async with memory_client(session, monkeypatch) as sdk:
        entered = await sdk.call_tool('move_unit', {'source': bench(0), 'target': board(3, 2)})
        assert not entered.isError
        guards = [c['location'] for c in entered.structuredContent['unit_changes'] if c['after'] and c['after']['champion'] == 'sandguard']
        assert len(guards) == 2
        before = gameplay(session)
        rejected = await sdk.call_tool('move_unit', {'source': guards[0], 'target': bench(8)})
        assert rejected.isError and rejected.structuredContent['code'] == 'unsupported_action'
        assert gameplay(session) == before
        moved = await sdk.call_tool('move_unit', {'source': guards[0], 'target': board(6, 3)})
        assert not moved.isError
        removed = await sdk.call_tool('move_unit', {'source': board(3, 2), 'target': bench(8)})
        assert not removed.isError
        assert any(c['before'] and c['before']['champion'] == 'sandguard' and c['after'] is None for c in removed.structuredContent['unit_changes'])
        await sdk.call_tool('close_game')


def test_movement_lifecycle_budget_and_detached_receipt(tmp_path):
    from tft_mcp.session import GameSession
    idle = GameSession(tmp_path / 'idle.jsonl', tmp_path / 'idle-native')
    with pytest.raises(SessionError) as error:
        idle.move_unit(source=bench(0), target=board(0, 0))
    assert error.value.code == 'no_game'
    session = session_fixture(tmp_path)
    install_units(session, [(0, 'garen', 1, [])])
    source, target = bench(0), board(0, 0)
    receipt = session.move_unit(source=source, target=target)
    source['slot'] = 8
    target['x'] = 6
    receipt['unit_changes'][0]['after']['items'].append('bf_sword')
    assert receipt['source'] == bench(0) and receipt['target'] == board(0, 0)
    assert session.get_board()['slots'][0]['unit']['items'] == []
    for _ in range(13):
        session.controlled_action([0, 0, 0])
    with pytest.raises(SessionError) as error:
        session.move_unit(source=board(0, 0), target=board(6, 3))
    assert error.value.code == 'budget_exhausted'
    session.state = 'terminal'
    with pytest.raises(SessionError) as error:
        session.move_unit(source=board(0, 0), target=board(6, 3))
    assert error.value.code == 'game_terminal'
    with pytest.raises(SessionError) as error:
        session.move_unit(source=board(0, 0), target=board(6, 3), extra=True)
    assert error.value.code == 'invalid_input'


@pytest.mark.anyio
async def test_production_movement_audit_failure_then_fresh_retry(tmp_path):
    from support import client
    async with client(tmp_path) as sdk:
        await sdk.call_tool('start_game', {'seed': 0})
        await sdk.call_tool('end_turn')
        offers = (await sdk.call_tool('get_shop')).structuredContent['slots']
        gold = (await sdk.call_tool('get_economy')).structuredContent['gold']
        offer = next(o for o in offers if o['purchase_cost'] <= gold)
        purchased = await sdk.call_tool('buy_unit', {'shop_slot': offer['slot']})
        source = next(c['location'] for c in purchased.structuredContent['unit_changes'] if c['after'])
        before = [(await sdk.call_tool(read)).structuredContent for read in ('get_game_status', 'get_board', 'get_bench', 'get_traits')]
        audit = tmp_path / 'audit.jsonl'
        saved = audit.read_bytes()
        audit.unlink()
        audit.mkdir()
        rejected = await sdk.call_tool('move_unit', {'source': source, 'target': board(6, 3)})
        assert rejected.isError and rejected.structuredContent['code'] == 'log_unavailable'
        audit.rmdir()
        audit.write_bytes(saved)
        assert [(await sdk.call_tool(read)).structuredContent for read in ('get_game_status', 'get_board', 'get_bench', 'get_traits')] == before
        assert not (await sdk.call_tool('move_unit', {'source': source, 'target': board(6, 3)})).isError
        await sdk.call_tool('close_game')


@pytest.mark.parametrize('corruption', ['count', 'guard_link', 'guard_type'])
def test_inconsistent_native_metadata_rejects_before_movement(tmp_path, corruption):
    import pickle
    session = session_fixture(tmp_path)
    install_units(session, [(0, 'azir', 1, [])])
    session.move_unit(source=bench(0), target=board(3, 2))
    player = session.game.player_manager.player_states['player_0']
    if corruption == 'count':
        player.num_units_in_play = 3
    elif corruption == 'guard_link':
        player.board[3][2].sandguard_overlord_coordinates[0] = [3, 2]
    else:
        x, y = player.board[3][2].sandguard_overlord_coordinates[0]
        player.board[x][y].target_dummy = False
    before = pickle.dumps(session.game)
    with pytest.raises(SessionError) as error:
        session.move_unit(source=board(3, 2), target=bench(0))
    assert error.value.code == 'internal_error'
    assert pickle.dumps(session.game) == before


def test_guard_can_swap_with_its_board_azir_and_update_linkage(tmp_path):
    session = session_fixture(tmp_path)
    install_units(session, [(0, 'azir', 1, [])])
    session.move_unit(source=bench(0), target=board(3, 2))
    player = session.game.player_manager.player_states['player_0']
    coord = list(player.board[3][2].sandguard_overlord_coordinates[0])
    session.move_unit(source=board(*coord), target=board(3, 2))
    player = session.game.player_manager.player_states['player_0']
    assert player.board[3][2].name == 'sandguard'
    assert player.board[coord[0]][coord[1]].name == 'azir'
    assert [3, 2] in player.board[coord[0]][coord[1]].sandguard_overlord_coordinates
    session.move_unit(source=board(*coord), target=bench(0))
    assert session.game.player_manager.player_states['player_0'].board[3][2] is None

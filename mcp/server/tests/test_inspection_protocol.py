import pytest

from support import client, INSPECTION_TOOLS


@pytest.mark.anyio
async def test_board_discovery_and_seeded_coordinates(tmp_path):
    async with client(tmp_path) as session:
        tools = {tool.name: tool for tool in (await session.list_tools()).tools}
        assert 'get_board' in tools
        assert tools['get_board'].inputSchema['additionalProperties'] is False
        started = (await session.call_tool('start_game', {'seed': 0})).structuredContent
        result = await session.call_tool('get_board', {})
        assert not result.isError
        board = result.structuredContent
        assert set(board) == {'game_id', 'player_id', 'round', 'slots', 'num_units_in_play', 'max_units'}
        assert board['game_id'] == started['game_id']
        assert board['player_id'] == 'player_0'
        assert board['round'] == 1
        assert len(board['slots']) == 28
        assert board['slots'][0]['location'] == {'kind': 'board', 'x': 0, 'y': 0}
        assert board['slots'][27]['location'] == {'kind': 'board', 'x': 6, 'y': 3}
        assert all(slot['unit'] is None for slot in board['slots'])


@pytest.mark.anyio
async def test_bench_has_explicit_empty_reserve_slots(tmp_path):
    async with client(tmp_path) as session:
        await session.call_tool('start_game', {'seed': 0})
        result = await session.call_tool('get_bench', {})
        assert not result.isError
        assert result.structuredContent['slots'] == [
            {'location': {'kind': 'bench', 'slot': slot}, 'unit': None} for slot in range(9)]
        assert set(result.structuredContent) == {'game_id', 'player_id', 'round', 'slots'}


@pytest.mark.anyio
async def test_shop_uses_real_seeded_offers_and_native_purchase_prices(tmp_path):
    from tft_mcp.session import GameSession
    from Simulator.game.pool_stats import cost_star_values

    reference = GameSession(tmp_path / 'reference.jsonl', tmp_path / 'reference-native')
    reference.start_game(0)
    player = reference.game.player_manager.player_states['player_0']
    expected = [(unit.name, unit.stars, cost_star_values[unit.cost - 1][unit.stars - 1])
                if unit else None for unit in player.shop_champions]
    async with client(tmp_path) as session:
        await session.call_tool('start_game', {'seed': 0})
        result = await session.call_tool('get_shop', {})
        assert not result.isError
        assert len(result.structuredContent['slots']) == 5
        actual = [(slot['unit']['champion'], slot['unit']['stars'], slot['purchase_cost'])
                  if slot['unit'] else None for slot in result.structuredContent['slots']]
        assert actual == expected
        assert [slot['slot'] for slot in result.structuredContent['slots']] == [0, 1, 2, 3, 4]


@pytest.mark.anyio
async def test_inventory_has_ten_explicit_empty_slots(tmp_path):
    async with client(tmp_path) as session:
        await session.call_tool('start_game', {'seed': 0})
        result = await session.call_tool('get_items', {})
        assert not result.isError
        assert result.structuredContent['slots'] == [{'slot': slot, 'item': None} for slot in range(10)]


@pytest.mark.anyio
async def test_economy_reports_own_scalars_and_shared_budget(tmp_path):
    from tft_mcp.session import GameSession
    reference = GameSession(tmp_path / 'reference.jsonl', tmp_path / 'reference-native')
    reference.start_game(0)
    player = reference.game.player_manager.player_states['player_0']
    async with client(tmp_path) as session:
        status = (await session.call_tool('start_game', {'seed': 0})).structuredContent
        result = await session.call_tool('get_economy', {})
        assert not result.isError
        assert result.structuredContent == {'game_id': status['game_id'], 'player_id': 'player_0', 'round': 1,
            'gold': player.gold, 'health': player.health, 'level': player.level, 'exp': player.exp,
            'planning_budget': status['planning_budget']}


@pytest.mark.anyio
async def test_traits_report_stored_native_counts_without_recomputation(tmp_path):
    from tft_mcp.session import GameSession
    reference = GameSession(tmp_path / 'reference.jsonl', tmp_path / 'reference-native')
    reference.start_game(0)
    player = reference.game.player_manager.player_states['player_0']
    async with client(tmp_path) as session:
        await session.call_tool('start_game', {'seed': 0})
        result = await session.call_tool('get_traits', {'player_id': 'player_0'})
        assert not result.isError
        assert result.structuredContent['traits'] == [
            {'trait_id': key, 'count': player.team_composition[key], 'tier': player.team_tiers[key]}
            for key in sorted(player.team_composition)]
        assert result.structuredContent['traits']


@pytest.mark.anyio
async def test_round_matches_current_session_round(tmp_path):
    async with client(tmp_path) as session:
        status = (await session.call_tool('start_game', {'seed': 0})).structuredContent
        result = await session.call_tool('get_round', {})
        assert not result.isError
        assert result.structuredContent == {'game_id': status['game_id'], 'round': 1}
        status = (await session.call_tool('end_turn', {})).structuredContent
        assert (await session.call_tool('get_round', {})).structuredContent == {
            'game_id': status['game_id'], 'round': 2}


@pytest.mark.anyio
async def test_inspection_errors_are_strict_and_restart_clears_state(tmp_path):
    async with client(tmp_path) as session:
        tools = {tool.name: tool for tool in (await session.list_tools()).tools}
        assert set(INSPECTION_TOOLS) <= tools.keys()
        for name in INSPECTION_TOOLS:
            assert tools[name].inputSchema['additionalProperties'] is False
            assert tools[name].outputSchema['additionalProperties'] is False
            result = await session.call_tool(name, {})
            assert result.isError and result.structuredContent['code'] == 'no_game'
            result = await session.call_tool(name, {'extra': True})
            assert result.isError and result.structuredContent['code'] == 'invalid_input'
        for name in ('get_board', 'get_traits'):
            for value in (None, True, 1, [], {}):
                result = await session.call_tool(name, {'player_id': value})
                assert result.isError and result.structuredContent['code'] == 'invalid_input'
            result = await session.call_tool(name, {'player_id': 'player_1'})
            assert result.isError and result.structuredContent['code'] == 'no_game'
        initial = (await session.call_tool('start_game', {'seed': 0})).structuredContent
        for name in ('get_board', 'get_traits'):
            for value in ('unknown', ''):
                result = await session.call_tool(name, {'player_id': value})
                assert result.isError and result.structuredContent['code'] == 'invalid_player'
                assert result.structuredContent['details']['supported_ids'] == [f'player_{index}' for index in range(8)]
        assert (await session.call_tool('get_game_status', {})).structuredContent == initial
        await session.call_tool('close_game', {})
        for name in INSPECTION_TOOLS:
            result = await session.call_tool(name, {})
            assert result.isError and result.structuredContent['code'] == 'no_game'
        fresh = (await session.call_tool('start_game', {'seed': 0})).structuredContent
        assert fresh['game_id'] != initial['game_id']
        assert (await session.call_tool('get_economy', {})).structuredContent['planning_budget'] == {'capacity': 14, 'remaining': 14}


@pytest.mark.anyio
async def test_terminal_inspection_retains_elimination_round_until_close(tmp_path):
    async with client(tmp_path) as session:
        await session.call_tool('start_game', {'seed': 0})
        for _ in range(30):
            status = (await session.call_tool('end_turn', {})).structuredContent
            if status['state'] == 'terminal':
                break
        assert status['state'] == 'terminal'
        own = {}
        for name in INSPECTION_TOOLS:
            result = await session.call_tool(name, {})
            assert not result.isError, result.structuredContent
            own[name] = result.structuredContent
            assert (await session.call_tool(name, {})).structuredContent == own[name]
        assert own['get_economy']['health'] <= 0
        assert own['get_economy']['planning_budget'] is None
        assert own['get_round']['round'] == status['round']
        assert len({own[name]['round'] for name in INSPECTION_TOOLS if name != 'get_round'}) == 1
        assert own['get_board']['round'] < own['get_round']['round']
        assert (await session.call_tool('get_game_status', {})).structuredContent == status
        await session.call_tool('close_game', {})
        result = await session.call_tool('get_board', {})
        assert result.isError and result.structuredContent['code'] == 'no_game'
        await session.call_tool('start_game', {'seed': 0})
        assert (await session.call_tool('get_board', {})).structuredContent['round'] == 1


@pytest.mark.anyio
async def test_failed_inspection_audit_preserves_game_and_allows_retry(tmp_path):
    async with client(tmp_path) as session:
        status = (await session.call_tool('start_game', {'seed': 0})).structuredContent
        board = (await session.call_tool('get_board', {})).structuredContent
        audit = tmp_path / 'audit.jsonl'
        saved = audit.read_bytes()
        audit.unlink()
        audit.mkdir()
        failed = await session.call_tool('get_board', {})
        assert failed.isError and failed.structuredContent['code'] == 'log_unavailable'
        audit.rmdir()
        audit.write_bytes(saved)
        assert (await session.call_tool('get_board', {})).structuredContent == board
        assert (await session.call_tool('get_game_status', {})).structuredContent == status

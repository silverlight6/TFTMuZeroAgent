import pytest

from test_protocol import client, anyio_backend


@pytest.mark.anyio
async def test_public_players_are_discoverable_strict_and_available_until_close(tmp_path):
    async with client(tmp_path) as session:
        tools = {tool.name: tool for tool in (await session.list_tools()).tools}
        assert "get_players" in tools
        tool = tools["get_players"]
        assert tool.inputSchema == {"type": "object", "properties": {}, "additionalProperties": False}
        assert tool.outputSchema["additionalProperties"] is False
        assert tool.outputSchema["properties"]["players"]["items"]["additionalProperties"] is False
        result = await session.call_tool("get_players", {})
        assert result.isError and result.structuredContent["code"] == "no_game"
        result = await session.call_tool("get_players", {"player_id": "player_1"})
        assert result.isError and result.structuredContent["code"] == "invalid_input"
        started = (await session.call_tool("start_game", {"seed": 0})).structuredContent
        result = await session.call_tool("get_players", {})
        assert not result.isError
        assert result.structuredContent == {"game_id": started["game_id"], "players": [
            {"player_id": f"player_{index}", "controlled": index == 0, "status": "alive",
             "health": 100, "level": 1, "placement": None} for index in range(8)]}
        await session.call_tool("close_game", {})
        result = await session.call_tool("get_players", {})
        assert result.isError and result.structuredContent["code"] == "no_game"


@pytest.mark.anyio
async def test_living_opponents_are_visible_without_changing_later_gameplay(tmp_path):
    (tmp_path / 'scouting').mkdir()
    (tmp_path / 'reference').mkdir()
    async with client(tmp_path / 'scouting') as scout, client(tmp_path / 'reference') as reference:
        for session in (scout, reference):
            await session.call_tool('start_game', {'seed': 0})
        for _ in range(4):
            for name in ('get_board', 'get_traits'):
                result = await scout.call_tool(name, {'player_id': 'player_1'})
                assert not result.isError, result.structuredContent
                assert result.structuredContent['player_id'] == 'player_1'
                if name == 'get_board':
                    assert len(result.structuredContent['slots']) == 28
                    assert result.structuredContent['slots'][27]['location'] == {'kind': 'board', 'x': 6, 'y': 3}
                else:
                    assert result.structuredContent['traits']
            await scout.call_tool('get_players', {})
            actual = (await scout.call_tool('end_turn', {})).structuredContent
            expected = (await reference.call_tool('end_turn', {})).structuredContent
            assert {k: v for k, v in actual.items() if k != 'game_id'} == {k: v for k, v in expected.items() if k != 'game_id'}
            for name in ('get_board', 'get_traits', 'get_players'):
                actual = (await scout.call_tool(name, {})).structuredContent
                expected = (await reference.call_tool(name, {})).structuredContent
                assert {k: v for k, v in actual.items() if k != 'game_id'} == {k: v for k, v in expected.items() if k != 'game_id'}
        assert any(slot['unit'] for slot in (await scout.call_tool('get_board', {'player_id': 'player_1'})).structuredContent['slots'])


@pytest.mark.anyio
async def test_terminal_public_list_and_removed_winner_preserve_own_categories(tmp_path):
    async with client(tmp_path) as session:
        await session.call_tool('start_game', {'seed': 0})
        initial = (await session.call_tool('get_players', {})).structuredContent
        observed_removal = False
        for _ in range(30):
            status = (await session.call_tool('end_turn', {})).structuredContent
            public = (await session.call_tool('get_players', {})).structuredContent
            assert [row['player_id'] for row in public['players']] == [row['player_id'] for row in initial['players']]
            for row in public['players']:
                if row['status'] == 'eliminated' and not row['controlled']:
                    observed_removal = True
                    result = await session.call_tool('get_board', {'player_id': row['player_id']})
                    assert result.isError and result.structuredContent['code'] == 'player_eliminated'
            if status['state'] == 'terminal':
                break
        assert status['state'] == 'terminal' and observed_removal
        assert sorted(row['placement'] for row in public['players']) == list(range(1, 9))
        winner = next(row for row in public['players'] if row['status'] == 'winner')
        assert winner['placement'] == 1 and not winner['controlled']
        assert all(row['status'] != 'alive' for row in public['players'])
        for name in ('get_board', 'get_traits'):
            result = await session.call_tool(name, {'player_id': winner['player_id']})
            assert result.isError and result.structuredContent['code'] == 'player_eliminated'
            assert result.structuredContent['details']['player_id'] == winner['player_id']
            assert 'get_players' in result.structuredContent['message']
            own = await session.call_tool(name, {})
            assert not own.isError
            assert own.structuredContent['round'] < status['round']
        assert (await session.call_tool('get_players', {})).structuredContent == public
        assert (await session.call_tool('get_game_status', {})).structuredContent == status
        await session.call_tool('close_game', {})
        assert (await session.call_tool('get_players', {})).structuredContent['code'] == 'no_game'
        await session.call_tool('start_game', {'seed': 0})
        fresh = (await session.call_tool('get_players', {})).structuredContent
        assert fresh['game_id'] != public['game_id']
        assert fresh['players'] == initial['players']


@pytest.mark.anyio
async def test_failed_public_audit_preserves_game_and_allows_retry(tmp_path):
    async with client(tmp_path) as session:
        status = (await session.call_tool('start_game', {'seed': 0})).structuredContent
        calls = [('get_players', {}), ('get_board', {'player_id': 'player_1'}), ('get_traits', {'player_id': 'player_1'})]
        expected = [(await session.call_tool(name, args)).structuredContent for name, args in calls]
        audit = tmp_path / 'audit.jsonl'
        saved = audit.read_bytes()
        audit.unlink()
        audit.mkdir()
        for name, args in calls:
            failed = await session.call_tool(name, args)
            assert failed.isError and failed.structuredContent['code'] == 'log_unavailable'
        audit.rmdir()
        audit.write_bytes(saved)
        for (name, args), result in zip(calls, expected):
            assert (await session.call_tool(name, args)).structuredContent == result
        assert (await session.call_tool('get_game_status', {})).structuredContent == status

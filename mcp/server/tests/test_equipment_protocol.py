import pytest

from test_protocol import client
from test_movement import bench, board


@pytest.mark.anyio
async def test_stdio_equipment_discovery_strict_schema_and_natural_item(tmp_path):
    from tft_mcp.transport import LOCATION_SCHEMA, UNIT_CHANGE_SCHEMA, ITEM_CHANGE_SCHEMA, STATUS_SCHEMA
    async with client(tmp_path) as sdk:
        tools = {t.name: t for t in (await sdk.list_tools()).tools}
        tool = tools['equip_item']
        assert tool.inputSchema['properties'] == {'item_slot': {'type': 'integer', 'minimum': 0, 'maximum': 9}, 'target': LOCATION_SCHEMA}
        assert tool.inputSchema['additionalProperties'] is False
        assert set(tool.inputSchema['required']) == {'item_slot', 'target'}
        assert set(tool.outputSchema['required']) == {'item_slot', 'item_id', 'target', 'unit_changes', 'item_changes', 'kayn_form', 'status'}
        assert tool.outputSchema['additionalProperties'] is False
        assert tool.outputSchema['properties']['unit_changes']['items'] == UNIT_CHANGE_SCHEMA
        assert tool.outputSchema['properties']['item_changes']['items'] == ITEM_CHANGE_SCHEMA
        assert tool.outputSchema['properties']['status'] == STATUS_SCHEMA
        assert (await sdk.call_tool('equip_item', {'item_slot': 0, 'target': bench(0)})).structuredContent['code'] == 'no_game'
        for arguments in [{}, {'item_slot': True, 'target': bench(0)}, {'item_slot': 1.0, 'target': bench(0)}, {'item_slot': -1, 'target': bench(0)}, {'item_slot': 10, 'target': bench(0)}, {'item_slot': 0, 'target': None}, {'item_slot': 0, 'target': {'kind': 'bench', 'slot': False}}, {'item_slot': 0, 'target': {**board(0, 0), 'slot': 0}}, {'item_slot': 0, 'target': bench(0), 'player_id': 'player_1'}]:
            result = await sdk.call_tool('equip_item', arguments)
            assert result.isError and result.structuredContent['code'] == 'invalid_input'
        await sdk.call_tool('start_game', {'seed': 0})
        await sdk.call_tool('end_turn')
        await sdk.call_tool('buy_unit', {'shop_slot': 0})
        await sdk.call_tool('buy_unit', {'shop_slot': 1})
        await sdk.call_tool('end_turn')
        items = (await sdk.call_tool('get_items')).structuredContent['slots']
        assert items[0]['item'] == 'sparring_gloves'
        units = (await sdk.call_tool('get_board')).structuredContent['slots']
        target = next(s['location'] for s in units if s['unit'])
        before = [(await sdk.call_tool(read)).structuredContent for read in ['get_game_status', 'get_board', 'get_items']]
        audit = tmp_path / 'audit.jsonl'
        saved = audit.read_bytes()
        audit.unlink()
        audit.mkdir()
        rejected = await sdk.call_tool('equip_item', {'item_slot': 0, 'target': target})
        assert rejected.isError and rejected.structuredContent['code'] == 'log_unavailable'
        audit.rmdir()
        audit.write_bytes(saved)
        assert [(await sdk.call_tool(read)).structuredContent for read in ['get_game_status', 'get_board', 'get_items']] == before
        result = await sdk.call_tool('equip_item', {'item_slot': 0, 'target': target})
        assert not result.isError, result.structuredContent
        receipt = result.structuredContent
        assert receipt['item_id'] == 'sparring_gloves'
        assert receipt['unit_changes'][0]['after']['items'] == ['sparring_gloves']
        assert receipt['item_changes'] == [{'slot': 0, 'before': 'sparring_gloves', 'after': None}]
        assert receipt['status']['round'] == 3 and receipt['status']['planning_budget']['remaining'] == 13
        assert (await sdk.call_tool('equip_item', {'item_slot': 0, 'target': target})).structuredContent['code'] == 'empty_slot'
        await sdk.call_tool('close_game')


@pytest.mark.anyio
@pytest.mark.parametrize('item,equipment,expected', [('bf_sword', ['bf_sword'], ['deathblade']), ('magnetic_remover', ['bf_sword'], []), ('reforger', ['bf_sword'], []), ('champion_duplicator', [], []), ('thieves_gloves', [], None), ('kayn_rhast', [], [])])
async def test_sdk_memory_rare_equipment_native_fixture(tmp_path, monkeypatch, item, equipment, expected):
    from contextlib import asynccontextmanager
    import anyio
    from mcp import ClientSession
    from mcp.shared.memory import create_client_server_memory_streams
    import tft_mcp.transport as transport
    from test_buy_sell import install_units, session_fixture
    from test_shop_xp import gameplay
    session = session_fixture(tmp_path)
    is_kayn = item == 'kayn_rhast'
    player = install_units(session, **({'board': [((6, 3), 'kayn', 1, equipment)]} if is_kayn else {'bench': [(8, 'garen', 1, equipment)]}))
    player.item_bench = [item] + [None] * 9
    target = board(6, 3) if is_kayn else bench(8)
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
                result = await sdk.call_tool('equip_item', {'item_slot': 0, 'target': target})
                assert not result.isError, result.structuredContent
                receipt = result.structuredContent
                assert receipt['item_id'] == item and receipt['target'] == target
                assert receipt['status']['planning_budget']['remaining'] == 13
                if item == 'champion_duplicator':
                    assert receipt['unit_changes'][0]['after']['stars'] == 1
                elif item == 'kayn_rhast':
                    assert receipt['kayn_form'] == item
                elif expected is not None:
                    assert receipt['unit_changes'][0]['after']['items'] == expected
                else:
                    assert len(receipt['unit_changes'][0]['after']['items']) == 3
                before = gameplay(session)
                rejected = await sdk.call_tool('equip_item', {'item_slot': 0, 'target': target})
                assert rejected.isError and rejected.structuredContent['code'] == 'empty_slot'
                assert gameplay(session) == before
                await sdk.call_tool('close_game')
            tasks.cancel_scope.cancel()


@pytest.mark.anyio
async def test_stdio_equipment_replay_ignores_reads_and_rejections(tmp_path):
    import json
    journeys = []
    for extra in [False, True]:
        path = tmp_path / str(extra)
        path.mkdir()
        receipts = []
        async with client(path) as sdk:
            await sdk.call_tool('start_game', {'seed': 0})
            await sdk.call_tool('end_turn')
            await sdk.call_tool('buy_unit', {'shop_slot': 0})
            await sdk.call_tool('buy_unit', {'shop_slot': 1})
            await sdk.call_tool('end_turn')
            slots = (await sdk.call_tool('get_board')).structuredContent['slots']
            target = next(s['location'] for s in slots if s['unit'])
            if extra:
                for read in ['get_board', 'get_bench', 'get_items', 'get_traits', 'get_game_status', 'search_items']:
                    assert not (await sdk.call_tool(read)).isError
                assert (await sdk.call_tool('equip_item', {'item_slot': False, 'target': target})).isError
            result = await sdk.call_tool('equip_item', {'item_slot': 0, 'target': target})
            assert not result.isError
            receipt = result.structuredContent
            receipt['status'].pop('game_id')
            receipts.append(receipt)
            sold = await sdk.call_tool('sell_unit', {'location': target})
            assert not sold.isError and sold.structuredContent['returned_items'] == ['sparring_gloves']
            slots = (await sdk.call_tool('get_bench')).structuredContent['slots']
            target = next(s['location'] for s in slots if s['unit'])
            if extra:
                assert (await sdk.call_tool('equip_item', {'item_slot': 9, 'target': target})).structuredContent['code'] == 'empty_slot'
                await sdk.call_tool('get_shop')
                await sdk.call_tool('get_items')
            result = await sdk.call_tool('equip_item', {'item_slot': 0, 'target': target})
            assert not result.isError
            receipt = result.structuredContent
            receipt['status'].pop('game_id')
            receipts.append(receipt)
            state = []
            for read in ['get_board', 'get_bench', 'get_items', 'get_traits', 'get_economy', 'get_shop']:
                data = (await sdk.call_tool(read)).structuredContent
                data.pop('game_id')
                state.append(data)
            await sdk.call_tool('close_game')
        events = [json.loads(line) for line in (path / 'audit.jsonl').read_text().splitlines()]
        progression = [{key: e[key] for key in ('player_id', 'round', 'kind', 'action', 'placements')} for e in events if e['event'] == 'progression']
        journeys.append((receipts, state, progression))
    assert journeys[0] == journeys[1]

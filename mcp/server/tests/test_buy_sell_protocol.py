import jsonschema
import pytest

from test_protocol import client, anyio_backend


@pytest.mark.anyio
async def test_stdio_buy_sell_discovery_and_seeded_journey(tmp_path):
    async with client(tmp_path) as session:
        tools = {tool.name: tool for tool in (await session.list_tools()).tools}
        assert {'buy_unit', 'sell_unit'} <= tools.keys()
        for name in ('buy_unit', 'sell_unit'):
            assert tools[name].inputSchema['additionalProperties'] is False
            assert tools[name].outputSchema['additionalProperties'] is False
        await session.call_tool('start_game', {'seed': 0})
        status = (await session.call_tool('end_turn', {})).structuredContent
        shop = (await session.call_tool('get_shop', {})).structuredContent
        gold = (await session.call_tool('get_economy', {})).structuredContent['gold']
        offer = next(slot for slot in shop['slots'] if slot['unit'] and slot['purchase_cost'] <= gold)
        bought = await session.call_tool('buy_unit', {'shop_slot': offer['slot']})
        assert not bought.isError, bought.structuredContent
        receipt = bought.structuredContent
        jsonschema.validate(receipt, tools['buy_unit'].outputSchema)
        assert receipt['purchased'] == offer['unit']
        assert receipt['gold_spent'] == offer['purchase_cost']
        assert receipt['status']['planning_budget']['remaining'] == 13
        assert receipt['status']['round'] == status['round']
        sold = await session.call_tool('sell_unit', {'location': receipt['unit_changes'][0]['location']})
        assert not sold.isError, sold.structuredContent
        jsonschema.validate(sold.structuredContent, tools['sell_unit'].outputSchema)
        assert sold.structuredContent['sold'] == offer['unit']
        assert sold.structuredContent['status']['planning_budget']['remaining'] == 12


@pytest.mark.anyio
async def test_stdio_strict_requests_rejections_and_recovery(tmp_path):
    async with client(tmp_path) as session:
        valid = [('buy_unit', {'shop_slot': 0}), ('sell_unit', {'location': {'kind': 'bench', 'slot': 0}})]
        for name, arguments in valid:
            result = await session.call_tool(name, arguments)
            assert result.isError and result.structuredContent['code'] == 'no_game'
        requests = [('buy_unit', args) for args in ({}, {'shop_slot': None}, {'shop_slot': True}, {'shop_slot': 1.0}, {'shop_slot': -1}, {'shop_slot': 5}, {'shop_slot': 0, 'player_id': 'player_1'})]
        requests += [('sell_unit', {'location': location}) for location in (None, {}, {'kind': 'bench', 'slot': False}, {'kind': 'bench', 'slot': 1.0}, {'kind': 'bench', 'slot': 9}, {'kind': 'bench', 'slot': 0, 'x': 0}, {'kind': 'board', 'x': 7, 'y': 0}, {'kind': 'board', 'x': 0}, {'kind': 'board', 'x': 0, 'y': None})]
        requests += [('sell_unit', {}), ('sell_unit', {'location': {'kind': 'bench', 'slot': 0}, 'player_id': 'player_1'})]
        for name, arguments in requests:
            result = await session.call_tool(name, arguments)
            assert result.isError and result.structuredContent['code'] == 'invalid_input', (name, arguments, result)
        initial = (await session.call_tool('start_game', {'seed': 0})).structuredContent
        for name, arguments, code in [('buy_unit', {'shop_slot': 0}, 'insufficient_gold'), ('sell_unit', {'location': {'kind': 'board', 'x': 0, 'y': 0}}, 'empty_slot')]:
            result = await session.call_tool(name, arguments)
            assert result.isError and result.structuredContent['code'] == code
            assert result.structuredContent['details']
        assert (await session.call_tool('get_game_status', {})).structuredContent == initial
        await session.call_tool('end_turn', {})
        shop = (await session.call_tool('get_shop', {})).structuredContent
        gold = (await session.call_tool('get_economy', {})).structuredContent['gold']
        slot = next(offer['slot'] for offer in shop['slots'] if offer['unit'] and offer['purchase_cost'] <= gold)
        audit = tmp_path / 'audit.jsonl'
        saved = audit.read_bytes()
        audit.unlink()
        audit.mkdir()
        result = await session.call_tool('buy_unit', {'shop_slot': slot})
        assert result.isError and result.structuredContent['code'] == 'log_unavailable'
        audit.rmdir()
        audit.write_bytes(saved)
        assert (await session.call_tool('get_shop', {})).structuredContent == shop
        result = await session.call_tool('buy_unit', {'shop_slot': slot})
        assert not result.isError
        result = await session.call_tool('buy_unit', {'shop_slot': slot})
        assert result.isError and result.structuredContent['code'] == 'empty_slot'
        await session.call_tool('close_game', {})
        for name, arguments in valid:
            result = await session.call_tool(name, arguments)
            assert result.isError and result.structuredContent['code'] == 'no_game'


@pytest.mark.anyio
async def test_stdio_purchase_native_autofill_combat_and_board_sale(tmp_path):
    async with client(tmp_path) as session:
        await session.call_tool('start_game', {'seed': 0})
        await session.call_tool('end_turn', {})
        shop = (await session.call_tool('get_shop', {})).structuredContent
        gold = (await session.call_tool('get_economy', {})).structuredContent['gold']
        offer = next(slot for slot in shop['slots'] if slot['unit'] and slot['purchase_cost'] <= gold)
        purchased = await session.call_tool('buy_unit', {'shop_slot': offer['slot']})
        assert not purchased.isError
        status = (await session.call_tool('end_turn', {})).structuredContent
        board = (await session.call_tool('get_board', {})).structuredContent
        unit = next(slot for slot in board['slots'] if slot['unit'])
        gold = (await session.call_tool('get_economy', {})).structuredContent['gold']
        sale = await session.call_tool('sell_unit', {'location': unit['location']})
        assert not sale.isError, sale.structuredContent
        receipt = sale.structuredContent
        tools = {tool.name: tool for tool in (await session.list_tools()).tools}
        jsonschema.validate(receipt, tools['sell_unit'].outputSchema)
        assert receipt['sold'] == unit['unit']
        assert receipt['status']['round'] == status['round']
        assert receipt['status']['planning_budget']['remaining'] == 13
        assert (await session.call_tool('get_economy', {})).structuredContent['gold'] == gold + receipt['gold_gained']
        after = (await session.call_tool('get_board', {})).structuredContent
        assert next(slot['unit'] for slot in after['slots'] if slot['location'] == unit['location']) is None

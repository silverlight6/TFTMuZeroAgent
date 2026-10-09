import pytest

from test_protocol import client, anyio_backend


@pytest.mark.anyio
async def test_idle_item_tools_discovery_and_results(tmp_path):
    async with client(tmp_path) as session:
        tools = {tool.name: tool for tool in (await session.list_tools()).tools}
        assert {'search_items', 'get_item'} <= tools.keys()
        assert tools['get_item'].inputSchema['required'] == ['item_id']
        assert tools['search_items'].inputSchema['additionalProperties'] is False
        searched = await session.call_tool('search_items', {'query': 'BUFF', 'kind': 'equipment'})
        assert not searched.isError
        assert searched.structuredContent == {'items': [{'item_id': 'blue_buff', 'kind': 'equipment', 'craftable': True}]}
        item = await session.call_tool('get_item', {'item_id': 'blue_buff'})
        assert not item.isError
        assert item.structuredContent['recipe'] == ['tear_of_the_goddess', 'tear_of_the_goddess']


@pytest.mark.anyio
async def test_item_strict_filters_errors_and_live_queries(tmp_path):
    from Simulator.battle import item_stats

    async with client(tmp_path) as session:
        for tool, arguments, field, value in [
            ('search_items', {'query': None}, 'query', None),
            ('search_items', {'query': 4}, 'query', 4),
            ('search_items', {'kind': None}, 'kind', None),
            ('search_items', {'kind': 'other'}, 'kind', 'other'),
            ('search_items', {'kind': []}, 'kind', []),
            ('search_items', {'extra': True}, 'extra', True),
            ('get_item', {}, 'item_id', None),
            ('get_item', {'item_id': False}, 'item_id', False),
            ('get_item', {'item_id': 'blue_buff', 'extra': 1}, 'extra', 1),
        ]:
            result = await session.call_tool(tool, arguments)
            assert result.isError
            assert result.structuredContent['code'] == 'invalid_input'
            assert result.structuredContent['details'] == {'field': field, 'value': value}
        for item_id in ['BLUE_BUFF', 'Blue Buff', 'unknown']:
            result = await session.call_tool('get_item', {'item_id': item_id})
            assert result.isError
            assert result.structuredContent == {'code': 'unknown_item', 'message': f'Unknown item: {item_id}',
                                                'details': {'item_id': item_id}}
        assert (await session.call_tool('search_items', {'query': 'no_such_item'})).structuredContent == {'items': []}
        for item_id, stats in item_stats.items.items():
            result = await session.call_tool('get_item', {'item_id': item_id})
            assert not result.isError
            assert result.structuredContent['base_stats'] == stats
            assert result.structuredContent['recipe'] == item_stats.item_builds.get(item_id)
        before = (await session.call_tool('start_game', {'seed': 123})).structuredContent
        for kind in ['component', 'equipment', 'consumable']:
            result = await session.call_tool('search_items', {'kind': kind})
            assert not result.isError
            assert result.structuredContent['items']
            assert all(item['kind'] == kind for item in result.structuredContent['items'])
        assert (await session.call_tool('get_game_status', {})).structuredContent == before

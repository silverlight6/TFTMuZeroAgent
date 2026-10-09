import pytest

from test_protocol import client


@pytest.mark.anyio
async def test_idle_champion_search_and_filters(tmp_path):
    async with client(tmp_path) as session:
        result = await session.call_tool('search_champions', {'query': 'JAR', 'cost': 2, 'trait_id': 'keeper'})
        assert not result.isError
        assert result.structuredContent == {'champions': [{'champion_id': 'jarvaniv', 'cost': 2, 'traits': ['warlord', 'keeper']}]}
        assert (await session.call_tool('search_champions', {'query': 'no-such-champion'})).structuredContent == {'champions': []}


@pytest.mark.anyio
async def test_champion_rules_and_exact_errors(tmp_path):
    async with client(tmp_path) as session:
        result = await session.call_tool('get_champion', {'champion_id': 'kayn'})
        assert not result.isError
        champion = result.structuredContent
        assert champion['cost'] == 5
        assert champion['special_attributes'] == {'chosen': {'eligible_traits': ['shade'], 'bonus': {'stat': 'SP', 'value': 0.3}}, 'kayn_forms': ['kayn_shadowassassin', 'kayn_rhast']}
        assert champion['star_costs'] == [{'stars': 1, 'gold': 5}, {'stars': 2, 'gold': 14}, {'stars': 3, 'gold': 44}]
        assert champion['description'] is None and champion['ability_description'] is None
        error = await session.call_tool('get_champion', {'champion_id': 'Kayn'})
        assert error.isError
        assert error.structuredContent['code'] == 'unknown_champion'
        assert error.structuredContent['details'] == {'champion_id': 'Kayn'}


@pytest.mark.anyio
async def test_trait_rules_search_and_membership(tmp_path):
    async with client(tmp_path) as session:
        result = await session.call_tool('search_traits', {'query': 'NIN'})
        assert result.structuredContent == {'traits': [{'trait_id': 'ninja', 'thresholds': [1, 4]}]}
        ninja = (await session.call_tool('get_trait', {'trait_id': 'ninja'})).structuredContent
        assert ninja['activation'] == 'exact'
        assert ninja['effects']['AD'] == [0, 50, 140]
        assert ninja['champion_ids'] == ['akali', 'kennen', 'shen', 'zed']
        members = (await session.call_tool('get_trait_champions', {'trait_id': 'ninja'})).structuredContent
        assert [champion['champion_id'] for champion in members['champions']] == ninja['champion_ids']
        fortune = (await session.call_tool('get_trait', {'trait_id': 'fortune'})).structuredContent
        assert 'fortune_returns' in fortune['effects']
        error = await session.call_tool('get_trait', {'trait_id': 'blacksmith'})
        assert error.isError and error.structuredContent['code'] == 'unknown_trait'


@pytest.mark.anyio
async def test_catalog_discovery_strict_inputs_and_empty_filters(tmp_path):
    async with client(tmp_path) as session:
        tools = {tool.name: tool for tool in (await session.list_tools()).tools}
        names = {'search_champions', 'get_champion', 'search_traits', 'get_trait', 'get_trait_champions'}
        assert names <= tools.keys()
        for name in names:
            assert tools[name].inputSchema['additionalProperties'] is False
            assert tools[name].outputSchema is not None
        cases = [('search_champions', {'cost': value}) for value in [True, False, 1.0, 0, 6, '2', None]]
        cases += [('search_champions', {'trait_id': 'Ninja'}), ('search_champions', {'query': None}),
                  ('search_traits', {'query': 3}), ('search_traits', {'extra': 1})]
        for name in ['get_champion', 'get_trait', 'get_trait_champions']:
            field = 'champion_id' if name == 'get_champion' else 'trait_id'
            cases += [(name, {}), (name, {field: None}), (name, {field: 4}), (name, {field: 'ninja', 'extra': 1})]
        for name, arguments in cases:
            result = await session.call_tool(name, arguments)
            assert result.isError, (name, arguments)
            assert result.structuredContent['code'] == 'invalid_input'
            assert set(result.structuredContent['details']) == {'field', 'value'}
        assert (await session.call_tool('search_champions', {'cost': 1, 'trait_id': 'ninja'})).structuredContent == {'champions': []}
        assert (await session.call_tool('search_traits', {'query': 'unknown'})).structuredContent == {'traits': []}
        for name in ['get_trait', 'get_trait_champions']:
            result = await session.call_tool(name, {'trait_id': 'Ninja'})
            assert result.isError
            assert result.structuredContent == {'code': 'unknown_trait', 'message': 'Unknown trait: Ninja.', 'details': {'trait_id': 'Ninja'}}
        assert (await session.call_tool('get_game_status', {})).structuredContent['state'] == 'idle'


def test_all_catalog_definitions_are_source_consistent_and_reads_are_pure():
    from copy import deepcopy
    import pickle
    import random

    import numpy as np
    from Simulator.battle import origin_class_stats, stats
    from Simulator.game import pool_stats
    from tft_mcp.session import GameSession

    session = GameSession()
    definitions = [stats, origin_class_stats, pool_stats]
    before = pickle.dumps([{key: deepcopy(value) for key, value in vars(module).items()
                            if isinstance(value, (dict, list))} for module in definitions])
    rng_before = pickle.dumps((random.getstate(), np.random.get_state()))
    champions = session.search_champions()['champions']
    assert [entry['champion_id'] for entry in champions] == sorted(stats.BASE_CHAMPION_LIST)
    for entry in champions:
        name = entry['champion_id']
        result = session.get_champion(name)
        assert result['traits'] == origin_class_stats.origin_class[name]
        assert result['cost'] == stats.COST[name]
        assert [row['gold'] for row in result['star_costs']] == pool_stats.cost_star_values[stats.COST[name] - 1]
        for key, value in vars(stats).items():
            if isinstance(value, dict) and name in value and key != 'COST':
                projection = result['base_stats'] if key in ['AD', 'AS', 'HEALTH', 'ARMOR', 'MR', 'MANA', 'MAXMANA', 'RANGE'] else result['rule_parameters']
                assert projection[key] == value[name], (name, key)
    assert [entry['trait_id'] for entry in session.search_traits()['traits']] == sorted(origin_class_stats.tiers)
    for name, thresholds in origin_class_stats.tiers.items():
        result = session.get_trait(name)
        assert result['thresholds'] == thresholds
        expected_members = sorted(champion for champion in stats.BASE_CHAMPION_LIST if name in origin_class_stats.origin_class[champion])
        assert result['champion_ids'] == expected_members
        assert [entry['champion_id'] for entry in session.get_trait_champions(name)['champions']] == expected_members
        for key, value in vars(origin_class_stats).items():
            if isinstance(value, dict) and name in value and key != 'tiers':
                assert result['effects'][key] == value[name]
    original = session.get_champion('kayn')
    changed = session.get_champion('kayn')
    changed['rule_parameters']['ABILITY_DMG'][1] = -100
    changed['special_attributes']['chosen']['bonus']['value'] = -100
    changed['traits'].clear()
    assert session.get_champion('kayn') == original
    original = session.get_trait('fortune')
    changed = session.get_trait('fortune')
    changed['effects']['fortune_returns'].clear()
    changed['thresholds'].clear()
    assert session.get_trait('fortune') == original
    after = pickle.dumps([{key: deepcopy(value) for key, value in vars(module).items()
                           if isinstance(value, (dict, list))} for module in definitions])
    assert after == before
    assert pickle.dumps((random.getstate(), np.random.get_state())) == rng_before
    assert session.get_game_status()['state'] == 'idle'


@pytest.mark.anyio
async def test_catalog_queries_preserve_running_status(tmp_path):
    async with client(tmp_path) as session:
        started = await session.call_tool('start_game', {'seed': 123})
        for name, arguments in [('search_champions', {}), ('get_champion', {'champion_id': 'jarvaniv'}),
                                ('search_traits', {}), ('get_trait', {'trait_id': 'fortune'}),
                                ('get_trait_champions', {'trait_id': 'keeper'})]:
            assert not (await session.call_tool(name, arguments)).isError
        assert (await session.call_tool('get_game_status', {})).structuredContent == started.structuredContent


@pytest.mark.anyio
async def test_tools_describe_installed_simulator_without_set_selector(tmp_path):
    import re
    async with client(tmp_path) as session:
        for tool in (await session.list_tools()).tools:
            assert re.search(r'\bset\s*\d+\b', tool.description, re.IGNORECASE) is None
            assert not {'set', 'set_id', 'set_number'} & set(tool.inputSchema.get('properties', {}))
        result = await session.call_tool('start_game', {'seed': 0, 'set': 4})
        assert result.isError
        assert result.structuredContent['code'] == 'invalid_input'


def test_special_metadata_omits_forms_absent_from_installed_definitions(monkeypatch):
    from Simulator.battle import item_stats
    from tft_mcp.session import GameSession
    with monkeypatch.context() as patch:
        patch.delitem(item_stats.items, 'kayn_rhast')
        result = GameSession().get_champion('kayn')
    assert result['special_attributes']['kayn_forms'] == ['kayn_shadowassassin']

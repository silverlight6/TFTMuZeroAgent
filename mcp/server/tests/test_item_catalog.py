from tft_mcp.session import GameSession


def test_idle_item_search_filters_and_order():
    session = GameSession()
    assert session.search_items(query='BUFF', kind='equipment') == {'items': [
        {'item_id': 'blue_buff', 'kind': 'equipment', 'craftable': True}]}
    assert session.search_items(query='does_not_exist') == {'items': []}
    assert [item['item_id'] for item in session.search_items(kind='component')['items']] == [
        'bf_sword', 'chain_vest', 'giants_belt', 'needlessly_large_rod', 'negatron_cloak',
        'recurve_bow', 'sparring_gloves', 'spatula', 'tear_of_the_goddess']


def test_item_recipe_effects_and_defensive_copies():
    session = GameSession()
    blue = session.get_item('blue_buff')
    assert blue['recipe'] == ['tear_of_the_goddess', 'tear_of_the_goddess']
    assert session.get_item('mages_cap')['granted_trait'] == 'mage'
    guardian = session.get_item('guardian_angel')
    assert guardian['base_stats']['will_revive'] == [[None], ['guardian_angel']]
    assert guardian['effects'] == {'heal': 400, 'cooldown': 2000}
    guardian['base_stats']['will_revive'][1].append('mutated')
    assert session.get_item('guardian_angel')['base_stats']['will_revive'] == [[None], ['guardian_angel']]


def test_catalog_queries_preserve_definitions_and_rng_for_every_item(tmp_path):
    from copy import deepcopy
    import pickle
    import random
    import numpy as np
    from Simulator.battle import item_stats

    session = GameSession(tmp_path / 'audit.jsonl')
    session.start_game(123)
    status = deepcopy(session.get_game_status())
    players = session.game.player_manager.player_states
    gameplay = pickle.dumps({player_id: (player.shop, player.item_bench, player.gold,
                             player.health, player.board, player.bench)
                             for player_id, player in players.items()})
    definitions = deepcopy({name: value for name, value in vars(item_stats).items()
                            if not name.startswith("_") and isinstance(value, (dict, list))})
    rng = pickle.dumps((random.getstate(), np.random.get_state(), session.python_rng, session.baseline_rng))
    for _ in range(2):
        assert [item['item_id'] for item in session.search_items()['items']] == sorted(item_stats.items)
        for item_id, source_stats in item_stats.items.items():
            item = session.get_item(item_id)
            assert item['base_stats'] == source_stats
            assert item['recipe'] == item_stats.item_builds.get(item_id)
            assert item['craftable'] == (item_id in item_stats.item_builds)
            assert item['description'] is None
            assert item['unavailable_fields'] == ['description']
            for name, table in definitions.items():
                if isinstance(table, dict) and name not in {'items', 'item_builds', 'trait_items'}:
                    if item_id in table:
                        assert item['effects'][name] == table[item_id]
            expected_builds = {result: parts for result, parts in item_stats.item_builds.items() if item_id in parts}
            assert {entry['item_id']: entry['components'] for entry in item['builds_into']} == expected_builds
    assert definitions == {name: value for name, value in vars(item_stats).items() if not name.startswith("_") and isinstance(value, (dict, list))}
    assert rng == pickle.dumps((random.getstate(), np.random.get_state(), session.python_rng, session.baseline_rng))
    assert session.get_game_status() == status
    assert gameplay == pickle.dumps({player_id: (player.shop, player.item_bench, player.gold,
                                    player.health, player.board, player.bench)
                                    for player_id, player in players.items()})
    session.close_game()


def test_all_consumables_and_thieves_gloves_have_source_constraints():
    session = GameSession()
    consumables = session.search_items(kind='consumable')['items']
    assert [item['item_id'] for item in consumables] == [
        'champion_duplicator', 'kayn_rhast', 'kayn_shadowassassin', 'magnetic_remover', 'reforger']
    for summary in consumables:
        item = session.get_item(summary['item_id'])
        assert item['kind'] == 'consumable'
        assert item['recipe'] is None and not item['craftable']
        assert item['constraints'][0] == 'Requires a present unit that is not a target dummy.'
    assert 'default-star' in ' '.join(session.get_item('champion_duplicator')['constraints'])
    assert 'before this consumable' in ' '.join(session.get_item('magnetic_remover')['constraints'])
    assert 'spatula remains spatula' in ' '.join(session.get_item('reforger')['constraints'])
    for form in ['kayn_rhast', 'kayn_shadowassassin']:
        assert 'not guaranteed' in ' '.join(session.get_item(form)['constraints'])
    assert 'two distinct random items' in ' '.join(session.get_item('thieves_gloves')['constraints'])

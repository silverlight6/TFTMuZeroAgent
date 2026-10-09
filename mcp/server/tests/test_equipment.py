import pytest

from test_buy_sell import install_units, session_fixture
from test_movement import board, bench
from tft_mcp.session import SessionError


@pytest.mark.parametrize('target', [board(6, 3), bench(8)])
def test_equipment_receipt_native_order_and_one_action(tmp_path, target):
    session = session_fixture(tmp_path)
    player = install_units(session, **({'board': [((6, 3), 'garen', 1, ['bf_sword'])]} if target['kind'] == 'board' else {'bench': [(8, 'garen', 1, ['bf_sword'])]}))
    player.item_bench = [None] * 9 + ['giant_slayer']
    receipt = session.equip_item(item_slot=9, target=target)
    assert set(receipt) == {'item_slot', 'item_id', 'target', 'unit_changes', 'item_changes', 'kayn_form', 'status'}
    assert receipt['unit_changes'][0]['after']['items'] == ['giant_slayer', 'bf_sword']
    assert receipt['item_changes'] == [{'slot': 9, 'before': 'giant_slayer', 'after': None}]
    assert receipt['status']['planning_budget']['remaining'] == 13
    assert receipt['status']['round'] == 1
    assert session.game.agent_selection == 'player_0'


@pytest.mark.parametrize('equipment,incoming,result', [
    (['bf_sword'], 'bf_sword', ['deathblade']),
    (['spatula'], 'spatula', ['force_of_nature']),
    (['giant_slayer', 'warmogs_armor', 'bf_sword'], 'bf_sword', ['giant_slayer', 'warmogs_armor', 'deathblade']),
    ([], 'duelists_zeal', ['duelists_zeal']),
    (['spatula'], 'recurve_bow', ['duelists_zeal']),
])
def test_supported_recipes_and_trait_grants(tmp_path, equipment, incoming, result):
    session = session_fixture(tmp_path)
    player = install_units(session, bench=[(0, 'garen', 1, equipment)])
    player.item_bench = [incoming] + [None] * 9
    receipt = session.equip_item(item_slot=0, target=bench(0))
    unit = receipt['unit_changes'][0]['after']
    assert unit['items'] == result
    assert unit['traits'] == ['warlord', 'vanguard'] + (['duelist'] if 'duelists_zeal' in result else [])


@pytest.mark.parametrize('equipment,incoming,code', [
    (['sparring_gloves'], 'sparring_gloves', 'unsupported_action'),
    (['giant_slayer', 'warmogs_armor', 'deathblade'], 'bf_sword', 'capacity_exceeded'),
    ([], 'warlords_banner', 'incompatible_item'),
    (['thieves_gloves', 'giant_slayer', 'deathblade'], 'bf_sword', 'unsupported_action'),
])
def test_native_unsafe_equipment_rejects_atomically(tmp_path, equipment, incoming, code):
    import pickle
    session = session_fixture(tmp_path)
    player = install_units(session, bench=[(0, 'garen', 1, equipment)])
    player.item_bench = [incoming] + [None] * 9
    before = pickle.dumps(session.game)
    with pytest.raises(SessionError) as error:
        session.equip_item(item_slot=0, target=bench(0))
    assert error.value.code == code
    assert pickle.dumps(session.game) == before


@pytest.mark.parametrize('target', [board(6, 3), bench(8)])
def test_direct_gloves_draw_and_track_actual_location(tmp_path, target):
    from Simulator.battle.item_stats import thieves_gloves_items
    session = session_fixture(tmp_path)
    player = install_units(session, **({'board': [((6, 3), 'garen', 1, [])]} if target['kind'] == 'board' else {'bench': [(8, 'garen', 1, [])]}))
    player.item_bench = ['thieves_gloves'] + [None] * 9
    receipt = session.equip_item(item_slot=0, target=target)
    items = receipt['unit_changes'][0]['after']['items']
    assert items[0] == 'thieves_gloves' and len(set(items)) == 3
    assert set(items[1:]) <= set(thieves_gloves_items)
    player = session.game.player_manager.player_states['player_0']
    assert player.thieves_gloves_loc == ([[6, 3]] if target['kind'] == 'board' else [[8, -1]])


@pytest.mark.parametrize('consumable', ['magnetic_remover', 'reforger'])
def test_removal_and_reforge_native_order_and_categories(tmp_path, consumable):
    from Simulator.battle.item_stats import starting_items, thieves_gloves_items
    session = session_fixture(tmp_path)
    player = install_units(session, bench=[(0, 'garen', 1, ['spatula', 'bf_sword', 'giant_slayer'])])
    player.item_bench = [None, consumable, None, None] + ['chain_vest'] * 6
    receipt = session.equip_item(item_slot=1, target=bench(0))
    items = session.game.player_manager.player_states['player_0'].item_bench
    assert items[0] == 'spatula' and items[1] is None
    assert receipt['unit_changes'][0]['after']['items'] == []
    if consumable == 'magnetic_remover':
        assert items[2:4] == ['bf_sword', 'giant_slayer']
    else:
        assert items[2] in starting_items and items[2] != 'bf_sword'
        assert items[3] in thieves_gloves_items and items[3] != 'giant_slayer'


@pytest.mark.parametrize('reverse', [False, True])
def test_remover_safely_assigned_trait_suffix_in_any_equipment_order(tmp_path, reverse):
    session = session_fixture(tmp_path)
    player = install_units(session, bench=[(0, 'garen', 1, [])])
    player.item_bench = ['duelists_zeal', 'mages_cap'] + [None] * 8
    session.equip_item(item_slot=0, target=bench(0))
    session.equip_item(item_slot=1, target=bench(0))
    session.game.player_manager.player_states['player_0'].item_bench[2] = 'magnetic_remover'
    if reverse:
        session.game.player_manager.player_states['player_0'].bench[0].items.reverse()
    receipt = session.equip_item(item_slot=2, target=bench(0))
    assert receipt['unit_changes'][0]['after']['traits'] == ['warlord', 'vanguard']
    assert receipt['unit_changes'][0]['after']['items'] == []


@pytest.mark.parametrize('equipment,item,free,code', [
    ([], 'magnetic_remover', 9, 'incompatible_item'),
    ([], 'reforger', 9, 'incompatible_item'),
    (['bf_sword'], 'magnetic_remover', 0, 'capacity_exceeded'),
    (['bf_sword'], 'reforger', 0, 'capacity_exceeded'),
    (['duelists_zeal'], 'magnetic_remover', 9, 'unsupported_action'),
    (['duelists_zeal'], 'reforger', 9, 'unsupported_action'),
    (['thieves_gloves', 'bf_sword', 'chain_vest'], 'magnetic_remover', 9, 'unsupported_action'),
    (['thieves_gloves', 'bf_sword', 'chain_vest'], 'reforger', 9, 'unsupported_action'),
    (['bf_sword'], 'thieves_gloves', 9, 'incompatible_item'),
])
def test_unsafe_consumables_and_preconsumption_capacity(tmp_path, equipment, item, free, code):
    import pickle
    session = session_fixture(tmp_path)
    player = install_units(session, bench=[(0, 'garen', 1, equipment)])
    player.item_bench = [item] + [None] * free + ['chain_vest'] * (9 - free)
    before = pickle.dumps(session.game)
    with pytest.raises(SessionError) as error:
        session.equip_item(item_slot=0, target=bench(0))
    assert error.value.code == code
    assert pickle.dumps(session.game) == before


@pytest.mark.parametrize('stars,chosen,cascade', [(3, False, False), (3, 'vanguard', False), (1, False, True)])
def test_duplicator_uses_fresh_constructor_and_safe_cascades(tmp_path, stars, chosen, cascade):
    session = session_fixture(tmp_path)
    units = [(0, 'garen', stars, ['giant_slayer'])]
    if cascade:
        units = [(0, 'garen', 1, ['bf_sword']), (1, 'garen', 1, []), (2, 'garen', 2, []), (3, 'garen', 2, [])]
    player = install_units(session, bench=units)
    player.bench[0].chosen = chosen
    player.item_bench = ['champion_duplicator'] + [None] * 9
    receipt = session.equip_item(item_slot=0, target=bench(0))
    slots = session.get_bench()['slots']
    if cascade:
        assert slots[0]['unit']['stars'] == 3
        assert session.get_items()['slots'][1]['item'] == 'bf_sword'
    else:
        fresh = slots[1]['unit']
        assert fresh['stars'] == (2 if chosen else 1)
        assert fresh['chosen'] == chosen and fresh['items'] == []
        assert slots[0]['unit']['stars'] == 3
    assert receipt['item_id'] == 'champion_duplicator'
    assert session.get_economy()['gold'] == 100


@pytest.mark.parametrize('full_bench,board_items,free,code', [(True, [], 9, 'capacity_exceeded'), (False, ['bf_sword'], 0, 'capacity_exceeded')])
def test_duplicator_requires_real_vacancy_and_safe_board_returns(tmp_path, full_bench, board_items, free, code):
    import pickle
    session = session_fixture(tmp_path)
    units = [(0, 'garen', 1, [])]
    if full_bench:
        units += [(i, name, 1, []) for i, name in enumerate(['garen', 'fiora', 'nami', 'vayne', 'nidalee', 'diana', 'lissandra', 'wukong'], 1)]
    player = install_units(session, bench=units, board=[] if full_bench else [((0, 0), 'garen', 1, board_items)])
    player.item_bench = ['champion_duplicator'] + [None] * free + ['chain_vest'] * (9 - free)
    before = pickle.dumps(session.game)
    with pytest.raises(SessionError) as error:
        session.equip_item(item_slot=0, target=bench(0))
    assert error.value.code == code
    assert pickle.dumps(session.game) == before


@pytest.mark.parametrize('form', ['kayn_rhast', 'kayn_shadowassassin'])
def test_kayn_literal_all_board_and_all_inventory_effects(tmp_path, form):
    session = session_fixture(tmp_path)
    player = install_units(session, bench=[(8, 'kayn', 1, [])], board=[((0, 0), 'kayn', 1, ['bf_sword']), ((6, 3), 'kayn', 2, [])], offer='kayn')
    player.item_bench = [form, 'kayn_rhast', 'kayn_shadowassassin', 'bf_sword'] + [None] * 6
    receipt = session.equip_item(item_slot=0, target=board(0, 0))
    assert receipt['kayn_form'] == form
    assert [c['location'] for c in receipt['unit_changes']] == [board(0, 0), board(6, 3)]
    assert all(c['after']['kayn_form'] == form for c in receipt['unit_changes'])
    assert session.get_bench()['slots'][8]['unit']['kayn_form'] is None
    assert session.get_shop()['slots'][0]['unit']['kayn_form'] is None
    assert session.get_items()['slots'][3]['item'] == 'bf_sword'
    session.game.player_manager.player_states['player_0'].item_bench[0] = form
    repeated = session.equip_item(item_slot=0, target=board(0, 0))
    assert repeated['unit_changes'] == []
    assert repeated['status']['planning_budget']['remaining'] == 12


@pytest.mark.parametrize('name,target,item,code', [('garen', bench(0), 'kayn_rhast', 'incompatible_item'), ('kayn', bench(0), 'kayn_rhast', 'unsupported_action'), ('sandguard', bench(0), 'bf_sword', 'unsupported_action')])
def test_special_target_restrictions(tmp_path, name, target, item, code):
    session = session_fixture(tmp_path)
    player = install_units(session, bench=[(0, name, 1, [])])
    player.item_bench = [item] + [None] * 9
    with pytest.raises(SessionError) as error:
        session.equip_item(item_slot=0, target=target)
    assert error.value.code == code


@pytest.mark.parametrize('item', ['thieves_gloves', 'reforger'])
@pytest.mark.parametrize('failure', ['native_action', 'observation', 'mask', 'baseline', 'noop', 'corrupt', 'receipt', 'postcondition', 'native', 'audit'])
def test_equipment_failure_rolls_back_full_graph_rng_logs_and_fresh_retry(tmp_path, monkeypatch, failure, item):
    import os
    import pickle
    import random
    import numpy as np
    from Simulator.battle import champion, origin_class
    from Simulator.game.player import Player
    from Simulator.simulators.tft_simulator import TFT_Simulator
    import tft_mcp.session as module
    from test_shop_xp import gameplay
    from test_movement import normalized_graph
    sessions = [session_fixture(tmp_path / name) for name in ('actual', 'reference')]
    for session in sessions:
        player = install_units(session, [(0, 'garen', 1, ['bf_sword'] if item == 'reforger' else [])])
        player.item_bench = [item] + [None] * 9
    session, reference = sessions
    accepted = session.game, session.baselines, session.module_state, session.baseline_rng, session.python_rng
    graph = pickle.dumps(session.game)
    before = gameplay(session)
    logs = session.audit_path.read_bytes(), (session.native_dir / 'log.txt').read_bytes()
    process = random.getstate(), pickle.dumps(np.random.get_state())
    bindings = champion.log, champion.test_multiple, origin_class.game_compositions, origin_class.game_comp_tiers
    with monkeypatch.context() as patch:
        native = Player.move_item
        def fail_native(player, *args):
            if failure == 'noop':
                return False
            result = native(player, *args)
            raise RuntimeError('after actual native equipment')
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
                    stream.write('partial candidate equipment log')
                raise OSError('native file fault')
            return result
        def fail_after(*args):
            raise RuntimeError('equipment receipt or postcondition fault')
        if failure in {'native_action', 'noop'}:
            patch.setattr(Player, 'move_item', fail_native)
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
            session.equip_item(item_slot=0, target=bench(0))
        assert error.value.code == ('log_unavailable' if failure in {'native', 'audit'} else 'internal_error')
    assert all(a is b for a, b in zip((session.game, session.baselines, session.module_state, session.baseline_rng, session.python_rng), accepted))
    assert pickle.dumps(session.game) == graph  # Includes shared pool, traits, encoders and masks.
    assert gameplay(session) == before
    assert logs == (session.audit_path.read_bytes(), (session.native_dir / 'log.txt').read_bytes())
    assert random.getstate() == process[0] and pickle.dumps(np.random.get_state()) == process[1]
    assert all(a is b for a, b in zip((champion.log, champion.test_multiple, origin_class.game_compositions, origin_class.game_comp_tiers), bindings))
    session.equip_item(item_slot=0, target=bench(0))
    reference.equip_item(item_slot=0, target=bench(0))
    assert gameplay(session) == gameplay(reference)
    assert normalized_graph(session.game) == normalized_graph(reference.game)
    game = session.game
    assert game.rng is game.combat_ctx.rng and game.pool_obj is game.player_manager.pool_obj
    assert game.game_round.PLAYERS is game.player_manager.player_states
    for key, player in game.player_manager.player_states.items():
        if player:
            assert game.player_manager.observation_states[key].player is player
            assert game.player_manager.action_handlers[key].player is player
            assert player.pool_obj is game.pool_obj



def test_four_star_ordinary_equipment_does_not_require_sale_price(tmp_path):
    session = session_fixture(tmp_path)
    player = install_units(session, bench=[(0, 'garen', 4, [])])
    player.item_bench = ['bf_sword'] + [None] * 9
    receipt = session.equip_item(item_slot=0, target=bench(0))
    assert receipt['unit_changes'][0]['after']['stars'] == 4
    assert receipt['unit_changes'][0]['after']['items'] == ['bf_sword']


def test_native_special_inventory_observation_failure_is_atomic(tmp_path):
    import pickle
    session = session_fixture(tmp_path)
    player = install_units(session, bench=[(0, 'garen', 1, [])])
    player.item_bench = ['bf_sword', 'magnetic_remover'] + [None] * 8
    before = pickle.dumps(session.game)
    logs = session.audit_path.read_bytes(), (session.native_dir / 'log.txt').read_bytes()
    with pytest.raises(SessionError) as error:
        session.equip_item(item_slot=0, target=bench(0))
    assert error.value.code == 'internal_error'
    assert isinstance(error.value.__cause__, AssertionError)
    assert pickle.dumps(session.game) == before
    assert logs == (session.audit_path.read_bytes(), (session.native_dir / 'log.txt').read_bytes())


@pytest.mark.parametrize('item,equipment', [('bf_sword', ['bf_sword']), ('duelists_zeal', []), ('champion_duplicator', []), ('thieves_gloves', []), ('magnetic_remover', ['bf_sword']), ('reforger', ['bf_sword'])])
@pytest.mark.parametrize('target', [board(6, 3), bench(8)])
def test_actual_native_result_observations_traits_and_masks(tmp_path, item, equipment, target):
    from copy import deepcopy
    import numpy as np
    from tft_mcp.session import freeze_player
    session = session_fixture(tmp_path)
    player = install_units(session, **({'board': [((6, 3), 'garen', 1, equipment)]} if target['kind'] == 'board' else {'bench': [(8, 'garen', 1, equipment)]}))
    player.item_bench = [item] + [None] * 9
    manager = session.game.player_manager
    with session.simulator_scope():
        player.update_team_tiers()
        observation_class = type(manager.observation_states['player_0'])
        action_class = type(manager.action_handlers['player_0'])
        saved_item = player.item_bench[0]
        player.item_bench[0] = None
        manager.observation_states['player_0'] = observation_class(player)
        player.item_bench[0] = saved_item
        manager.action_handlers['player_0'] = action_class(player)
        saved_rng = session.game.rng.py.getstate()
        native = deepcopy(player)
        native.item_bench[0] = None
        expected_observation = observation_class(native)
        native.item_bench[0] = item
        expected_mask = action_class(native)
        assert native.move_item(0, target.get('x', target.get('slot')), target.get('y', -1))
        native.actions_remaining -= 1
        from tft_mcp.session import location_flat
        action = [6, location_flat(target), 0]
        expected_observation.update_observation(action)
        expected_mask.update_action_mask(action)
        session.game.rng.py.setstate(saved_rng)
    # Restore the native draw state before comparing the same action through the adapter.
    session.equip_item(item_slot=0, target=target)
    player = session.game.player_manager.player_states['player_0']
    assert freeze_player(player, 1) == freeze_player(native, 1)
    manager = session.game.player_manager
    with session.simulator_scope():
        actual = manager.observation_states['player_0'].fetch_player_observation()
        expected = expected_observation.fetch_player_observation()
        for key in actual:
            np.testing.assert_array_equal(actual[key], expected[key])
        np.testing.assert_array_equal(manager.action_handlers['player_0'].fetch_action_mask(), expected_mask.fetch_action_mask())


@pytest.mark.parametrize('rejected', [False, True])
def test_duplicator_preserves_process_default_combat_context(tmp_path, rejected):
    import pickle
    from Simulator.battle.combat_context import get_ctx
    session = session_fixture(tmp_path)
    player = install_units(session, bench=[(0, 'garen', 1, [])])
    player.item_bench = ['champion_duplicator'] + [None] * 9
    if rejected:
        player.triple_catalog.clear()
    context = get_ctx()
    before = pickle.dumps(context)
    if rejected:
        with pytest.raises(SessionError) as error:
            session.equip_item(item_slot=0, target=bench(0))
        assert error.value.code == 'internal_error'
    else:
        session.equip_item(item_slot=0, target=bench(0))
    assert get_ctx() is context and pickle.dumps(context) == before


@pytest.mark.parametrize('item,chosen', [('champion_duplicator', 'vanguard'), ('champion_duplicator', False)])
def test_duplication_board_merge_and_chosen_results(tmp_path, item, chosen):
    session = session_fixture(tmp_path)
    stars = 2 if chosen else 1
    player = install_units(session, bench=[(0, 'garen', stars, [])], board=[((6, 3), 'garen', stars, ['bf_sword'])])
    player.board[6][3].chosen = chosen
    player.chosen = chosen
    player.item_bench = [item] + [None] * 9
    receipt = session.equip_item(item_slot=0, target=board(6, 3))
    assert session.get_board()['slots'][27]['unit']['stars'] == (3 if chosen else 2)
    assert session.get_board()['slots'][27]['unit']['chosen'] == chosen
    assert receipt['item_changes'] == [{'slot': 0, 'before': item, 'after': None}, {'slot': 1, 'before': None, 'after': 'bf_sword'}]
    assert session.get_bench()['slots'][0]['unit'] is None


def test_duplicator_native_azir_merge_changes_linked_guards(tmp_path):
    session = session_fixture(tmp_path)
    install_units(session, bench=[(0, 'azir', 1, []), (1, 'azir', 1, [])])
    session.move_unit(source=bench(1), target=board(3, 2))
    player = session.game.player_manager.player_states['player_0']
    player.item_bench = ['champion_duplicator'] + [None] * 9
    receipt = session.equip_item(item_slot=0, target=bench(0))
    assert session.get_board()['slots'][14]['unit']['stars'] == 2
    player = session.game.player_manager.player_states['player_0']
    assert player.num_units_in_play == 1
    assert len(player.board[3][2].sandguard_overlord_coordinates) == 2
    assert receipt['status']['planning_budget']['remaining'] == 12


def test_equipment_lifecycle_budget_detachment_and_mask_disagreement(tmp_path):
    import pickle
    from tft_mcp.session import GameSession
    session = GameSession(tmp_path / 'audit.jsonl')
    with pytest.raises(SessionError) as error:
        session.equip_item(item_slot=True, target=bench(0))
    assert error.value.code == 'invalid_input'
    with pytest.raises(SessionError) as error:
        session.equip_item(item_slot=0, target=bench(0))
    assert error.value.code == 'no_game'
    session.start_game(0)
    player = install_units(session, bench=[(0, 'garen', 1, [])])
    player.item_bench = ['bf_sword', 'chain_vest'] + [None] * 8
    # Deliberately stale native mask is not an authority for singular item legality.
    session.game.player_manager.action_handlers['player_0'].item_mask[:] = 0
    receipt = session.equip_item(item_slot=0, target=bench(0))
    receipt['unit_changes'][0]['after']['items'].append('giant_slayer')
    assert session.get_bench()['slots'][0]['unit']['items'] == ['bf_sword']
    session.game.actions_taken['player_0'] = 14
    before = pickle.dumps(session.game)
    with pytest.raises(SessionError) as error:
        session.equip_item(item_slot=1, target=bench(0))
    assert error.value.code == 'budget_exhausted'
    assert pickle.dumps(session.game) == before
    session.close_game()
    with pytest.raises(SessionError) as error:
        session.equip_item(item_slot=1, target=bench(0))
    assert error.value.code == 'no_game'


@pytest.mark.parametrize('corruption', ['noop', 'equipment', 'origin', 'catalog', 'gold', 'shop', 'inventory', 'gloves', 'capacity', 'count', 'copy_equipment'])
def test_corrupted_native_equipment_cannot_publish(tmp_path, monkeypatch, corruption):
    import pickle
    from Simulator.game.player import Player
    session = session_fixture(tmp_path)
    item = 'champion_duplicator' if corruption == 'copy_equipment' else 'bf_sword'
    player = install_units(session, bench=[(0, 'garen', 1, [])])
    player.item_bench = [item] + [None] * 9
    before = pickle.dumps(session.game)
    native = Player.move_item
    def corrupt(player, *args):
        if corruption == 'noop':
            return True
        result = native(player, *args)
        if corruption == 'equipment':
            player.bench[0].items.append('chain_vest')
        elif corruption == 'origin':
            player.bench[0].origin.pop()
        elif corruption == 'catalog':
            player.triple_catalog.clear()
        elif corruption == 'gold':
            player.gold += 1
        elif corruption == 'shop':
            player.shop[0] = None
        elif corruption == 'inventory':
            player.item_bench[0] = 'chain_vest'
        elif corruption == 'gloves':
            player.thieves_gloves_loc.append([0, -1])
        elif corruption == 'capacity':
            player.max_units += 1
        elif corruption == 'count':
            player.num_units_in_play += 1
        else:
            player.bench[0].items.append('giant_slayer')
        return result
    monkeypatch.setattr(Player, 'move_item', corrupt)
    with pytest.raises(SessionError) as error:
        session.equip_item(item_slot=0, target=bench(0))
    assert error.value.code == 'internal_error'
    assert pickle.dumps(session.game) == before


def test_missing_recipe_rejects_before_native_fallback(tmp_path, monkeypatch):
    from Simulator.battle import item_stats
    session = session_fixture(tmp_path)
    player = install_units(session, bench=[(0, 'garen', 1, ['bf_sword'])])
    player.item_bench = ['bf_sword'] + [None] * 9
    monkeypatch.delitem(item_stats.item_builds, 'deathblade')
    with pytest.raises(SessionError) as error:
        session.equip_item(item_slot=0, target=bench(0))
    assert error.value.code == 'incompatible_item'


def test_duplicate_supports_native_bench_whole_set_drop(tmp_path):
    session = session_fixture(tmp_path)
    player = install_units(session, bench=[(0, 'garen', 1, ['bf_sword', 'chain_vest']), (1, 'garen', 1, [])])
    player.item_bench = ['champion_duplicator'] + ['chain_vest'] * 9
    receipt = session.equip_item(item_slot=0, target=bench(0))
    assert receipt['item_changes'] == [{'slot': 0, 'before': 'champion_duplicator', 'after': None}]
    assert session.get_bench()['slots'][0]['unit']['stars'] == 2
    assert session.get_bench()['slots'][0]['unit']['items'] == []


@pytest.mark.parametrize('corruption', ['unknown_inventory', 'unknown_equipment', 'unknown_champion', 'cost', 'dummy', 'empty_target', 'empty_inventory'])
def test_impossible_or_empty_native_records_reject_atomically(tmp_path, corruption):
    import pickle
    session = session_fixture(tmp_path)
    player = install_units(session, bench=[(0, 'garen', 1, [])])
    player.item_bench = ['bf_sword'] + [None] * 9
    if corruption == 'unknown_inventory':
        player.item_bench[0] = 'unknown'
    elif corruption == 'unknown_equipment':
        player.bench[0].items = ['unknown']
    elif corruption == 'unknown_champion':
        player.bench[0].name = 'unknown'
    elif corruption == 'cost':
        player.bench[0].cost = 9
    elif corruption == 'dummy':
        player.bench[0].target_dummy = True
    elif corruption == 'empty_target':
        player.bench[0] = None
    else:
        player.item_bench[0] = None
    before = pickle.dumps(session.game)
    with pytest.raises(SessionError) as error:
        session.equip_item(item_slot=0, target=bench(0))
    expected = 'empty_slot' if corruption.startswith('empty') else 'unsupported_action' if corruption == 'dummy' else 'internal_error'
    assert error.value.code == expected
    assert pickle.dumps(session.game) == before


@pytest.mark.parametrize('origins', [['warlord', 'vanguard', 'mage'], ['vanguard', 'warlord', 'duelist'], ['warlord', 'vanguard', 'duelist', 'duelist']])
def test_remover_rejects_mismatched_trait_origin_prefix_or_suffix(tmp_path, origins):
    session = session_fixture(tmp_path)
    player = install_units(session, bench=[(0, 'garen', 1, ['duelists_zeal'])])
    player.bench[0].origin = origins
    player.item_bench = ['magnetic_remover'] + [None] * 9
    with pytest.raises(SessionError) as error:
        session.equip_item(item_slot=0, target=bench(0))
    assert error.value.code == 'unsupported_action'
    assert error.value.details['reason'] == 'trait_origin_suffix'


@pytest.mark.parametrize('mode', ['grant', 'remove', 'duplicate'])
def test_trait_corruption_after_native_equipment_rolls_back_and_retries(tmp_path, monkeypatch, mode):
    import pickle
    from Simulator.game.player import Player
    from test_shop_xp import gameplay
    from test_movement import normalized_graph
    sessions = [session_fixture(tmp_path / name) for name in ['actual', 'reference']]
    for session in sessions:
        if mode == 'duplicate':
            player = install_units(session, bench=[(0, 'garen', 1, [])], board=[((0, 0), 'garen', 1, [])])
            item = 'champion_duplicator'
        else:
            player = install_units(session, board=[((0, 0), 'garen', 1, [])])
            if mode == 'remove':
                player.item_bench = ['duelists_zeal'] + [None] * 9
                session.equip_item(item_slot=0, target=board(0, 0))
                player = session.game.player_manager.player_states['player_0']
            item = 'magnetic_remover' if mode == 'remove' else 'duelists_zeal'
        player.item_bench = [item] + [None] * 9
    actual, reference = sessions
    before = pickle.dumps(actual.game)
    refs = actual.game, actual.baselines, actual.module_state, actual.python_rng, actual.baseline_rng
    logs = actual.audit_path.read_bytes(), (actual.native_dir / 'log.txt').read_bytes()
    native = Player.move_item
    with monkeypatch.context() as patch:
        def corrupt(player, *args):
            result = native(player, *args)
            player.team_composition['duelist'] = 999
            player.team_tiers['duelist'] = 999
            return result
        patch.setattr(Player, 'move_item', corrupt)
        with pytest.raises(SessionError) as error:
            actual.equip_item(item_slot=0, target=board(0, 0))
        assert error.value.code == 'internal_error'
    assert pickle.dumps(actual.game) == before
    assert all(a is b for a, b in zip(refs, (actual.game, actual.baselines, actual.module_state, actual.python_rng, actual.baseline_rng)))
    assert logs == (actual.audit_path.read_bytes(), (actual.native_dir / 'log.txt').read_bytes())
    actual.equip_item(item_slot=0, target=board(0, 0))
    reference.equip_item(item_slot=0, target=board(0, 0))
    assert gameplay(actual) == gameplay(reference)
    assert normalized_graph(actual.game) == normalized_graph(reference.game)


@pytest.mark.parametrize('item', ['bf_sword', 'magnetic_remover', 'reforger', 'thieves_gloves'])
def test_initial_catalog_inconsistency_rejects_every_equipment_mode(tmp_path, item):
    import pickle
    session = session_fixture(tmp_path)
    player = install_units(session, bench=[(0, 'garen', 1, ['bf_sword'] if item in {'magnetic_remover', 'reforger'} else [])])
    player.triple_catalog.clear()
    player.item_bench = [item] + [None] * 9
    before = pickle.dumps(session.game)
    with pytest.raises(SessionError) as error:
        session.equip_item(item_slot=0, target=bench(0))
    assert error.value.code == 'internal_error'
    assert pickle.dumps(session.game) == before


def test_early_board_contributor_cascade_rejects_before_unsafe_native_reposition(tmp_path):
    import pickle
    session = session_fixture(tmp_path)
    player = install_units(session, bench=[(0, 'garen', 1, []), (1, 'garen', 2, []), (2, 'garen', 2, [])], board=[((0, 0), 'garen', 1, [])])
    player.item_bench = ['champion_duplicator'] + [None] * 9
    before = pickle.dumps(session.game)
    with pytest.raises(SessionError) as error:
        session.equip_item(item_slot=0, target=board(0, 0))
    assert error.value.code == 'unsupported_action'
    assert error.value.details['reason'] == 'early_board_duplicate_cascade'
    assert pickle.dumps(session.game) == before


def test_final_phase_board_contributor_cascade_remains_supported(tmp_path):
    session = session_fixture(tmp_path)
    player = install_units(session, bench=[(0, 'garen', 1, []), (1, 'garen', 1, []), (2, 'garen', 2, [])], board=[((0, 0), 'garen', 2, [])])
    player.item_bench = ['champion_duplicator'] + [None] * 9
    receipt = session.equip_item(item_slot=0, target=bench(0))
    assert session.get_board()['slots'][0]['unit']['stars'] == 3
    assert all(s['unit'] is None for s in session.get_bench()['slots'])
    assert receipt['status']['planning_budget']['remaining'] == 13


@pytest.mark.parametrize('field,value', [('num', True), ('level', True), ('num', 0)])
def test_impossible_catalog_types_reject_before_equipment(tmp_path, field, value):
    session = session_fixture(tmp_path)
    player = install_units(session, bench=[(0, 'garen', 1, [])])
    player.item_bench = ['bf_sword'] + [None] * 9
    player.triple_catalog[0][field] = value
    with pytest.raises(SessionError) as error:
        session.equip_item(item_slot=0, target=bench(0))
    assert error.value.code == 'internal_error'

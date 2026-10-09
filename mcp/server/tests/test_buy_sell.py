import pytest

from tft_mcp.session import GameSession, SessionError


def session_fixture(tmp_path):
    session = GameSession(tmp_path / 'audit.jsonl', tmp_path / 'native')
    session.start_game(0)
    session.game.player_manager.player_states['player_0'].gold = 100
    return session


def test_buy_then_sell_returns_detached_units_and_one_budget_slot(tmp_path):
    session = session_fixture(tmp_path)
    offer = session.get_shop()['slots'][0]
    receipt = session.buy_unit(shop_slot=0)
    assert receipt['purchased'] == offer['unit']
    assert receipt['gold_spent'] == offer['purchase_cost']
    assert receipt['status']['planning_budget']['remaining'] == 13
    assert receipt['status']['round'] == 1
    assert session.get_shop()['slots'][0]['unit'] is None
    change = receipt['unit_changes'][0]
    assert change['before'] is None
    sale = session.sell_unit(location=change['location'])
    assert sale['sold'] == receipt['purchased']
    assert sale['gold_gained'] == receipt['gold_spent']
    assert sale['returned_items'] == sale['dropped_items'] == []
    assert sale['status']['planning_budget']['remaining'] == 12
    assert session.game.agent_selection == 'player_0'
    receipt['purchased']['items'].append('bf_sword')
    assert sale['sold']['items'] == []


def install_units(session, bench=(), board=(), offer='garen', chosen=False):
    from Simulator.battle.champion import champion
    player = session.game.player_manager.player_states['player_0']
    with session.simulator_scope():
        player.bench = [None] * 9
        player.board = [[None] * 4 for _ in range(7)]
        player.triple_catalog = []
        player.num_units_in_play = 0
        player.thieves_gloves_loc = []
        for kind, units in [('bench', bench), ('board', board)]:
            for position, name, stars, items in units:
                unit = champion(name, stars=stars, itemlist=items)
                entry = next((e for e in player.triple_catalog if e['name'] == name and e['level'] == stars), None)
                if entry:
                    entry['num'] += 1
                else:
                    player.triple_catalog.append({'name': name, 'level': stars, 'num': 1})
                if kind == 'bench':
                    player.bench[position] = unit
                    unit.bench_loc = position
                    if items and items[0] == 'thieves_gloves':
                        player.thieves_gloves_loc.append([position, -1])
                else:
                    x, y = position
                    player.board[x][y] = unit
                    unit.x, unit.y = x, y
                    player.num_units_in_play += 1
                    if items and items[0] == 'thieves_gloves':
                        player.thieves_gloves_loc.append([x, y])
        unit = champion(offer, chosen=chosen)
        player.shop[0] = f'{offer}_{chosen}_c' if chosen else offer
        player.shop_champions[0] = unit
    return player


def test_merge_capacity_rejects_both_native_corruption_cases_atomically(tmp_path):
    from test_progression import gameplay
    for final_contributor in (False, True):
        session = session_fixture(tmp_path / str(final_contributor))
        bench = [(i, name, 1, []) for i, name in enumerate(['fiora', 'nami', 'vayne', 'nidalee', 'diana', 'lissandra', 'wukong', 'twistedfate', 'maokai'])] if final_contributor else []
        board = [((0, 0), 'garen', 1, [] if final_contributor else ['bf_sword']),
                 ((0, 1), 'garen', 1, ['bf_sword'] if final_contributor else [])]
        player = install_units(session, bench, board)
        player.item_bench = ['chain_vest'] * 10
        committed = session.game
        before = gameplay(session)
        with pytest.raises(SessionError) as error:
            session.buy_unit(shop_slot=0)
        assert error.value.code == 'capacity_exceeded'
        assert session.game is committed
        assert gameplay(session) == before


@pytest.mark.parametrize('full_bench,cascade,chosen', [(False, False, False), (True, False, False), (False, True, False), (False, False, 'vanguard')])
def test_native_merges_preserve_copies_prices_catalog_and_pool(tmp_path, full_bench, cascade, chosen):
    session = session_fixture(tmp_path)
    stars = 2 if chosen else 1
    bench = [(0, 'garen', stars, []), (1, 'garen', stars, [])]
    if cascade:
        bench += [(2, 'garen', 2, []), (3, 'garen', 2, [])]
    if full_bench:
        bench += [(i, name, 1, []) for i, name in enumerate(['fiora', 'nami', 'vayne', 'nidalee', 'diana', 'lissandra', 'wukong'], start=2)]
    player = install_units(session, bench=bench, chosen=chosen)
    from copy import deepcopy
    native = deepcopy(player)
    if full_bench:
        from Simulator.encoding.token.action import ActionToken
        with session.simulator_scope():
            assert ActionToken(player).buy_mask[0] == 0
    with session.simulator_scope():
        assert native.buy_shop_action(0)
    receipt = session.buy_unit(shop_slot=0)
    assert receipt['gold_spent'] == (3 if chosen else 1)
    expected_stars = 3 if cascade or chosen else 2
    assert session.get_bench()['slots'][0]['unit']['stars'] == expected_stars
    player = session.game.player_manager.player_states['player_0']
    assert player.triple_catalog == native.triple_catalog
    assert player.pool_obj.__dict__ == native.pool_obj.__dict__
    assert player.chosen == native.chosen
    assert receipt['item_changes'] == []
    if full_bench:
        assert len(receipt['unit_changes']) == 2


@pytest.mark.parametrize('cascade,bench_items,free', [(False, ['bf_sword'], 1), (True, [], 0), (False, ['bf_sword', 'recurve_bow'], 1)])
def test_capacity_carries_bench_returns_and_drops_into_board_and_cascade(tmp_path, cascade, bench_items, free):
    session = session_fixture(tmp_path)
    bench = [(0, 'garen', 1, bench_items)]
    board = [((0, 0), 'garen', 1, ['chain_vest'])]
    if cascade:
        bench = [(0, 'garen', 1, []), (1, 'garen', 1, []), (2, 'garen', 2, [])]
        board = [((0, 0), 'garen', 2, ['chain_vest'])]
    player = install_units(session, bench, board)
    player.item_bench = ['negatron_cloak'] * (10 - free) + [None] * free
    player.max_units = 9
    if len(bench_items) > free:
        receipt = session.buy_unit(shop_slot=0)
        assert receipt['item_changes'] == [{'slot': 9, 'before': None, 'after': 'chain_vest'}]
    else:
        with pytest.raises(SessionError) as error:
            session.buy_unit(shop_slot=0)
        assert error.value.code == 'capacity_exceeded'


@pytest.mark.parametrize('kind,items,free,returned,dropped', [
    ('bench', ['bf_sword', 'recurve_bow'], 1, [], ['bf_sword', 'recurve_bow']),
    ('bench', ['bf_sword', 'recurve_bow'], 2, ['bf_sword', 'recurve_bow'], []),
    ('board', ['bf_sword'], 1, ['bf_sword'], []),
    ('board', ['thieves_gloves', 'infinity_edge', 'rapid_firecannon'], 1, ['thieves_gloves'], []),
    ('bench', ['thieves_gloves', 'infinity_edge', 'rapid_firecannon'], 0, [], ['thieves_gloves']),
    ('bench', ['thieves_gloves', 'infinity_edge', 'rapid_firecannon'], 1, ['thieves_gloves'], []),
])
def test_sales_follow_native_equipment_return_drop_and_glove_rules(tmp_path, kind, items, free, returned, dropped):
    session = session_fixture(tmp_path)
    units = [(0 if kind == 'bench' else (2, 3), 'garen', 1, items)]
    player = install_units(session, bench=units if kind == 'bench' else [], board=units if kind == 'board' else [])
    player.item_bench = ['chain_vest'] * (10 - free) + [None] * free
    location = {'kind': 'bench', 'slot': 0} if kind == 'bench' else {'kind': 'board', 'x': 2, 'y': 3}
    from copy import deepcopy
    native = deepcopy(player)
    with session.simulator_scope():
        native.sell_action(28 if kind == 'bench' else 11)
    receipt = session.sell_unit(location=location)
    assert receipt['sold']['items'] == items
    assert receipt['returned_items'] == returned
    assert receipt['dropped_items'] == dropped
    current = session.game.player_manager.player_states['player_0']
    assert current.item_bench == native.item_bench
    assert current.triple_catalog == native.triple_catalog
    assert current.pool_obj.__dict__ == native.pool_obj.__dict__


def test_board_glove_capacity_and_dummy_sales_reject_without_change(tmp_path):
    from test_progression import gameplay
    session = session_fixture(tmp_path)
    player = install_units(session, board=[((0, 0), 'garen', 1, ['thieves_gloves', 'infinity_edge', 'rapid_firecannon'])])
    player.item_bench = ['bf_sword'] * 10
    before = gameplay(session)
    with pytest.raises(SessionError) as error:
        session.sell_unit(location={'kind': 'board', 'x': 0, 'y': 0})
    assert error.value.code == 'capacity_exceeded'
    assert gameplay(session) == before
    player.board[0][0].target_dummy = True
    with pytest.raises(SessionError) as error:
        session.sell_unit(location={'kind': 'board', 'x': 0, 'y': 0})
    assert error.value.code == 'unsupported_action'


def test_azir_sale_reports_native_sandguard_removals(tmp_path):
    session = session_fixture(tmp_path)
    player = install_units(session, board=[((3, 1), 'azir', 1, [])])
    with session.simulator_scope():
        player.board[3][1].sandguard_overlord_coordinates = player.find_azir_sandguards(3, 1)
    coords = player.board[3][1].sandguard_overlord_coordinates
    location = {'kind': 'board', 'x': 3, 'y': 1}
    receipt = session.sell_unit(location=location)
    assert receipt['gold_gained'] == 5
    assert [change['location'] for change in receipt['unit_changes']] == sorted(
        [location] + [{'kind': 'board', 'x': x, 'y': y} for x, y in coords], key=lambda loc: (loc['x'], loc['y']))
    assert all(change['after'] is None for change in receipt['unit_changes'])


def test_chosen_purchase_and_sale_use_native_price_and_clear_chosen(tmp_path):
    session = session_fixture(tmp_path)
    install_units(session, chosen='vanguard')
    receipt = session.buy_unit(shop_slot=0)
    assert receipt['purchased']['stars'] == 2
    assert receipt['gold_spent'] == 3
    assert session.game.player_manager.player_states['player_0'].chosen == 'vanguard'
    sold = session.sell_unit(location={'kind': 'bench', 'slot': 0})
    assert sold['gold_gained'] == 3
    assert session.game.player_manager.player_states['player_0'].chosen is False


@pytest.mark.parametrize('failure', ['native_action', 'observation', 'baseline', 'copy', 'receipt', 'native', 'audit'])
@pytest.mark.parametrize('action', ['buy', 'sell'])
def test_failed_action_discards_aggregate_rng_logs_and_retries(tmp_path, monkeypatch, failure, action):
    import os
    import random
    import numpy as np
    from Simulator.battle import champion, origin_class
    from Simulator.simulators.tft_simulator import TFT_Simulator
    import tft_mcp.session as module
    from test_progression import gameplay
    session = session_fixture(tmp_path / 'actual')
    reference = session_fixture(tmp_path / 'reference')
    for current in (session, reference):
        units = [(0, 'fiora', 1, ['bf_sword'])]
        if action == 'sell' and failure == 'copy':
            units.append((1, 'garen', 1, []))
        install_units(current, bench=units)
    method = lambda current: current.buy_unit(shop_slot=0) if action == 'buy' else current.sell_unit(location={'kind': 'bench', 'slot': 0})
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
            if failure == 'copy' and selected == 'player_0':
                player = game.player_manager.player_states['player_0']
                from copy import deepcopy
                player.bench[8] = deepcopy(next(unit for unit in player.bench if unit))
            if failure == 'native' and selected == 'player_0':
                with open('log.txt', 'a') as stream:
                    stream.write('partial candidate native log')
                raise OSError('native log write failed')
            return result
        if failure == 'native_action':
            from Simulator.game.player import Player
            name = 'buy_shop_action' if action == 'buy' else 'sell_action'
            original_action = getattr(Player, name)
            def fail_native(player, *args):
                original_action(player, *args)
                raise RuntimeError('after native mutation before observation update')
            patch.setattr(Player, name, fail_native)
        elif failure in {'observation', 'baseline', 'copy', 'native'}:
            # Failure injection exercises rollback, never substitutes a successful simulator.
            patch.setattr(TFT_Simulator, 'step', fail_step)
        elif failure == 'receipt':
            def fail_receipt(*args):
                raise RuntimeError('receipt construction failed after native progression')
            patch.setattr(module, 'unit_changes', fail_receipt)
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


def test_action_lifecycle_budget_full_bench_and_catalog_errors_preserve_state(tmp_path):
    from test_progression import gameplay
    session = session_fixture(tmp_path)
    player = install_units(session, bench=[(i, name, 1, []) for i, name in enumerate(['fiora', 'nami', 'vayne', 'nidalee', 'diana', 'lissandra', 'wukong', 'twistedfate', 'maokai'])])
    expected = gameplay(session)
    with pytest.raises(SessionError) as error:
        session.buy_unit(shop_slot=0)
    assert error.value.code == 'capacity_exceeded'
    assert gameplay(session) == expected
    player.triple_catalog[0]['num'] = 0
    with pytest.raises(SessionError) as error:
        session.sell_unit(location={'kind': 'bench', 'slot': 0})
    assert error.value.code == 'internal_error'
    for _ in range(14):
        session.controlled_action([0, 0, 0])
    for action in (lambda: session.buy_unit(shop_slot=0), lambda: session.sell_unit(location={'kind': 'bench', 'slot': 0})):
        with pytest.raises(SessionError) as error:
            action()
        assert error.value.code == 'budget_exhausted'
    session.state = 'terminal'
    for action in (lambda: session.buy_unit(shop_slot=0), lambda: session.sell_unit(location={'kind': 'bench', 'slot': 0})):
        with pytest.raises(SessionError) as error:
            action()
        assert error.value.code == 'game_terminal'
    for arguments in ({'shop_slot': True}, {'shop_slot': 1.0}, {'shop_slot': 0, 'player_id': 'player_1'}):
        with pytest.raises(SessionError) as error:
            session.buy_unit(**arguments)
        assert error.value.code == 'invalid_input'


def test_buy_sell_reads_and_rejections_do_not_change_deterministic_journey(tmp_path):
    from test_progression import gameplay
    sessions = [session_fixture(tmp_path / str(i)) for i in range(2)]
    for current in sessions:
        install_units(current)
    for index, current in enumerate(sessions):
        if index:
            current.get_board()
            current.get_items()
            current.get_shop()
            with pytest.raises(SessionError):
                current.sell_unit(location={'kind': 'bench', 'slot': 8})
        current.buy_unit(shop_slot=0)
        if index:
            current.get_economy()
            current.get_bench()
        current.sell_unit(location={'kind': 'bench', 'slot': 0})
        current.end_turn()
    assert gameplay(sessions[0]) == gameplay(sessions[1])


@pytest.mark.parametrize('corruption', ['price', 'sale_count'])
def test_inconsistent_unit_price_or_catalog_count_rejects_before_native(tmp_path, corruption):
    from test_progression import gameplay
    session = session_fixture(tmp_path)
    player = install_units(session, bench=[(0, 'garen', 1, [])])
    if corruption == 'price':
        player.shop_champions[0].cost = 2
        action = lambda: session.buy_unit(shop_slot=0)
    else:
        player.triple_catalog[0]['num'] = 2
        action = lambda: session.sell_unit(location={'kind': 'bench', 'slot': 0})
    before = gameplay(session)
    with pytest.raises(SessionError) as error:
        action()
    assert error.value.code == 'internal_error'
    assert gameplay(session) == before


@pytest.mark.parametrize('reason', ['four_star_sale', 'four_star_merge', 'chosen_cycle', 'sandguard'])
def test_unsupported_native_actions_reject_atomically(tmp_path, reason):
    from test_progression import gameplay
    session = session_fixture(tmp_path)
    stars = 3 if reason in {'four_star_merge', 'chosen_cycle'} else 1
    player = install_units(session, bench=[(0, 'garen', stars, []), (1, 'garen', stars, [])] if stars == 3 else [(0, 'garen', 1, [])])
    if reason == 'four_star_sale':
        player.bench[0].stars = 4
    if reason == 'sandguard':
        player.bench[0].target_dummy = True
        player.bench[0].name = 'sandguard'
    if reason in {'four_star_merge', 'chosen_cycle'}:
        player.shop_champions[0].stars = 3
        if reason == 'chosen_cycle':
            player.shop_champions[0].chosen = 'vanguard'
            player.shop[0] = 'garen_vanguard_c'
        action = lambda: session.buy_unit(shop_slot=0)
    else:
        action = lambda: session.sell_unit(location={'kind': 'bench', 'slot': 0})
    before = gameplay(session)
    with pytest.raises(SessionError) as error:
        action()
    assert error.value.code == 'unsupported_action'
    assert gameplay(session) == before


@pytest.mark.parametrize('starting_copies,expected_copies', [(0, 9), (28, 29)])
def test_promoted_sale_preserves_native_copy_quantity_and_pool_saturation(tmp_path, starting_copies, expected_copies):
    session = session_fixture(tmp_path)
    player = install_units(session, bench=[(0, 'garen', 3, [])])
    player.pool_obj.COST_1['garen'] = starting_copies
    player.pool_obj.update_stats(one=True)
    receipt = session.sell_unit(location={'kind': 'bench', 'slot': 0})
    assert receipt['gold_gained'] == 9
    assert session.game.pool_obj.COST_1['garen'] == expected_copies
    assert session.game.player_manager.player_states['player_0'].triple_catalog == []

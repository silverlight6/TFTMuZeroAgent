from copy import deepcopy
import pickle
import random

import numpy as np
import pytest

from tft_mcp.session import GameSession, SessionError, freeze_player
from test_inspection_protocol import INSPECTION_TOOLS


def started(tmp_path):
    session = GameSession(tmp_path / 'audit.jsonl', tmp_path / 'native')
    session.start_game(0)
    return session


def test_native_special_units_use_container_locations_and_detached_allowlist(tmp_path):
    from Simulator.battle.champion import champion
    session = started(tmp_path)
    player = session.game.player_manager.player_states['player_0']
    azir = champion('azir', chosen='keeper')
    azir.items = ['guardian_angel']
    azir.stars = 4
    azir.x, azir.y = 2, 2
    azir.sandguard_overlord_coordinates = [(6, 1)]
    azir.overlord = {'hidden': 'must never serialize'}
    kayn = champion('kayn')
    kayn.kayn_form = 'kayn_rhast'
    dummy = champion('sandguard', target_dummy=True)
    player.board[6][3] = azir
    player.board[0][2] = dummy
    player.bench[8] = kayn
    player.num_units_in_play = 2
    board = session.get_board()
    expected = {'champion': 'azir', 'stars': 4, 'items': ['guardian_angel'], 'chosen': 'keeper',
                'cost': 5, 'kayn_form': None, 'traits': ['warlord', 'keeper', 'emperor'],
                'target_dummy': False, 'sandguard_overlord_coordinates': [(6, 1)]}
    assert board['slots'][27] == {'location': {'kind': 'board', 'x': 6, 'y': 3}, 'unit': expected}
    assert board['slots'][2]['unit']['target_dummy'] is True
    assert board['num_units_in_play'] == 2
    assert session.get_bench()['slots'][8]['unit']['kayn_form'] == 'kayn_rhast'
    board['slots'][27]['unit']['items'].append('hidden')
    board['slots'][27]['unit']['sandguard_overlord_coordinates'].append((0, 0))
    assert azir.items == ['guardian_angel']
    assert azir.sandguard_overlord_coordinates == [(6, 1)]
    import json
    from jsonschema import validate
    from tft_mcp.transport import TOOLS
    schemas = {tool.name: tool.outputSchema for tool in TOOLS}
    for name in INSPECTION_TOOLS:
        validate(json.loads(json.dumps(getattr(session, name)())), schemas[name])


def test_read_purity_preserves_entire_game_rng_caches_aliases_and_later_progression(tmp_path):
    from Simulator.battle import champion, origin_class
    session = started(tmp_path / 'reads')
    reference = started(tmp_path / 'reference')
    player = session.game.player_manager.player_states['player_0']
    game = session.game
    native = session.native_dir
    baselines = session.baselines
    module_state = session.module_state
    baseline_rng = session.baseline_rng
    serialized = pickle.dumps(game)
    python_rng = random.getstate()
    numpy_rng = pickle.dumps(np.random.get_state())
    bindings = (champion.log, champion.test_multiple, origin_class.game_compositions, origin_class.game_comp_tiers)
    for _ in range(3):
        for name in INSPECTION_TOOLS:
            getattr(session, name)()
        with pytest.raises(SessionError):
            session.get_board('unknown')
    assert pickle.dumps(game) == serialized
    assert session.game is game
    assert session.native_dir is native
    assert session.baselines is baselines
    assert session.module_state is module_state
    assert session.baseline_rng is baseline_rng
    assert random.getstate() == python_rng
    assert pickle.dumps(np.random.get_state()) == numpy_rng
    assert all(actual is before for actual, before in zip(
        (champion.log, champion.test_multiple, origin_class.game_compositions, origin_class.game_comp_tiers), bindings))
    assert game.player_manager.observation_states['player_0'].player is player
    assert game.player_manager.action_handlers['player_0'].player is player
    for _ in range(2):
        for name in INSPECTION_TOOLS:
            getattr(session, name)()
        a, b = session.end_turn(), reference.end_turn()
        assert {k: v for k, v in a.items() if k != 'game_id'} == {k: v for k, v in b.items() if k != 'game_id'}
        for key, actual in session.game.player_manager.player_states.items():
            if actual is not None:
                assert freeze_player(actual, a['round']) == freeze_player(reference.game.player_manager.player_states[key], b['round'])
        assert pickle.dumps(session.game.rng) == pickle.dumps(reference.game.rng)
        assert pickle.dumps(session.baseline_rng) == pickle.dumps(reference.baseline_rng)


def test_exhausted_budget_and_stored_traits_are_read_without_recalculation(tmp_path):
    session = started(tmp_path)
    for _ in range(14):
        session.controlled_action([0, 0, 0])
    player = session.game.player_manager.player_states['player_0']
    player.team_composition['ninja'] = 3
    player.team_tiers['ninja'] = 7
    assert {'trait_id': 'ninja', 'count': 3, 'tier': 7} in session.get_traits()['traits']
    assert session.get_economy()['planning_budget'] == session.get_game_status()['planning_budget'] == {'capacity': 14, 'remaining': 0}
    before = pickle.dumps(session.game)
    for name in INSPECTION_TOOLS:
        getattr(session, name)()
    assert pickle.dumps(session.game) == before
    assert session.get_round()['round'] == 1


def test_terminal_winner_records_are_copied_and_missing_snapshot_is_an_error(tmp_path):
    from Simulator.battle.champion import champion
    session = started(tmp_path)
    session.game.player_manager.player_states['player_0'].bench[8] = champion('kayn', itemlist=['guardian_angel'])
    for key, player in session.game.player_manager.player_states.items():
        if key != 'player_0':
            player.health = -1000
    assert session.end_turn()['outcome']['controlled_placement'] == 1
    snapshot = deepcopy(session.terminal_snapshot)
    public = deepcopy(session.public_final)
    for name in INSPECTION_TOOLS:
        result = getattr(session, name)()
        result['round'] = -100
        if 'slots' in result:
            for slot in result['slots']:
                if slot.get('unit'):
                    slot['unit']['items'].append('mutated')
                    slot['unit']['traits'].clear()
                    slot['unit']['sandguard_overlord_coordinates'].append([0, 0])
            result['slots'].clear()
        if 'traits' in result:
            result['traits'].clear()
    assert session.terminal_snapshot == snapshot
    assert session.public_final == public
    assert session.get_economy()['planning_budget'] is None
    session.terminal_snapshot = None
    with pytest.raises(SessionError) as error:
        session.get_board()
    assert error.value.code == 'internal_error'


def test_empty_and_inconsistent_shop_slots_and_inventory_are_explicit(tmp_path):
    session = started(tmp_path)
    player = session.game.player_manager.player_states['player_0']
    player.shop[4] = None
    player.shop_champions[4] = None
    player.item_bench[9] = 'kayn_rhast'
    assert session.get_shop()['slots'][4] == {'slot': 4, 'unit': None, 'purchase_cost': None}
    assert session.get_items()['slots'][9] == {'slot': 9, 'item': 'kayn_rhast'}
    player.shop[4] = 'ahri'
    before = pickle.dumps(session.game)
    with pytest.raises(SessionError) as error:
        session.get_shop()
    assert error.value.code == 'internal_error'
    assert error.value.details == {'category': 'shop', 'slot': 4}
    assert pickle.dumps(session.game) == before


def test_nonempty_shop_offer_mismatch_is_rejected_without_mutation(tmp_path):
    session = started(tmp_path)
    player = session.game.player_manager.player_states['player_0']
    player.shop[0] = 'not_the_stored_champion'
    before = pickle.dumps(session.game)
    with pytest.raises(SessionError) as error:
        session.get_shop()
    assert error.value.code == 'internal_error'
    assert error.value.details == {'category': 'shop', 'slot': 0}
    assert pickle.dumps(session.game) == before


def test_terminal_negative_health_is_retained_without_clamping(tmp_path):
    session = started(tmp_path)
    for player in session.game.player_manager.player_states.values():
        player.health = -1000
    assert session.end_turn()['state'] == 'terminal'
    assert session.get_economy()['health'] == session.terminal_snapshot['economy']['health'] < 0
    assert session.get_economy()['planning_budget'] is None

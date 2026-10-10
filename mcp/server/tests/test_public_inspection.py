from copy import deepcopy
import json
import pickle
import random

from jsonschema import validate, ValidationError
import numpy as np
import pytest

from support import started_session as started
from tft_mcp.session import SessionError
from tft_mcp.transport import TOOLS


def test_public_projection_uses_keys_and_excludes_nested_private_state(tmp_path):
    from Simulator.battle.champion import champion
    session = started(tmp_path)
    player = session.game.player_manager.player_states['player_1']
    player.player_num = 777
    player.gold, player.exp = 987654, 876543
    player.shop[0] = 'hidden_shop'
    player.item_bench[0] = 'hidden_inventory'
    player.bench[0] = champion('ahri')
    unit = champion('azir', chosen='keeper')
    unit.items = ['guardian_angel']
    unit.sandguard_overlord_coordinates = [(6, 1)]
    unit.overlord = {'secret': 'hidden_combat'}
    unit.target = {'private': player}
    unit.x, unit.y = 0, 0
    player.board[6][3] = unit
    player.num_units_in_play = 1
    player.team_composition['ninja'], player.team_tiers['ninja'] = 3, 7
    board = session.get_board('player_1')
    assert board['slots'][27]['location'] == {'kind': 'board', 'x': 6, 'y': 3}
    assert board['slots'][27]['unit'] == {
        'champion': 'azir', 'stars': 2, 'items': ['guardian_angel'], 'chosen': 'keeper',
        'cost': 5, 'kayn_form': None, 'traits': ['warlord', 'keeper', 'emperor'],
        'target_dummy': False, 'sandguard_overlord_coordinates': [(6, 1)]}
    traits = session.get_traits('player_1')
    assert {'trait_id': 'ninja', 'count': 3, 'tier': 7} in traits['traits']
    players = session.get_players()
    assert players['players'][1] == {'player_id': 'player_1', 'controlled': False,
        'status': 'alive', 'health': 100, 'level': 1, 'placement': None}
    schemas = {tool.name: tool.outputSchema for tool in TOOLS}
    for name, result in [('get_board', board), ('get_traits', traits), ('get_players', players)]:
        encoded = json.loads(json.dumps(result))
        validate(encoded, schemas[name])
        assert all(secret not in json.dumps(encoded) for secret in ('987654', '876543', 'hidden_shop', 'hidden_inventory', 'hidden_combat'))
        with pytest.raises(ValidationError):
            validate({**encoded, 'gold': 987654}, schemas[name])
    for name, result, collection in [('get_players', players, 'players'), ('get_traits', traits, 'traits')]:
        leaked = json.loads(json.dumps(result))
        leaked[collection][0]['gold'] = 987654
        with pytest.raises(ValidationError):
            validate(leaked, schemas[name])
    for placement in (0, 9, True):
        leaked = deepcopy(players)
        leaked['players'][1]['placement'] = placement
        with pytest.raises(ValidationError):
            validate(leaked, schemas['get_players'])
    for secret_path in ('unit', 'location'):
        leaked = json.loads(json.dumps(board))
        leaked['slots'][27][secret_path]['private'] = 'hidden'
        with pytest.raises(ValidationError):
            validate(leaked, schemas['get_board'])
    board['slots'][27]['unit']['items'].clear()
    board['slots'][27]['unit']['traits'].clear()
    board['slots'][27]['unit']['sandguard_overlord_coordinates'].clear()
    traits['traits'].clear()
    players['players'][1]['health'] = -999
    assert unit.items == ['guardian_angel']
    assert unit.origin == ['warlord', 'keeper', 'emperor']
    assert unit.sandguard_overlord_coordinates == [(6, 1)]
    assert player.health == 100
    assert session.get_traits('player_1')['traits']


def test_public_reads_preserve_complete_graph_rng_caches_and_bindings(tmp_path):
    from Simulator.battle import champion, origin_class
    session = started(tmp_path)
    game = session.game
    manager = game.player_manager
    player = manager.player_states['player_1']
    aggregate = (game, session.baselines, session.module_state, session.baseline_rng,
                 session.terminal_snapshot, session.public_final, session.placements)
    before = pickle.dumps(aggregate)
    process_rng = (random.getstate(), pickle.dumps(np.random.get_state()))
    bindings = (champion.log, champion.test_multiple, origin_class.game_compositions, origin_class.game_comp_tiers)
    for _ in range(3):
        session.get_players()
        for player_id in manager.player_ids:
            session.get_board(player_id)
            session.get_traits(player_id)
        for name in ('get_board', 'get_traits'):
            with pytest.raises(SessionError) as error:
                getattr(session, name)('unknown')
            assert error.value.code == 'invalid_player'
    assert pickle.dumps(aggregate) == before
    assert session.game is game
    assert manager.observation_states['player_1'].player is player
    assert manager.action_handlers['player_1'].player is player
    assert random.getstate() == process_rng[0]
    assert pickle.dumps(np.random.get_state()) == process_rng[1]
    assert all(a is b for a, b in zip(bindings,
        (champion.log, champion.test_multiple, origin_class.game_compositions, origin_class.game_comp_tiers)))


def test_native_removed_players_and_winner_keep_final_public_records(tmp_path):
    session = started(tmp_path)
    for key, player in session.game.player_manager.player_states.items():
        if key != 'player_0':
            player.health = -1000
    status = session.end_turn()
    assert status['outcome']['controlled_placement'] == 1
    snapshot = deepcopy(session.terminal_snapshot)
    public = deepcopy(session.public_final)
    result = session.get_players()
    assert [row['player_id'] for row in result['players']] == [f'player_{i}' for i in range(8)]
    assert sorted(row['placement'] for row in result['players']) == list(range(1, 9))
    for row in result['players']:
        key = row['player_id']
        assert row['placement'] == session.placements[key]
        assert row['health'] == public[key]['health']
        assert row['level'] == public[key]['level']
        assert row['status'] == ('winner' if key == 'player_0' else 'eliminated')
        if key != 'player_0':
            for name, category in [('get_board', 'board'), ('get_traits', 'traits')]:
                with pytest.raises(SessionError) as error:
                    getattr(session, name)(key)
                assert error.value.code == 'player_eliminated'
                assert error.value.details == {'player_id': key, 'category': category}
                assert 'get_players' in str(error.value)
        row['health'] = 123456
    assert session.public_final == public
    assert session.terminal_snapshot == snapshot
    assert session.get_board()['round'] == snapshot['round']
    assert session.get_traits()['round'] == snapshot['round']

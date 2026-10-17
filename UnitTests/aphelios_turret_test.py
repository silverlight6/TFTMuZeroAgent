"""Aphelios turrets are extra board units that can be placed during positioning."""

from Simulator import config
from Simulator.battle.champion import champion
from Simulator.encoding.token.basic_observation import ObservationToken
from Simulator.game.game_round import Game_Round
from Simulator.game.player import Player
from Simulator.game.player_manager import PlayerManager
from Simulator.game.pool import pool
from Simulator.game.step_function import Step_Function
from Simulator.simulators.tft_simulator import TFTConfig
from Simulator.utils import x_y_to_1d_coord
import numpy as np


def _player(player_num=0, max_units=4) -> Player:
    p = Player(pool(), player_num)
    p.gold = 1000
    p.max_units = max_units
    return p


def _occupied(player):
    return [
        (x, y, player.board[x][y].name)
        for x in range(7)
        for y in range(4)
        if player.board[x][y]
    ]


def _turret_hex(player):
    for x in range(7):
        for y in range(4):
            unit = player.board[x][y]
            if unit and unit.name == 'aphelios_turret':
                return x, y, unit
    return None


def test_aphelios_turret_spawns_as_extra_unit():
    p = _player()
    p.buy_champion(champion('aphelios'))
    assert p.move_bench_to_board(0, 0, 0)

    occupied = _occupied(p)
    names = [name for _, _, name in occupied]
    assert names.count('aphelios') == 1
    assert names.count('aphelios_turret') == 1
    assert p.num_units_in_play == 1

    tx, ty, turret = _turret_hex(p)
    assert turret.target_dummy
    assert turret.overlord is p.board[0][0]
    assert [tx, ty] in p.board[0][0].turret_coordinates


def test_aphelios_turret_can_be_moved_during_positioning():
    p = _player()
    p.buy_champion(champion('aphelios'))
    p.move_bench_to_board(0, 0, 0)
    tx, ty, _ = _turret_hex(p)

    dest_x, dest_y = 6, 3
    if (tx, ty) == (dest_x, dest_y):
        dest_x, dest_y = 5, 3
    assert p.move_board_to_board(tx, ty, dest_x, dest_y)
    assert p.board[dest_x][dest_y].name == 'aphelios_turret'
    assert p.board[0][0].name == 'aphelios'
    assert [dest_x, dest_y] in p.board[0][0].turret_coordinates
    assert p.num_units_in_play == 1


def test_aphelios_turret_included_in_position_unit_list():
    p = _player()
    p.buy_champion(champion('aphelios'))
    p.move_bench_to_board(0, 0, 0)
    p.buy_champion(champion('garen'))
    p.move_bench_to_board(0, 2, 0)

    manager = PlayerManager(2, pool(), TFTConfig())
    step = Step_Function(manager)
    step.create_unit_list(p)
    names = [p.board[x][y].name for x, y in step.position_list]
    assert 'aphelios' in names
    assert 'aphelios_turret' in names
    assert 'garen' in names
    assert len(step.position_list) == 3
    assert p.num_units_in_play == 2

    turret_idx = names.index('aphelios_turret')
    dest = x_y_to_1d_coord(6, 3)
    action = np.full(12, 28, dtype=np.int64)
    action[turret_idx] = dest
    step.position_controller(action, p)
    assert p.board[6][3].name == 'aphelios_turret'


def test_benching_or_selling_aphelios_removes_turret():
    p = _player()
    p.buy_champion(champion('aphelios'))
    p.move_bench_to_board(0, 0, 0)
    assert _turret_hex(p) is not None

    assert p.move_board_to_bench(0, 0)
    assert _turret_hex(p) is None
    assert all(cell is None for row in p.board for cell in row)

    p.move_bench_to_board(0, 1, 1)
    assert _turret_hex(p) is not None
    p.sell_champion(p.board[1][1])
    assert _turret_hex(p) is None
    assert p.num_units_in_play == 0


def test_aphelios_turret_observation_and_combat():
    p1 = _player(0)
    p2 = _player(1)
    p1.buy_champion(champion('aphelios'))
    p1.move_bench_to_board(0, 0, 0)
    p2.buy_champion(champion('garen'))
    p2.move_bench_to_board(0, 0, 0)
    p1.opponent = p2
    p2.opponent = p1

    tokens = ObservationToken(p1).create_board_vector(p1)
    occupied_tokens = int(np.count_nonzero(tokens[:, 0]))
    assert occupied_tokens == 2

    pool_obj = pool()
    manager = PlayerManager(config.NUM_PLAYERS, pool_obj, TFTConfig())
    manager.reinit_player_set([p1, p2])
    game_round = Game_Round(p1, pool_obj, manager)
    result, damage = game_round.single_combat_phase([p1, p2])
    assert result in (0, 1, 2)
    assert isinstance(damage, (int, float, np.integer, np.floating))

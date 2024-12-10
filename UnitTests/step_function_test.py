import config
import pytest
import numpy as np
from Simulator import pool
from Simulator.battle_generator import BattleGenerator
from Simulator.player_manager import PlayerManager
from Simulator.observation.vector.observation import ObservationVector
from Simulator.step_function import Step_Function
from Simulator.tft_config import TFTConfig
from Simulator.utils import x_y_to_1d_coord


def test_position_controller():
    battle_generator = BattleGenerator()
    [player, opponent, other_players] = battle_generator.generate_battle()

    PLAYER = player
    PLAYER.opponent = opponent
    opponent.opponent = PLAYER

    pool_obj = pool.pool()
    player_manager = PlayerManager(config.NUM_PLAYERS, pool_obj, TFTConfig(observation_class=ObservationVector))
    player_manager.reinit_player_set([PLAYER] + list(other_players.values()))

    step_function = Step_Function(player_manager)

    original_coord = [0, 0]
    switch_unit = None
    switch_coord = 0
    switch_coord_check = [0, 0]
    first_unit = True

    # switch first 2
    for x in range(len(PLAYER.board)):
        for y in range(len(PLAYER.board[0])):
            if PLAYER.board[x][y]:
                if first_unit:
                    first_unit = False
                    original_coord = [x, y]
                else:
                    switch_unit = PLAYER.board[x][y]
                    switch_coord = x_y_to_1d_coord(x, y)
                    switch_coord_check = [x, y]
                    break
        if switch_coord != 0:
            break

    action = np.ones(12) * 28
    action[0] = switch_coord
    step_function.position_controller(action, PLAYER)

    assert PLAYER.board[original_coord[0]][original_coord[1]].name == switch_unit.name, \
        "Unit is not where it was sent to"
    assert PLAYER.board[switch_coord_check[0]][switch_coord_check[1]].name != switch_unit.name, \
        "Unit found at original location"

    first_unit = True
    list_of_actions = []
    list_of_champions = []
    # switch all to original square
    for x in range(len(PLAYER.board)):
        for y in range(len(PLAYER.board[0])):
            if PLAYER.board[x][y]:
                if first_unit:
                    first_unit = False
                    original_coord = [x, y]
                    list_of_actions = [28]
                    list_of_champions = [PLAYER.board[x][y].name]
                    switch_coord_check = [x, y]
                else:
                    list_of_actions.append(x_y_to_1d_coord(switch_coord_check[0], switch_coord_check[1]))
                    list_of_champions.append(PLAYER.board[x][y].name)
                    switch_coord_check = [x, y]

    action = np.ones(12) * 28
    action[0:len(list_of_actions)] = list_of_actions

    step_function.position_controller(action, PLAYER)

    assert PLAYER.board[original_coord[0]][original_coord[1]].name == list_of_champions[1], \
        "Unit is not where it was sent to"

    assert PLAYER.board[switch_coord_check[0]][switch_coord_check[1]].name != list_of_champions[-1], \
        "Unit found at original location"

    assert PLAYER.board[switch_coord_check[0]][switch_coord_check[1]].name == list_of_champions[0], \
        "Unit not found at final location"

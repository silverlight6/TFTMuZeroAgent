import config
import numpy as np
import time
from Models import MCTS_Util as utils
from Simulator.player_manager import PlayerManager
from Simulator import pool
from Simulator.tft_simulator import TFTConfig
from Simulator.battle_generator import base_level_config, BattleGenerator
from Simulator.utils import coord_to_x_y

def generate_battle(battle_config):
    battle_generator = BattleGenerator(battle_config)
    [player, opponent, other_players] = battle_generator.generate_battle()
    # Reinit to get around a ray memory bug.
    player.reinit_numpy_arrays()
    player.opponent = opponent
    opponent.opponent = player

    for player in other_players.values():
        player.reinit_numpy_arrays()

    pool_obj = pool.pool()
    player_manager = PlayerManager(config.NUM_PLAYERS, pool_obj,
                                   TFTConfig())
    player_manager.reinit_player_set([player] + list(other_players.values()))
    return player_manager, player

def generate_policy_logits():
    return np.random.uniform(low=-2, high=2, size=(1, config.POLICY_HEAD_SIZE))


# Test default mapping to ensure that it includes all possible in game actions
def default_mapping_test():
    default_mapping, _ = utils.create_default_mapping()
    default_mapping = default_mapping[0]
    local_counter = 0

    for b in range(37):
        for a in range(38):
            if a == 37:
                assert default_mapping[local_counter][0] == "4"
                assert default_mapping[local_counter][1:] == f"_{b}"
            else:
                assert default_mapping[local_counter][0] == "5"
                assert default_mapping[local_counter][1:] == f"_{a}_{b}"
            local_counter += 1
    for b in range(10):
        for a in range(38):
            assert default_mapping[local_counter][0] == "6"
            assert default_mapping[local_counter][1:] == f"_{a}_{b}"
            local_counter += 1
    for b in range(5):
        for a in range(38):
            assert default_mapping[local_counter][0] == "3"
            if a == 0:
                assert default_mapping[local_counter][1:] == f"_{b}"
            else:
                assert default_mapping[local_counter][1:] == f"_{b}_{a}"
            local_counter += 1
    for a in range(38):
        if a == 0:
            assert default_mapping[local_counter] == "0"
            assert default_mapping[local_counter + 1] == "1"
            assert default_mapping[local_counter + 2] == "2"
        else:
            assert default_mapping[local_counter] == f"0_{a}"
            assert default_mapping[local_counter + 1] == f"1_{a}"
            assert default_mapping[local_counter + 2] == f"2_{a}"
        local_counter += 3

# So let me figure out a few tests that I need to run.
def mask_basic_test():
    test_config = {**base_level_config, "test_mode": True, "set_test_position": True, "test_position": [0, 0]}
    player_manager, player = generate_battle(test_config)
    action_mask = player_manager.action_handlers[f"player_{player.player_num}"].fetch_action_mask()
    assert action_mask[54][0] == 1  # Refresh
    assert action_mask[53][0] == 1  # Level
    assert action_mask[52][0] == 1  # Pass
    assert action_mask[54][1] == 0  # Refresh out of range
    assert action_mask[53][1] == 0  # Level out of range
    assert action_mask[52][1] == 0  # Pass out of range
    assert action_mask[0][0] == 0  # Move action from [0, 0] to [0, 0]
    assert action_mask[0][1] == 1  # Move action from [0, 0] to [0, 1]
    if not player.board[0][1]:
        assert action_mask[1][2] == 0  # Move action from [0, 1] to [0, 2]
    else:
        assert action_mask[1][2] == 1

    flat_mask = np.reshape(action_mask, (-1))
    assert flat_mask[1976] == 1  # Should be a useless buy option

# Test to make sure that no empty square is being called movable.
def two_empty_move_test():
    test_config = {**base_level_config}
    player_manager, player = generate_battle(test_config)
    action_mask = player_manager.action_handlers[f"player_{player.player_num}"].fetch_action_mask()

    first_two_empty_positions = []
    for i in range(28):
        x, y = coord_to_x_y(i)
        if not player.board[x][y]:
            first_two_empty_positions.append(i)
    assert action_mask[first_two_empty_positions[0]][first_two_empty_positions[1]] == 0

# Test to ensure I can't move an item to a location with nothing on it.
def move_item_mask():
    test_config = {**base_level_config, "set_test_position": True, "test_position": [0, 0]}
    player_manager, player = generate_battle(test_config)
    action_mask = player_manager.action_handlers[f"player_{player.player_num}"].fetch_action_mask()
    for i in range(37, 47):
        for j in range(28):
            x, y = coord_to_x_y(j)
            if player.board[x][y]:
                assert action_mask[i][j] == 1
            else:
                assert action_mask[i][j] == 0

def thieves_glove_mask():
    test_config = {**base_level_config, "set_test_position": True, "test_position": [0, 0], "num_items": 1,
                   "thieves_gloves": True}
    player_manager, player = generate_battle(test_config)
    action_mask = player_manager.action_handlers[f"player_{player.player_num}"].fetch_action_mask()
    for i in range(37, 47):
        assert action_mask[i][0] == 0  # Make sure you can't move an item to a unit that has thieves glove

def exp_mask_test():
    test_config = {**base_level_config, "scenario_info": False}
    player_manager, player = generate_battle(test_config)
    action_mask = player_manager.action_handlers[f"player_{player.player_num}"].fetch_action_mask()
    for i in range(38):
        assert action_mask[53][i] == 0  # Level out of range

# Test to determine sell action works
def sell_mask():
    ...

# Test to see if kayn is mapping correctly with items.
# This test will be implemented later when training gets to where we start to see Kayn
def kayn_mask():
    ...

# Test to see if Azir is mapping correctly with items.
# This test will be implemented later when training gets to where we start to see Azir
def azir_mask():
    ...

# Since this file doesn't have many test in it, I am going to use it for masking as well
def test_list():
    default_mapping_test()
    mask_basic_test()
    exp_mask_test()
    two_empty_move_test()
    move_item_mask()
    thieves_glove_mask()

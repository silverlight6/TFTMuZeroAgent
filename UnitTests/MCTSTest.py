import numpy as np

import config
import collections
from Simulator.tft_simulator import TFT_Simulator
from Simulator.champion import champion
from Models.MCTS import MCTS


"""
List of tests currently failing on the player class.
Move bench to board, move board to bench with full units
No gold to buy units
"""

NetworkOutput = collections.namedtuple(
    'NetworkOutput',
    'value reward policy_logits hidden_state')


class Dummy_Model:
    def __init__(self):
        pass

    def initial_inference(self, observation):
        return {
            "value": np.random.rand(1),
            "value_logits": np.random.rand(config.ENCODER_NUM_STEPS),
            "reward": np.random.rand(1),
            "reward_logits": np.random.rand(config.ENCODER_NUM_STEPS),
            "policy_logits": np.random.rand(config.ACTION_ENCODING_SIZE),
            "hidden_state": np.random.rand(config.HIDDEN_STATE_SIZE)
        }

    def recurrent_inferrence(self, action):
        return {
            "value": np.random.rand(1),
            "value_logits": np.random.rand(config.ENCODER_NUM_STEPS),
            "reward": np.random.rand(1),
            "reward_logits": np.random.rand(config.ENCODER_NUM_STEPS),
            "policy_logits": np.random.rand(config.ACTION_ENCODING_SIZE),
            "hidden_state": np.random.rand(config.HIDDEN_STATE_SIZE)
        }


def setup():
    """Creates fresh player and pool"""
    env = TFT_Simulator(None)
    mcts = MCTS(Dummy_Model())
    return env, mcts


# test to make sure invalid actions are masked out during encoding
def encodeTest():
    # Set up the test with base settings
    tft_env, mcts_obj = setup()
    player1 = tft_env.PLAYERS["player_1"]
    observation_obj = tft_env.game_observations["player_1"]

    player1.reset_state()

    # Take a few actions to set up the needed environment
    player1.level = 2
    player1.max_units = 2
    player1.gold = 0

    # Test with 2 units on the field
    player1.add_to_bench(champion('zed'))
    player1.add_to_bench(champion('zilean'))

    player1.end_turn_actions()
    obs = observation_obj.observation("player_1", player1)
    _, _, mapping_str = mcts_obj.encode_action_to_str(np.random.rand(1, config.ACTION_ENCODING_SIZE), [obs["mask"]])

    # using test player, check if mapping contains illegal actions in each category
    # make invalid actions list to compare against
    invalid_actions = ["4", "5"]
    # invalid shop purchases
    for i in range(5):
        invalid_actions.append(f"1_{i}")
    # invalid board movements
    for i in range(2, 37):
        for j in range(i, 38):
            invalid_actions.append(f"2_{i}_{j}")
        # add invalid item movements while we're here
        for j in range(10):
            invalid_actions.append(f"3_{i}_{j}")
    # invalid item movements; player has no items
    for i in range(2):
        for j in range(10):
            invalid_actions.append(f"3_{i}_{j}")

    for action in mapping_str[0]:
        assert action not in invalid_actions, f"Agent is trying to make an illegal action! action: {action}"

    # Test with a unit on the bench
    player1.gold = 2
    player1.buy_champion(champion('sylas'))

    obs = observation_obj.observation("player_1", player1)
    _, _, mapping_str = mcts_obj.encode_action_to_str(np.random.rand(1, config.ACTION_ENCODING_SIZE), [obs["mask"]])

    assert "2_0_28" in mapping_str[0]

    # Test a swap command
    player1.move_bench_to_board(0, 0, 0)

    obs = observation_obj.observation("player_1", player1)
    _, _, mapping_str = mcts_obj.encode_action_to_str(np.random.rand(1, config.ACTION_ENCODING_SIZE), [obs["mask"]])

    assert "2_0_28" in mapping_str[0]

    # Fill up the bench
    player1.add_to_bench(champion('nami'))
    player1.add_to_bench(champion('nidalee'))
    player1.add_to_bench(champion('nunu'))
    player1.add_to_bench(champion('pyke'))
    player1.add_to_bench(champion('riven'))
    player1.add_to_bench(champion('sejuani'))
    player1.add_to_bench(champion('sett'))
    player1.add_to_bench(champion('nami'))

    obs = observation_obj.observation("player_1", player1)
    _, _, mapping_str = mcts_obj.encode_action_to_str(np.random.rand(1, config.ACTION_ENCODING_SIZE), [obs["mask"]])

    # All bench plus sell
    for i in range(28, 38):
        assert f"2_0_{i}" in mapping_str[0], f"Agent is not including action: " + f"2_0_{i}"
        assert f"2_1_{i}" in mapping_str[0], f"Agent is not including action: " + f"2_1_{i}"

    # Test single glove case
    player1.add_to_item_bench('sparring_gloves')
    player1.add_to_item_bench('sunfire_cape')
    player1.add_to_item_bench('sparring_gloves')
    player1.move_item(1, 0, 0)
    player1.move_item(2, 0, 0)

    obs = observation_obj.observation("player_1", player1)
    _, _, mapping_str = mcts_obj.encode_action_to_str(np.random.rand(1, config.ACTION_ENCODING_SIZE), [obs["mask"]])

    assert '3_0_0' not in mapping_str[0]

    # Test tripling
    player1.add_to_bench(champion('nami'))

    obs = observation_obj.observation("player_1", player1)
    _, _, mapping_str = mcts_obj.encode_action_to_str(np.random.rand(1, config.ACTION_ENCODING_SIZE), [obs["mask"]])

    assert "2_2_36" not in mapping_str[0]

    # Testing giving it some gold
    tft_env.step_function.generate_shop("player_1", player1)
    player1.start_round(5)

    obs = observation_obj.observation("player_1", player1)
    _, _, mapping_str = mcts_obj.encode_action_to_str(np.random.rand(1, config.ACTION_ENCODING_SIZE), [obs["mask"]])

    print(mapping_str)
    assert "1_0" in mapping_str[0]


# make sure default mapping doesn't have invalid actions
def defaultMappingTest():
    pass


def list_of_tests():
    encodeTest()
    defaultMappingTest()

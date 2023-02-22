import config
import numpy as np
from Simulator.pool import pool
from Simulator.champion import champion
from Simulator.player import player
from Simulator.step_function import Step_Function
from Simulator.observation import Observation
from Models.MCTS import MCTS
from typing import List
import collections
import config
import numpy as np
import tensorflow as tf
import time

class Dummy:
    def __init__(self, observation):
        pass
    def initial_inference(self, observation):
        pass
    def recurrent_inferrence(self, observation):
        pass

def setup() -> tuple[player, MCTS]:
    """Creates fresh player and pool"""
    base_pool = pool()
    player1 = player(base_pool, 0)
    player1.level = 2
    player1.gold = 0
    player1.add_to_bench(champion('kayn'))
    player1.add_to_bench(champion('zilean'))
    player1.generate_bench_vector()
    player1.generate_board_vector()
    player1.generate_chosen_vector()
    player1.generate_item_vector()
    player1.end_turn_actions()
    mcts = MCTS(Dummy(None))
    ob = Observation()
    sf = Step_Function(base_pool, ob)
    return player1, mcts, ob, sf

# test to make sure invalid actions are masked out during encoding
def encodeTest():
    player1, mcts, ob, sf = setup()
    _, _, mapping_str = mcts.encode_action_to_str(np.random.rand(1, 1081), \
        [[player1.decision_mask, player1.shop_mask, player1.board_mask, \
          player1.bench_mask, player1.item_mask, player1.util_mask]])
    sf.generate_shop(player1)
    ob.generate_shop_vector(sf.shops[player1.player_num], player1)
    # using test player, check if mapping contains illegal actions in each category
    # make invalid actions list to compare against
    invalid_actions = []
    invalid_actions.append("4")
    invalid_actions.append("5")
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
    
    print(mapping_str)
    print(invalid_actions)

# make sure default mapping doesn't have invalid actions
def defaultMappingTest():
    pass

def list_of_tests():
    encodeTest()
    defaultMappingTest()
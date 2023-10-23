import collections
import numpy as np
import config
from Simulator.stats import COST
from Simulator.origin_class import team_traits, game_comp_tiers

'''
Includes the vector of the shop, bench, board, and item list.
Add a vector for each player composition makeup at the start of the round.
action vector = [Decision, shop, champion_bench, item_bench, x_axis, y_axis, x_axis 2, y_axis 2]
'''


class Observation:
    def __init__(self):
        self.shop_vector = np.zeros(45)
        self.shop_mask = np.ones(5, dtype=np.int8)
        self.game_comp_vector = np.zeros(208)
        self.dummy_observation = np.zeros(config.OBSERVATION_SIZE)
        self.cur_player_observations = collections.deque(maxlen=config.OBSERVATION_TIME_STEPS *
                                                         config.OBSERVATION_TIME_STEP_INTERVAL)
        self.other_player_observations = {"player_" + str(player_id): np.zeros(740)
                                          for player_id in range(config.NUM_PLAYERS)}
        self.turn_since_update = 0.01
        self.moves_left_in_turn = 1

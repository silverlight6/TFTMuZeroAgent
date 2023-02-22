import collections
import numpy as np
import config
from Simulator.stats import COST
from Simulator.origin_class import team_traits, game_comp_tiers
import Simulator.utils as utils


# Includes the vector of the shop, bench, board, and item list.
# Add a vector for each player composition makeup at the start of the round.
# action vector = [Decision, shop, champion_bench, item_bench, x_axis, y_axis, x_axis 2, y_axis 2]
class Observation:
    def __init__(self):
        self.game_comp_vector = np.zeros(208)
        self.dummy_observation = np.zeros(config.OBSERVATION_SIZE)
        self.cur_player_observations = collections.deque(maxlen=config.OBSERVATION_TIME_STEPS *
                                                                config.OBSERVATION_TIME_STEP_INTERVAL)
        self.other_player_observations = {"player_" + str(player_id): np.zeros(280)
                                          for player_id in range(config.NUM_PLAYERS)}

    
    #TODO Save elements when not updated, to save computations and just return the elements

    def get_lobo_observation(self, curr_player, curr_shop, players): #NAME WIP
        # get_lobo_observation took 0.0009965896606445312 seconds to finish
        #Return observation of size 8246
        other_players_obs = []
        for player_id in players.keys():
            other_player = players[player_id]
            other_player_obs = list(np.zeros((1026,)))
            if other_player != curr_player:
                if other_player:
                    other_player_obs = self.get_lobo_public_observation(other_player)
                other_players_obs += other_player_obs
        return np.concatenate([
            self.get_lobo_public_observation(curr_player),
            self.get_lobo_private_observation(curr_player, curr_shop),
            other_players_obs            
         ], axis=-1)

    def get_lobo_public_observation(self, player):
        return player.get_public_observation()

    def get_lobo_private_observation(self, player, shop):
        return np.concatenate([
            player.get_private_observation(),
            self.get_shop_vector_obs(shop)            
         ], axis=-1)

    def get_shop_vector_obs(self, shop):
        # each champion has 6 bit for the name, 1 bit for the chosen.
        # 5 of them makes it 35.
        output_array = []
        for x in range(0, len(shop)):
            input_array = np.zeros(7)
            if shop[x] and shop[x] != ' ':
                name = shop[x]
                if name.endswith("_c"):
                    name = name.split('_')[0]
                    input_array[6] = 1
                c_index = list(COST.keys()).index(name)
                input_array[0:6] = utils.champ_binary_encode(c_index)
            output_array += list(input_array)
        return output_array

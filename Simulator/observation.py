import collections
import numpy as np
import config
from Simulator.stats import COST
from Simulator.origin_class import team_traits, game_comp_tiers


# Includes the vector of the shop, bench, board, and item list.
# Add a vector for each player composition makeup at the start of the round.
# action vector = [Decision, shop, champion_bench, item_bench, x_axis, y_axis, x_axis 2, y_axis 2]
class Observation:
    def __init__(self):
        self.shop_vector = np.zeros(45)
        self.game_comp_vector = np.zeros(208)
        self.dummy_observation = np.zeros(config.OBSERVATION_SIZE)
        self.cur_player_observations = collections.deque(maxlen=config.OBSERVATION_TIME_STEPS *
                                                                config.OBSERVATION_TIME_STEP_INTERVAL)
        self.other_player_observations = {"player_" + str(player_id): np.zeros(280)
                                          for player_id in range(config.NUM_PLAYERS)}

    def observation(self, player_id, player, action_vector):
        shop_vector = self.shop_vector
        game_state_vector = self.game_comp_vector
        complete_game_state_vector = np.concatenate([shop_vector,
                                                     player.board_occupation_vector,
                                                     player.bench_occupation_vector,
                                                     player.champions_owned_vector,
                                                     player.chosen_vector,
                                                     player.item_vector,
                                                     player.player_vector,
                                                     game_state_vector,
                                                     action_vector], axis=-1)

        # initially fill the queue with duplicates of first observation
        # so we can still sample when there aren't enough time steps yet
        maxLen = config.OBSERVATION_TIME_STEPS * config.OBSERVATION_TIME_STEP_INTERVAL
        if len(self.cur_player_observations) == 0:
            for _ in range(maxLen):
                self.cur_player_observations.append(complete_game_state_vector)

        # enqueue the latest observation and pop the oldest (performed automatically by deque with maxLen configured)
        self.cur_player_observations.append(complete_game_state_vector)

        # sample every N time steps at M intervals, where maxLen of queue = M*N
        cur_player_observation = np.array([self.cur_player_observations[i]
                                           for i in range(0, maxLen, config.OBSERVATION_TIME_STEP_INTERVAL)]).flatten()

        other_player_observation_list = []
        for k, v in self.other_player_observations.items():
            if k != player_id:
                other_player_observation_list.append(v)
        other_player_observation = np.array(other_player_observation_list).flatten()

        total_observation = np.concatenate((cur_player_observation, other_player_observation))
        return total_observation

    def single_step_observation(self, player_id, player, action_vector):
        shop_vector = self.shop_vector
        complete_game_state_vector = np.concatenate([shop_vector,
                                                     player.board_occupation_vector,
                                                     player.bench_occupation_vector,
                                                     player.champions_owned_vector,
                                                     player.chosen_vector,
                                                     player.item_vector,
                                                     player.player_vector], axis=-1)

        # initially fill the queue with duplicates of first observation
        # so we can still sample when there aren't enough time steps yet
        maxLen = config.OBSERVATION_TIME_STEPS * config.OBSERVATION_TIME_STEP_INTERVAL
        if len(self.cur_player_observations) == 0:
            for _ in range(maxLen):
                self.cur_player_observations.append(complete_game_state_vector)

        # enqueue the latest observation and pop the oldest (performed automatically by deque with maxLen configured)
        self.cur_player_observations.append(complete_game_state_vector)

        # sample every N time steps at M intervals, where maxLen of queue = M*N
        cur_player_observation = np.array([self.cur_player_observations[i]
                                           for i in range(0, maxLen, config.OBSERVATION_TIME_STEP_INTERVAL)]).flatten()

        other_player_observation_list = []
        for k, v in self.other_player_observations.items():
            if k != player_id:
                other_player_observation_list.append(v)
        other_player_observation = np.array(other_player_observation_list).flatten()

        total_observation = np.concatenate((cur_player_observation, other_player_observation))
        return total_observation

    def generate_other_player_vectors(self, cur_player, players):
        for player_id in players:
            other_player = players[player_id]
            if other_player and other_player != cur_player:
                other_player_vector = np.concatenate([other_player.board_occupation_vector,
                                                      other_player.champions_owned_vector,
                                                      other_player.bench_occupation_vector,
                                                      other_player.chosen_vector,
                                                      other_player.item_vector,
                                                      other_player.player_vector], axis=-1)
                self.other_player_observations[player_id] = other_player_vector

    def generate_game_comps_vector(self):
        output = np.zeros(208)
        for i in range(len(game_comp_tiers)):
            tiers = np.array(list(game_comp_tiers[i].values()))
            tierMax = np.max(tiers)
            if tierMax != 0:
                tiers = tiers / tierMax
            output[i * 26: i * 26 + 26] = tiers
        self.game_comp_vector = output

    def generate_shop_vector(self, shop):
        # each champion has 6 bit for the name, 1 bit for the chosen.
        # 5 of them makes it 35.
        output_array = np.zeros(45)
        shop_chosen = False
        chosen_shop_index = -1
        chosen_shop = ''
        for x in range(0, len(shop)):
            input_array = np.zeros(8)
            if shop[x]:
                chosen = 0
                if shop[x].endswith("_c"):
                    chosen_shop_index = x
                    chosen_shop = shop[x]
                    c_shop = shop[x].split('_')
                    shop[x] = c_shop[0]
                    chosen = 1
                    shop_chosen = c_shop[1]
                i_index = list(COST.keys()).index(shop[x])
                # This should update the item name section of the vector
                for z in range(6, 0, -1):
                    if i_index > 2 ** (z - 1):
                        input_array[6 - z] = 1
                        i_index -= 2 ** (z - 1)
                input_array[7] = chosen
            # Input chosen mechanics once I go back and update the chosen mechanics.
            output_array[8 * x: 8 * (x + 1)] = input_array
        if shop_chosen:
            if shop_chosen == 'the':
                shop_chosen = 'the_boss'
            i_index = list(team_traits.keys()).index(shop_chosen)
            # This should update the item name section of the vector
            for z in range(5, 0, -1):
                if i_index > 2 * z:
                    output_array[45 - z] = 1
                    i_index -= 2 * z
            shop[chosen_shop_index] = chosen_shop
        self.shop_vector = output_array

    
    #TODO Save elements when not updated, to save computations and just return the elements

    def get_lobo_observation(self, curr_player, curr_shop, players): #NAME WIP
        #Return observation of size 8238
        other_players_obs = []
        for player_id in players.keys():
            other_player = players[player_id]
            other_player_obs = np.zeros((1025,))
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
        return player.public_observation()

    def get_lobo_private_observation(self, player, shop):
        return np.concatenate([
            player.private_observation(),
            self.get_shop_vector_obs(shop)            
         ], axis=-1)

    def get_shop_vector_obs(self, shop):
        # each champion has 6 bit for the name, 1 bit for the chosen.
        # 5 of them makes it 35.
        output_array = []
        for x in range(0, len(shop)):
            input_array = np.zeros(7)
            if shop[x]:
                name = shop[x]
                if name.endswith("_c"):
                    name = name.split('_')[0]
                    input_array[6] = 1
                c_index = list(COST.keys()).index(name)
                input_array[0:6] = self.champ_binary_encode(c_index)
            output_array += list(input_array)
        return output_array

    def champ_binary_encode(self, n):
        return list(np.unpackbits(np.array([n],np.uint8))[2:8])

    def item_binary_encode(self, n):
        return list(np.unpackbits(np.array([n],np.uint8))[2:8])
    
    def champ_one_hot_encode(self, n):
        return self.CHAMPION_ONE_HOT_ENCODING[n]
    
    def item_one_hot_encode(self, n):
        return self.BASIC_ITEMS_ONE_HOT_ENCODING[n]

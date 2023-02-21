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
        self.shop_mask = np.ones(5, dtype=np.int8)
        self.game_comp_vector = np.zeros(208)
        self.dummy_observation = np.zeros(config.OBSERVATION_SIZE)
        self.cur_player_observations = collections.deque(maxlen=config.OBSERVATION_TIME_STEPS *
                                                                config.OBSERVATION_TIME_STEP_INTERVAL)
        self.other_player_observations = {"player_" + str(player_id): np.zeros(306)
                                          for player_id in range(config.NUM_PLAYERS)}

    def observation(self, player_id, player, action_vector=np.array([])):
        # Fetch the shop vector and game comp vector
        shop_vector = self.shop_vector
        game_state_vector = self.game_comp_vector
        # Concatenate all vector based player information
        game_state_tensor = np.concatenate([shop_vector,
                                            player.bench_vector,
                                            player.chosen_vector,
                                            player.item_vector,
                                            player.player_public_vector,
                                            player.player_private_vector,
                                            player.board_vector,
                                            game_state_vector,
                                            action_vector], axis=-1)

        # Initially fill the queue with duplicates of first observation
        # so we can still sample when there aren't enough time steps yet
        maxLen = config.OBSERVATION_TIME_STEPS * config.OBSERVATION_TIME_STEP_INTERVAL
        if len(self.cur_player_observations) == 0:
            for _ in range(maxLen):
                self.cur_player_observations.append(game_state_tensor)

        # Enqueue the latest observation and pop the oldest (performed automatically by deque with maxLen configured)
        self.cur_player_observations.append(game_state_tensor)

        # # sample every N time steps at M intervals, where maxLen of queue = M*N
        # cur_player_observation = np.array([self.cur_player_observations[i]
        #                               for i in range(0, maxLen, config.OBSERVATION_TIME_STEP_INTERVAL)]).flatten()

        cur_player_tensor_observation = []
        for i in range(0, maxLen, config.OBSERVATION_TIME_STEP_INTERVAL):
            tensor = self.cur_player_observations[i]
            cur_player_tensor_observation.append(tensor)
        cur_player_tensor_observation = np.asarray(cur_player_tensor_observation).flatten()

        # Fetch other player data
        other_player_tensor_observation_list = []
        for k, v in self.other_player_observations.items():
            if k != player_id:
                other_player_tensor_observation_list.append(v)
        other_player_tensor_observation = np.array(other_player_tensor_observation_list).flatten()

        # Gather all vectors into one place
        total_tensor_observation = np.concatenate((cur_player_tensor_observation, other_player_tensor_observation))

        # Fetch and concatenate mask
        mask = (player.decision_mask, player.shop_mask, player.board_mask, player.bench_mask, player.item_mask)
        return {"tensor": total_tensor_observation, "mask": mask}

    def generate_other_player_vectors(self, cur_player, players):
        for player_id in players:
            other_player = players[player_id]
            if other_player and other_player != cur_player:
                other_player_vector = np.concatenate([other_player.bench_vector,
                                                      other_player.chosen_vector,
                                                      other_player.item_vector,
                                                      other_player.player_public_vector], axis=-1)
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

    def generate_shop_vector(self, shop, player):
        # each champion has 6 bit for the name, 1 bit for the chosen.
        # 5 of them makes it 35.
        output_array = np.zeros(45)
        shop_chosen = False
        chosen_shop_index = -1
        chosen_shop = ''
        shop_costs = np.zeros(5)
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
                    if COST[shop[x]] == 1:
                        shop_costs[x] = 3
                    else:
                        shop_costs[x] = 3 * COST[shop[x]] - 1
                else:
                    shop_costs[x] = COST[shop[x]]
                i_index = list(COST.keys()).index(shop[x])
                if i_index == 0:
                    self.shop_mask[x] = 0
                # This should update the item name section of the vector
                for z in range(6, 0, -1):
                    if i_index > 2 ** (z - 1):
                        input_array[6 - z] = 1
                        i_index -= 2 ** (z - 1)
                input_array[7] = chosen
                self.shop_mask[x] = 1

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
        player.shop_costs = shop_costs
        player.shop_mask = self.shop_mask

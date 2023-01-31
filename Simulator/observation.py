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
        self.other_player_observations = {"player_" + str(player_id): [np.zeros(280), np.zeros((49, 16))]
                                          for player_id in range(config.NUM_PLAYERS)}

    def observation(self, player_id, player, action_vector=np.array([])):
        shop_vector = self.shop_vector
        game_state_vector = self.game_comp_vector
        game_state_tensor = np.concatenate([shop_vector,
                                            player.bench_vector,
                                            player.chosen_vector,
                                            player.item_vector,
                                            player.player_public_vector,
                                            player.player_private_vector,
                                            game_state_vector,
                                            action_vector], axis=-1)

        game_state_image = player.board_image

        # initially fill the queue with duplicates of first observation
        # so we can still sample when there aren't enough time steps yet
        maxLen = config.OBSERVATION_TIME_STEPS * config.OBSERVATION_TIME_STEP_INTERVAL
        if len(self.cur_player_observations) == 0:
            for _ in range(maxLen):
                self.cur_player_observations.append([game_state_tensor, game_state_image])

        # enqueue the latest observation and pop the oldest (performed automatically by deque with maxLen configured)
        self.cur_player_observations.append([game_state_tensor, game_state_image])

        # # sample every N time steps at M intervals, where maxLen of queue = M*N
        # cur_player_observation = np.array([self.cur_player_observations[i]
        #                               for i in range(0, maxLen, config.OBSERVATION_TIME_STEP_INTERVAL)]).flatten()

        cur_player_tensor_observation = []
        cur_player_image_observation = []
        for i in range(0, maxLen, config.OBSERVATION_TIME_STEP_INTERVAL):
            tensor, image = self.cur_player_observations[i]
            cur_player_tensor_observation.append(tensor)
            cur_player_image_observation.append(image)
        cur_player_tensor_observation = np.asarray(cur_player_tensor_observation).flatten()
        cur_player_image_observation = np.asarray(cur_player_image_observation)

        other_player_tensor_observation_list = []
        other_player_image_observation_list = []
        for k, v in self.other_player_observations.items():
            if k != player_id:
                other_player_tensor_observation_list.append(v[0])
                other_player_image_observation_list.append(v[1])
        other_player_tensor_observation = np.array(other_player_tensor_observation_list).flatten()
        other_player_image_observation = np.array(other_player_image_observation_list)

        total_tensor_observation = np.concatenate((cur_player_tensor_observation, other_player_tensor_observation))
        total_image_observation = np.concatenate((cur_player_image_observation, other_player_image_observation))
        total_image_observation = np.transpose(total_image_observation, (1, 2, 0))
        return [total_tensor_observation, total_image_observation]

    def generate_other_player_vectors(self, cur_player, players):
        for player_id in players:
            other_player = players[player_id]
            if other_player and other_player != cur_player:
                other_player_vector = np.concatenate([other_player.bench_vector,
                                                      other_player.chosen_vector,
                                                      other_player.item_vector,
                                                      other_player.player_public_vector], axis=-1)
                other_player_game_state_image = other_player.board_image
                print(other_player_vector.shape)
                self.other_player_observations[player_id] = [other_player_vector, other_player_game_state_image]

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

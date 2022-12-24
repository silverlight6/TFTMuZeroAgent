import config
import numpy as np
from Simulator import pool
from Simulator.player import player as player_class
from Simulator.step_function import Step_Function
from Simulator.game_round import Game_Round
from Simulator.observation import Observation


class TFT_Simulator:
    def __init__(self):
        self.pool_obj = pool.pool()
        self.PLAYERS = [player_class(self.pool_obj, i) for i in range(config.NUM_PLAYERS)]
        self.game_observations = [Observation() for _ in range(config.NUM_PLAYERS)]
        self.NUM_DEAD = 0
        self.num_players = config.NUM_PLAYERS
        self.player_rewards = [0 for _ in range(config.NUM_PLAYERS)]
        self.last_observation = [[] for _ in range(config.NUM_PLAYERS)]
        self.last_action = [[] for _ in range(config.NUM_PLAYERS)]
        self.last_value = [[] for _ in range(config.NUM_PLAYERS)]
        self.last_policy = [[] for _ in range(config.NUM_PLAYERS)]
        self.previous_reward = [0 for _ in range(config.NUM_PLAYERS)]

        self.step_function = Step_Function(self.pool_obj, self.game_observations)
        self.game_round = Game_Round(self.PLAYERS, self.pool_obj, self.step_function)

    def calculate_reward(self, player, previous_reward):
        print("This never gets called")
        self.player_rewards[player.player_num] = player.reward - previous_reward
        average = 0
        for i in range(config.NUM_PLAYERS):
            if i != player.player_num and self.PLAYERS[i]:
                average += self.player_rewards[i]
        if self.NUM_DEAD < config.NUM_PLAYERS - 1:
            average = average / (config.NUM_PLAYERS - self.NUM_DEAD - 1)
        return player.reward - previous_reward - average

    def check_dead(self):
        num_alive = 0
        for i, player in enumerate(self.PLAYERS):
            if player:
                if player.health <= 0:
                    self.NUM_DEAD += 1
                    self.pool_obj.return_hero(player)

                    self.PLAYERS[i] = None
                    self.game_round.update_players(self.PLAYERS)
                else:
                    num_alive += 1
        return num_alive

    def get_observations_objects(self):
        return [self.game_observations for i in range(config.NUM_PLAYERS)]

    def get_observation(self, buffers):
        observation_list = []
        previous_action = []
        for i in range(config.NUM_PLAYERS):
            if self.PLAYERS[i]:
                # TODO
                # store game state vector later
                observation, game_state_vector = self.game_observations[self.PLAYERS[i].player_num] \
                    .observation(self.PLAYERS[i], buffers[self.PLAYERS[i].player_num], self.PLAYERS[i].action_vector)
                observation_list.append(observation)
                buffers[self.PLAYERS[i].player_num].store_observation(game_state_vector)
                previous_action.append(buffers[self.PLAYERS[i].player_num].get_prev_action())
            else:
                dummy_observation = Observation()
                observation = dummy_observation.dummy_observation(buffers[i])
                observation_list.append(observation)
                previous_action.append(9)
        observation_list = np.squeeze(np.array(observation_list))
        previous_action = np.array(previous_action)
        return observation_list, previous_action

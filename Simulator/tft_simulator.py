import config
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from Simulator import pool
from Simulator.player import player as player_class
from Simulator.step_function import Step_Function
from Simulator.game_round import Game_Round
from Simulator.observation import Observation


class TFT_Simulator(gym.Env):
    metadata = {}

    def __init__(self):
        self.pool_obj = pool.pool()
        self.PLAYERS = [player_class(self.pool_obj, i) for i in range(config.NUM_PLAYERS)]
        self.observation_space = spaces.MultiDiscrete([config.OBSERVATION_SIZE for _ in range(config.NUM_PLAYERS)])
        self.action_space = spaces.MultiDiscrete([config.ACTION_DIM for _ in range(config.NUM_PLAYERS)])

        self.game_observations = [Observation() for _ in range(config.NUM_PLAYERS)]
        self.render_mode = None

        self.NUM_DEAD = 0
        self.num_players = config.NUM_PLAYERS
        self.player_rewards = [0 for _ in range(config.NUM_PLAYERS)]

        self.step_function = Step_Function(self.pool_obj, self.game_observations)
        self.game_round = Game_Round(self.PLAYERS, self.pool_obj, self.step_function)
        self.actions_taken = 0
        self.game_round.play_game_round()
        self.game_round.play_game_round()
        self.episode_done = False

    def check_dead(self):
        num_alive = 0
        for i, player in enumerate(self.PLAYERS):
            if player:
                if player.health <= 0:
                    self.NUM_DEAD += 1
                    self.game_round.NUM_DEAD = self.NUM_DEAD
                    self.pool_obj.return_hero(player)

                    self.PLAYERS[i] = None
                    self.game_round.update_players(self.PLAYERS)
                else:
                    num_alive += 1
        return num_alive

    def get_observations_objects(self):
        return [self.game_observations for i in range(config.NUM_PLAYERS)]

    def get_observation(self):
        observation_list = []
        for i in range(config.NUM_PLAYERS):
            if self.PLAYERS[i]:
                # TODO
                # store game state vector later
                observation, _ = self.game_observations[self.PLAYERS[i].player_num] \
                    .observation(self.PLAYERS[i], self.PLAYERS[i].action_vector)
                observation_list.append(observation)
            else:
                dummy_observation = Observation()
                observation = dummy_observation.dummy_observation()
                observation_list.append(observation)
        observation_list = np.squeeze(np.array(observation_list))
        return observation_list

    def reset(self, seed=None, options=None):
        self.pool_obj = pool.pool()
        self.PLAYERS = [player_class(self.pool_obj, i) for i in range(config.NUM_PLAYERS)]
        self.game_observations = [Observation() for _ in range(config.NUM_PLAYERS)]
        self.NUM_DEAD = 0
        self.player_rewards = [0 for _ in range(config.NUM_PLAYERS)]

        self.step_function = Step_Function(self.pool_obj, self.game_observations)
        self.game_round = Game_Round(self.PLAYERS, self.pool_obj, self.step_function)
        self.actions_taken = 0
        self.game_round.play_game_round()
        self.game_round.play_game_round()
        self.episode_done = False
        observation = self.get_observation()
        return observation, {"players": self.PLAYERS}

    def render(self):
        ...

    def step(self, action):
        rewards = self.step_function.batch_controller(action, self.PLAYERS, self.game_observations)

        self.actions_taken += 1
        # If at the end of the turn
        if self.actions_taken == config.ACTIONS_PER_TURN:
            # Take a game action and reset actions taken
            self.actions_taken = 0
            self.game_round.play_game_round()

            # Check if the game is over
            if self.check_dead() == 1 or self.game_round.current_round > 48:
                self.episode_done = True
                # Anyone left alive (should only be 1 player unless time limit) wins the game
                for player in self.PLAYERS:
                    if player:
                        player.won_game()
        observations = self.get_observation()

        return observations, rewards, self.episode_done, False, {"players": self.PLAYERS}

import config
import functools
import gym
import numpy as np
from gym import spaces
from Simulator import pool
from Simulator.player import player as player_class
from Simulator.step_function import Step_Function
from Simulator.game_round import Game_Round
from Simulator.observation import Observation
from pettingzoo.utils.env import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers


def env(render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    local_env = raw_env()

    # this wrapper helps error handling for discrete action spaces
    # local_env = wrappers.AssertOutOfBoundsWrapper(local_env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    local_env = wrappers.OrderEnforcingWrapper(local_env)
    return local_env


def raw_env():
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    local_env = TFT_Simulator(env_config=None)
    local_env = parallel_to_aec(local_env)
    return local_env


class TFT_Simulator(ParallelEnv):
    metadata = {}

    def __init__(self, env_config):
        self.pool_obj = pool.pool()
        self.PLAYERS = [player_class(self.pool_obj, i) for i in range(config.NUM_PLAYERS)]

        self.game_observations = [Observation() for _ in range(config.NUM_PLAYERS)]
        self.render_mode = None

        self.NUM_DEAD = 0
        self.num_players = config.NUM_PLAYERS
        self.player_rewards = [0 for _ in range(config.NUM_PLAYERS)]

        self.step_function = Step_Function(self.pool_obj, self.game_observations)
        self.game_round = Game_Round(self.PLAYERS, self.pool_obj, self.step_function)
        self.actions_taken = 0
        self.actions_taken_this_turn = 0
        self.game_round.play_game_round()
        self.game_round.play_game_round()
        self.episode_done = False

        self.possible_agents = ["player_" + str(r) for r in range(config.NUM_PLAYERS)]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents)))))
        self.agents = self.possible_agents[:]

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> gym.Space:
        return spaces.Discrete(config.OBSERVATION_SIZE)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> gym.Space:
        return spaces.Discrete(config.ACTION_DIM)

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
        return [self.game_observations for _ in range(config.NUM_PLAYERS)]

    def get_observation(self, player):
        if player:
            # TODO
            # store game state vector later
            observation, _ = self.game_observations[player.player_num].observation(player, player.action_vector)
        else:
            dummy_observation = Observation()
            observation = dummy_observation.dummy_observation
        return observation

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
        observation = []
        for player in self.PLAYERS:
            observation.append(self.get_observation(player))
        return np.asarray(observation), {"players": self.PLAYERS}

    def render(self):
        ...

    def step(self, action):
        reward = 0
        if action.ndim == 0:
            reward = self.step_function.action_controller(action, self.PLAYERS, self.game_observations,
                                                          self.actions_taken_this_turn)
        elif action.ndim == 1:
            reward = self.step_function.batch_2d_controller(action, self.PLAYERS, self.game_observations,
                                                            self.actions_taken_this_turn)

        observation = self.get_observation(self.PLAYERS[self.actions_taken_this_turn])

        self.actions_taken_this_turn += 1
        if self.actions_taken_this_turn == 8:
            self.actions_taken_this_turn = 0
            self.actions_taken += 1

        # If at the end of the turn
        if self.actions_taken == config.ACTIONS_PER_TURN:
            # Take a game action and reset actions taken
            self.actions_taken = 0
            self.game_round.play_game_round()
            # reset for the next turn
            for p in self.PLAYERS:
                if p:
                    p.turn_taken = False

            # Check if the game is over
            if self.check_dead() == 1 or self.game_round.current_round > 48:
                self.episode_done = True
                # Anyone left alive (should only be 1 player unless time limit) wins the game
                for player in self.PLAYERS:
                    if player:
                        player.won_game()

        terminated = False
        if self.PLAYERS[self.actions_taken_this_turn] is None:
            terminated = True

        return observation, reward, self.episode_done or terminated, {"players": self.PLAYERS}

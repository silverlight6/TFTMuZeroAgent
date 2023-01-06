import config
import functools
import gym
import numpy as np
from typing import Dict
from gym.spaces import Discrete
from Simulator import pool
from Simulator.player import player as player_class
from Simulator.step_function import Step_Function
from Simulator.game_round import Game_Round
from Simulator.observation import Observation
from pettingzoo.utils.env import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers, agent_selector


def env():
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
    # local_env = wrappers.OrderEnforcingWrapper(local_env)
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
        self.PLAYERS = {"player_" + str(player_id): player_class(self.pool_obj, player_id)
                        for player_id in range(config.NUM_PLAYERS)}
        self.game_observations = {"player_" + str(player_id): Observation() for player_id in range(config.NUM_PLAYERS)}
        self.render_mode = None

        self.NUM_DEAD = 0
        self.num_players = config.NUM_PLAYERS
        self.player_rewards = {"player_" + str(player_id): 0 for player_id in range(config.NUM_PLAYERS)}

        self.step_function = Step_Function(self.pool_obj, self.game_observations)
        self.game_round = Game_Round(self.PLAYERS, self.pool_obj, self.step_function)
        self.actions_taken = 0
        self.actions_taken_this_turn = 0
        self.game_round.play_game_round()
        self.game_round.play_game_round()
        self.episode_done = False

        self.possible_agents = ["player_" + str(r) for r in range(config.NUM_PLAYERS)]
        self.agents = self.possible_agents[:]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents)))))
        self._agent_selector = agent_selector(self.possible_agents)
        self.agent_selection = self.possible_agents[0]

        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: {} for agent in self.agents}
        self.observations = {agent: {} for agent in self.agents}
        self.actions = {agent: {} for agent in self.agents}

        self.observation_spaces: Dict = dict(
            zip(self.agents,
                [Discrete(config.OBSERVATION_SIZE) for _ in self.possible_agents])
        )

        self.action_spaces = {agent: Discrete(config.ACTION_DIM) for agent in self.agents}

        super().__init__()
        print("At the end of init")

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> Discrete:
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> gym.Space:
        return self.action_spaces[agent]

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

    def observe(self, agent):
        print("Why hello there")
        return dict(self.observations[agent])

    def reset(self, seed=None, options=None):
        self.pool_obj = pool.pool()
        self.PLAYERS = {"player_" + str(player_id): player_class(self.pool_obj, player_id)
                        for player_id in range(config.NUM_PLAYERS)}
        self.game_observations = {"player_" + str(player_id): Observation() for player_id in range(config.NUM_PLAYERS)}
        self.NUM_DEAD = 0
        self.player_rewards = {"player_" + str(player_id): 0 for player_id in range(config.NUM_PLAYERS)}

        self.step_function = Step_Function(self.pool_obj, self.game_observations)
        self.game_round = Game_Round(self.PLAYERS, self.pool_obj, self.step_function)
        self.actions_taken = 0
        self.game_round.play_game_round()
        self.game_round.play_game_round()
        self.episode_done = False

        self.agents = self.possible_agents.copy()
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        for player_id in self.PLAYERS.keys():
            player = self.PLAYERS[player_id]
            self.observations[player_id] = self.game_observations[
                player_id].observation(player, player.action_vector)
            self.rewards[player_id] = 0
            self._cumulative_rewards[player_id] = 0
            self.dones[player_id] = False
            self.infos[player_id] = {}
            self.actions[player_id] = {}
        # print(self.observations)
        print("After reset")
        return self.observations

    def render(self):
        ...

    def close(self):
        self.reset()

    def step(self, action):
        action_list = np.asarray(list(action.values()))
        print(action_list)
        if action_list.ndim == 1:
            self.step_function.action_controller(action, self.PLAYERS, self.game_observations)
        elif action_list.ndim == 2:
            self.step_function.batch_2d_controller(action, self.PLAYERS, self.game_observations)

        self.actions_taken_this_turn += 1
        if self.actions_taken_this_turn == 8:
            self.actions_taken_this_turn = 0
            self.actions_taken += 1

        for player_id in self.observations.keys():
            self.observations[player_id] = self.game_observations[
                player_id].observation(self.PLAYERS[player_id], self.PLAYERS[player_id].action_vector)

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

        # terminated = False
        # if self.PLAYERS[self.actions_taken_this_turn] is None:
        #     terminated = True

        return self.observations, self.rewards, self.dones, {agent: False for agent in self.agents}, self.infos

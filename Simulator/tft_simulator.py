import config
import functools
import gym
import numpy as np
from typing import Dict
from gym.spaces import MultiDiscrete, Discrete, Box
from Simulator import pool
from Simulator.player import player as player_class
from Simulator.step_function import Step_Function
from Simulator.game_round import Game_Round
from Simulator.observation import Observation
from pettingzoo.utils.env import AECEnv
from pettingzoo.utils import wrappers, agent_selector
from pettingzoo.utils.conversions import parallel_wrapper_fn


def env():
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    local_env = TFT_Simulator(env_config=None)

    # this wrapper helps error handling for discrete action spaces
    # local_env = wrappers.AssertOutOfBoundsWrapper(local_env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    local_env = wrappers.OrderEnforcingWrapper(local_env)
    return local_env


parallel_env = parallel_wrapper_fn(env)


class TFT_Simulator(AECEnv):
    metadata = {"is_parallelizable": True, "name": "tft-set4-v0"}

    def __init__(self, env_config):
        self.pool_obj = pool.pool()
        self.PLAYERS = {"player_" + str(player_id): player_class(self.pool_obj, player_id)
                        for player_id in range(config.NUM_PLAYERS)}
        self.game_observations = {"player_" + str(player_id): Observation() for player_id in range(config.NUM_PLAYERS)}
        self.render_mode = None

        self.NUM_DEAD = 0
        self.num_players = config.NUM_PLAYERS
        self.previous_rewards = {"player_" + str(player_id): 0 for player_id in range(config.NUM_PLAYERS)}

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
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: {} for agent in self.agents}
        self.observations = {agent: {} for agent in self.agents}
        self.actions = {agent: {} for agent in self.agents}

        self.observation_spaces: Dict = dict(
            zip(self.agents,
                [Box(low=(-5.0), high=5.0, shape=(config.OBSERVATION_SIZE,),
                     dtype=np.float32) for _ in self.possible_agents])
        )

        self.action_spaces = {agent: Discrete(config.ACTION_DIM) for agent in self.agents}

        super().__init__()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> MultiDiscrete:
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> gym.Space:
        return self.action_spaces[agent]

    def check_dead(self):
        num_alive = 0
        for key, player in self.PLAYERS.items():
            if player:
                if player.health <= 0:
                    self.NUM_DEAD += 1
                    self.game_round.NUM_DEAD = self.NUM_DEAD
                    self.pool_obj.return_hero(player)
                    self.PLAYERS[key] = None
                    self.game_round.update_players(self.PLAYERS)
                else:
                    num_alive += 1
        return num_alive

    def observe(self, player_id):
        return self.observations[player_id]

    def reset(self, seed=None, options=None):

        self.pool_obj = pool.pool()
        self.PLAYERS = {"player_" + str(player_id): player_class(self.pool_obj, player_id)
                        for player_id in range(config.NUM_PLAYERS)}
        self.game_observations = {"player_" + str(player_id): Observation() for player_id in range(config.NUM_PLAYERS)}
        self.NUM_DEAD = 0
        self.previous_rewards = {"player_" + str(player_id): 0 for player_id in range(config.NUM_PLAYERS)}

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
            self.terminations[player_id] = False
            self.infos[player_id] = {}
            self.actions[player_id] = {}

        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()

        super().__init__()
        # return self.observations

    def render(self):
        ...

    def close(self):
        self.reset()

    def step(self, action):
        if self.terminations[self.agent_selection]:
            self.agent_selection = self._agent_selector.next()
            # self._was_dead_step(action)
            return
        action = np.asarray(action)
        if action.ndim == 1:
            self.step_function.action_controller(action, self.PLAYERS, self.game_observations)
        elif action.ndim == 2:
            self.step_function.batch_2d_controller(action, self.PLAYERS, self.game_observations)

        if self._agent_selector.is_last():
            self.actions_taken += 1
        else:
            self._clear_rewards()

        for player_id in self.observations.keys():
            if self.PLAYERS[player_id] is None:
                if not self.terminations[player_id]:
                    self.terminations[player_id] = True
                    print(self.agents)
            else:
                self.observations[player_id] = self.game_observations[
                    player_id].observation(self.PLAYERS[player_id], self.PLAYERS[player_id].action_vector)
                self.rewards[player_id] = self.PLAYERS[player_id].reward - self.previous_rewards[player_id]
                self.previous_rewards[player_id] = self.PLAYERS[player_id].reward
                self._cumulative_rewards[player_id] = self._cumulative_rewards[player_id] + self.rewards[player_id]

        # If at the end of the turn
        if self.actions_taken == config.ACTIONS_PER_TURN:
            # Take a game action and reset actions taken
            self.actions_taken = 0
            self.game_round.play_game_round()

            # Check if the game is over
            if self.check_dead() == 1 or self.game_round.current_round > 48:
                self.episode_done = True
                # Anyone left alive (should only be 1 player unless time limit) wins the game
                for player_id in self.PLAYERS.keys():
                    if self.PLAYERS[player_id]:
                        self.PLAYERS[player_id].won_game()
                        self.terminations[player_id] = True

        self._accumulate_rewards()
        self.agent_selection = self._agent_selector.next()

        # return self.observations, self.rewards, self.terminations, self.infos

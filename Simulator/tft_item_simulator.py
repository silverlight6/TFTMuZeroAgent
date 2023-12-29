import config
import time
import functools
import numpy as np
from Simulator import pool
from Simulator.observation.vector.observation import ObservationVector
from Simulator.step_function import Step_Function
from Simulator.game_round import Game_Round
from Simulator.player import Player as player_class
from gymnasium.spaces import MultiDiscrete, Dict, Box
from pettingzoo.utils.env import ParallelEnv
from pettingzoo.utils import agent_selector


class TFT_Item_Simulator(ParallelEnv):
    """
    Environment for training a model that takes in two players a token of item movements and which items should be
    moved.
    Moves the items for the provided player then plays a single battle. Returns reward and ends episode.
    All episodes are 1 step.
    Trains with both no other player information available as well as having other player information available.
    """
    metadata = {"render_mode": [], "name": "TFT_Item_Simulator_s4_v0"}

    def __init__(self, data_generator):
        self.pool_obj = pool.pool()
        self.data_generator = data_generator
        self.PLAYER = player_class(self.pool_obj, 0)
        self._agent_ids = ["player_0"]
        self.possible_agents = ["player_0"]
        self.agents = self.possible_agents[:]

        self.game_observations = ObservationVector()
        self.render_mode = None

        self.step_function = Step_Function(self.pool_obj, self.game_observations)
        self.game_round = Game_Round(self.PLAYER, self.pool_obj, self.step_function)

        self.item_guide = np.zeros((10, 3))

        self._agent_selector = agent_selector(self.possible_agents)
        self.agent_selection = self.possible_agents[0]

        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}

        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}

        self.infos = {agent: {} for agent in self.agents}

        self.action_spaces = dict(
            zip(
                self.agents,
                [
                    MultiDiscrete(np.ones(10) * 29) for _ in self.agents
                ],
            )
        )

        super().__init__()

    def reset(self, seed=None, return_info=False, options=None):
        while self.data_generator.q_size() < config.MINIMUM_POP_AMOUNT:
            time.sleep(2)

        [player, opponent, other_players, item_guide] = self.data_generator.pop()
        self.item_guide = item_guide
        self.PLAYER = player
        self.PLAYER.reinit_numpy_arrays()
        self.PLAYER.opponent = opponent
        self.game_observations = ObservationVector()

        self.agents = ["player_0"]

        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()

        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}

        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}

        self.infos = {agent: {} for agent in self.agents}

        self.observation_spaces = dict(
            zip(
                self.agents,
                [
                    Dict({
                        "shop": Box(-2, 2, (config.SHOP_INPUT_SIZE,), np.float32),
                        "board": Box(-2, 2, (config.BOARD_INPUT_SIZE,), np.float32),
                        "bench": Box(-2, 2, (config.BENCH_INPUT_SIZE,), np.float32),
                        "states": Box(-2, 2, (config.STATE_INPUT_SIZE,), np.float32),
                        "game_comp": Box(-2, 2, (config.COMP_INPUT_SIZE,), np.float32),
                        "other_players": Box(-2, 2, (config.OTHER_PLAYER_ITEM_POS_SIZE,), np.float32)
                    }) for _ in self.agents
                ],
            )
        )

        self.action_spaces = dict(
            zip(
                self.agents,
                [
                    MultiDiscrete(np.ones(10) * 29) for _ in self.agents
                ],
            )
        )

        if other_players:
            self.game_observations.generate_other_player_vectors(self.PLAYER, other_players)

        observation = self.game_observations.observation(self.PLAYER.player_num, self.PLAYER)["tensor"]

        return {agent: {"observation": observation} for agent in self.agents}, self.infos

    def render(self):
        ...

    def close(self):
        self.reset()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        """
        Method comply with pettingzoo and rllib requirements as well as Gymnasiums
        """
        return Dict({
            "shop": Box(-2, 2, (config.SHOP_INPUT_SIZE,), np.float32),
            "board": Box(-2, 2, (config.BOARD_INPUT_SIZE,), np.float32),
            "bench": Box(-2, 2, (config.BENCH_INPUT_SIZE,), np.float32),
            "states": Box(-2, 2, (config.STATE_INPUT_SIZE,), np.float32),
            "game_comp": Box(-2, 2, (config.COMP_INPUT_SIZE,), np.float32),
            "other_players": Box(-2, 2, (config.OTHER_PLAYER_ITEM_POS_SIZE,), np.float32),
        })

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
            """
            Another method to keep in line with petting zoo and rllib requirements
            """
            return self.action_spaces[agent]

    def step(self, action):
        if action is not None:
            self.step_function.batch_item_controller(action, self.PLAYER, self.item_guide)
        self.game_round.single_combat_phase([self.PLAYER, self.PLAYER.opponent])
        self.rewards[self.agent_selection] = self.PLAYER.reward
        self.terminations = {a: True for a in self.agents}
        self._cumulative_rewards[self.agent_selection] = self.rewards[self.agent_selection]
        self.agents.remove(self.agent_selection)

        return {}, self.rewards, self.terminations, self.truncations, self.infos

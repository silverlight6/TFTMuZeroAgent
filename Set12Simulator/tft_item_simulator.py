import config
import time
import numpy as np
import gymnasium as gym
from Set12Simulator import pool
from Set12Simulator.observation.vector.observation import ObservationVector
from Set12Simulator.step_function import Step_Function
from Set12Simulator.game_round import Game_Round, log_to_file
from Set12Simulator.player import Player as player_class
from Set12Simulator.player_manager import PlayerManager
from Set12Simulator.tft_simulator import TFTConfig
from Set12Simulator.battle_generator import BattleGenerator
from gymnasium.spaces import MultiDiscrete, Dict, Box, Tuple
from gymnasium.envs.registration import EnvSpec


class TFT_Item_Simulator(gym.Env):
    """
    Environment for training a model that takes in two players a token of item movements and which items should be
    moved.
    Moves the items for the provided player then plays a single battle. Returns reward and ends episode.
    All episodes are 1 step.
    Trains with both no other player information available as well as having other player information available.
    """
    metadata = {"render_mode": [], "name": "TFT_Item_Simulator_s4_v0"}

    def __init__(self, data_generator):
        self._skip_env_checking = True
        self.pool_obj = pool.pool()
        self.data_generator = data_generator
        self.PLAYER = player_class(self.pool_obj, 0)

        self.render_mode = None

        self.reward = 0

        self.action_space = MultiDiscrete(np.ones(10) * 29)

        self.observation_space = Dict({
            "observations": Dict({
                "player": Dict({
                    "scalars": Box(-2, 2, (config.SCALAR_INPUT_SIZE,), np.float32),
                    "shop": Box(-2, 2, (config.SHOP_INPUT_SIZE,), np.float32),
                    "board": Box(-2, 2, (config.BOARD_INPUT_SIZE,), np.float32),
                    "bench": Box(-2, 2, (config.BENCH_INPUT_SIZE,), np.float32),
                    "items": Box(-2, 2, (config.ITEMS_INPUT_SIZE,), np.float32),
                    "traits": Box(-2, 2, (config.TRAIT_INPUT_SIZE,), np.float32),
                }),
                "opponents": Tuple(
                    Dict({
                        "scalars": Box(-2, 2, (config.OTHER_PLAYER_SCALAR_SIZE,), np.float32),
                        "board": Box(-2, 2, (config.BOARD_INPUT_SIZE,), np.float32),
                        "traits": Box(-2, 2, (config.TRAIT_INPUT_SIZE,), np.float32),
                    }) for _ in range(config.NUM_PLAYERS))
            }),
            "action_mask": Box(0.0, 1.0, shape=(10, 29,))
        })

        self.spec = EnvSpec(
            id="TFT_Item_Simulator_s4_v0",
            max_episode_steps=1
        )

        self.battle_generator = BattleGenerator()

        super().__init__()

    def reset(self, seed=None, return_info=False, options=None):
        if self.data_generator.q_size() >= config.MINIMUM_POP_AMOUNT:
            [player, opponent, other_players, item_guide] = self.data_generator.pop()
        else:
            [player, opponent, other_players] = self.battle_generator.generate_battle()
            item_guide = np.ones((10, 2))

        self.item_guide = item_guide
        self.PLAYER = player
        self.PLAYER.reinit_numpy_arrays()
        self.PLAYER.opponent = opponent
        opponent.opponent = self.PLAYER

        self.player_manager = PlayerManager(config.NUM_PLAYERS,
                                            self.pool_obj, TFTConfig(observation_class=ObservationVector))
        if other_players:
            for player in other_players.values():
                player.reinit_numpy_arrays()
            self.player_manager.reinit_player_set([self.PLAYER] + list(other_players.values()))
        else:
            self.player_manager.reinit_player_set([self.PLAYER])

        self.step_function = Step_Function(self.player_manager)
        self.game_round = Game_Round(self.PLAYER, self.pool_obj, self.step_function)

        self.reward = 0

        initial_observation = self.player_manager.fetch_observation(f"player_{self.PLAYER.player_num}")
        observation = {
            "observations": {key: initial_observation[key]
                             for key in ["player", "opponents"]},
            "action_mask": initial_observation["action_mask"][37:47]
        }
        return observation, {}

    def render(self):
        ...

    def close(self):
        self.reset()

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

    def action_space(self, agent):
            """
            Another method to keep in line with petting zoo and rllib requirements
            """
            return self.action_spaces[agent]

    def step(self, action):
        self.PLAYER.printComp()
        log_to_file(self.PLAYER)
        self.game_round.single_combat_phase([self.PLAYER, self.PLAYER.opponent])
        initial_reward = self.PLAYER.reward
        self.PLAYER.reward = 0
        if action is not None:
            self.step_function.item_controller(action, self.PLAYER, self.item_guide)
        self.game_round.single_combat_phase([self.PLAYER, self.PLAYER.opponent])
        self.reward = self.PLAYER.reward - initial_reward

        initial_observation = self.player_manager.fetch_observation(f"player_{self.PLAYER.player_num}")
        observation = {
            "observations": {key: initial_observation[key]
                             for key in ["player", "opponents"]},
            "action_mask": initial_observation["action_mask"][37:47]
        }
        self.PLAYER.printComp()
        log_to_file(self.PLAYER)

        return observation, self.reward, True, False, {}

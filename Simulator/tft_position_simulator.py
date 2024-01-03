import config
import time
import numpy as np
import gymnasium as gym
from Simulator import pool
from Simulator.observation.vector.observation import ObservationVector
from Simulator.step_function import Step_Function
from Simulator.game_round import Game_Round, log_to_file
from Simulator.player import Player as player_class
from Simulator.player_manager import PlayerManager
from Simulator.tft_simulator import TFTConfig
from Simulator.battle_generator import BattleGenerator
from Simulator.utils import coord_to_x_y
from gymnasium.spaces import MultiDiscrete, Dict, Box, Tuple
from gymnasium.envs.registration import EnvSpec


class TFT_Position_Simulator(gym.Env):
    """
    Environment for training the positioning model.
    Takes in a set of movement commands to reorganize the board, executes those commands and then does a battle.
    Reward is the reward from the battle.
    Each episode is a single step.
    """
    metadata = {"render_mode": [], "name": "TFT_Position_Simulator_s4_v0"}

    def __init__(self, data_generator):
        self.pool_obj = pool.pool()
        self.data_generator = data_generator
        self.PLAYER = player_class(self.pool_obj, 0)

        self.render_mode = None

        self.reward = 0

        self.action_space = MultiDiscrete(np.ones(12) * 29)

        self.observation_space = Dict({
            "observation": Dict({
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
            "action_mask": MultiDiscrete(np.ones((12, 29)) * 2)
        })

        self.spec = EnvSpec(
            id="TFT_Position_Simulator_s4_v0",
            max_episode_steps=1
        )

        self.battle_generator = BattleGenerator()

        super().__init__()

    def reset(self, seed=None, return_info=False, options=None):
        if self.data_generator.q_size() >= config.MINIMUM_POP_AMOUNT:
            [player, opponent, other_players] = self.data_generator.pop()
        else:
            [player, opponent, other_players] = self.battle_generator.generate_battle()

        self.PLAYER = player
        self.PLAYER.reinit_numpy_arrays()
        self.PLAYER.opponent = opponent
        opponent.opponent = self.PLAYER

        for player in other_players.values():
            player.reinit_numpy_arrays()

        self.player_manager = PlayerManager(config.NUM_PLAYERS, self.pool_obj,
                                            TFTConfig(observation_class=ObservationVector))
        self.player_manager.reinit_player_set([self.PLAYER] + list(other_players.values()))

        self.step_function = Step_Function(self.player_manager)

        self.game_round = Game_Round(self.PLAYER, self.pool_obj, self.player_manager)

        self.reward = 0
        print("Checkpoint -3")

        initial_observation = self.player_manager.fetch_observation(f"player_{self.PLAYER.player_num}")
        observation = {
            "observation": {key: initial_observation[key]
                            for key in ["player", "opponents"]},
            "action_mask": self.full_mask_to_action_mask(initial_observation["action_mask"])
        }
        print("Checkpoint -4")

        return observation, {}

    def render(self):
        ...

    def close(self):
        self.reset()

    def step(self, action):
        self.PLAYER.printComp()
        log_to_file(self.PLAYER)
        self.game_round.single_combat_phase([self.PLAYER, self.PLAYER.opponent])
        initial_reward = self.PLAYER.reward
        self.PLAYER.reward = 0
        if action is not None:
            self.step_function.batch_position_controller(action, self.PLAYER)
        self.game_round.single_combat_phase([self.PLAYER, self.PLAYER.opponent])
        self.reward = self.PLAYER.reward - initial_reward

        initial_observation = self.player_manager.fetch_observation(f"player_{self.PLAYER.player_num}")
        observation = {
            "observation": {key: initial_observation[key]
                            for key in ["player", "opponents"]},
            "action_mask": self.full_mask_to_action_mask(initial_observation["action_mask"])
        }
        self.PLAYER.printComp()
        log_to_file(self.PLAYER)

        return observation, self.reward, True, False, {}

    def full_mask_to_action_mask(self, mask):
        action_mask = np.ones((12, 29))
        idx = 0
        for coord in range(len(self.PLAYER.board) * len(self.PLAYER.board[0])):
            x1, y1 = coord_to_x_y(coord)
            if self.PLAYER.board[x1][y1]:
                action_mask[idx][0:28] = mask[coord][0:28]
                idx += 1

        while idx < 12:
            action_mask[idx][0:28] = np.zeros(28)
            idx += 1

        return action_mask

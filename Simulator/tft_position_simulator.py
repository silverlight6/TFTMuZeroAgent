import copy

import config
import gymnasium as gym
import numpy as np

from gymnasium.envs.registration import EnvSpec
from gymnasium.spaces import MultiDiscrete, Dict, Box
from Simulator import pool
from Simulator.position_leveling_system import PositionLevelingSystem
from Simulator.game_round import Game_Round, log_to_file, log_to_file_start
from Simulator.observation.token.basic_observation import ObservationToken
from Simulator.player_manager import PlayerManager
from Simulator.step_function import Step_Function
from Simulator.tft_config import TFTConfig
from Simulator.utils import coord_to_x_y


class TFT_Position_Simulator(gym.Env):
    """
    Environment for training the positioning model.
    Takes in a set of movement commands to reorganize the board, executes those commands and then does a battle.
    Reward is the reward from the battle.
    Each episode is a single step.
    """
    metadata = {"render_mode": [], "name": "TFT_Position_Simulator_s4_v0"}

    def __init__(self, data_generator=None, index=None):
        self._skip_env_checking = True

        self.data_generator = data_generator

        self.render_mode = None

        self.reward = 0
        self.max_reward = 1
        self.max_action_count = 12

        self.action_space = MultiDiscrete(np.ones(self.max_action_count) * 29)

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
                "opponents": Box(-2, 2, (config.OTHER_PLAYER_INPUT_SIZE,), np.float32)
            }),
            "action_mask": Box(0.0, 1.0, shape=(12, 29,))
        })

        self.spec = EnvSpec(
            id="TFT_Position_Simulator_s4_v0",
            max_episode_steps=1
        )

        # Object that creates random battles. Used when the buffer is empty.
        self.leveling_system = PositionLevelingSystem()
        self.index = index
        self.observation_class = ObservationToken

        self.multi_step = config.MULTI_STEP_POSITION
        self.action_count = 0

        super().__init__()

    def reset(self, seed=None, return_info=False, options=None):
        if self.data_generator and self.data_generator.q_size() >= config.MINIMUM_POP_AMOUNT:
            [player, opponent, other_players] = self.data_generator.pop()
        else:
            if config.PRESET_BATTLE:
                [player, opponent, other_players] = self.leveling_system.generate_preset_battle()
            else:
                [player, opponent, other_players] = self.leveling_system.generate_battle()

        pool_obj = pool.pool()
        # Objects for the player manager
        self.PLAYER = player
        # Reinit to get around a ray memory bug.
        self.PLAYER.reinit_numpy_arrays()
        self.PLAYER.opponent = opponent
        opponent.opponent = self.PLAYER

        for player in other_players.values():
            player.reinit_numpy_arrays()

        self.player_manager = PlayerManager(config.NUM_PLAYERS, pool_obj,
                                            TFTConfig(observation_class=self.observation_class,
                                                      num_players=config.NUM_PLAYERS))
        self.player_manager.reinit_player_set([self.PLAYER] + list(other_players.values()))

        self.step_function = Step_Function(self.player_manager)

        self.game_round = Game_Round(self.PLAYER, pool_obj, self.player_manager)

        self.reward = 0
        log_to_file_start()

        self.PLAYER.printt("Position Simulator before movement")
        self.PLAYER.printComp()
        log_to_file(self.PLAYER)
        self.step_function.create_unit_list(self.PLAYER)
        # self.game_round.single_combat_phase([self.PLAYER, self.PLAYER.opponent])

        # Single step environment so this fetch will be the observation for the entire step.
        initial_observation = self.player_manager.fetch_position_observation(f"player_{self.PLAYER.player_num}")
        observation = {
            "observations": self.observation_class.observation_to_position_input(initial_observation, self.action_count),
            "action_mask": self.full_mask_to_action_mask(self.PLAYER, initial_observation["action_mask"], 'reset')
        }
        self.action_count = 0
        self.max_action_count = self.PLAYER.num_units_in_play

        return observation, {"num_units": self.max_action_count}

    def render(self):
        ...

    def close(self):
        self.reset()

    """
    Description - There are two battles as part of this simulation.
                    The first is to make sure that we are aware of what the reward is for the fight without changes.
                    The second is to see if the changes made by the model improved the fight or not. 
                    No difference between the fight results in a reward of 0. 
                    Losing by less the second time means an improvement even if the agent still lost. 
                    When this model trains on data provided from the self-play games and is trained to some degree,
                    both boards will be moderately well positioned. The idea is it should find a maximum where it can
                    no longer improve the positioning of the board from what it is given. 
    """
    def step(self, action):
        if action is not None:
            if config.MUZERO_POSITION:
                self.step_function.multi_step_position_controller(action, self.action_count)
                self.action_count += 1
            else:
                self.step_function.position_controller(action, self.PLAYER)
        if not config.MUZERO_POSITION or self.action_count == self.PLAYER.num_units_in_play:
            # initial_reward = self.PLAYER.reward
            self.PLAYER.reward = 0
            self.game_round.single_combat_phase([self.PLAYER, self.PLAYER.opponent])
            # self.reward = self.PLAYER.reward - initial_reward
            self.reward = self.PLAYER.reward
            # if np.abs(self.reward) > self.max_reward:
            #     self.max_reward = np.abs(self.reward)
            # self.reward = self.reward / self.max_reward + 1
            termination = True
        else:
            self.reward = 0
            termination = False

        initial_observation = self.player_manager.fetch_position_observation(f"player_{self.PLAYER.player_num}")
        observation = {
            "observations": self.observation_class.observation_to_position_input(initial_observation, self.action_count),
            "action_mask": self.full_mask_to_action_mask(self.PLAYER, initial_observation["action_mask"], 'step')
        }

        self.PLAYER.print("Position Simulator after movement")
        self.PLAYER.printComp()
        log_to_file(self.PLAYER)

        return observation, self.reward, termination, False, {"num_units": self.max_action_count}

    """
    Description - This method is intended to be used in the MCTS Tree when you need to do local simulations but not 
                simulations that would return a termination or an observation. 
    """
    def fake_step(self, action, unit_number):
        copied_player = copy.deepcopy(self.PLAYER)
        if action is not None:
            action_count = 0
            while unit_number < self.max_action_count:
                self.step_function.fake_multi_step_position_controller(action[action_count], copied_player, unit_number)
                action_count += 1
                unit_number += 1
        # initial_reward = copied_player.reward
        copied_player.reward = 0
        self.game_round.single_combat_phase([copied_player, copied_player.opponent])
        # reward = copied_player.reward - initial_reward
        reward = copied_player.reward
        # print(f"rewarding reward {reward} for unit number {unit_number} with action {action} on {self.action_count} turn")
        return reward

    # Building the action mask, the from_place is in case I need information for debugging.
    def full_mask_to_action_mask(self, player, mask, from_place='step'):
        action_mask = np.ones((12, 29), dtype=np.float32)
        action_mask[:, 0:28] = np.zeros((12, 28), dtype=np.float32)
        idx = 0
        for coord in range(len(player.board) * len(player.board[0])):
            x1, y1 = coord_to_x_y(coord)
            if player.board[x1][y1]:
                action_mask[idx, 0:28] = mask[coord, 0:28]
                idx += 1

        return action_mask

    def level_up(self):
        self.leveling_system.level_up()

# Turns the 3 separate vectors that belong to the opponent into one.
def opponents_to_one_vector(opponents):
    opponents_vector = np.array([])
    for player in opponents:
        for key in player.keys():
            opponents_vector = np.append(opponents_vector, player[key])
    return np.array(opponents_vector, dtype=np.float32)

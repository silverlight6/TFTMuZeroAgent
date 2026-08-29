import copy
import random

import gymnasium as gym
import numpy as np
from Simulator import config

from gymnasium.spaces import Box, Dict, MultiDiscrete
from Simulator.game import pool
from Simulator.generators.position_leveling_system import PositionLevelingSystem
from Simulator.game.game_round import Game_Round, log_to_file, log_to_file_start
from Simulator.encoding.token.basic_observation import ObservationToken
from Simulator.game.player_manager import PlayerManager
from Simulator.game.step_function import Step_Function
from Simulator.simulators.tft_simulator import TFTConfig
from Simulator.simulators.ui import GameState, is_porosight_render
from Simulator.utils import coord_to_x_y


class TFT_Position_Simulator(gym.Env):
    """
    Environment for training the positioning model.
    Takes in a set of movement commands to reorganize the board, executes those commands and then does a battle.
    Reward is the reward from the battle.
    Each episode is a single step.
    """
    metadata = {"render_modes": ["porosight"], "name": "TFT_Position_Simulator_s4_v0"}

    def __init__(self, data_generator=None, index=None, multi_step=False,
                 preset_battle=False, step_until_units_placed=False, single_player=False,
                 render_mode=None, render_path="Games"):
        super().__init__()
        self.data_generator = data_generator
        self.preset_battle = preset_battle
        self.step_until_units_placed = step_until_units_placed

        self.render_mode = render_mode
        self.render_path = render_path

        self.reward = 0
        self.max_reward = 1
        self.max_action_count = 12

        self.action_space = MultiDiscrete(np.ones(self.max_action_count, dtype=np.int64) * 29)

        self.spec = None

        # Object that creates random battles. Used when the buffer is empty.
        self.leveling_system = PositionLevelingSystem(single_player=single_player)
        self.index = index
        self.observation_class = ObservationToken

        self.multi_step = multi_step
        self.action_count = 0
        self.observation_space = Dict({
            "observations": self.observation_class.position_observation_space(num_players=config.NUM_PLAYERS),
            "action_mask": Box(0.0, 1.0, (self.max_action_count, 29), np.float32),
        })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        if self.data_generator and self.data_generator.q_size() >= config.MINIMUM_POP_AMOUNT:
            [player, opponent, other_players] = self.data_generator.pop()
        else:
            if self.preset_battle:
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

        if is_porosight_render(self.render_mode):
            suffix = f"_env{self.index}" if self.index is not None else ""
            self.game_state = GameState.for_position(
                self.PLAYER, opponent, other_players, self.render_path, file_suffix=suffix
            )

        return observation, {"num_units": self.max_action_count}

    def render(self):
        ...

    def close(self):
        pass

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
            if self.step_until_units_placed:
                self.step_function.multi_step_position_controller(action, self.action_count)
                self.action_count += 1
            else:
                self.step_function.position_controller(action, self.PLAYER)
        if not self.step_until_units_placed or self.action_count == self.PLAYER.num_units_in_play:
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

        if is_porosight_render(self.render_mode):
            agent = f"player_{self.PLAYER.player_num}"
            dest = int(np.asarray(action).reshape(-1)[0]) if action is not None else 0
            self.game_state.store_board_change(agent, from_loc=0, to_loc=dest)
            if termination:
                result = "win" if self.PLAYER.reward > 0 else ("tie" if self.PLAYER.reward == 0 else "loss")
                self.game_state.store_direct_battle(
                    agent, opponent=self.PLAYER.opponent, result=result, round_num=1, emit_action=True
                )
                self.game_state.write_json()

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

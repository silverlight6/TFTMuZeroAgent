import random

import numpy as np
import gymnasium as gym
from Simulator import config

from Simulator.game import pool
from Simulator.game.game_round import log_to_file_start
from Simulator.game.single_player_game_round import Game_Round
from Simulator.encoding.token.basic_observation import ObservationToken
from Simulator.game.player_manager import PlayerManager
from Simulator.game.step_function import Step_Function
from Simulator.simulators.tft_simulator import TFTConfig
from Simulator.simulators.ui import GameState, is_porosight_render
from gymnasium.spaces import Dict


class TFT_Single_Player_Simulator(gym.Env):
    """
    Environment for training the positioning model.
    Takes in a set of movement commands to reorganize the board, executes those commands and then does a battle.
    Reward is the reward from the battle.
    Each episode is a single step.
    """
    metadata = {"render_modes": ["porosight"], "name": "TFT_Single_Player_Simulator_s4_v0"}

    def __init__(self, tft_config: TFTConfig = None, index=None):
        super().__init__()
        tft_config = tft_config or TFTConfig(num_players=1)
        self.tft_config = tft_config
        self.index = index

        self.render_mode = tft_config.render_mode
        self.render_path = tft_config.render_path
        self.action_class = tft_config.action_class
        self.reward = 0
        self.action_space = tft_config.action_class.action_space()

        self.spec = None

        # Object that creates random battles. Used when the buffer is empty.
        self.observation_class = ObservationToken

        self.multi_step = tft_config.multi_step_position
        self.action_count = 0
        self.max_actions_per_round = tft_config.max_actions_per_round
        mask_space = getattr(tft_config.action_class, "action_mask_space", None)
        self.observation_space = Dict({
            "observations": self.observation_class.player_observation_space(),
            "action_mask": mask_space() if mask_space else tft_config.action_class.action_space(),
        })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        pool_obj = pool.pool()
        self.player_manager = PlayerManager(config.NUM_PLAYERS, pool_obj,
                                            TFTConfig(observation_class=self.observation_class,
                                                      action_class=self.action_class,
                                                      num_players=1))
        # Objects for the player manager
        self.PLAYER = self.player_manager.player_states['player_0']

        self.step_function = Step_Function(self.player_manager)

        self.game_round = Game_Round(self.PLAYER, pool_obj, self.player_manager)

        self.reward = 0
        log_to_file_start()

        self.info = {
            "player_0": {
                "state_empty": False,
                # "player": self.player_manager.player_states["player_0"],
                # "shop": self.player_manager.player_states["player_0"].shop,
                # "start_turn": False,
                "game_round": 0,
                # "save_battle": False,
                # "actions_taken": 0,
            }
        }

        # Single step environment so this fetch will be the observation for the entire step.
        # --- TFT Starting Game State ---
        self.game_round.play_game_round()  # Does first carousel and first minion wave
        self.player_manager.refresh_player_shop("player_0")
        self.player_manager.update_game_round()

        if is_porosight_render(self.render_mode):
            suffix = f"_env{self.index}" if self.index is not None else ""
            self.game_state = GameState.for_single_player(
                {"player_0": self.PLAYER},
                self.game_round,
                self.render_path,
                self.action_class,
                file_suffix=suffix,
            )

        initial_observation = self.player_manager.fetch_observation('player_0')
        return {
            "observations": initial_observation["player"],
            "action_mask": np.asarray(initial_observation["action_mask"]).reshape(-1).astype(np.int8),
        }, self.info

    def render(self):
        ...

    def close(self):
        pass

    def taken_max_actions(self, player_id):
        return self.action_count >= self.max_actions_per_round

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
        # Perform action and update observations
        action = np.asarray(action)
        decoded = self.action_class.decode_env_action(action)
        self.step_function.perform_action('player_0', decoded)

        self.action_count += 1
        if is_porosight_render(self.render_mode):
            self.game_state.store_action("player_0", action)

        self.info = {
            "state_empty": self.player_manager.player_states['player_0'].state_empty(),
            # "player": self.player_manager.player_states['player_0'],
            # "shop": self.player_manager.player_states['player_0'].shop,
            "game_round": self.game_round.current_round,
            # "start_turn": False,
            # "actions_taken": self.action_count,
            # "save_battle": self.game_round.save_current_battle['player_0']
        }
        round_result = True

        if self.taken_max_actions('player_0'):
            round_result = self.game_round.play_game_round()
            if is_porosight_render(self.render_mode):
                self.game_state.store_single_player_battle("player_0", won=round_result)

            if round_result:
                self.reward += 1
                self.action_count = 0
                self.game_round.start_round()
                self.player_manager.update_game_round()
                if is_porosight_render(self.render_mode):
                    self.game_state.store_game_round()
                self.info['player_0'] = {
                    "state_empty": False,
                    # "player": self.player_manager.player_states['player_0'],
                    "game_round": self.game_round.current_round,
                    # "shop": self.player_manager.player_states['player_0'].shop,
                    # "start_turn": True,
                    # "save_battle": self.game_round.save_current_battle['player_0']
                }
                if self.game_round.current_round >= len(self.game_round.game_rounds):
                    if is_porosight_render(self.render_mode):
                        self.game_state.write_json()
                    round_result = False
            else:
                self.reward -= 1
                if is_porosight_render(self.render_mode):
                    self.game_state.write_json()

        initial_observation = self.player_manager.fetch_observation('player_0')
        observation = {
            "observations": initial_observation["player"],
            "action_mask": np.asarray(initial_observation["action_mask"]).reshape(-1).astype(np.int8),
        }

        return observation, self.reward, not round_result, False, self.info

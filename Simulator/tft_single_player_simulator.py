import config
import gymnasium as gym

from gymnasium.envs.registration import EnvSpec
from Simulator import pool
from Simulator.game_round import log_to_file_start
from Simulator.single_player_game_round import Game_Round
from Simulator.observation.token.basic_observation import ObservationToken
from Simulator.player_manager import PlayerManager
from Simulator.step_function import Step_Function
from Simulator.tft_simulator import TFTConfig


class TFT_Single_Player_Simulator(gym.Env):
    """
    Environment for training the positioning model.
    Takes in a set of movement commands to reorganize the board, executes those commands and then does a battle.
    Reward is the reward from the battle.
    Each episode is a single step.
    """
    metadata = {"render_mode": [], "name": "TFT_Single_Player_Simulator_s4_v0"}

    def __init__(self, tft_config: TFTConfig):
        self._skip_env_checking = True

        self.render_mode = None
        self.reward = 0
        self.action_space = tft_config.action_class.action_space()

        self.spec = EnvSpec(
            id="TFT_Single_Player_Simulator_s4_v0",
            max_episode_steps=1
        )

        # Object that creates random battles. Used when the buffer is empty.
        self.observation_class = ObservationToken

        self.multi_step = config.MULTI_STEP_POSITION
        self.action_count = 0
        self.max_actions_per_round = tft_config.max_actions_per_round

        super().__init__()

    def reset(self, seed=None, return_info=False, options=None):

        pool_obj = pool.pool()
        self.player_manager = PlayerManager(config.NUM_PLAYERS, pool_obj,
                                            TFTConfig(observation_class=self.observation_class, num_players=1))
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
                # "game_round": 1,
                # "save_battle": False,
                # "actions_taken": 0,
            }
        }

        # Single step environment so this fetch will be the observation for the entire step.
        # --- TFT Starting Game State ---
        self.game_round.play_game_round()  # Does first carousel and first minion wave
        self.player_manager.refresh_player_shop("player_0")
        self.player_manager.update_game_round()

        initial_observation = self.player_manager.fetch_observation('player_0')
        return {
            "observations": initial_observation["player"],
            "action_mask": initial_observation["action_mask"]
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
        if action.ndim == 0:
            self.step_function.perform_1d_action('player_0', action)
        else:
            self.step_function.perform_action('player_0', action)

        self.action_count += 1

        self.info = {
            "state_empty": self.player_manager.player_states['player_0'].state_empty(),
            # "player": self.player_manager.player_states['player_0'],
            # "shop": self.player_manager.player_states['player_0'].shop,
            # "game_round": self.game_round.current_round,
            # "start_turn": False,
            # "actions_taken": self.action_count,
            # "save_battle": self.game_round.save_current_battle['player_0']
        }
        round_result = True

        if self.taken_max_actions('player_0'):
            round_result = self.game_round.play_game_round()

            if round_result:
                self.reward += 1
                self.action_count = 0
                self.game_round.start_round()
                self.player_manager.update_game_round()
                self.info['player_0'] = {
                    "state_empty": False,
                    # "player": self.player_manager.player_states['player_0'],
                    # "game_round": self.game_round.current_round,
                    # "shop": self.player_manager.player_states['player_0'].shop,
                    # "start_turn": True,
                    # "save_battle": self.game_round.save_current_battle['player_0']
                }
        else:
            self.reward -= 1

        initial_observation = self.player_manager.fetch_observation('player_0')
        observation = {
            "observations": initial_observation["player"],
            "action_mask": initial_observation["action_mask"]
        }

        return observation, self.reward, not round_result, False, self.info

    def observation_space(self):
        return None

import config
import gymnasium as gym
import numpy as np

from gymnasium.envs.registration import EnvSpec
from gymnasium.spaces import MultiDiscrete, Dict, Box
from Set12Simulator import pool
from Set12Simulator.position_leveling_system import PositionLevelingSystem
from Set12Simulator.game_round import Game_Round, log_to_file, log_to_file_start
from Set12Simulator.observation.vector.observation import ObservationVector
from Set12Simulator.observation.token.basic_observation import ObservationToken
from Set12Simulator.player_manager import PlayerManager
from Set12Simulator.step_function import Step_Function
from Set12Simulator.tft_simulator import TFTConfig
from Set12Simulator.utils import coord_to_x_y


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

        self.action_space = MultiDiscrete(np.ones(12) * 29)

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

        super().__init__()

    def reset(self, seed=None, return_info=False, options=None):
        if self.data_generator and self.data_generator.q_size() >= config.MINIMUM_POP_AMOUNT:
            [player, opponent, other_players] = self.data_generator.pop()
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
                                            TFTConfig(observation_class=self.observation_class))
        self.player_manager.reinit_player_set([self.PLAYER] + list(other_players.values()))

        self.step_function = Step_Function(self.player_manager)

        self.game_round = Game_Round(self.PLAYER, pool_obj, self.player_manager)

        self.reward = 0
        log_to_file_start()

        # Single step environment so this fetch will be the observation for the entire step.
        initial_observation = self.player_manager.fetch_position_observation(f"player_{self.PLAYER.player_num}")
        observation = {
            "observations": self.observation_class.observation_to_input(initial_observation),
            "action_mask": self.full_mask_to_action_mask(self.PLAYER, initial_observation["action_mask"], 'reset')
        }

        return observation, {}

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
        self.PLAYER.printt("Position Simulator before movement")
        self.PLAYER.printComp()
        log_to_file(self.PLAYER)
        self.game_round.single_combat_phase([self.PLAYER, self.PLAYER.opponent])
        initial_reward = self.PLAYER.reward
        self.PLAYER.reward = 0
        if action is not None:
            self.step_function.position_controller(action, self.PLAYER)
        self.game_round.single_combat_phase([self.PLAYER, self.PLAYER.opponent])
        self.reward = self.PLAYER.reward - initial_reward
        if np.abs(self.reward) > self.max_reward:
            self.max_reward = np.abs(self.reward)
        self.reward = self.reward / self.max_reward

        initial_observation = self.player_manager.fetch_position_observation(f"player_{self.PLAYER.player_num}")
        observation = {
            "observations": self.observation_class.observation_to_input(initial_observation),
            "action_mask": self.full_mask_to_action_mask(self.PLAYER, initial_observation["action_mask"], 'step')
        }
        self.PLAYER.print("Position Simulator after movement")
        self.PLAYER.printComp()
        log_to_file(self.PLAYER)

        return observation, self.reward, True, False, {}

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

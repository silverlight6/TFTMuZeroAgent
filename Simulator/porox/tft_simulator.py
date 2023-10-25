import functools

from gymnasium.spaces import MultiDiscrete, Box, Dict, Tuple

from pettingzoo.utils.env import ParallelEnv
from pettingzoo.utils import wrappers

import numpy as np

from Simulator import pool
from Simulator.game_round import Game_Round

from Simulator.porox.player import Player as player_class
from Simulator.porox.player_manager import PlayerManager
from Simulator.porox.step_function import Step_Function
from Simulator.porox.observation.observation_helper import Observation


def env():
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer pettingzoo documentation.
    """
    local_env = TFT_Simulator(env_config=None)

    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    local_env = wrappers.OrderEnforcingWrapper(local_env)
    return local_env


class TFT_Simulator(ParallelEnv):
    metadata = {"is_parallelizable": True, "name": "tft-set4-v0"}

    def __init__(self,
                 num_players: int = 8,
                 max_actions_per_round: int = 15,
                 reward_type: str = "winloss",
                 render_mode: str = None,
                 ):

        # --- PettingZoo AECEnv Variables ---
        self.possible_agents = ["player_" +
                                str(r) for r in range(num_players)]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.render_mode = render_mode

        # --- Config Variables ---
        self.num_players = num_players
        self.max_actions_per_round = max_actions_per_round
        self.reward_type = reward_type

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return None

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        """
        Action Space is an 5x11x38 Dimension MultiDiscrete Tensor
                     11
           |P|L|R|B|B|B|B|B|B|B|S| 
           |b|b|b|B|B|B|B|B|B|B|S|
        5  |b|b|b|B|B|B|B|B|B|B|S| x 38
           |b|b|b|B|B|B|B|B|B|B|S|
           |I|I|I|I|I|I|I|I|I|I|S|

        P = Pass Action
        L = Level Action
        R = Refresh Action
        B = Board Slot
        b = Bench Slot
        I = Item Slot
        S = Shop Slot

        Pass, Level, Refresh, and Shop are single action spaces,
        meaning we only use the first dimension of the MultiDiscrete Space

        Board, Bench, and Item are multi action spaces,
        meaning we use all 3 dimensions of the MultiDiscrete Space

        0-26 -> Board Slots
        27-36 -> Bench Slots
        37 -> Sell Slot

        Board and Bench use all 38 dimensions,
        Item only uses 37 dimensions, as you cannot sell an item

        """
        return MultiDiscrete([5, 11, 38])

    def render(self):
        pass

    def observe(self, agent):
        return None

    def close(self):
        pass

    def reset(self):
        # --- PettingZoo AECEnv Variables ---
        self.agents = self.possible_agents[:]
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}

        # --- TFT Reward Related Variables ---
        self.previous_rewards = {
            "player_" + str(player_id): 0 for player_id in range(self.num_players)
        }
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}

        # --- TFT Game State Related Variables ---
        self.num_dead = 0
        self.num_alive = self.num_players

        # --- TFT Game Related Variables ---
        self.pool_obj = pool.pool()

        # --- TFT Player Related Variables ---
        self.player_manager = PlayerManager(self.num_players)
        self.player_game_states = {
            "player_" + str(player_id): {
                "actions_taken": 0,
            }
            for player_id in range(self.num_players)
        }

        # --- TFT Game Round Related Variables ---
        self.step_function = Step_Function(self.pool_obj)
        self.game_round = Game_Round(
            self.player_states, self.pool_obj, self.step_function)

        # --- TFT Starting Game State ---
        self.game_round.play_game_round()

        for key, p in self.player_states.items():
            self.step_function.generate_shop(key, p)

        observations = self.player_manager.generate_observations()
        infos = self.player_game_states

        return observations, infos

    # --- Utility Functions ---

    # -- Query Functions --
    def is_alive(self, player_id):
        return not self.terminations[player_id]

    def taking_actions(self, player_id):
        return not self.truncations[player_id]

    def taken_max_actions(self, player_id):
        return self.player_game_states[player_id]["actions_taken"] >= self.max_actions_per_round

    def round_done(self):
        return all(self.truncations.values())

    def game_over(self):
        return self.num_alive <= 1 or self.game_round.current_round > 48

    # -- Update Functions --
    def reset_max_actions(self):
        for player_id in self.player_game_states:
            self.player_game_states[player_id]["actions_taken"] = 0
            self.truncations[player_id] = False

    def calculate_winloss(self, placement):
        MAX_REWARD = 400
        STEP = 100

        return MAX_REWARD - (placement - 1) * STEP

    def update_dead(self):
        for player_id, player in self.player_states.items():
            if self.is_alive(player_id) and \
                    player.health <= 0:
                self.num_dead += 1
                self.num_alive -= 1

                self.game_round.NUM_DEAD = self.num_dead
                self.pool_obj.return_hero(player)

                self.rewards[player_id] = self.calculate_winloss(
                    self.num_alive + 1)
                self._cumulative_rewards[player_id] = self.rewards[player_id]

                self.player_states[player_id] = None
                self.game_round.update_players(self.player_states)

                self.terminations[player_id] = True

    # --- Step Function ---

    def step(self, actions):
        """
        Actions is a dictionary of actions from each agent.
        Ex:
            {
                "player_0": "[0, 0, 0]", - Pass action
                "player_1": "[1, 0, 0]", - Level action
                "player_2": "[2, 0, 0]", - Refresh action
                "player_3": "[3, X1, 0]", - Buy action
                "player_4": "[4, X1, 0]", - Sell action
                "player_5": "[5, X1, X2]", - Move action
                "player_6": "[6, X1, X2]", - Item action
                ...
            }
        """
        # Perform actions
        for player_id, action in actions.items():
            if self.is_alive(player_id) and self.taking_actions(player_id):
                # Perform action
                self.player_manager.perform_action(player_id, action)

                self.player_game_states[player_id]["actions_taken"] += 1

                if self.taken_max_actions(player_id):
                    self.truncations[player_id] = True

                # Update observations
                self.player_observations[player_id] = self.player_states[player_id].observation(
                )

        # Check if round is over
        if self.round_done():
            # Battle
            self.game_round.play_game_round()

            # Update dead
            self.update_dead()

            # Check if the game is over
            if self.game_over():
                for player_id in self.agents:
                    if not self.terminations[player_id]:
                        self.rewards[player_id] = self.calculate_winloss(1)
                        self._cumulative_rewards[player_id] = self.rewards[player_id]
                        # Without this the reward is reset
                        self.player_states[player_id] = None

                self.terminations = {a: True for a in self.agents}

            # Update observations and start next round
            if not all(self.terminations.values()):
                # Reset Game Info
                self.reset_max_actions()
                self.game_round.start_round()

                for player_id in self.agents:
                    if self.is_alive(player_id):
                        self.player_observations[player_id] = self.player_states[player_id].observation(
                        )

        obs = {
            player_id: self.player_observations[player_id] for player_id in actions}
        rewards = {
            player_id: self.rewards[player_id] for player_id in actions}
        terminations = {
            player_id: self.terminations[player_id] for player_id in actions}
        truncations = {
            player_id: self.truncations[player_id] for player_id in actions}
        infos = {
            player_id: self.player_game_states[player_id] for player_id in actions}

        return obs, rewards, terminations, truncations, infos

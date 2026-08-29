import random
import time
from Simulator import config
import numpy as np
import gymnasium as gym
from Simulator.game import pool
from Simulator.observation.vector.observation import ObservationVector
from Simulator.game.step_function import Step_Function
from Simulator.game.game_round import Game_Round, log_to_file
from Simulator.game.player import Player as player_class
from Simulator.game.player_manager import PlayerManager
from Simulator.simulators.tft_simulator import TFTConfig
from Simulator.simulators.ui import GameState, is_porosight_render
from Simulator.generators.battle_generator import BattleGenerator
from gymnasium.spaces import Box, Dict, MultiDiscrete, Tuple


class TFT_Item_Simulator(gym.Env):
    """
    Environment for training a model that takes in two players a token of item movements and which items should be
    moved.
    Moves the items for the provided player then plays a single battle. Returns reward and ends episode.
    All episodes are 1 step.
    Trains with both no other player information available as well as having other player information available.
    """
    metadata = {"render_modes": ["porosight"], "name": "TFT_Item_Simulator_s4_v0"}

    def __init__(self, data_generator=None, index=None, render_mode=None, render_path="Games"):
        super().__init__()
        self.pool_obj = pool.pool()
        self.data_generator = data_generator
        self.PLAYER = player_class(self.pool_obj, 0)
        self.index = index

        self.render_mode = render_mode
        self.render_path = render_path

        self.reward = 0

        self.action_space = MultiDiscrete(np.ones(10, dtype=np.int64) * 29)

        self.spec = None

        self.battle_generator = BattleGenerator()
        n_opp = config.NUM_PLAYERS - 1
        self.observation_space = Dict({
            "observations": Dict({
                "player": ObservationVector.player_observation_space(),
                "opponents": Tuple(
                    tuple(ObservationVector.public_observation_space() for _ in range(n_opp))
                ),
            }),
            "action_mask": Box(0.0, 1.0, (10, 38), np.float32),
        })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        if self.data_generator and self.data_generator.q_size() >= config.MINIMUM_POP_AMOUNT:
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
            "observations": {
                "player": initial_observation["player"],
                "opponents": tuple(initial_observation["opponents"]),
            },
            "action_mask": initial_observation["action_mask"][37:47]
        }
        if is_porosight_render(self.render_mode):
            suffix = f"_env{self.index}" if self.index is not None else ""
            self.game_state = GameState.for_item(
                self.PLAYER, opponent, other_players, self.render_path, file_suffix=suffix
            )
        return observation, {}

    def render(self):
        ...

    def close(self):
        pass

    def step(self, action):
        self.PLAYER.printComp()
        log_to_file(self.PLAYER)
        self.game_round.single_combat_phase([self.PLAYER, self.PLAYER.opponent])
        initial_reward = self.PLAYER.reward
        if is_porosight_render(self.render_mode):
            before = "win" if initial_reward > 0 else ("tie" if initial_reward == 0 else "loss")
            self.game_state.store_direct_battle(
                f"player_{self.PLAYER.player_num}",
                opponent=self.PLAYER.opponent,
                result=before,
                round_num=0,
                emit_action=False,
            )
        self.PLAYER.reward = 0
        if action is not None:
            self.step_function.item_controller(action, self.PLAYER, self.item_guide)
        if is_porosight_render(self.render_mode):
            self.game_state.store_item_assignments(f"player_{self.PLAYER.player_num}", action)
        self.game_round.single_combat_phase([self.PLAYER, self.PLAYER.opponent])
        self.reward = self.PLAYER.reward - initial_reward
        if is_porosight_render(self.render_mode):
            after = "win" if self.PLAYER.reward > 0 else ("tie" if self.PLAYER.reward == 0 else "loss")
            self.game_state.store_direct_battle(
                f"player_{self.PLAYER.player_num}",
                opponent=self.PLAYER.opponent,
                result=after,
                round_num=1,
                emit_action=True,
            )
            self.game_state.write_json()

        initial_observation = self.player_manager.fetch_observation(f"player_{self.PLAYER.player_num}")
        observation = {
            "observations": {
                "player": initial_observation["player"],
                "opponents": tuple(initial_observation["opponents"]),
            },
            "action_mask": initial_observation["action_mask"][37:47]
        }
        self.PLAYER.printComp()
        log_to_file(self.PLAYER)

        return observation, self.reward, True, False, {}

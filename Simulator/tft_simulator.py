import functools
from dataclasses import dataclass

from pettingzoo.utils import wrappers, agent_selector
from pettingzoo.utils.env import AECEnv
from pettingzoo.utils.conversions import parallel_wrapper_fn

from Simulator import pool
from Simulator.game_round import Game_Round

from Simulator.player_manager import PlayerManager
from Simulator.step_function import Step_Function
from Simulator.ui import GameState

from Simulator.observation.interface import ObservationBase, ActionBase
from Simulator.observation.token.observation import ObservationToken
from Simulator.observation.token.action import ActionToken

import time

@dataclass
class TFTConfig:
    num_players: int = 8
    max_actions_per_round: int = 15
    reward_type: str = "winloss"
    render_mode: str = None  # "json" or None
    render_path: str = "Games"
    observation_class: ObservationBase = ObservationToken
    action_class: ActionBase = ActionToken

def env(config: TFTConfig = TFTConfig()):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer pettingzoo documentation.
    """
    local_env = TFT_Simulator(config)

    local_env = wrappers.OrderEnforcingWrapper(local_env)
    return local_env

def parallel_env(config: TFTConfig = TFTConfig()):
    def env():
        local_env = TFT_Simulator(config)
        local_env = wrappers.OrderEnforcingWrapper(local_env)
        return local_env
    
    return parallel_wrapper_fn(env)()

class TFT_Simulator(AECEnv):
    metadata = {"is_parallelizable": True, "name": "tft-set4-v0"}

    def __init__(self, config: TFTConfig):

        # --- PettingZoo AECEnv Variables ---
        self.possible_agents = ["player_" +
                                str(r) for r in range(config.num_players)]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        self.config = config

        self.render_mode = config.render_mode
        self.render_path = config.render_path

        # --- Config Variables ---
        self.num_players = config.num_players
        self.max_actions_per_round = config.max_actions_per_round
        self.reward_type = config.reward_type
        
        # --- Observation and Action Classes ---
        self.observation_class = config.observation_class
        self.action_class = config.action_class

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return None

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_class.action_space()

    def render(self):
        if self.render_mode is not None:
            ...

    def observe(self, agent):
        return self.player_manager.fetch_observation(agent)

    def close(self):
        pass

    def reset(self, seed=None, options=None):
        # --- PettingZoo AECEnv Variables ---
        self.agents = self.possible_agents[:]
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}

        # --- TFT Reward Related Variables ---
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}

        # --- TFT Game State Related Variables ---
        self.num_dead = 0
        self.num_alive = self.num_players

        # --- TFT Game Related Variables ---
        self.pool_obj = pool.pool()

        # --- TFT Player Related Variables ---
        self.player_manager = PlayerManager(self.num_players, self.pool_obj, self.config)
        self.step_function = Step_Function(self.player_manager)

        # --- TFT Game Round Related Variables ---
        self.game_round = Game_Round(self.player_manager.player_states, self.pool_obj, self.player_manager)
        
        # --- TFT Starting Game State ---
        self.game_round.play_game_round()  # Does first carousel and first minion wave
        self.player_manager.generate_shops(self.agents)
        self.player_manager.update_game_round()

        self.infos = {
            "player_" + str(player_id): {
                "state_empty": False,
                "player": self.player_manager.player_states["player_" + str(player_id)],
                "shop": self.player_manager.player_states["player_" + str(player_id)].shop,
                "start_turn": False,
                "game_round": 1,
                "save_battle": False,
                "actions_taken": 0,
            } for player_id in range(self.num_players)
        }

        # --- Game State for Render ---
        if self.render_mode is not None:
            self.game_state = GameState(self.player_manager.player_states, self.game_round, self.render_path,
                                        action_class=self.action_class)
        
        # --- Agent Selector API ---
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        self.actions_taken = {agent: 0 for agent in self.agents}

        self.truncated_agents = []

    # -- Query Functions --
    def is_alive(self, player_id):
        return not self.terminations[player_id]

    def taking_actions(self, player_id):
        return not self.truncations[player_id]

    def taken_max_actions(self, player_id):
        return self.actions_taken[player_id] >= self.max_actions_per_round

    def round_done(self):
        if all(self.truncations.values()):
            for truncated_agent in self.truncated_agents:
                if truncated_agent not in self.agents:
                    self.agents.append(truncated_agent)
            self.truncations = {agent: False for agent in self.agents}
            self.truncated_agents = []
            self.agents.sort()
            self._agent_selector.reinit(self.agents)
            return True
        return False

    def game_over(self):
        return self.num_alive <= 1 or self.game_round.current_round > 48

    # -- Update Functions --
    def reset_max_actions(self):
        for player_id in self.agents:
            if self.is_alive(player_id):
                self.actions_taken[player_id] = 0
                self.truncations[player_id] = False

    def calculate_winloss(self, placement):
        MAX_REWARD = 40
        STEP = 5

        return MAX_REWARD - (placement - 1) * STEP

    def update_dead(self):
        killed_agents = []
        for player_id, player in self.player_manager.player_states.items():
            if self.is_alive(player_id) and player.health <= 0:
                self.num_dead += 1
                self.num_alive -= 1

                self.rewards[player_id] = self.calculate_winloss(self.num_alive + 1)
                self.rewards[player_id] += player.reward
                self._cumulative_rewards[player_id] = self.rewards[player_id]

                self.player_manager.kill_player(player_id)

                self.game_round.NUM_DEAD = self.num_dead
                self.game_round.update_players(self.player_manager.player_states)

                self.terminations[player_id] = True
                killed_agents.append(player_id)
        return killed_agents

    # --- Step Function ---

    def step(self, action):
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

        A regular game round consists of the following:
            1. Players shops refresh, unless locked (not implemented yet)
            2. Players perform actions
                - In this env, players can only take MAX_ACTIONS_PER_ROUND actions per round
                - If a player takes MAX_ACTIONS_PER_ROUND actions, they are truncated
            3. Players battle after all alive players have taken their actions
        """
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        # Perform action and update observations
        if action.ndim == 0:
            self.step_function.perform_1d_action(agent, action)
        else:
            self.step_function.perform_action(agent, action)

        self.actions_taken[agent] += 1
        if self.render_mode is not None:
            self.game_state.store_action(agent, action)

        self.infos[agent] = {
            "state_empty": self.player_manager.player_states[agent].state_empty(),
            "player": self.player_manager.player_states[agent],
            "shop": self.player_manager.player_states[agent].shop,
            "game_round": self.game_round.current_round,
            "start_turn": False,
            "actions_taken": self.actions_taken[agent]
        }

        self._clear_rewards()

        _non_trunc_agents = self.agents[:]
        if self.taken_max_actions(agent):
            self.truncations[agent] = True
            self.truncated_agents.append(agent)

            _non_trunc_agents.remove(agent)

        if self._agent_selector.is_last():
            # TODO: Update rewards
            # If round is over
            if self.round_done():
                self.game_round.play_game_round()

                if self.render_mode is not None:
                    self.game_state.store_battles()

                killed_agents = self.update_dead()

                # Check if the game is over
                if self.game_over():
                    if self.render_mode is not None:
                        self.game_state.write_json()

                    for player_id in self.agents:
                        if not self.terminations[player_id]:
                            self.rewards[player_id] = self.calculate_winloss(1)
                            self.rewards[player_id] += self.player_manager.player_states[player_id].reward
                            self._cumulative_rewards[player_id] = self.rewards[player_id]
                            self.player_manager.kill_player(player_id)
                    
                    self.terminations = {a: True for a in self.agents}
                    
                # Update observations and start the next round
                if not all(self.terminations.values()):
                    self.reset_max_actions()
                    self.game_round.start_round()
                    self.player_manager.update_game_round()
                    
                    if self.render_mode is not None:
                        self.game_state.store_game_round()

                    for player_id in self.agents:
                        if not self.terminations[player_id] and not self.truncations[player_id]:
                            self.rewards[player_id] = self.player_manager.player_states[player_id].reward
                            self._cumulative_rewards[player_id] = self.rewards[player_id]
                            self.infos[player_id] = {
                                "state_empty": False,
                                "player": self.player_manager.player_states[player_id],
                                "game_round": self.game_round.current_round,
                                "shop": self.player_manager.player_states[player_id].shop,
                                "start_turn": True,
                                "save_battle": self.game_round.save_current_battle[player_id]
                            }

                _live_agents = self.agents[:]
                # Update agent_selector if agents died this round
                for killed_agent in killed_agents:
                    _live_agents.remove(killed_agent)
                    del self.player_manager.player_states[killed_agent]

                if len(killed_agents) > 0 and _live_agents:
                    _live_agents.sort()
                    self._agent_selector.reinit(_live_agents)

            elif self.truncated_agents and _non_trunc_agents:
                _non_trunc_agents.sort()
                self._agent_selector.reinit(_non_trunc_agents)

        if len(self._agent_selector.agent_order):
            self.agent_selection = self._agent_selector.next()

        if self._agent_selector.is_first():
            self._deads_step_first()



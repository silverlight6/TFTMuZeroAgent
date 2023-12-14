import functools
from dataclasses import dataclass

from pettingzoo.utils import wrappers, agent_selector
from pettingzoo.utils.env import AECEnv
from pettingzoo.utils.conversions import parallel_wrapper_fn

from Simulator import pool
from Simulator.game_round import Game_Round

from Simulator.porox.player_manager import PlayerManager
from Simulator.porox.ui import GameState

from Simulator.porox.observation import ObservationBase, ActionBase, ObservationVector, ActionVector

@dataclass
class TFTConfig:
    num_players: int = 8
    max_actions_per_round: int = 15
    reward_type: str = "winloss"
    render_mode: str = None # "json" or None
    render_path: str = "Games"
    observation_class: ObservationBase = ObservationVector
    action_class: ActionBase = ActionVector

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
        
        self.update_config(config)
        
    def update_config(self, config: TFTConfig):
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

    def reset(self, seed = None, options = None):
        # --- PettingZoo AECEnv Variables ---
        self.agents = self.possible_agents[:]
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {
            "player_" + str(player_id): {
                "actions_taken": 0,
            } for player_id in range(self.num_players)
        }

        # --- TFT Reward Related Variables ---
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}

        # --- TFT Game State Related Variables ---
        self.num_dead = 0
        self.num_alive = self.num_players

        # --- TFT Game Related Variables ---
        self.pool_obj = pool.pool()

        # --- TFT Player Related Variables ---
        self.player_manager = PlayerManager(self.num_players,
                                            self.pool_obj, 
                                            self.config)

        # --- TFT Game Round Related Variables ---
        self.game_round = Game_Round(
            self.player_manager.player_states, self.pool_obj, self.player_manager)
        
        # --- TFT Starting Game State ---
        self.game_round.play_game_round() # Does first carousel and first minion wave
        self.player_manager.generate_shops(self.agents)
        self.player_manager.update_game_round()
        
        # --- Game State for Render ---
        if self.render_mode is not None:
            self.game_state = GameState(self.player_manager.player_states, self.game_round, self.render_path, action_class=self.action_class)
        
        # --- Agent Selector API ---
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    # -- Query Functions --
    def is_alive(self, player_id):
        return not self.terminations[player_id]

    def taking_actions(self, player_id):
        return not self.truncations[player_id]

    def taken_max_actions(self, player_id):
        return self.infos[player_id]["actions_taken"] >= self.max_actions_per_round

    def round_done(self):
        return all(self.truncations.values())

    def game_over(self):
        return self.num_alive <= 1 or self.game_round.current_round > 48

    # -- Update Functions --
    def reset_max_actions(self):
        for player_id in self.infos:
            if self.is_alive(player_id):
                self.infos[player_id]["actions_taken"] = 0
                self.truncations[player_id] = False

    def calculate_winloss(self, placement):
        MAX_REWARD = 400
        STEP = 100

        return MAX_REWARD - (placement - 1) * STEP

    def update_dead(self):
        killed_agents = []
        for player_id, player in self.player_manager.player_states.items():
            if self.is_alive(player_id) and \
                    player.health <= 0:
                self.num_dead += 1
                self.num_alive -= 1

                self.rewards[player_id] = self.calculate_winloss(
                    self.num_alive + 1)

                self.player_manager.kill_player(player_id)

                self.game_round.NUM_DEAD = self.num_dead
                self.game_round.update_players(self.player_manager.player_states)

                self.terminations[player_id] = True
                self.truncations[player_id] = True
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
        self.player_manager.perform_action(agent, action)

        if self.render_mode is not None:
            self.game_state.store_action(agent, action)
        
        # Update actions taken and truncate if needed
        self.infos[agent]["actions_taken"] += 1
        if self.taken_max_actions(agent):
            self.truncations[agent] = True
            
        if self._agent_selector.is_last():
            # TODO: Update rewards
            ...
            
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
                            self.player_manager.kill_player(player_id)
                    
                    self.terminations = {a: True for a in self.agents}
                    
                # Update observations and start the next round
                if not all(self.terminations.values()):
                    self.reset_max_actions()
                    self.game_round.start_round()
                    self.player_manager.update_game_round()
                    
                    if self.render_mode is not None:
                        self.game_state.store_game_round()
                    
                # Update agent_selector if agents died this round
                if len(killed_agents) > 0:
                    _live_agents = [a for a in self.agents if not self.terminations[a]]
                    self.agents = _live_agents
                    self._agent_selector.reinit(self.agents)

                    
        else:
            self._clear_rewards()
                    
        if len(self._agent_selector.agent_order):
            self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()
        
import functools
from dataclasses import dataclass
import config
import numpy as np

from gymnasium.spaces import MultiDiscrete, Box, Dict, Tuple

from pettingzoo.utils import wrappers, agent_selector
from pettingzoo.utils.env import AECEnv
from pettingzoo.utils.conversions import parallel_wrapper_fn

from Simulator import pool
from Simulator.game_round import Game_Round

from Simulator.porox.player_manager import PlayerManager



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
        self.render_mode = config.render_mode
        self.render_path = config.render_path

        # --- Config Variables ---
        self.num_players = num_players
        self.max_actions_per_round = max_actions_per_round
        self.reward_type = reward_type

        # --- TFT Game Related Variables ---
        self.pool_obj = pool.pool()

        # --- TFT Player Related Variables ---
        self.player_manager = PlayerManager(self.num_players, self.pool_obj)

        self.agents = self.possible_agents[:]
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {"state_empty": False, "player": self.player_manager.player_states[agent], "game_round": 1,
                              "shop": self.player_manager.player_states[agent].shop, "start_turn": True,
                              "actions_taken": 0, "save_battle": False}
                      for agent in self.agents}

        self.state = {agent: {} for agent in self.agents}
        # self.observations = {agent: {} for agent in self.agents}
        self.actions = {agent: {} for agent in self.agents}

        self.default_agent = {agent: False for agent in self.agents}

        # --- TFT Reward Related Variables ---
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}

        # --- TFT Game State Related Variables ---
        self.num_dead = 0
        self.num_alive = self.num_players

        # --- TFT Game Round Related Variables ---
        self.game_round = Game_Round(self.player_manager.player_states, self.pool_obj, self.player_manager)
        # --- TFT Starting Game State ---
        self.game_round.play_game_round()
        self.player_manager.generate_shops(self.agents)
        self.player_manager.update_game_round()

        self.actions_taken = 0

        self.observation_spaces = self.observation_spaces = dict(
            zip(
                self.agents,
                [
                    Dict({
                        "tensor": Dict({
                            "shop": Box(low=-5, high=5, shape=(config.SHOP_INPUT_SIZE,), dtype=np.float32),
                            "board": Box(low=-5, high=5, shape=(config.BOARD_INPUT_SIZE,), dtype=np.float32),
                            "bench": Box(low=-5, high=5, shape=(config.BENCH_INPUT_SIZE,), dtype=np.float32),
                            "states": Box(low=-5, high=5, shape=(config.STATE_INPUT_SIZE,), dtype=np.float32),
                            "game_comp": Box(low=-5, high=5, shape=(config.COMP_INPUT_SIZE,), dtype=np.float32),
                            "other_players": Box(low=-5, high=5, shape=(config.OTHER_PLAYER_INPUT_SIZE,),
                                                 dtype=np.float32)
                        }),
                        "mask": Tuple((
                            Box(low=-2, high=2, shape=(6,), dtype=np.int8),
                            Box(low=-2, high=2, shape=(5,), dtype=np.int8),
                            Box(low=-2, high=2, shape=(28,), dtype=np.int8),
                            Box(low=-2, high=2, shape=(9,), dtype=np.int8),
                            Box(low=-2, high=2, shape=(10,), dtype=np.int8),
                            Box(low=-2, high=2, shape=(3,), dtype=np.int8),
                            Box(low=-2, high=2, shape=(37,), dtype=np.int8),
                            Box(low=-2, high=2, shape=(37,), dtype=np.int8),
                            Box(low=-2, high=2, shape=(10,), dtype=np.int8),
                            Box(low=-2, high=2, shape=(28,), dtype=np.int8),
                            Box(low=-2, high=2, shape=(28,), dtype=np.int8)))
                    }) for _ in self.agents
                ],
            )
        )

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return None

    @functools.lru_cache(maxsize=None)
    def action_space_v1(self, agent):
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
    
    @functools.lru_cache(maxsize=None)
    def action_space_v2(self, agent):
        """
        v2 Action Space is an 55x38 Dimension MultiDiscrete Tensor to keep my sanity
        
        v2 action space: (55, 38)
            55 
        1 | | | | | ... | | x 38
        
        55 :
        0-27 -> Board Slots (28)
        28-36 -> Bench Slots (9)
        37-46 -> Item Bench Slots (10)
        47-51 -> Shop Slots (5)
        52 -> Pass
        53 -> Level
        54 -> Refresh
        
        38 :
        0-27 -> Board Slots
        28-36 -> Bench Slots
        37 -> Sell Slot
        
        """
        return MultiDiscrete([55, 38])
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_space_v2(agent)

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
        self.infos = {agent: {"state_empty": False, "player": self.player_manager.player_states[agent], "game_round": 1,
                              "shop": self.player_manager.player_states[agent].shop, "start_turn": True,
                              "actions_taken": 0, "save_battle": False}
                      for agent in self.agents}

        # --- TFT Reward Related Variables ---
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}

        # --- TFT Game State Related Variables ---
        self.num_dead = 0
        self.num_alive = self.num_players

        # --- TFT Game Related Variables ---
        self.pool_obj = pool.pool()

        # --- TFT Player Related Variables ---
        self.player_manager = PlayerManager(self.num_players, self.pool_obj)

        # --- TFT Game Round Related Variables ---
        self.game_round = Game_Round(
            self.player_manager.player_states, self.pool_obj, self.player_manager)

        # --- TFT Starting Game State ---
        self.game_round.play_game_round() # Does first carousel and first minion wave
        self.player_manager.generate_shops(self.agents)
        self.player_manager.update_game_round()

        # --- Game State for Render ---
        if self.render_mode is not None:
            self.game_state = GameState(self.player_manager.player_states, self.game_round, self.render_path)

        # --- Agent Selector API ---
        self._agent_selector.reinit(self.agents)
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

                self.rewards[player_id] = self.calculate_winloss(self.num_alive + 1) \
                                          + self.player_manager.player_states[player_id].reward
                self._cumulative_rewards[player_id] = self.rewards[player_id]

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
        self.infos[agent] = {"state_empty": self.player_manager.player_states[self.agent_selection].state_empty(),
                             "player": self.player_manager.player_states[self.agent_selection],
                             "game_round": self.game_round.current_round,
                             "shop": self.player_manager.player_states[agent].shop,
                             "start_turn": False,
                             "actions_taken": self.infos[agent]["actions_taken"] + 1}

        self._clear_rewards()

        if self.taken_max_actions(agent):
            self.truncations[agent] = True

        if self._agent_selector.is_last():

            # TODO: Update rewards
            ...
            self.actions_taken += 1
            
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

                # Update observations and start the next round
                if not all(self.terminations.values()):
                    self.reset_max_actions()
                    self.game_round.start_round()
                    self.player_manager.update_game_round()

            for player_id in self.player_manager.player_states.keys():
                if self.player_manager.player_states[player_id]:
                    self.rewards[player_id] = self.player_manager.player_states[player_id].reward
                    self._cumulative_rewards[player_id] = self.rewards[player_id]

        if len(self._agent_selector.agent_order):
            self.agent_selection = self._agent_selector.next()

        self._deads_step_first()
        
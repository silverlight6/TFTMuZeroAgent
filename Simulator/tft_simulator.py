import config
import functools
import time
import gymnasium as gym
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box, Dict, Tuple
from Simulator import pool
from Simulator.player import Player as player_class
from Simulator.step_function import Step_Function
from Simulator.game_round import Game_Round
from Simulator.observation import Observation
from pettingzoo.utils.env import AECEnv
from pettingzoo.utils import wrappers, agent_selector
from pettingzoo.utils.conversions import parallel_wrapper_fn


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


parallel_env = parallel_wrapper_fn(env)


class TFT_Simulator(AECEnv):
    metadata = {"is_parallelizable": True, "name": "tft-set4-v0"}

    def __init__(self, env_config):
        self.pool_obj = pool.pool()
        self.PLAYERS = {"player_" + str(player_id): player_class(self.pool_obj, player_id)
                        for player_id in range(config.NUM_PLAYERS)}
        self.game_observations = {"player_" + str(player_id): Observation() for player_id in range(config.NUM_PLAYERS)}
        self.render_mode = None

        self.NUM_DEAD = 0
        self.num_players = config.NUM_PLAYERS
        self.previous_rewards = {"player_" + str(player_id): 0 for player_id in range(config.NUM_PLAYERS)}

        self.step_function = Step_Function(self.pool_obj, self.game_observations)
        self.game_round = Game_Round(self.PLAYERS, self.pool_obj, self.step_function)
        self.actions_taken = 0
        self.actions_taken_this_turn = 0
        self.game_round.play_game_round()
        for key, p in self.PLAYERS.items():
            self.step_function.generate_shop(key, p)
        self.step_function.generate_shop_vectors(self.PLAYERS)

        self.possible_agents = ["player_" + str(r) for r in range(config.NUM_PLAYERS)]
        self.agents = self.possible_agents[:]
        self.kill_list = []
        # This can likely be deleted, but I'm unsure if petting zoo uses this.
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents)))))
        self._agent_selector = agent_selector(self.possible_agents)
        self.agent_selection = self.possible_agents[0]

        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {"state_empty": False} for agent in self.agents}
        # Is this variable needed for petting zoo?
        self.state = {agent: {} for agent in self.agents}
        self.observations = {agent: {} for agent in self.agents}
        self.actions = {agent: {} for agent in self.agents}

        # For MuZero
        # self.observation_spaces: Dict = dict(
        #     zip(self.agents,
        #         [Box(low=(-5.0), high=5.0, shape=(config.NUM_PLAYERS, config.OBSERVATION_SIZE,),
        #              dtype=np.float32) for _ in self.possible_agents])
        # )

        # For PPO
        self.observation_spaces = dict(
            zip(
                self.agents,
                [
                    Dict({
                        "tensor": Box(low=0, high=10.0, shape=(config.OBSERVATION_SIZE,), dtype=np.float64),
                        "mask": Tuple((MultiDiscrete(np.ones(6) * 2, dtype=np.int8), 
                                       MultiDiscrete(np.ones(5) * 2, dtype=np.int8),
                                       MultiDiscrete(np.ones(28) * 2, dtype=np.int8),
                                       MultiDiscrete(np.ones(9) * 2, dtype=np.int8),
                                       MultiDiscrete(np.ones(10) * 2, dtype=np.int8)))
                    }) for _ in self.agents
                ],
            )
        )

        # For MuZero
        # self.action_spaces = {agent: MultiDiscrete([config.ACTION_DIM for _ in range(config.NUM_PLAYERS)])
        #                       for agent in self.agents}

        # For PPO
        self.action_spaces = {agent: MultiDiscrete(np.ones(config.ACTION_DIM))
                              for agent in self.agents}
        super().__init__()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> gym.spaces.Space:
        return self.action_spaces[agent]

    def check_dead(self):
        num_alive = 0
        for key, player in self.PLAYERS.items():
            if player:
                if player.health <= 0:
                    self.NUM_DEAD += 1
                    self.game_round.NUM_DEAD = self.NUM_DEAD
                    self.pool_obj.return_hero(player)
                    self.kill_list.append(key)
                else:
                    num_alive += 1
        return num_alive

    def observe(self, player_id):
        return self.observations[player_id]

    def reset(self, seed=None, options=None):
        self.pool_obj = pool.pool()
        self.PLAYERS = {"player_" + str(player_id): player_class(self.pool_obj, player_id)
                        for player_id in range(config.NUM_PLAYERS)}
        self.game_observations = {"player_" + str(player_id): Observation() for player_id in range(config.NUM_PLAYERS)}
        self.NUM_DEAD = 0
        self.previous_rewards = {"player_" + str(player_id): 0 for player_id in range(config.NUM_PLAYERS)}

        self.step_function = Step_Function(self.pool_obj, self.game_observations)
        self.game_round = Game_Round(self.PLAYERS, self.pool_obj, self.step_function)
        self.actions_taken = 0
        self.actions_taken_this_turn = 0
        self.game_round.play_game_round()
        for key, p in self.PLAYERS.items():
            self.step_function.generate_shop(key, p)
        self.step_function.generate_shop_vectors(self.PLAYERS)

        self.agents = self.possible_agents.copy()
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}

        self.infos = {agent: {"state_empty": False} for agent in self.agents}
        self.actions = {agent: {} for agent in self.agents}

        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}

        self.observations = {agent: self.game_observations[agent].observation(
            agent, self.PLAYERS[agent], self.PLAYERS[agent].action_vector) for agent in self.agents}

        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()

        super().__init__()
        return self.observations

    def render(self):
        ...

    def close(self):
        self.reset()

    def step(self, action):
        # step for dead agents
        if self.terminations[self.agent_selection]:
            self._was_dead_step(action)
            return
        action = np.asarray(action)
        if action.ndim == 0:
            self.step_function.action_controller(action, self.PLAYERS[self.agent_selection], self.PLAYERS,
                                                 self.agent_selection, self.game_observations)
        elif action.ndim == 1:
            self.step_function.batch_2d_controller(action, self.PLAYERS[self.agent_selection], self.PLAYERS,
                                                   self.agent_selection, self.game_observations)

        # Also called in many environments but the line above this does the same thing but better
        # self._accumulate_rewards()
        self._clear_rewards()
        if self._agent_selector.is_last():

            # if we don't use this line, rewards will compound per step
            # (e.g. if player 1 gets reward in step 1, he will get rewards in steps 2-8)

            self.infos[self.agent_selection] = {"state_empty": self.PLAYERS[self.agent_selection].state_empty()}

            self.terminations = {a: False for a in self.agents}
            self.truncations = {a: False for a in self.agents}

            self.actions_taken += 1

            if self.actions_taken < config.ACTIONS_PER_TURN:
                for agent in self.agents:
                    self.observations[agent] = self.game_observations[agent].observation(
                        agent, self.PLAYERS[agent], self.PLAYERS[agent].action_vector)

            # If at the end of the turn
            if self.actions_taken >= config.ACTIONS_PER_TURN:
                # Take a game action and reset actions taken
                self.actions_taken = 0
                self.game_round.play_game_round()

                # Check if the game is over
                if self.check_dead() <= 1 or self.game_round.current_round > 48:
                    # Anyone left alive (should only be 1 player unless time limit) wins the game
                    for player_id in self.agents:
                        if self.PLAYERS[player_id] and self.PLAYERS[player_id].health > 0:
                            self.PLAYERS[player_id].won_game()
                            self.rewards[player_id] = 70 + self.PLAYERS[player_id].reward
                            self._cumulative_rewards[player_id] = self.rewards[player_id]
                            self.PLAYERS[player_id] = None  # Without this the reward is reset

                    self.terminations = {a: True for a in self.agents}

                self.infos = {a: {"state_empty": False} for a in self.agents}

                _live_agents = self.agents[:]
                for k in self.kill_list:
                    self.terminations[k] = True
                    _live_agents.remove(k)
                    self.rewards[k] = (7 - len(_live_agents)) * 10 + self.PLAYERS[k].reward
                    self._cumulative_rewards[k] = self.rewards[k]
                    self.PLAYERS[k] = None
                    self.game_round.update_players(self.PLAYERS)

                if len(self.kill_list) > 0:
                    self._agent_selector.reinit(_live_agents)
                self.kill_list = []

                if not all(self.terminations.values()):
                    self.game_round.start_round()

                    for agent in _live_agents:
                        self.observations[agent] = self.game_observations[agent].observation(
                            agent, self.PLAYERS[agent], self.PLAYERS[agent].action_vector)

            for player_id in self.PLAYERS:
                if self.PLAYERS[player_id]:
                    self.rewards[player_id] = self.PLAYERS[player_id].reward
                    self._cumulative_rewards[player_id] = self.rewards[player_id]

        # I think this if statement is needed in case all the agents die to the same minion round. a little sad.
        if len(self._agent_selector.agent_order):
            self.agent_selection = self._agent_selector.next()

        # Probably not needed but doesn't hurt?
        self._deads_step_first()
        # return self.observations, self.rewards, self.terminations, self.truncations, self.infos

import config
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from Simulator import pool
from Simulator.player import player as player_class
from Simulator.step_function import Step_Function
from Simulator.game_round import Game_Round
from Simulator.observation import Observation


class Single_Player_Game(gym.Env):

    def __init__(self):
        self.env = gym.make('TFTSet4-v0')
        self.env.reset()
        self.observation_space = spaces.MultiDiscrete([config.OBSERVATION_SIZE for _ in range(8)])
        self.action_space = spaces.Discrete(10+5+9+10+7+4+7+4)

    def reset(self, seed=None, options=None):
        return self.env.reset()

    def render(self):
        ...

    def step(self, action):
        random_actions = action = np.random.rand(7,10+5+9+10+7+4+7+4)
        eight_players_actions = np.concatenate((action, random_actions), axis=0)
        eight_players_actions = eight_players_actions > 0.5
        observation, reward, terminated, truncated, info = self.env.step(eight_players_actions)
        return observation, reward[0], terminated, truncated, info

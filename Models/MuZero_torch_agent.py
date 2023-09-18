import torch
import config
import collections
import numpy as np
import time
import os
import MCTS
from Models.MuZero_torch_model import MuZeroNetwork
class MuZeroAgent:
    def __init__(self, action_size, obs_size, simulations):
        self.action_size = action_size
        self.obs_size = obs_size
        self.simulations = simulations
        self.mcts = MCTS.MCTS(sample_size=80, action_size=self.action_size)
        self.model = MuZeroNetwork()

    def select_action(self, observation, mask):
        return self.mcts.generate_action(self.simulations, observation=observation)

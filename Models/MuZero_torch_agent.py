import torch
import config
import collections
import numpy as np
import time
import os
import MCTS
from Models.MuZero_torch_model import MuZeroNetwork
class MuZeroAgent:
    def __init__(self, action_size, obs_size, simulations, shared_information):
        self.action_size = action_size
        self.obs_size = obs_size
        self.simulations = simulations
        self.mcts = MCTS.MCTS(sample_size=80, action_size=self.action_size)
        self.replay_buffer = shared_information[0]
        self.shared_weights = shared_information[1]
        self.model = MuZeroNetwork()

    def select_action(self, observation, mask, reward):
        return self.mcts.generate_action(self.simulations, observation=observation)

import torch
import config
import collections
import numpy as np
import time
import os
import MCTS
from Models.MuZero_torch_model import MuZeroNetwork
from Models.replay_muzero_buffer import ReplayBuffer
class MuZeroAgent:
    def __init__(self, action_size, action_limits, obs_size, simulations, weights, global_buffer):
        self.action_size = action_size
        self.obs_size = obs_size
        self.simulations = simulations
        self.action_limits = action_limits
        self.mcts = MCTS.MCTS(sample_size=80, action_size=self.action_size, action_limits=self.action_limits, policy_size=1000)
        self.replay_buffer = ReplayBuffer(global_buffer)
        self.shared_weights = weights
        self.model = MuZeroNetwork()

    def select_action(self, observation, mask, reward, terminated):
        if all(terminated):
            self.replay_buffer.move_buffer_to_global()
        action, policy = self.mcts.generate_action(self.simulations, observation=observation)
        for n in range(len(observation)):
            self.replay_buffer.store_step(observation[n], action[n], reward[n], policy[n])
        return action

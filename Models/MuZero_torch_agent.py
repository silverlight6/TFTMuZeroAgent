import ray
import torch
from Models.MCTS_torch import MCTS
from Simulator.utils import hp_from_obs, round_from_obs, t_f_c_from_obs
import config
import collections
import numpy as np
import time
import os
from Models.MuZero_torch_model import MuZeroNetwork
from Models.replay_muzero_buffer import ReplayBuffer

class MuZeroAgent:
    def __init__(self, action_size, action_limits, obs_size, simulations, global_buffer, weights=None):
        self.action_size = action_size
        self.obs_size = obs_size
        self.simulations = simulations
        self.action_limits = action_limits
        self.global_buffer = global_buffer
        self.shared_weights = weights
        self.model = MuZeroNetwork()
        self.mcts = MCTS(sample_size=80, action_size=self.action_size, action_limits=self.action_limits, policy_size=1000, network=self.model)
        if weights is not None:
            self.model.load_state_dict(weights)
        self.model.to('cuda')
        self.hp = 100
        self.replay_buffers = []

    def select_action(self, observation, mask, reward, terminated):
        while len(self.replay_buffers) < observation.shape[0]:
            self.replay_buffers.append(ReplayBuffer(self.global_buffer))
        action, policy = self.mcts.generate_action(self.simulations, observation=observation, mask=mask)
        if np.any(terminated):
            for n in range(len(terminated)):
                if terminated[n]:
                    self.replay_buffers[n].store_step(observation[n], action[n], reward[n], policy[n])
                    self.replay_buffers[n].move_buffer_to_global()
                    # print(f'Muzero {n} ended with reward {reward[n]}')
        for n in range(len(observation)):
            if not terminated[n]:
                self.replay_buffers[n].store_step(observation[n], action[n], reward[n], policy[n])
        turns_left = t_f_c_from_obs(observation[0])
        round = round_from_obs(observation[0])
        if turns_left == config.ACTIONS_PER_TURN and round > 1:
            hp = hp_from_obs(observation[0])
            self.global_buffer.store_combat.remote([observation[n], hp >= self.hp])
        action = np.random.randint(self.action_limits, size=(observation.shape[0], self.action_size))
        return action
    
    def get_weights(self):
        return self.model.get_weights()
    
class BaseMuZeroAgent(MuZeroAgent):
    pass
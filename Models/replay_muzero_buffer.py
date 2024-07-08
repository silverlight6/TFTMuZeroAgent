import numpy as np
import config
import random
from global_buffer import GlobalBuffer
from Models.MCTS_Util import split_sample_set
import ray


class ReplayBuffer:
    def __init__(self, g_buffer: GlobalBuffer):
        self.observations = []
        self.rewards = []
        self.policys = []
        self.actions = []
        self.g_buffer = g_buffer

    def reset(self):
        self.observations = []
        self.rewards = []
        self.policys = []
        self.actions = []

    def store_step(self, observation, action, reward, policy):
        # Records a single step of gameplay experience
        # First few are self-explanatory
        # done is boolean if game is done after taking said action
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.policys.append(policy)

    def get_reward_sequence(self):
        return self.rewards
    
    def set_reward_sequence(self, rewards):
        self.rewards = rewards

    def get_len(self):
        return len(self.observations)

    def move_buffer_to_global(self):
        replay_set = []

        for current_start in range(config.UNROLL_STEPS, len(self.observations)):
            value = float(self.rewards[-1])
            replay_set.append([self.observations[current_start-config.UNROLL_STEPS],
                               self.actions[current_start-config.UNROLL_STEPS:current_start],
                               [value] * config.UNROLL_STEPS,
                            #    self.rewards[current_start-config.UNROLL_STEPS:current_start],
                               [value] * config.UNROLL_STEPS, # TODO Check if this is actually used
                               self.policys[current_start-config.UNROLL_STEPS:current_start]])
        ray.get(self.g_buffer.store_episode.remote(replay_set)) #TODO MAKE THIS SO IT ONLY CALLS GLOBAL ONCE

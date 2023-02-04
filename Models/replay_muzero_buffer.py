import numpy as np
import config
import random
from global_buffer import GlobalBuffer
import Simulator.utils as utils

class ReplayBuffer:
    def __init__(self, g_buffer: GlobalBuffer):
        self.gameplay_experiences = []
        self.rewards = []
        self.policy_distributions = []
        self.observation_history = []
        self.action_history = []
        self.action_mask = []
        self.g_buffer = g_buffer

    def store_replay_buffer(self, observation, action, reward, policy):
        # Records a single step of gameplay experience
        # First few are self-explanatory
        # done is boolean if game is done after taking said action
        self.gameplay_experiences.append(observation)
        self.action_history.append(action)
        self.rewards.append(reward)
        self.action_mask.append(utils.generate_masking(action))
        self.policy_distributions.append(policy)

    def store_observation(self, observation):
        self.observation_history.append(observation)

    def len_observation_buffer(self):
        return len(self.observation_history)

    def get_prev_observation(self, i):
        # take the sample from i num from the end of list
        return self.observation_history[i * -1]

    def get_prev_action(self):
        if self.action_history:
            return self.action_history[-1]
        else:
            return 9

    def get_reward_sequence(self):
        return self.rewards
    
    def set_reward_sequence(self, rewards):
        self.rewards = rewards

    def store_global_buffer(self):
        # Putting this if case here in case the episode length is less than 72 which is 8 more than the batch size
        # In general, we are having episodes of 200 or so but the minimum possible is close to 20
        # samples_per_player = config.SAMPLES_PER_PLAYER \
        #     if (len(self.gameplay_experiences) - config.UNROLL_STEPS) > config.SAMPLES_PER_PLAYER \
        #     else len(self.gameplay_experiences) - config.UNROLL_STEPS
        # config.UNROLL_STEPS because I don't want to sample the very end of the range
        # samples = random.sample(range(0, len(self.gameplay_experiences) - config.UNROLL_STEPS), samples_per_player)
        num_steps = len(self.gameplay_experiences)

        action_set = []
        value_mask_set = []
        reward_mask_set = []
        policy_mask_set = []
        value_set = []
        reward_set = []
        policy_set = []
        action_mask_set = []

        for current_index in range(num_steps):
            #### POSSIBLE EXTENSION -- set up value_approximation storing
            # value = value_approximations[bootstrap_index] * discount**td_steps
            value = self.rewards[-1] * (config.DISCOUNT ** (num_steps - current_index))

            # reward_mask = 1.0 if current_index > sample else 0.0
            reward_mask = 1.0
            action_set.append(np.asarray(self.action_history[current_index].copy()))
            action_mask_set.append(np.asarray(self.action_mask[current_index].copy()))
            value_mask_set.append(1.0)
            reward_mask_set.append(reward_mask)
            policy_mask_set.append(1.0)
            value_set.append(value)
            # This is current_index - 1 in the Google's code but in my version
            # This is simply current_index since I store the reward with the same time stamp
            reward_set.append(self.rewards[current_index])
            policy_set.append(self.policy_distributions[current_index].copy())
        # print(value_set)
        sample_set = [self.gameplay_experiences.copy(), action_set, value_mask_set, reward_mask_set,
                        policy_mask_set, value_set, reward_set, policy_set, action_mask_set]
        self.g_buffer.store_replay_sequence.remote(sample_set)

        self.gameplay_experiences = []
        self.rewards = []
        self.policy_distributions = []
        self.observation_history = []
        self.action_history = []
        self.action_mask = []

    

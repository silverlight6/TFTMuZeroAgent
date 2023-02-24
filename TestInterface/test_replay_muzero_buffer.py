import numpy as np
import config
import random
from global_buffer import GlobalBuffer


class ReplayBuffer:
    def __init__(self, g_buffer: GlobalBuffer):
        self.gameplay_experiences = []
        self.rewards = []
        self.policy_distributions = []
        self.action_history = []
        self.g_buffer = g_buffer

    def store_replay_buffer(self, observation, action, reward, policy):
        # Records a single step of gameplay experience
        # First few are self-explanatory
        # done is boolean if game is done after taking said action
        self.gameplay_experiences.append(observation)
        self.action_history.append(action)
        self.rewards.append(reward)
        self.policy_distributions.append(policy)

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
        samples_per_player = config.SAMPLES_PER_PLAYER \
            if (len(self.gameplay_experiences) - config.UNROLL_STEPS) > config.SAMPLES_PER_PLAYER \
            else len(self.gameplay_experiences) - config.UNROLL_STEPS
        if samples_per_player > 0:
            # config.UNROLL_STEPS because I don't want to sample the very end of the range
            samples = random.sample(range(0, len(self.gameplay_experiences) - config.UNROLL_STEPS), samples_per_player)
            num_steps = len(self.gameplay_experiences)
            reward_correction = []
            prev_reward = 0
            for reward in self.rewards:
                reward_correction.append(reward - prev_reward)
                prev_reward = reward
            for sample in samples:
                # Hard coding because I would be required to do a transpose if I didn't
                # and that takes a lot of time.
                action_set = []
                value_mask_set = []
                reward_mask_set = []
                policy_mask_set = []
                value_set = []
                reward_set = []
                policy_set = []

                for current_index in range(sample, sample + config.UNROLL_STEPS + 1):
                    #### POSSIBLE EXTENSION -- set up value_approximation storing
                    # value = value_approximations[bootstrap_index] * discount**td_steps
                    value = 0.0

                    for i, reward in enumerate(reward_correction[current_index:]):
                        value += reward * config.DISCOUNT ** i

                    reward_mask = 1.0 if current_index > sample else 0.0
                    if current_index < num_steps - 1:
                        if current_index != sample:
                            action_set.append(np.asarray(self.action_history[current_index]))
                        else:
                            # To weed this out later when sampling the global buffer
                            action_set.append([0, 0, 0])
                        value_mask_set.append(1.0)
                        reward_mask_set.append(reward_mask)
                        policy_mask_set.append(1.0)
                        value_set.append(value)
                        # This is current_index - 1 in the Google's code but in my version
                        # This is simply current_index since I store the reward with the same time stamp
                        reward_set.append(reward_correction[current_index])
                        policy_set.append(self.policy_distributions[current_index])
                    elif current_index == num_steps - 1:
                        action_set.append([0, 0, 0])
                        value_mask_set.append(1.0)
                        reward_mask_set.append(reward_mask)
                        policy_mask_set.append(0.0)
                        # This is 0.0 in Google's code but thinking this should be the same as the reward?
                        # The value of the terminal state should equal
                        # the value of the cumulative reward at the given state.
                        value_set.append(0.0)
                        reward_set.append(reward_correction[current_index])
                        # 0 is ok here because this get masked out anyway
                        policy_set.append(self.policy_distributions[0])
                    else:
                        # States past the end of games is treated as absorbing states.
                        action_set.append([0, 0, 0])
                        value_mask_set.append(1.0)
                        reward_mask_set.append(0.0)
                        policy_mask_set.append(0.0)
                        value_set.append(0.0)
                        reward_set.append(0.0)
                        policy_set.append(self.policy_distributions[0])
                sample_set = [self.gameplay_experiences[sample], action_set, value_mask_set, reward_mask_set,
                              policy_mask_set, value_set, reward_set, policy_set]
                self.g_buffer.store_replay_sequence(sample_set)

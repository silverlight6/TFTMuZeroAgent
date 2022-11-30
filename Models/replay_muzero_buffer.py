import numpy as np
import config
import random
from global_buffer import GlobalBuffer


class ReplayBuffer:
    def __init__(self, g_buffer: GlobalBuffer):
        self.gameplay_experiences = []
        self.rewards = []
        self.policy_distributions = []
        self.observation_history = []
        self.action_history = []
        self.g_buffer = g_buffer

    def store_replay_buffer(self, observation, action, reward, policy):
        # Records a single step of gameplay experience
        # First few are self-explanatory
        # done is boolean if game is done after taking said action
        # commenting out because MuZero supports a wide range of rewards, not just -1 to 1
        # reward = np.clip(reward, -1.0, 1.0)
        self.gameplay_experiences.append(observation)
        self.action_history.append(action)
        self.rewards.append(reward)
        self.policy_distributions.append(policy)

    def store_observation(self, observation):
        self.observation_history.append(observation)

    def len_observation_buffer(self):
        return len(self.observation_history)

    def get_prev_observation(self, i):
        # take the sample from i num from the end of list
        return self.observation_history[i * -1]

    def get_observation_shape(self):
        # Hard coding this because the need to know this value before any observation are
        # Generated in the case of no successful actions completed yet in the game which
        # Is very likely at the start of the game.
        return config.OBSERVATION_SHAPE

    def store_global_buffer(self):
        # Putting this if case here in case the episode length is less than 72 which is 8 more than the batch size
        # In general, we are having episodes of 200 or so but the minimum possible is close to 20
        samples_per_player = config.SAMPLES_PER_PLAYER \
            if (len(self.gameplay_experiences) - 8) > config.SAMPLES_PER_PLAYER else len(self.gameplay_experiences) - 8
        if samples_per_player > 0:
            # 8 because I don't want to sample the very end of the range
            samples = random.sample(range(0, len(self.gameplay_experiences) - 8), samples_per_player)
            td_steps = config.UNROLL_STEPS
            num_steps = len(self.gameplay_experiences)
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
                    bootstrap_index = current_index + td_steps
                    #### POSSIBLE EXTENSION -- set up value_approximation storing
                    # value = value_approximations[bootstrap_index] * discount**td_steps
                    value = 0.0

                    for i, reward in enumerate(self.rewards[current_index:bootstrap_index]):
                        value += reward * config.DISCOUNT ** i

                    reward_mask = 1.0 if current_index > sample else 0.0
                    if current_index < num_steps - 1:
                        if current_index != sample:
                            action_set.append(np.asarray(self.action_history[current_index]))
                        else:
                            # To weed this out later when sampling the global buffer
                            action_set.append([0, 0, 0, 0, 0])
                        value_mask_set.append(1.0)
                        reward_mask_set.append(reward_mask)
                        policy_mask_set.append(1.0)
                        value_set.append(value)
                        # This is current_index - 1 in the Google's code but in my version
                        # This is simply current_index since I store the reward with the same time stamp
                        reward_set.append(self.rewards[current_index])
                        policy_set.append(self.policy_distributions[current_index])
                    elif current_index == num_steps - 1:
                        action_set.append([7, 0, 0, 0, 0])
                        value_mask_set.append(1.0)
                        reward_mask_set.append(reward_mask)
                        policy_mask_set.append(0.0)
                        # This is 0.0 in Google's code but thinking this should be the same as the reward?
                        # The value of the terminal state should equal
                        # the value of the cumulative reward at the given state.
                        value_set.append(0.0)
                        reward_set.append(self.rewards[current_index])
                        # 0 is ok here because this get masked out anyway
                        policy_set.append(self.policy_distributions[0])
                    else:
                        # States past the end of games is treated as absorbing states.
                        action_set.append([0, 0, 0, 0, 0])
                        value_mask_set.append(1.0)
                        reward_mask_set.append(0.0)
                        policy_mask_set.append(0.0)
                        value_set.append(0.0)
                        reward_set.append(0.0)
                        policy_set.append(self.policy_distributions[0])
                sample_set = [self.gameplay_experiences[sample], action_set, value_mask_set, reward_mask_set,
                              policy_mask_set, value_set, reward_set, policy_set]
                self.g_buffer.store_replay_sequence(sample_set)

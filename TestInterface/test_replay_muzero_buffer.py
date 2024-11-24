import numpy as np
import config
from TestInterface.test_global_buffer import GlobalBuffer
from Models.MCTS_Util import split_sample_decide


class ReplayBuffer:
    def __init__(self, g_buffer: GlobalBuffer, key: str):
        self.gameplay_experiences = []
        self.rewards = []
        self.policy_distributions = []
        self.string_samples = []
        self.action_history = []
        self.root_values = []
        self.team_tiers = []
        self.team_champions = []
        self.g_buffer = g_buffer
        self.key = key
        self.ending_position = -1

    def store_replay_buffer(self, observation, action, reward, policy, string_samples,
                            root_value, team_tiers, team_champions):
        # Records a single step of gameplay experience
        # First few are self-explanatory
        # done is boolean if game is done after taking said action
        self.gameplay_experiences.append(observation)
        self.action_history.append(action)
        np.clip(reward, config.MINIMUM_REWARD, config.MAXIMUM_REWARD)
        self.rewards.append(reward)
        self.policy_distributions.append(policy)
        self.string_samples.append(string_samples)
        self.root_values.append(root_value)
        self.team_tiers.append(team_tiers)
        self.team_champions.append(team_champions)

    def store_gumbel_buffer(self, observation, action, reward, policy, root_value):
        self.gameplay_experiences.append(observation)
        self.action_history.append(action)
        np.clip(reward, config.MINIMUM_REWARD, config.MAXIMUM_REWARD)
        self.rewards.append(reward)
        self.policy_distributions.append(policy)
        self.root_values.append(root_value)

    def reset(self):
        self.gameplay_experiences = []
        self.rewards = []
        self.policy_distributions = []
        self.string_samples = []
        self.action_history = []
        self.root_values = []
        self.team_tiers = []
        self.team_champions = []

    def get_prev_action(self):
        if self.action_history:
            return self.action_history[-1]
        else:
            return 9

    def get_reward_sequence(self):
        return self.rewards

    def set_reward_sequence(self, rewards):
        self.rewards = rewards

    def get_ending_position(self):
        return self.ending_position

    def set_ending_position(self, ending_position):
        self.ending_position = ending_position

    def store_global_buffer(self):
        # Putting this if case here in case the episode length is less than 72 which is 8 more than the batch size
        # In general, we are having episodes of 200 or so but the minimum possible is close to 20
        samples_per_player = config.SAMPLES_PER_PLAYER \
            if (len(self.gameplay_experiences) - config.UNROLL_STEPS) > config.SAMPLES_PER_PLAYER \
            else len(self.gameplay_experiences) - config.UNROLL_STEPS
        # if samples_per_player > 0 and (self.ending_position > 6 or self.ending_position < 3):
        if samples_per_player > 0:
            # config.UNROLL_STEPS because I don't want to sample the very end of the range
            # The other way says we can sample the end of the array to make sure we recognize when we are at the end
            # samples = random.sample(range(0, len(self.gameplay_experiences) -
            #   config.UNROLL_STEPS), samples_per_player)
            # samples = range(0, len(self.gameplay_experiences) - config.UNROLL_STEPS)
            samples = range(0, len(self.gameplay_experiences))
            num_steps = len(self.gameplay_experiences)
            reward_correction = []
            prev_reward = 0
            output_sample_set = []
            for reward in self.rewards:
                reward_correction.append(reward - prev_reward)  # Getting instant rewards not cumulative
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
                sample_set = []
                priority_set = []
                tier_set = []
                final_tier_set = []
                champion_set = []

                for current_index in range(sample, sample + config.UNROLL_STEPS + 1):
                    if config.TD_STEPS > 0:
                        bootstrap_index = current_index + config.TD_STEPS
                    else:
                        bootstrap_index = len(reward_correction)
                    if config.TD_STEPS > 0 and bootstrap_index < len(self.root_values):
                        value = self.root_values[bootstrap_index] * config.DISCOUNT ** config.TD_STEPS
                    else:
                        value = 0.0
                    # bootstrapping value back from rewards
                    for i, reward_corrected in enumerate(reward_correction[current_index:bootstrap_index]):
                        value += reward_corrected * config.DISCOUNT ** i

                    priority = 0.001
                    if current_index < num_steps:
                        priority = np.maximum(priority, np.abs(self.root_values[current_index] - value))
                    priority_set.append(priority)

                    reward_mask = 1.0 if current_index > sample else 0.0
                    if current_index < num_steps - 1:
                        if current_index != sample:
                            action_set.append(np.asarray(self.action_history[current_index]))
                        else:
                            if config.CHAMP_DECIDER:
                                action_set.append([0 for _ in range(len(config.CHAMP_DECIDER_ACTION_DIM))])
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
                        sample_set.append(self.string_samples[current_index])
                        tier_set.append(self.team_tiers[current_index])
                        final_tier_set.append(self.team_tiers[-1])
                        champion_set.append(self.team_champions[current_index])
                    elif current_index == num_steps - 1:
                        if config.CHAMP_DECIDER:
                            action_set.append([0 for _ in range(len(config.CHAMP_DECIDER_ACTION_DIM))])
                        else:
                            # To weed this out later when sampling the global buffer
                            action_set.append([0, 0, 0])
                        value_mask_set.append(0.0)
                        reward_mask_set.append(reward_mask)
                        policy_mask_set.append(0.0)
                        # This is 0.0 in Google's code but thinking this should be the same as the reward?
                        # The value of the terminal state should equal
                        # the value of the cumulative reward at the given state.
                        value_set.append(0.0)
                        reward_set.append(reward_correction[current_index])
                        # 0 is ok here because this get masked out anyway
                        policy_set.append(self.policy_distributions[0])
                        sample_set.append(self.string_samples[0])
                        tier_set.append(self.team_tiers[-1])
                        final_tier_set.append(self.team_tiers[-1])
                        champion_set.append(self.team_champions[-1])
                    else:
                        # States past the end of games is treated as absorbing states.
                        if config.CHAMP_DECIDER:
                            action_set.append([0 for _ in range(len(config.CHAMP_DECIDER_ACTION_DIM))])
                        else:
                            # To weed this out later when sampling the global buffer
                            action_set.append([0, 0, 0])
                        # I'm pretty sure this should be 0 and not 1.
                        value_mask_set.append(0.0)
                        reward_mask_set.append(0.0)
                        policy_mask_set.append(0.0)
                        value_set.append(0.0)
                        reward_set.append(0.0)
                        policy_set.append(self.policy_distributions[0])
                        sample_set.append(self.string_samples[0])
                        tier_set.append(self.team_tiers[-1])
                        final_tier_set.append(self.team_tiers[-1])
                        champion_set.append(self.team_champions[-1])

                for i in range(len(sample_set)):
                    split_mapping, split_policy = split_sample_decide(sample_set[i], policy_set[i])
                    sample_set[i] = split_mapping
                    policy_set[i] = split_policy

                # formula for priority over unroll steps,
                # adding small randomness to get around a priority queue error where it crashes if you add two items
                # with identical priorities
                priority = priority_set[0]
                div = -priority
                for i in priority_set:
                    div += i
                priority = 1 / (priority / div) + np.random.rand() * 0.00001

                # priority = 1 / priority because priority queue stores in ascending order.
                output_sample_set.append([priority, [self.gameplay_experiences[sample], action_set, value_mask_set,
                                                     reward_mask_set, policy_mask_set, value_set, reward_set,
                                                     policy_set, sample_set, tier_set, final_tier_set, champion_set]])
            self.g_buffer.store_replay_sequence([output_sample_set, self.ending_position])

    def store_global_position_buffer(self):
        # Putting this if case here in case the episode length is less than 72 which is 8 more than the batch size
        # In general, we are having episodes of 200 or so but the minimum possible is close to 20

        samples_per_player = config.SAMPLES_PER_PLAYER \
            if (len(self.gameplay_experiences) - config.UNROLL_STEPS) > config.SAMPLES_PER_PLAYER \
            else len(self.gameplay_experiences) - config.UNROLL_STEPS + 1
        # if samples_per_player > 0 and (self.ending_position > 6 or self.ending_position < 3):
        if samples_per_player > 0:
            # config.UNROLL_STEPS because I don't want to sample the very end of the range
            # The other way says we can sample the end of the array to make sure we recognize when we are at the end
            # samples = random.sample(range(0, len(self.gameplay_experiences) -
            #   config.UNROLL_STEPS), samples_per_player)
            # samples = range(0, len(self.gameplay_experiences) - config.UNROLL_STEPS)
            samples = range(0, len(self.gameplay_experiences))
            num_steps = len(self.gameplay_experiences)
            reward_correction = []
            prev_reward = 0
            output_sample_set = []
            for reward in self.rewards:
                reward_correction.append(reward - prev_reward)  # Getting instant rewards not cumulative
                prev_reward = reward
            for sample in samples:
                # Hard coding because I would be required to do a transpose if I didn't
                # and that takes a lot of time.
                action_set = []
                value_mask_set = []
                policy_mask_set = []
                value_set = []
                policy_set = []

                for current_index in range(sample, sample + config.UNROLL_STEPS + 1):
                    if config.TD_STEPS > 0:
                        bootstrap_index = current_index + config.TD_STEPS
                    else:
                        bootstrap_index = len(reward_correction)
                    if config.TD_STEPS > 0 and bootstrap_index < len(self.root_values):
                        value = self.root_values[bootstrap_index] * config.DISCOUNT ** config.TD_STEPS
                    else:
                        value = 0.0
                    # bootstrapping value back from rewards
                    for i, reward_corrected in enumerate(reward_correction[current_index:bootstrap_index]):
                        value += reward_corrected * config.DISCOUNT ** i

                    if current_index < num_steps - 1:
                        action_set.append(self.action_history[current_index])
                        policy_mask_set.append(1.0)
                        value_set.append(value)
                        # This is current_index - 1 in the Google's code but in my version
                        # This is simply current_index since I store the reward with the same time stamp
                        policy_set.append(self.policy_distributions[current_index])
                        value_mask_set.append(1.0)
                    elif current_index == num_steps - 1:
                        action_set.append(28)

                        policy_mask_set.append(0.0)
                        # This is 0.0 in Google's code but thinking this should be the same as the reward?
                        # The value of the terminal state should equal
                        # the value of the cumulative reward at the given state.
                        value_set.append(0.0)
                        # 0 is ok here because this get masked out anyway
                        policy_set.append(self.policy_distributions[0])
                        value_mask_set.append(0.0)
                    else:
                        # States past the end of games is treated as absorbing states.
                        action_set.append(28)
                        # I'm pretty sure this should be 0 and not 1.

                        policy_mask_set.append(0.0)
                        value_set.append(0.0)
                        policy_set.append(self.policy_distributions[0])
                        value_mask_set.append(0.0)

                output_sample_set.append([self.gameplay_experiences[sample], action_set, value_mask_set,
                                                     policy_mask_set, value_set, policy_set])
            self.g_buffer.store_replay_sequence([output_sample_set, self.ending_position])


import numpy as np
import random

import config


class ReplayBuffer:
    def __init__(self, g_buffer):
        self.gameplay_experiences = []
        self.rewards = []
        self.policy_distributions = []
        self.observation_history = []
        self.action_history = []
        self.prev_actions = []
        self.zero_rewards = []
        self.non_zero_rewards = []
        self.g_buffer = g_buffer
        self.prev_action = np.asarray([0, 0, 0, 0, 0, 0, 0, 0])

    def sample_sequence(self, sequence_size):
        # -1 for the case if start pos is the terminated frame.
        # (Then +1 not to start from terminated frame.)
        start_pos = np.random.randint(
            0, len(self.gameplay_experiences) - sequence_size - 1
        )
        state_batch, logit_batch, action_batch, reward_batch = [], [], [], []

        for i in range(sequence_size):
            frame = self.gameplay_experiences[start_pos + i]
            if len(frame) == 4:
                state_batch.append(frame[0])
                logit_batch.append(frame[1])
                action_batch.append(frame[2])
                reward_batch.append(frame[3])
            else:
                print("I have an issue, len != 3")
            if start_pos + i >= len(self.gameplay_experiences):
                break

        return [
            state_batch,
            logit_batch,
            np.swapaxes(np.array(action_batch), 1, 0),
            np.array(reward_batch),
        ]

    def sample_gameplay_batch(self):
        # samples a batch of gameplay experience for training.
        # Returns: a list of gameplay experiences.

        state_batch, logit_batch, action_batch, reward_batch = [], [], [], []
        for i in range(len(self.gameplay_experiences) - 1, -1, -1):
            if len(self.gameplay_experiences[i]) != 4:
                self.gameplay_experiences.remove(self.gameplay_experiences[i])

        for gameplay_experience in list(self.gameplay_experiences):
            state_batch.append(gameplay_experience[0])
            logit_batch.append(gameplay_experience[1])
            action_batch.append(gameplay_experience[2])
            reward_batch.append(gameplay_experience[3])
        return [
            state_batch,
            logit_batch,
            np.swapaxes(np.array(action_batch), 1, 0),
            np.array(reward_batch),
        ]

    def store_replay_buffer(self, observation, action, reward, policy):
        # Records a single step of gameplay experience
        # First few are self-explanatory
        # done is boolean if game is done after taking said action
        reward = np.clip(reward, -1.0, 1.0)
        # self.gameplay_experiences.append([observation, logits, action, reward])
        self.gameplay_experiences.append(observation)
        self.action_history.append(action)
        self.rewards.append(reward)
        self.policy_distributions.append(policy)
        if len(self.action_history) > 1:
            self.prev_actions.append(self.action_history[-2])
        else:
            self.prev_actions.append(0)

    # Currently not using reward prediction due to the rich nature of my environment.
    def sample_rp_sequence(self):
        """
        Sample 4 successive frames for reward prediction.
        """
        if np.random.randint(2) == 0:
            from_zero = True
        else:
            from_zero = False

        if len(self.zero_rewards) == 0:
            # zero rewards container was empty
            from_zero = False
        elif len(self.non_zero_rewards) == 0:
            # non zero rewards container was empty
            from_zero = True

        if from_zero:
            index = np.random.randint(len(self.zero_rewards))
            end_frame_index = self.non_zero_rewards[index]
        else:
            index = np.random.randint(len(self.non_zero_rewards))
            end_frame_index = self.non_zero_rewards[index]

        start_frame_index = end_frame_index - 3
        # This does not feel required
        raw_start_frame_index = start_frame_index - len(self.gameplay_experiences)

        sample_state, sample_logit, sample_action, sample_reward = [], [], [], []

        for i in range(4):
            frame = self.gameplay_experiences[raw_start_frame_index + i]
            sample_state.append(frame[0])
            sample_logit.append(frame[1])
            sample_action.append(frame[2])
            sample_reward.append(frame[3])

        return [
            sample_state,
            sample_logit,
            np.swapaxes(np.array(sample_action), 0, 1),
            np.array(sample_reward),
        ]

    def store_global_buffer(self):
        # Putting this if case here in case the episode length is less than 72 which is 8 more than the batch size
        # In general, we are having episodes of 200 or so but the minimum possible is close to 20
        samples_per_player = (
            config.SAMPLES_PER_PLAYER
            if (len(self.gameplay_experiences) - config.UNROLL_STEPS)
            > config.SAMPLES_PER_PLAYER
            else len(self.gameplay_experiences) - config.UNROLL_STEPS
        )

        if samples_per_player > 0:
            # config.UNROLL_STEPS because I don't want to sample the very end of teh range
            samples = random.sample(
                range(0, len(self.gameplay_experiences) - config.UNROLL_STEPS),
                samples_per_player,
            )
            num_steps = len(self.gameplay_experiences)
            for sample in samples:
                action_set = []
                prev_action_set = []
                value_mask_set = []
                reward_mask_set = []
                policy_mask_set = []
                value_set = []
                reward_set = []
                policy_set = []

                for current_index in range(sample, sample + config.UNROLL_STEPS + 1):
                    value = 0.0

                    for i, reward in enumerate(self.rewards[current_index:]):
                        value += reward * config.DISCOUNT**i

                    reward_mask = 1.0 if current_index > sample else 0.0

                    if current_index < num_steps - 1:
                        if current_index != sample:
                            action_set.append(
                                np.asarray(self.action_history[current_index])
                            )
                        else:
                            action_set.append([0])
                        value_mask_set.append(1.0)
                        reward_mask_set.append(reward_mask)
                        policy_mask_set.append(1.0)
                        value_set.append(value)
                        reward_set.append(self.rewards[current_index])

                        policy_set.append(self.policy_distributions[current_index])
                    elif current_index == num_steps - 1:
                        action_set.append(9)
                        value_mask_set.append(1.0)
                        reward_mask_set.append(reward_mask)
                        policy_mask_set.append(0.0)
                        value_set.append(0.0)
                        reward_set.append(self.rewards[current_index])
                        policy_set.append(self.policy_distributions[0])
                    else:
                        # States past the end of games is treated as absorbing states.
                        action_set.append(0)
                        value_mask_set.append(1.0)
                        reward_mask_set.append(0.0)
                        policy_mask_set.append(0.0)
                        value_set.append(0.0)
                        reward_set.append(0.0)
                        policy_set.append(self.policy_distribtuions[0])
                prev_action_set.append(self.prev_actions[current_index])
                sample_set = [
                    self.gameplay_experiences[sample],
                    action_set,
                    value_mask_set,
                    reward_mask_set,
                    policy_mask_set,
                    value_set,
                    reward_set,
                    policy_set,
                    prev_action_set,
                ]
                self.g_buffer.store_replay_sequence.remote(sample_set)

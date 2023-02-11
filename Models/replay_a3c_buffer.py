import numpy as np
import random

import config


class ReplayBuffer:
    def __init__(self, g_buffer):
        self.gameplay_experiences = []
        self.zero_rewards = []
        self.non_zero_rewards = []
        self.g_buffer = g_buffer
        self.prev_action = np.asarray([0, 0, 0, 0, 0, 0, 0, 0])

    def sample_sequence(self, sequence_size):
        # -1 for the case if start pos is the terminated frame.
        # (Then +1 not to start from terminated frame.)
        start_pos = np.random.randint(0, len(self.gameplay_experiences) - sequence_size - 1)
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

        return [state_batch, logit_batch, np.swapaxes(np.array(action_batch), 1, 0), np.array(reward_batch)]

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
        return [state_batch, logit_batch, np.swapaxes(np.array(action_batch), 1, 0), np.array(reward_batch)]

    def store_replay_buffer(self, state, action, reward, logits):
        # Records a single step of gameplay experience
        # First few are self-explanatory
        # done is boolean if game is done after taking said action
        reward = np.clip(reward, -1.0, 1.0)
        self.gameplay_experiences.append([state, logits, action, reward])
        frame_index = len(self.gameplay_experiences)
        # if frame_index >= 3:
        #     if reward == 0:
        #         self.zero_rewards.append(frame_index)
        #     else:
        #         self.non_zero_rewards.append(frame_index)

        # TODO: Move this section to store_global_replay, adjust AI_Interface accordingly, sample randomly
        # Add experience to the global buffer
        if frame_index % config.A3C_SEQUENCE_LENGTH == 0 and frame_index > 0:
            # first create the batch
            state_batch, logit_batch, action_batch, reward_batch, p_action_batch = [], [], [], [], []
            for gameplay_experience in list(self.gameplay_experiences)[-config.A3C_SEQUENCE_LENGTH:]:
                state_batch.append(gameplay_experience[0])
                logit_batch.append(gameplay_experience[1])
                action_batch.append(gameplay_experience[2])
                reward_batch.append(gameplay_experience[3])
                p_action_batch.append(self.prev_action)
                self.prev_action = gameplay_experience[2]

            state_batch = np.asarray(state_batch)
            action_batch = np.asarray(action_batch)
            reward_batch = np.asarray(reward_batch)
            p_action_batch = np.asarray(p_action_batch)
            self.g_buffer.store_replay_sequence.remote([state_batch, logit_batch, action_batch,
                                                        reward_batch, p_action_batch])
        self.prev_action = action

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

        return [sample_state, sample_logit, np.swapaxes(np.array(sample_action), 0, 1), np.array(sample_reward)]
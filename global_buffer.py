from collections import deque
import numpy as np

import config


class GlobalBuffer:
    def __init__(self):
        self.gameplay_experiences = deque(maxlen=25000)
        self.batch_size = config.BATCH_SIZE

    def sample_batch(self):
        # Returns: a batch of gameplay experiences without regard to which agent.
        observation_batch, action_history_batch, target_value_batch, target_reward_batch = [], [], [], []
        target_policy_a_batch, value_mask_batch, reward_mask_batch, policy_mask_batch = [], [], [], []
        target_policy_b_batch, target_policy_c_batch, target_policy_d_batch, target_policy_e_batch = [], [], [], []

        for gameplay_experience in range(self.batch_size):
            observation, action_history, value_mask, reward_mask, policy_mask,\
                value, reward, policy = self.gameplay_experiences.popleft()
            observation_batch.append(observation)
            action_history_batch.append(action_history[1:])
            value_mask_batch.append(value_mask)
            reward_mask_batch.append(reward_mask)
            policy_mask_batch.append(policy_mask)
            target_value_batch.append(value)
            target_reward_batch.append(reward)
            # print(policy)
            pol_a, pol_b, pol_c, pol_d, pol_e = [], [], [], [], []
            for i in range(len(policy)):
                pol_a.append(policy[i][0][0].numpy())
                pol_b.append(policy[i][1][0].numpy())
                pol_c.append(policy[i][2][0].numpy())
                pol_d.append(policy[i][3][0].numpy())
                pol_e.append(policy[i][4][0].numpy())
            target_policy_a_batch.append(pol_a)
            target_policy_b_batch.append(pol_b)
            target_policy_c_batch.append(pol_c)
            target_policy_d_batch.append(pol_d)
            target_policy_e_batch.append(pol_e)

        observation_batch = np.squeeze(np.asarray(observation_batch))
        # print(action_history_batch)
        action_history_batch = np.swapaxes(np.asarray(action_history_batch), 1, 2)
        target_value_batch = np.asarray(target_value_batch).astype('float32')
        target_reward_batch = np.asarray(target_reward_batch).astype('float32')
        value_mask_batch = np.asarray(value_mask_batch).astype('float32')
        reward_mask_batch = np.asarray(reward_mask_batch).astype('float32')
        policy_mask_batch = np.asarray(policy_mask_batch).astype('float32')
        target_policy_a_batch = np.asarray(target_policy_a_batch).astype('float32')
        target_policy_b_batch = np.asarray(target_policy_b_batch).astype('float32')
        target_policy_c_batch = np.asarray(target_policy_c_batch).astype('float32')
        target_policy_d_batch = np.asarray(target_policy_d_batch).astype('float32')
        target_policy_e_batch = np.asarray(target_policy_e_batch).astype('float32')
        target_policy_batch = [target_policy_a_batch, target_policy_b_batch, target_policy_c_batch,
                               target_policy_d_batch, target_policy_e_batch]

        return [observation_batch, action_history_batch, value_mask_batch, reward_mask_batch, policy_mask_batch,
                target_value_batch, target_reward_batch, target_policy_batch]

    def store_replay_sequence(self, sample):
        # Records a single step of gameplay experience
        # First few are self-explanatory
        # done is boolean if game is done after taking said action
        self.gameplay_experiences.append(sample)

    def available_batch(self):
        queue_length = len(self.gameplay_experiences)
        if queue_length >= self.batch_size:
            return True
        return False

    # Leaving this transpose method here in case some model other than
    # MuZero requires this in the future.
    def transpose(self, matrix):
        rows = len(matrix)
        columns = len(matrix[0])

        matrix_T = []
        for j in range(columns):
            row = []
            for i in range(rows):
                row.append(matrix[i][j])
            matrix_T.append(row)

        return matrix_T

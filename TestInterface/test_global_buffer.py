import config
import numpy as np
from collections import deque

class GlobalBuffer:
    def __init__(self):
        self.gameplay_experiences = deque(maxlen=config.GLOBAL_BUFFER_SIZE)
        self.average_position = deque(maxlen=config.GLOBAL_BUFFER_SIZE)
        self.batch_size = config.BATCH_SIZE

    def sample_batch(self):
        """
        Prepares a batch for training. All preprocessing done here to avoid problems with data transfer between CPU
        and GPU causing slowdowns.

        Conditions:
            - There is enough data in the buffer to train on.

        Returns:
            A prepared batch ready for training.
        """
        obs_tensor_batch, action_history_batch, target_value_batch, policy_mask_batch = [], [], [], []
        target_reward_batch, target_policy_batch, value_mask_batch, reward_mask_batch = [], [], [], []
        sample_set_batch, tier_batch, final_tier_batch, champion_batch, position_batch = [], [], [], [], []
        importance_weights = []
        for batch_num in range(self.batch_size):
            if not config.GUMBEL:
                [observation, action_history, value_mask, reward_mask, policy_mask, value, reward, policy,
                 sample_set, tier_set, final_tier_set, champion_set], priority = self.gameplay_experiences.pop()
            else:
                [observation, action_history, policy_mask, value, reward, policy], \
                    priority = self.gameplay_experiences.pop()
            obs_tensor_batch.append(observation)
            action_history_batch.append(action_history)
            policy_mask_batch.append(policy_mask)
            target_value_batch.append(value)
            target_reward_batch.append(reward)
            target_policy_batch.append(policy)
            if not config.GUMBEL:
                value_mask_batch.append(value_mask)
                reward_mask_batch.append(reward_mask)
                sample_set_batch.append(sample_set)
                tier_batch.append(tier_set)
                final_tier_batch.append(final_tier_set)
                champion_batch.append(champion_set)
                position, _ = self.average_position.extractMax()
                position_batch.append(position)
            importance_weights.append(1 / self.batch_size / priority)

        observation_batch = self.reshape_observation(obs_tensor_batch)
        action_history_batch = np.asarray(action_history_batch)
        target_value_batch = np.asarray(target_value_batch).astype('float32')
        target_reward_batch = np.asarray(target_reward_batch).astype('float32')
        if not config.GUMBEL:
            value_mask_batch = np.asarray(value_mask_batch).astype('float32')
            reward_mask_batch = np.asarray(reward_mask_batch).astype('float32')
            position_batch = np.asarray(position_batch)
            position_batch = np.mean(position_batch)
        else:
            target_policy_batch = np.asarray(target_policy_batch).astype('float32')
        policy_mask_batch = np.asarray(policy_mask_batch).astype('float32')
        importance_weights_batch = np.asarray(importance_weights).astype('float32')
        importance_weights_batch = importance_weights_batch / np.max(importance_weights_batch)

        if not config.GUMBEL:
            data_list = [
                observation_batch, action_history_batch, value_mask_batch, reward_mask_batch,
                policy_mask_batch, target_value_batch, target_reward_batch, target_policy_batch, sample_set_batch,
                importance_weights_batch, tier_batch, final_tier_batch, champion_batch, np.array(position_batch)
            ]
        else:
            data_list = [
                observation_batch, action_history_batch, policy_mask_batch, target_value_batch, target_reward_batch,
                target_policy_batch, importance_weights_batch
            ]
        return np.array(data_list, dtype=object)

    def sample_position_batch(self):
        """
        Prepares a batch for training. All preprocessing done here to avoid problems with data transfer between CPU
        and GPU causing slowdowns.

        Conditions:
            - There is enough data in the buffer to train on.

        Returns:
            A prepared batch ready for training.
        """
        obs_tensor_batch, action_history_batch, target_value_batch, policy_mask_batch = [], [], [], []
        target_policy_batch, value_mask_batch, position_batch = [], [], []
        for batch_num in range(self.batch_size):
            # Setting the position gameplay_experiences get and position get next to each other to minimize
            # the number of multiprocessing errors that could occur by having them too far apart.
            # If these two commands become out of sync, it would cause the position logging to not match the rest
            # of the training logs, but it would not break training.
            [observation, action_history, value_mask, policy_mask, value, policy] = self.gameplay_experiences.pop()
            position, _ = self.average_position.pop()
            position_batch.append(position)
            obs_tensor_batch.append(observation)
            action_history_batch.append(action_history)
            policy_mask_batch.append(policy_mask)
            target_value_batch.append(value)
            target_policy_batch.append(policy)
            value_mask_batch.append(value_mask)

        observation_batch = self.reshape_observation(obs_tensor_batch)
        action_history_batch = np.asarray(action_history_batch)
        target_value_batch = np.asarray(target_value_batch).astype('float32')
        value_mask_batch = np.asarray(value_mask_batch).astype('float32')
        target_policy_batch = np.asarray(target_policy_batch).astype('float32')
        policy_mask_batch = np.asarray(policy_mask_batch).astype('float32')

        data_list = [
            observation_batch, action_history_batch, value_mask_batch, policy_mask_batch, target_value_batch,
            target_policy_batch
        ]
        return np.array(data_list, dtype=object)

    def reshape_observation(self, obs_batch):
        obs_reshaped = {}

        for obs in obs_batch:
            for key in obs:
                if key not in obs_reshaped:
                    obs_reshaped[key] = []
                obs_reshaped[key].append(obs[key])

        for key in obs_reshaped:
            obs_reshaped[key] = np.stack(obs_reshaped[key], axis=0)

        return obs_reshaped

    def store_replay_sequence(self, samples):
        """
        Description:
            Async method to store data into the global buffer. Some quick checking to ensure data validity.

        Args:
            samples (list): All samples from one game from one agent.
        """
        for sample in samples[0]:
            self.gameplay_experiences.append(sample)
            self.average_position.append(samples)

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

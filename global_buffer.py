import ray
import config
import time
import numpy as np
from collections import deque


@ray.remote
class GlobalBuffer:
    def __init__(self):
        self.gameplay_experiences = deque(maxlen=25000)
        self.batch_size = config.BATCH_SIZE

    def sample_batch(self):
        # Returns: a batch of gameplay experiences without regard to which agent.
        observation_batch, action_history_batch, target_value_batch, target_reward_batch = [], [], [], []
        target_policy_batch, value_mask_batch, reward_mask_batch, policy_mask_batch = [], [], [], []
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
            target_policy_batch.append(policy)

        observation_batch = np.squeeze(np.asarray(observation_batch))
        action_history_batch = np.asarray(action_history_batch)
        target_value_batch = np.asarray(target_value_batch).astype('float32')
        target_reward_batch = np.asarray(target_reward_batch).astype('float32')
        value_mask_batch = np.asarray(value_mask_batch).astype('float32')
        reward_mask_batch = np.asarray(reward_mask_batch).astype('float32')
        policy_mask_batch = np.asarray(policy_mask_batch).astype('float32')
        target_policy_batch = np.asarray(target_policy_batch).astype('float32')

        return [observation_batch, action_history_batch, value_mask_batch, reward_mask_batch, policy_mask_batch,
                target_value_batch, target_reward_batch, target_policy_batch]

    def store_replay_sequence(self, sample):
        # Records a single step of gameplay experience
        # First few are self-explanatory
        # done is boolean if game is done after taking said action
        self.gameplay_experiences.append(sample)

    def sample_a3c_batch(self):
    # Returns: a batch of gameplay experiences without regard to which agent.
        tensor_batch, image_batch, action_history_batch, target_value_batch, target_reward_batch = [], [], [], [], []
        target_policy_batch, value_mask_batch, reward_mask_batch, policy_mask_batch = [], [], [], []
        prev_action_batch = []
        for gameplay_experience in range(self.batch_size):
            observation, action_history, value_mask, reward_mask, policy_mask,\
                value, reward, policy, prev_action = self.gameplay_experiences.popleft()
            tensor_batch.append(observation[0])
            image_batch.append(observation[1])
            action_history_batch.append(action_history[1:])
            value_mask_batch.append(value_mask)
            reward_mask_batch.append(reward_mask)
            policy_mask_batch.append(policy_mask)
            target_value_batch.append(value)
            target_reward_batch.append(reward)
            target_policy_batch.append(policy)
            prev_action_batch.append(prev_action)

        tensor_batch = np.asarray(tensor_batch).astype('float32')
        image_batch = np.asarray(image_batch).astype('float32')
        action_history_batch = np.asarray(action_history_batch)
        target_value_batch = np.asarray(target_value_batch).astype('float32')
        target_reward_batch = np.asarray(target_reward_batch).astype('float32')
        value_mask_batch = np.asarray(value_mask_batch).astype('float32')
        reward_mask_batch = np.asarray(reward_mask_batch).astype('float32')
        policy_mask_batch = np.asarray(policy_mask_batch).astype('float32')
        target_policy_batch = np.asarray(target_policy_batch).astype('float32')
        prev_action_batch = np.asarray(prev_action_batch).astype('float32')

        return [[tensor_batch, image_batch], action_history_batch, value_mask_batch, reward_mask_batch, policy_mask_batch,
                target_value_batch, target_reward_batch, target_policy_batch, prev_action_batch]

    def store_replay_a3c_sequence(self, state, logits, action, reward, prev_action, prev_reward):
        # Records a single step of gameplay experience
        # First few are self-explanatory
        # done is boolean if game is done after taking said action
        reward = np.clip(reward, -1.0, 1.0)
        logits = self.transpose(logits)
        self.gameplay_experiences.append([state, logits, action, reward, prev_action, prev_reward])

    def available_batch(self):
        queue_length = len(self.gameplay_experiences)
        if queue_length >= self.batch_size:
            time.sleep(5)
            return True
        time.sleep(20)
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

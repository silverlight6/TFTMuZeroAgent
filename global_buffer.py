from collections import deque
import numpy as np

import config


class GlobalBuffer:
    def __init__(self):
        self.gameplay_experiences = deque(maxlen=25000)
        self.batch_size = config.BATCH_SIZE

    def sample_batch(self):
        # Returns: a batch of gameplay experiences without regard to which agent.
        observation_batch, history_batch, value_batch, reward_batch, policy_batch = [], [], [], [], []

        for gameplay_experience in range(self.batch_size):
            observation, history, value, reward, policy = self.gameplay_experiences.popleft()
            observation_batch.append(observation)
            history_batch.append(history)
            value_batch.append(value)
            reward_batch.append(reward)
            policy_batch.append(policy)

        return [observation_batch, history_batch, value_batch, reward_batch, policy_batch]

    def store_replay_sequence(self, observation, history, value, reward, policy):
        # Records a single step of gameplay experience
        # First few are self-explanatory
        # done is boolean if game is done after taking said action
        reward = np.clip(reward, -1.0, 1.0)
        observation = self.transpose(observation)
        self.gameplay_experiences.append([observation, history, value, reward, policy])

    def available_batch(self):
        queue_length = len(self.gameplay_experiences)
        if queue_length >= self.batch_size:
            return True
        return False

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

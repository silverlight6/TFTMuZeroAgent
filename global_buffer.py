from collections import deque
import numpy as np


class GlobalBuffer:
    def __init__(self, batch_size=120):
        self.gameplay_experiences = deque(maxlen=25000)
        self.batch_size = batch_size

    def sample_batch(self):
        # Returns: a batch of gameplay experiences without regard to which agent.
        state_batch, logit_batch, action_batch, reward_batch, p_action_batch, p_reward_batch = [], [], [], [], [], []

        for gameplay_experience in range(self.batch_size):
            state, logit, action, reward, prev_action, prev_reward = self.gameplay_experiences.popleft()
            state_batch.append(state)
            logit_batch.append(logit)
            action_batch.append(action)
            reward_batch.append(reward)
            p_action_batch.append(prev_action)
            p_reward_batch.append(prev_reward)

        return [state_batch, logit_batch, np.swapaxes(np.array(action_batch), 2, 1),
                np.swapaxes(np.array(reward_batch), 0, 1), np.swapaxes(np.array(p_action_batch), 2, 1),
                np.swapaxes(np.array(p_reward_batch), 0, 1)]

    def store_replay_sequence(self, state, logits, action, reward, prev_action, prev_reward):
        # Records a single step of gameplay experience
        # First few are self-explanatory
        # done is boolean if game is done after taking said action
        reward = np.clip(reward, -1.0, 1.0)
        logits = self.transpose(logits)
        self.gameplay_experiences.append([state, logits, action, reward, prev_action, prev_reward])

    def available_batch(self):
        queue_length = len(self.gameplay_experiences)
        # print(queue_length)
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

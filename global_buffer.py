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
        action_mask_batch = []
        for gameplay_experience in range(len(self.gameplay_experiences)):
            observation, action_history, value_mask, reward_mask, policy_mask,\
                value, reward, policy, action_mask = self.gameplay_experiences.popleft()
            observation_batch.append(observation)
            action_history_batch.append(action_history[1:])
            value_mask_batch.append(value_mask)
            reward_mask_batch.append(reward_mask)
            policy_mask_batch.append(policy_mask)
            target_value_batch.append(value)
            target_reward_batch.append(reward)
            target_policy_batch.append(policy)
            action_mask_batch.append(action_mask)

        for x in range(len(observation_batch)):
            observation_batch[x] = observation_batch[x][-1]
        max = 0
        for x in range(len(target_value_batch)):
            if len(target_value_batch[x]) > max:
                max = len(target_value_batch[x])

        for x in range(len(target_value_batch)):
            target_value_batch[x] += [0] * (max - len(target_value_batch[x]))
            target_reward_batch[x] += [0] * (max - len(target_reward_batch[x]))
            value_mask_batch[x] += [0] * (max - len(value_mask_batch[x]))
            reward_mask_batch[x] += [0] * (max - len(reward_mask_batch[x]))
            policy_mask_batch[x] += [0] * (max - len(policy_mask_batch[x]))
            target_policy_batch[x] += [np.zeros(config.ACTION_DIM[0] + config.ACTION_DIM[1] + config.ACTION_DIM[2])] * (max - len(target_policy_batch[x]))
            action_mask_batch[x] += [np.zeros(config.ACTION_DIM[0] + config.ACTION_DIM[1] + config.ACTION_DIM[2])] * (max - len(action_mask_batch[x]))
            action_history_batch[x] += ["0"] * ((max - 1) - len(action_history_batch[x]))
                # observation_batch[x] += [np.zeros(config.OBSERVATION_SIZE)] * (max - len(observation_batch[x]))
        

        # for x in range(len(action_history_batch)):
        #     print(len(action_history_batch[x]))
            # action_history_batch[x] = np.asarray(action_history_batch[x])

        observation_batch = np.squeeze(np.asarray(observation_batch))
        action_history_batch = np.asarray(action_history_batch)
        # print(action_history_batch[:][0])
        target_value_batch = np.asarray(target_value_batch).astype('float32')
        target_reward_batch = np.asarray(target_reward_batch).astype('float32')
        value_mask_batch = np.asarray(value_mask_batch).astype('float32')
        reward_mask_batch = np.asarray(reward_mask_batch).astype('float32')
        policy_mask_batch = np.asarray(policy_mask_batch).astype('float32')
        target_policy_batch = np.asarray(target_policy_batch).astype('float32')
        action_mask_batch = np.asarray(action_mask_batch).astype('float32')

        return [observation_batch, action_history_batch, value_mask_batch, reward_mask_batch, policy_mask_batch,
                target_value_batch, target_reward_batch, target_policy_batch, action_mask_batch]

    def store_replay_sequence(self, sample):
        # Records a single step of gameplay experience
        # First few are self-explanatory
        # done is boolean if game is done after taking said action
        self.gameplay_experiences.append(sample)

    def available_batch(self):
        queue_length = len(self.gameplay_experiences)
        if queue_length >= self.batch_size:
            print("BATCH READY")
            time.sleep(5)
            return True
        print("BATCH NOT READY", queue_length)
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

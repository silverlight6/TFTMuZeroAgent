import ray
import config
import time
import numpy as np
from collections import deque


@ray.remote(num_gpus=0.1)
class GlobalBuffer:
    def __init__(self, storage_ptr):
        self.gameplay_experiences = deque(maxlen=25000)
        self.batch_size = config.BATCH_SIZE
        self.storage_ptr = storage_ptr
        self.average_position = deque(maxlen=25000)

    # Might be a bug with the action_batch not always having correct dims
    def sample_batch(self):
        # Returns: a batch of gameplay experiences without regard to which agent.
        obs_tensor_batch, action_history_batch, target_value_batch, policy_mask_batch = [], [], [], []
        target_reward_batch, target_policy_batch, value_mask_batch, reward_mask_batch = [], [], [], []
        sample_set_batch, tier_batch, final_tier_batch, champion_batch, position_batch = [], [], [], [], []
        for batch_num in range(self.batch_size):
            # Setting the position gameplay_experiences get and position get next to each other to try to minimize
            # The number of multiprocessing errors that could occur by having them be too far apart.

            observation, action_history, value_mask, reward_mask, policy_mask, value, reward, policy, \
               sample_set, tier_set, final_tier_set, champion_set = self.gameplay_experiences.popleft()
            position_batch.append(self.average_position.popleft())
            obs_tensor_batch.append(observation)
            action_history_batch.append(action_history[1:])
            value_mask_batch.append(value_mask)
            reward_mask_batch.append(reward_mask)
            policy_mask_batch.append(policy_mask)
            target_value_batch.append(value)
            target_reward_batch.append(reward)
            target_policy_batch.append(policy)
            sample_set_batch.append(sample_set)
            tier_batch.append(tier_set)
            final_tier_batch.append(final_tier_set)
            champion_batch.append(champion_set)

        # observation_batch = np.squeeze(np.asarray(obs_tensor_batch))
        observation_batch = self.reshape_observation(obs_tensor_batch)
        action_history_batch = np.asarray(action_history_batch)
        target_value_batch = np.asarray(target_value_batch).astype('float32')
        target_reward_batch = np.asarray(target_reward_batch).astype('float32')
        value_mask_batch = np.asarray(value_mask_batch).astype('float32')
        reward_mask_batch = np.asarray(reward_mask_batch).astype('float32')
        policy_mask_batch = np.asarray(policy_mask_batch).astype('float32')

        return [observation_batch, action_history_batch, value_mask_batch, reward_mask_batch, policy_mask_batch,
                target_value_batch, target_reward_batch, target_policy_batch, sample_set_batch, tier_batch,
                final_tier_batch, champion_batch], np.mean(position_batch)
        
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

    def store_replay_sequence(self, sample, position):
        self.gameplay_experiences.append(sample)
        self.average_position.append(position)

    def available_batch(self):
        queue_length = len(self.gameplay_experiences)
        if queue_length >= self.batch_size and not ray.get(self.storage_ptr.get_trainer_busy.remote()):
            self.storage_ptr.set_trainer_busy.remote(True)
            print("QUEUE_LENGTH {} at time {}".format(queue_length, time.time_ns()))
            return True
        time.sleep(1)
        # print("QUEUE_LENGTH_SLEEPY {} at time {}".format(queue_length, time.time_ns()))
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

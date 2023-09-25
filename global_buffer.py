import ray
import config
import time
import numpy as np
from collections import deque


@ray.remote
class GlobalBuffer:
    def __init__(self):
        self.gameplay_experiences = deque(maxlen=100000)
        self.batch_size = config.BATCH_SIZE

    # Might be a bug with the action_batch not always having correct dims
    def sample_batch(self):
        # Returns: a batch of gameplay experiences without regard to which agent.
        observation_batch = []
        action_batch = []
        value_batch = []
        reward_batch = []
        policy_batch = []
        for _ in range(self.batch_size):
            observation, action, value, reward, policy = self.gameplay_experiences.popleft()
            observation_batch.append(observation)
            action_batch.append(action)
            value_batch.append(value)
            reward_batch.append(reward)
            policy_batch.append(policy)

        observation_batch = np.array(observation_batch)
        action_batch = np.array(action_batch)
        value_batch = np.array(value_batch)
        reward_batch = np.array(reward_batch)
        policy_batch = np.array(policy_batch)
        print("observation ", observation_batch.shape)
        print("action ", action_batch.shape)
        print("value ", value_batch.shape)
        print("reward ", reward_batch.shape)
        print("policy ", policy_batch.shape)
        exit()
        # policy_mask_batch = np.asarray(policy_mask_batch).astype('float32')

        return [observation_batch, action_batch, value_batch, reward_batch, policy_batch]

    def store_replay_sequence(self, sample):
        self.gameplay_experiences.extend(sample)

    def available_batch(self):
        queue_length = len(self.gameplay_experiences)
        print("Queue ", queue_length)
        if queue_length >= self.batch_size:
            print("Queue ", queue_length)
            return True
        time.sleep(5)
        return False

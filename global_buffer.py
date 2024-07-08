import ray
import config
import time
import numpy as np
from collections import deque


@ray.remote
class GlobalBuffer:
    def __init__(self):
        self.gameplay_experiences = deque(maxlen=100000)
        self.combat_experiences = deque(maxlen=100000)

    # Might be a bug with the action_batch not always having correct dims
    def sample_gameplay_batch(self, batch_size):
        # Returns: a batch of gameplay experiences without regard to which agent.
        observation_batch = []
        action_batch = []
        value_batch = []
        reward_batch = []
        policy_batch = []
        for _ in range(batch_size):
            observation, action, value, reward, policy = self.gameplay_experiences.pop()
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
        # print("observation ", observation_batch.shape)
        # print("action ", action_batch.shape)
        # print("value ", value_batch.shape)
        # print("reward ", reward_batch.shape)
        # print("policy ", policy_batch.shape)
        # policy_mask_batch = np.asarray(policy_mask_batch).astype('float32')

        return [observation_batch, action_batch, value_batch, reward_batch, policy_batch]
    
    def sample_combat_batch(self, batch_size):
        # Returns: a batch of gameplay experiences without regard to which agent.
        observation_batch = []
        result_batch = []
        for _ in range(batch_size):
            observation, result = self.combat_experiences.pop()
            observation_batch.append(observation)
            result_batch.append(result)

        observation_batch = np.array(observation_batch)
        result_batch = np.array(result_batch)

        return [observation_batch, result_batch]

    def store_episode(self, sample):
        self.gameplay_experiences.extend(sample)
    
    def store_combat(self, sample):
        self.combat_experiences.append(sample)

    def available_gameplay_batch(self, batch_size):
        queue_length = len(self.gameplay_experiences)
        # print("QUEUE SIZE: ", queue_length)
        if queue_length >= batch_size:
            return True
        return False
    
    def available_combat_batch(self, batch_size):
        queue_length = len(self.combat_experiences)
        if queue_length >= batch_size:
            return True
        return False

import numpy as np
import config

class RandomAgent:
    def __init__(self, action_size):
        self.action_size = action_size

    def select_action(self, observation, mask):
        return np.random.rand(observation.shape[0], self.action_size)

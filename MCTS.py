import numpy as np

class MCTS:
  
  def __init__(self, sample_size, action_size, action_limits, policy_size):
    self.sample_size = sample_size
    self.action_size = action_size
    self.action_limits = action_limits
    self.policy_size = policy_size

  def generate_action(self, n_simulations, observation):
    # PLACEHOLDER
    return np.random.randint(self.action_limits, size=(observation.shape[0], self.action_size)), np.random.randint([1], size=(observation.shape[0], self.policy_size))

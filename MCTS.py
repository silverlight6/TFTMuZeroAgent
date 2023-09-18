import numpy as np

class MCTS:
  
  def __init__(self, sample_size, action_size):
    self.sample_size = sample_size
    self.action_size = action_size

  def generate_action(self, n_simulations, observation):
    # PLACEHOLDER
    return np.random.rand(observation.shape[0], self.action_size)

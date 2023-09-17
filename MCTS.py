import numpy as np

class MCTS:
  
  def __init__(self, sample_size, action_size):
    self.sample_size = sample_size
    self.action_size = action_size

  def generate_action(self, n_simulations):
    # PLACEHOLDER
    return np.random.randn(self.action_size)

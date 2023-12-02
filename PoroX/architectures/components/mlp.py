from typing import Sequence
from flax import linen as nn

class MLP(nn.Module):
    features: Sequence[int]
    
    def setup(self):
        layers = []
        for i, size in enumerate(self.features):
            layers.append(nn.Dense(size))
            if i != len(self.features) - 1:
                layers.append(nn.gelu)
        self.mlp = nn.Sequential(layers=layers)
    
    def __call__(self, x):
        return self.mlp(x)
        
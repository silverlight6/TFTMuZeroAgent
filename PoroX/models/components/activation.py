import flax.linen as nn

def swiglu(x, beta=1.0):
    """
    From the great Noam Shazeer: https://arxiv.org/abs/2002.05202v1
    """
    y = nn.gelu(x)
    return x * nn.sigmoid(beta * x) + (1 - nn.sigmoid(beta * x)) * x

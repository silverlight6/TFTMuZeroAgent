from typing import Optional
from flax import linen as nn
from flax import struct

from PoroX.architectures.components.fc import FFNSwiGLU

@struct.dataclass
class EncoderConfig:
    num_blocks: int = 2
    num_heads: int = 8
    qkv_features: Optional[int] = None
    hidden_dim: Optional[int] = None

# Transformer Encoder
class EncoderBlock(nn.Module):
    """
    Transformer Encoder Block Using:
    1. LayerNorm
    2. MHSA
    3. Residual Connection
    4. LayerNorm
    5. FFN
    6. Residual Connection
    """
    config: EncoderConfig
    
    @nn.compact
    def __call__(self, x):
        y = nn.RMSNorm()(x)
        y = nn.MultiHeadDotProductAttention(
            num_heads=self.config.num_heads,
            qkv_features=self.config.qkv_features,
            kernel_init=nn.initializers.xavier_uniform(),
        )(y)
        x = x + y
        
        y = nn.RMSNorm()(x)
        y = FFNSwiGLU(
            hidden_dim=self.config.hidden_dim
        )(y)
        x = x + y
        
        return x
    
class Encoder(nn.Module):
    """
    Simple Transformer Encoder
    """
    config: EncoderConfig
    
    @nn.compact
    def __call__(self, x):
        for layer in range(self.config.num_blocks):
            x = EncoderBlock(self.config)(x)
        
        x = nn.RMSNorm()(x)
        
        return x
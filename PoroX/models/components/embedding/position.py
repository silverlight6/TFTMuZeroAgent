from typing import Optional
import flax.linen as nn
import jax.numpy as jnp

class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding
    Taken directly from:
    https://github.com/google/flax/blob/main/examples/lm1b/models.py
    """
    max_len: int
    embedding_size: int
    
    min_scale: float = 1.0
    max_scale: float = 1.0e4
    
    dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        
        def init_sinusoudal(max_len, embedding_size, min_scale, max_scale, dtype):
            pe = jnp.zeros((max_len, embedding_size), dtype=dtype)
            position = jnp.arange(0, max_len)[:, jnp.newaxis]
            scale_factor = -jnp.log(max_scale / min_scale) / (embedding_size // 2 - 1)
            div_term = min_scale * jnp.exp(jnp.arange(0, embedding_size // 2) * scale_factor)
            pe[:, :, embedding_size // 2] = jnp.sin(position * div_term)
            pe[:, embedding_size // 2 : 2 * (embedding_size // 2)] = jnp.cos(position * div_term)
            pe = pe[jnp.newaxis, :, :] # [1, max_len, embedding_size]
            return pe

        self.pe = init_sinusoudal(
            self.max_len,
            self.embedding_size,
            self.min_scale, 
            self.max_scale, 
            self.dtype
        )
    
    def __call__(self, x):
        return x + self.pe[:, :x.shape[-2], :]
    
class LearnedPositionalEncoding(nn.Module):
    """
    Learned Positional Encoding
    """
    max_len: Optional[int] = None
    embedding_size: Optional[int] = None
    
    @nn.compact
    def __call__(self, x):
        num_elements = self.max_len or x.shape[-2]
        features = self.embedding_size or x.shape[-1]

        pos = jnp.arange(x.shape[-2])

        pe = nn.Embed(
            num_embeddings=num_elements,
            features=features
        )(pos)
        
        return x + pe
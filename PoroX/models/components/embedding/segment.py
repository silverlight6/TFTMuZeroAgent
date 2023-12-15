from typing import Optional
import flax.linen as nn
from flax import struct
import jax
import jax.numpy as jnp

from functools import partial

from PoroX.models.components.fc import MLP, FFNSwiGLU

# -- Config -- #
    
@struct.dataclass
class SegmentConfig:
    segments: jnp.ndarray
    num_elements: Optional[int] = None
    hidden_dim: Optional[int] = None

# -- Encoding -- #
# def expand_segments(num_elements, segments):
def expand_segments(segments):
    segments = jnp.array(segments)
    num_segments = len(segments)
    expanded_segment_vector_length = jnp.sum(segments)
    expanded_segment_vector = jnp.repeat(
        a=jnp.arange(num_segments),
        repeats=segments,
        total_repeat_length=expanded_segment_vector_length
    )
    
    # Ignore the following. It's not worth the time to fight the jit compiler...

    # Just fill in the rest of the segment vector with the padding segment
    # padding_segment = jnp.ones(num_elements - len(expanded_segment_vector)) * (num_segments)

    # segment_vector = jnp.concatenate([
    #     expanded_segment_vector,
    #     padding_segment
    # ]).astype(jnp.int16)
    
    return expanded_segment_vector

class SegmentEncoding(nn.Module):
    config: SegmentConfig
    """
    Create segment encodings based on an array of segment lengths and add them to the input
    Ex: [3, 3, 3] -> [0, 0, 0, 1, 1, 1, 2, 2, 2]
    
    If the length of the value vector is longer than the sum of the segments, then we add a new segment
    Ex: [3, 3, 3] -> [0, 0, 0, 1, 1, 1, 2, 2, 2]  (length of 9)
        input: jnp.ones(10) (need one more segment)
        segment_vector: [0, 0, 0, 1, 1, 1, 2, 2, 2, 3]
        
    We might not know what the length of the final segment is, so we can just pad it with the last segment
    
    The player segments might look like this: [28, 9, 5, 10, 1]
    28 board, 9 bench, 5 shop, 10 items, 1 trait, N scalars, so padding segment will take care of the scalars

    """
    @nn.compact
    def __call__(self, x):
        num_segments = len(self.config.segments)
        out_dim = x.shape[-1]

        segment_encoding = nn.Embed(
            num_embeddings=num_segments,
            features=out_dim)(self.config.segments)
        
        return x + segment_encoding
    
# -- FFN -- #
# TODO: Generalize this to work with a segments config
# UPDATE: It's not worth the time to fight the jit compiler...
class GlobalPlayerSegmentFFN(nn.Module):
    config: SegmentConfig
    
    @nn.compact
    def __call__(self, x):
        board   = x[..., :28, :]
        bench   = x[..., 28:37, :]
        shop    = x[..., 37:42, :]
        items   = x[..., 42:52, :]
        traits  = x[..., 52:53, :]
        playerIDs = x[..., 53:57, :]
        scalars = x[..., 57:, :]
        
        def ffn(hidden_dim=self.config.hidden_dim):
            return FFNSwiGLU(hidden_dim)

        board_fc    = ffn()(board)
        bench_fc    = ffn()(bench)
        shop_fc     = ffn()(shop)
        items_fc    = ffn()(items)
        traits_fc   = ffn()(traits)
        ids_fc      = ffn()(playerIDs)
        scalars_fc  = ffn()(scalars)

        return jnp.concatenate([
            board_fc,
            bench_fc,
            shop_fc,
            items_fc,
            traits_fc,
            ids_fc,
            scalars_fc
        ], axis=-2)

class PlayerSegmentFFN(nn.Module):
    config: SegmentConfig

    @nn.compact
    def __call__(self, x):
        board   = x[..., :28, :]
        bench   = x[..., 28:37, :]
        shop    = x[..., 37:42, :]
        items   = x[..., 42:52, :]
        traits  = x[..., 52:53, :]
        scalars = x[..., 53:, :]
        
        def ffn(hidden_dim=self.config.hidden_dim):
            return FFNSwiGLU(hidden_dim)

        board_fc    = ffn()(board)
        bench_fc    = ffn()(bench)
        shop_fc     = ffn()(shop)
        items_fc    = ffn()(items)
        traits_fc   = ffn()(traits)
        scalars_fc  = ffn()(scalars)

        return jnp.concatenate([
            board_fc,
            bench_fc,
            shop_fc,
            items_fc,
            traits_fc,
            scalars_fc
        ], axis=-2)
        
class OpponentSegmentFFN(nn.Module):
    config: SegmentConfig

    @nn.compact
    def __call__(self, x):
        board   = x[..., :28, :]
        bench   = x[..., 28:37, :]
        items   = x[..., 37:47, :]
        traits  = x[..., 47:48, :]
        scalars = x[..., 48:, :]

        def ffn(hidden_dim=self.config.hidden_dim):
            return FFNSwiGLU(hidden_dim)

        board_fc    = ffn()(board)
        bench_fc    = ffn()(bench)
        items_fc    = ffn()(items)
        traits_fc   = ffn()(traits)
        scalars_fc  = ffn()(scalars)

        return jnp.concatenate([
            board_fc,
            bench_fc,
            items_fc,
            traits_fc,
            scalars_fc
        ], axis=-2)
        
"""
Unused. I actually hate the jax jit compiler.
"""

def create_ranges(num_elements, segments):
    segment_ranges = jnp.concatenate([
        jnp.array([0]),
        jnp.cumsum(segments),
        jnp.array([num_elements])
    ])

    # Need to turn this from [0, 3, 6, 9] to [[0, 3], [3, 6], [6, 9]]
    segment_ranges = jnp.stack([
        segment_ranges[:-1],
        segment_ranges[1:]
    ], axis=-1)

    return segment_ranges


class SegmentFFN(nn.Module):
    config: SegmentConfig

    @nn.compact
    def __call__(self, x):
        num_segments = len(self.config.segments) + 1
        num_elements = self.config.num_elements or x.shape[-2]
        output_size = self.config.output_size or x.shape[-1]
        
        segment_ranges = self.param(
            "segment_ranges",
            lambda rng, ne, s: create_ranges(ne, s),
            num_elements,
            self.config.segments
        )
        
        if output_size != x.shape[-1]:
            placeholder = jnp.zeros(x.shape[:-1] + (output_size,))
        else:
            placeholder = x
            
        for range in segment_ranges:
            start, end = range
            # segment_x = x[..., start:end, :]
            segment_x = jax.lax.dynamic_slice_in_dim(x, start, end - start, axis=-2)
            segment_x = FFNSwiGLU(self.config.hidden_dim)(segment_x)
            
            placeholder = placeholder.at[..., start:end, :].set(segment_x)
        
        return placeholder

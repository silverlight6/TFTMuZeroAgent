# Scalar Encoder to convert scalar values into an N-dimensional token
# Taken from https://github.com/google-research/google-research/blob/master/muzero/core.py

from functools import partial
import jax.numpy as jnp
from jax import jit

class ScalarEncoder:
    """
    Converts scalar values into an N-dimensional token and back.
    Allows for batch encoding and decoding.
    """
    def __init__(self, min_value: float, max_value: float, num_steps: int):
        """
        Args:
            min_value: Minimum value of the scalar
            max_value: Maximum value of the scalar
            num_steps: Size of the encoding token
        """
        self.min_value = min_value
        self.max_value = max_value
        self.num_steps = num_steps
        self.value_range = max_value - min_value
        self.step_size = self.value_range / (num_steps - 1)
        self.step_range_int = jnp.arange(num_steps, dtype=jnp.int16)
        self.step_range_float = self.step_range_int.astype(jnp.float16)

    @partial(jit, static_argnums=(0,))
    def encode(self, value: float) -> jnp.ndarray:
        """Encode a scalar value into a token
        Args:
            value: Scalar value to encode
        Returns:
            ndarray: Encoded token
        """
        value = jnp.expand_dims(value , -1)
        clipped_value = jnp.clip(value, self.min_value, self.max_value)
        above_min = clipped_value - self.min_value
        num_steps = above_min / self.step_size
        lower_step = jnp.floor(num_steps)
        upper_mod = num_steps - lower_step
        upper_step = lower_step + 1
        lower_mod = 1.0 - upper_mod

        def create_encoding(step, mod):
            return jnp.equal(step, self.step_range_int).astype(jnp.float16) * mod

        lower_encoding = create_encoding(lower_step, lower_mod)
        upper_encoding = create_encoding(upper_step, upper_mod)
        

        return lower_encoding + upper_encoding

    @partial(jit, static_argnums=(0,))
    def decode(self, logits: jnp.ndarray) -> float:
        """Decode a token into a scalar value
        Args:
            logits: Encoded token
        Returns:
            float: Decoded scalar value
        """
        num_steps = jnp.sum(logits * self.step_range_float, -1)
        above_min = num_steps * self.step_size
        value = above_min + self.min_value
        return value

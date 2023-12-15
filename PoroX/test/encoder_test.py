import pytest
from jax.experimental import checkify
import jax.numpy as jnp
import numpy as np
import jax
import time

from PoroX.models.components.scalar_encoder import ScalarEncoder
from PoroX.test.utils import profile
    
@pytest.fixture
def encoder():
    return ScalarEncoder(
        min_value=0,
        max_value=200,
        num_steps=60
    )
    
def test_scalar_encoder_correctness(encoder, key):
    """
    Test that the scalar encoder encodes and decodes values correctly.
    There will be some loss of precision due to the encoding process, so we allow for a small amount of error.
    """
    initial_values = jax.random.uniform(key, shape=(100,), minval=encoder.min_value, maxval=encoder.max_value, dtype=jnp.float16)
    logits = encoder.encode(initial_values)
    values = encoder.decode(logits)
    checkify.check(jnp.allclose(initial_values, values, atol=0.1), "Scalar Encoder does not work properly! Diff (initial, decoded):\n {diff}", diff=jnp.stack((initial_values, values), axis=1))
    
def test_scalar_encoder_performance(encoder, key):
    """
    Test that the scalar encoder is fast enough to be used in a real-time environment.
    """
    initial_values = jax.random.uniform(key, shape=(1000,), minval=encoder.min_value, maxval=encoder.max_value, dtype=jnp.float16)
    
    @jax.jit
    def encode_decode(values):
        logits = encoder.encode(values)
        return encoder.decode(logits)
    
    N = 1000
    profile(N, encode_decode, initial_values)
import pytest
from jax.experimental import checkify
import jax.numpy as jnp
import jax
import time

from PoroX.architectures.components.scalar_encoder import ScalarEncoder
    
@pytest.fixture
def encoder():
    return ScalarEncoder(
        min_value=0,
        max_value=200,
        num_steps=60
    )
    
@pytest.fixture
def key():
    return jax.random.PRNGKey(10)
    
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
    N = 10000
    initial_values = jax.random.uniform(key, shape=(1000,), minval=encoder.min_value, maxval=encoder.max_value, dtype=jnp.float16)
    
    total_time = 0.
    for _ in range(N):
        start = time.time()
        logits = encoder.encode(initial_values)
        values = encoder.decode(logits)
        end = time.time() - start
        total_time += end
        
    avg = total_time / N
        
    print(f'{N} loops, {avg} per loop')
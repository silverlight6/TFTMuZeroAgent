import pytest
import jax
import jax.numpy as jnp
import numpy as np
import time

"""
Just testing different jax functions.
"""

@pytest.fixture
def random_obs():
    N = 20
    S = (100,100)
    observations = []
    for _ in range(N):
        observation = {"observation": np.random.random(S)}
        observations.append(observation)
    return observations

def test_pytree_naive(random_obs):
    obs = random_obs
    @jax.jit
    def convert_to_array(obs):
        return jnp.asarray(obs["observation"])
    
    start = time.time()
    

    combined = []
    for obs in obs:
        array = convert_to_array(obs)
        combined.append(array)

    combined = jnp.stack(combined)
    
    end = time.time() - start
    print(f'Naive: {end}')
    
def test_pytree_jax(random_obs):
    obs = random_obs
    @jax.jit
    def transpose(obs):
        return jax.tree_map(lambda *xs: jnp.stack(xs), *obs)
    
    # Do this to force compilation
    combined = transpose(obs)

    start = time.time()
    
    combined = transpose(obs)

    end = time.time() - start
    print(f'Jax: {end}')
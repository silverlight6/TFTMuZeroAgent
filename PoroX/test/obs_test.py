import jax.numpy as jnp

import PoroX.modules.batch_utils as batch_utils
from PoroX.test.utils import profile

def test_batch_obs(first_obs):
    # Test Logic
    obs = batch_utils.collect_obs(first_obs)
    print(obs.players.champions.shape)
    print(obs.action_mask.shape)
    print(obs.opponents.champions.shape)
    
    # Profile
    N = 1000
    profile(N, batch_utils.collect_obs, first_obs)
    
def test_batch_obs_shared(first_obs):
    obs = batch_utils.collect_shared_obs(first_obs)
    print(obs.players.champions.shape)
    print(obs.action_mask.shape)
    print(obs.opponents.champions.shape)
    
    # Profile
    N = 1000
    profile(N, batch_utils.collect_shared_obs, first_obs)
    
def test_expand():
    # Test Logic
    x = jnp.ones((8, 23, 40))
    y = jnp.ones((8, 28, 10))
    z = jnp.ones((8, 23, 2))
    collection = [x, y, z]
    collection = batch_utils.expand(collection, axis=-3)
    print(collection[0].shape)
    
def test_game_batch_obs(first_obs):
    # Copy first_obs N times
    num_games = 10
    obs_list = [first_obs] * num_games
    
    # Test Logic
    obs = batch_utils.collect_env_obs(obs_list)
    print(obs.players.champions.shape)
    print(obs.action_mask.shape)
    # print(obs.opponents.champions.shape)

    # Profile
    N = 100
    profile(N, batch_utils.collect_env_obs, obs_list)
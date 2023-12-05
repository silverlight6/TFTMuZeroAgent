import time

import jax
import jax.numpy as jnp

import PoroX.modules.batch_utils as batch_utils
from PoroX.test.utils import profile

def test_batch_obs(first_obs):
    # Test Logic
    obs = batch_utils.collect_obs(first_obs)
    print(obs.player.champions.shape)
    print(obs.action_mask.shape)
    print(obs.opponents.champions.shape)
    
    # Profile
    N = 1000
    profile(N, batch_utils.collect_obs, first_obs)
    
def test_game_batch_obs(first_obs):
    # Copy first_obs N times
    num_games = 10
    obs_list = [first_obs] * num_games
    
    # Test Logic
    obs = batch_utils.collect_env_obs(obs_list)
    print(obs.player.champions.shape)
    print(obs.action_mask.shape)
    print(obs.opponents.champions.shape)

    # Profile
    N = 100
    profile(N, batch_utils.collect_env_obs, obs_list)
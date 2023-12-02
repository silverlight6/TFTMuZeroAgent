import pytest
import time

import jax
import jax.numpy as jnp
from jax import flatten_util

from Simulator.porox.tft_simulator import parallel_env, TFTConfig

from Simulator.porox.player import Player
from Simulator import pool
from PoroX.modules.observation import PoroXObservation
from PoroX.test.fixtures import obs, env, sample_action

@jax.jit
def transpose(obs):
    return jax.tree_map(lambda *xs: jnp.stack(xs), *obs)

def test_obs_speed(obs):
    N = 10000
    total = 0
    
    # Precompile JIT
    public = obs.fetch_public_observation()
    player = obs.fetch_player_observation()
    
    for _ in range(N):
        start = time.time()
        public = obs.fetch_public_observation()
        player = obs.fetch_player_observation()
        total += time.time() - start
        
    avg = total / N
    print(f"{N} loops, {avg} per loop")
    
def test_transpose_speed(env):
    start = time.time()
    
    obs_list = []

    obs, infos = env.reset()
    terminated = {agent: False for agent in env.agents}
    truncated = {agent: False for agent in env.agents}
    
    total_action_time = 0
    N = 0
    
    obs_list.append(obs)

    while not all(terminated.values()):
        action_start = time.time()
        actions = {
            agent: sample_action(env, obs, agent)
            for agent in env.agents
            if (
                (agent in terminated and not terminated[agent])
                or (agent in truncated and not truncated[agent])
            )
        }
        obs, rew, terminated, truncated, info = env.step(actions)
        total_action_time += time.time() - action_start
        N += 1
        obs_list.append(obs)
    
    print(f"Total time: {time.time() - start}")
    print(f"{N} actions, Avg action time: {total_action_time / N}")
    
    # Save obs_list to disk
    import pickle
    with open('obs_list.pkl', 'wb') as f:
        pickle.dump(obs_list, f)
        
def test_transpose_logic():
    import pickle
    with open('obs_list.pkl', 'rb') as f:
        obs_list = pickle.load(f)
        
    import chex
    
    @chex.dataclass(frozen=True)
    class Observation:
        player: chex.ArrayDevice
        action_mask: chex.ArrayDevice
        opponents: chex.ArrayDevice

    new_obs_list = []
    
    @jax.jit
    def transpose(collection):
        return jax.tree_map(lambda *xs: jnp.stack(xs), *collection)
    
    @jax.jit
    def transpose_concat(collection):
        return jax.tree_map(lambda *xs: jnp.concatenate(xs), *collection)
    
    total = len(obs_list)
    start = time.time()
    for i, obs in enumerate(obs_list):
        new_obs = [
            Observation(
                player=o["player"],
                action_mask=o["action_mask"],
                opponents=transpose(o["opponents"])
            )
            for o in obs.values()
        ]
        try:
            new_obs_list.append(transpose(new_obs))
        except Exception as e:
            print(new_obs)
    print(f'Converted arrays: {time.time() - start}')
    
    start = time.time()
    t = transpose_concat(new_obs_list)
    print(f'Transposed arrays: {time.time() - start}')
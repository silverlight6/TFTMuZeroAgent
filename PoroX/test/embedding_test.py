import pytest
import time

import jax

from PoroX.architectures.components.embedding import ChampionEmbedding, PlayerEmbedding
from PoroX.architectures.mctx_agent import RepresentationNetwork
import PoroX.modules.batch_utils as batch_utils
from PoroX.test.fixtures import obs, player, key, first_obs, env

def profile(N, f, *params):
    total = 0
    for _ in range(N):
        start = time.time()
        x = f(*params)
        total += time.time() - start
    avg = total / N
    print(f'{N} loops, {avg} per loop')

def test_champion_embedding(first_obs, key):
    obs = batch_utils.collect_obs(first_obs)
    champion_vectors = obs.player.champions
    
    champion_embedding = ChampionEmbedding()
    variables = champion_embedding.init(key, champion_vectors)

    @jax.jit
    def apply(variables, champion_vectors):
        return champion_embedding.apply(variables, champion_vectors)
    
    x = apply(variables, champion_vectors)
    print(x.shape)
    
    N=10000
    profile(N, apply, variables, champion_vectors)
    
def test_player_embedding(first_obs, key):
    obs = batch_utils.collect_obs(first_obs)
    players = obs.player
    
    player_embedding = PlayerEmbedding()
    variables = player_embedding.init(key, players)

    @jax.jit
    def apply(variables, players):
        return player_embedding.apply(variables, players)
    
    x = apply(variables, players)
    print(x.shape)
    
    N=10000
    profile(N, apply, variables, players)
    
def test_opponent_embedding(first_obs, key):
    obs = batch_utils.collect_obs(first_obs)
    opponents = obs.opponents
    
    player_embedding = PlayerEmbedding()
    variables = player_embedding.init(key, opponents)

    @jax.jit
    def apply(variables, players):
        return player_embedding.apply(variables, players)

    x = apply(variables, opponents)
    print(x.shape)
    
    N=10000
    profile(N, apply, variables, opponents)
    
def test_representation_network(first_obs, key):
    obs = batch_utils.collect_obs(first_obs)
    
    repr_network = RepresentationNetwork()
    variables = repr_network.init(key, obs)

    @jax.jit
    def apply(variables, obs):
        return repr_network.apply(variables, obs)

    x = apply(variables, obs)
    print(x[0].shape)
    
    N=10000
    profile(N, apply, variables, obs)
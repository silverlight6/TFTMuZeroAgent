import pytest
import time

import jax
import jax.numpy as jnp
from clu import parameter_overview


from PoroX.modules.observation import PlayerObservation, BatchedObservation

from PoroX.architectures.components.embedding import (
    ChampionEmbedding, PlayerEmbedding, EmbeddingConfig,
    SegmentEncoding, PlayerSegmentFFN, SegmentConfig
)
from PoroX.architectures.mctx_agent import RepresentationNetwork
from PoroX.architectures.config import test_config

import PoroX.modules.batch_utils as batch_utils
from PoroX.test.utils import profile

def test_champion_embedding(first_obs, key):
    obs = batch_utils.collect_obs(first_obs)
    champion_vectors = obs.players.champions
    
    config = EmbeddingConfig()
    champion_embedding = ChampionEmbedding(config=config)
    variables = champion_embedding.init(key, champion_vectors)

    @jax.jit
    def apply(variables, champion_vectors):
        return champion_embedding.apply(variables, champion_vectors)
    
    x = apply(variables, champion_vectors)
    print(x.shape)
    
    N=1000
    profile(N, apply, variables, champion_vectors)
    
def test_player_embedding(first_obs, key):
    obs = batch_utils.collect_obs(first_obs)
    players = obs.players
    
    config = EmbeddingConfig()
    player_embedding = PlayerEmbedding(config=config)
    variables = player_embedding.init(key, players)

    @jax.jit
    def apply(variables, players):
        return player_embedding.apply(variables, players)
    
    x = apply(variables, players)
    print(x.shape)
    
    N=1000
    profile(N, apply, variables, players)
    
def test_opponent_embedding(first_obs, key):
    obs = batch_utils.collect_obs(first_obs)
    opponents = obs.opponents
    
    config = EmbeddingConfig()
    player_embedding = PlayerEmbedding(config=config)
    variables = player_embedding.init(key, opponents)

    @jax.jit
    def apply(variables, players):
        return player_embedding.apply(variables, players)

    x = apply(variables, opponents)
    print(x.shape)
    
    N=1000
    profile(N, apply, variables, opponents)
    
def test_representation_network(first_obs, key):
    obs = batch_utils.collect_shared_obs(first_obs)
    
    repr_network = RepresentationNetwork(config=test_config)
    variables = repr_network.init(key, obs)

    @jax.jit
    def apply(variables, obs):
        return repr_network.apply(variables, obs)

    x = apply(variables, obs)
    print(x.shape)
    
    N=10
    profile(N, apply, variables, obs)
    
def test_params_representation_network(first_obs, key):
    obs = batch_utils.collect_obs(first_obs)

    repr_network = RepresentationNetwork(config=test_config)
    variables = repr_network.init(key, obs)
    
    print(parameter_overview.get_parameter_overview(variables))
    
def test_representation_network_gpu(first_obs, key):
    print("Number of GPUs:", jax.device_count())
    print("GPU Backend:", jax.default_backend())
    # TODO: Gotta figure out how to get flax to work with jax-metal...
    
def test_segment_embedding(first_obs, key):
    obs = batch_utils.collect_obs(first_obs)
    e_config = EmbeddingConfig()
    s_config = SegmentConfig(segments=jnp.array([75]))

    player_embedding = PlayerEmbedding(config=e_config)
    pv = player_embedding.init(key, obs.players)
    x = player_embedding.apply(pv, obs.players)
    
    player_ffn = PlayerSegmentFFN(config=s_config)
    mv = player_ffn.init(key, x)
    x = player_ffn.apply(mv, x)
    print(x.shape)
    num_elements = x.shape[-2]
    
    segment_embedding = SegmentEncoding(s_config)
    ev = segment_embedding.init(key, x)

    @jax.jit
    def apply(ev, mv, x):
        x = player_ffn.apply(mv, x)
        x = segment_embedding.apply(ev, x)
        return x
    
    y = apply(ev, mv, x)
    print(y.shape)
    
    N=1000
    profile(N, apply, ev, mv, x)
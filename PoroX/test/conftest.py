import pytest
import jax

from Simulator.porox.tft_simulator import parallel_env, TFTConfig

from PoroX.modules.observation import PoroXObservation

from Simulator.porox.player import Player
from Simulator import pool

from PoroX.test.utils import sample_action

@pytest.fixture(scope='session', autouse=True)
def player():
    pool_pointer = pool.pool()
    player_id = 0
    player = Player(pool_pointer=pool_pointer, player_num=player_id)
    
    # TODO: Add some champions, items, traits, etc.

    return player

@pytest.fixture(scope='session', autouse=True)
def obs(player):
    return PoroXObservation(player)

@pytest.fixture(scope='session', autouse=True)
def env():
    config = TFTConfig(observation_class=PoroXObservation)
    return parallel_env(config)
    
@pytest.fixture(scope='session', autouse=True)
def key():
    return jax.random.PRNGKey(10)

@pytest.fixture(scope='session', autouse=True)
def first_obs(env):
    """Gets the first observation after a random action is taken."""
    obs, infos = env.reset()
    terminated = {agent: False for agent in env.agents}
    truncated = {agent: False for agent in env.agents}
    actions = {
        agent: sample_action(env, obs, agent)
        for agent in env.agents
        if (
            (agent in terminated and not terminated[agent])
            or (agent in truncated and not truncated[agent])
        )
    }
    obs,rew,terminated,truncated,info = env.step(actions)
    
    return obs
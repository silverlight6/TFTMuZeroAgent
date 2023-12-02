import pytest
import jax
import random
import numpy as np

from pettingzoo.utils.env import ActionType, AgentID, ObsType, ParallelEnv

from Simulator.porox.tft_simulator import parallel_env, TFTConfig

from PoroX.modules.observation import PoroXObservation

from Simulator.porox.player import Player
from Simulator import pool

@pytest.fixture
def player():
    pool_pointer = pool.pool()
    player_id = 0
    player = Player(pool_pointer=pool_pointer, player_num=player_id)
    
    # TODO: Add some champions, items, traits, etc.

    return player

@pytest.fixture
def obs(player):
    return PoroXObservation(player)

@pytest.fixture
def env():
    config = TFTConfig(observation_class=PoroXObservation)
    return parallel_env(config)
    
@pytest.fixture
def key():
    return jax.random.PRNGKey(10)

@pytest.fixture
def first_obs(env):
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
    
# --- Utils ---
def sample_action(
    env: ParallelEnv[AgentID, ObsType, ActionType],
    obs: dict[AgentID, ObsType],
    agent: AgentID,
) -> ActionType:
    agent_obs = obs[agent]
    if isinstance(agent_obs, dict) and "action_mask" in agent_obs:
        legal_actions = np.flatnonzero(agent_obs["action_mask"])
        if len(legal_actions) == 0:
            return 0
        return random.choice(legal_actions)
    return env.action_space(agent).sample()
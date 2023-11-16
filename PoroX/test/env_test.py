from __future__ import annotations
import pytest
import random
import warnings

import numpy as np

from pettingzoo.test.api_test import missing_attr_warning
from pettingzoo.utils.conversions import (
    aec_to_parallel_wrapper,
    parallel_to_aec_wrapper,
    turn_based_aec_to_parallel_wrapper,
)
from pettingzoo.utils.env import ActionType, AgentID, ObsType, ParallelEnv
from pettingzoo.utils.wrappers import BaseWrapper
from pettingzoo.test import parallel_api_test

from Simulator.porox.tft_simulator import parallel_env, TFTConfig

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

@pytest.fixture
def env():
    return parallel_env()

def test_parallel_api(env):
    parallel_api_test(env)
    
def test_ui_render():
    config = TFTConfig(render_mode="json")
    env = parallel_env(config)
    
    obs, infos = env.reset()
    
    terminated = {agent: False for agent in env.agents}
    truncated = {agent: False for agent in env.agents}
    
    
    while not all(terminated.values()):
        actions = {
            agent: sample_action(env, obs, agent)
            for agent in env.agents
            if (
                (agent in terminated and not terminated[agent])
                or (agent in truncated and not truncated[agent])
            )
        }
        
        obs, rew, terminated, truncated, info = env.step(actions)
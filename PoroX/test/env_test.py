from __future__ import annotations
import pytest
import random
import warnings
import numpy as np
import time

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

from PoroX.modules.observation import PoroXObservation
from PoroX.test.fixtures import env, sample_action


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
        
def test_env_speed(env):
    start = time.time()

    obs, infos = env.reset()
    terminated = {agent: False for agent in env.agents}
    truncated = {agent: False for agent in env.agents}
    
    total_action_time = 0
    N = 0

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
    
    print(f"Total time: {time.time() - start}")
    print(f"Avg action time: {total_action_time / N}")
from __future__ import annotations
import pytest
import time
from pettingzoo.test import parallel_api_test

from Simulator.tft_simulator import parallel_env, TFTConfig
from PoroX.test.utils import sample_action


# disable for now
# @pytest.mark.skip
def test_parallel_api(env):
    parallel_api_test(env)
    
@pytest.mark.skip
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
        
@pytest.mark.skip
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

import pytest
import time
import jax
import random
import numpy as np

from pettingzoo.utils.env import ActionType, AgentID, ObsType, ParallelEnv

from Simulator.tft_config import TFTConfig
from Simulator.tft_simulator import parallel_env

from PoroX.modules.observation import PoroXObservation

from Simulator.player import Player
from Simulator import pool

# --- Utils ---
def sample_action(
    env: ParallelEnv[AgentID, ObsType, ActionType],
    obs: dict,
    agent: AgentID,
) -> ActionType:
    agent_obs = obs[agent]
    if isinstance(agent_obs, dict) and "action_mask" in agent_obs:
        legal_actions = np.flatnonzero(agent_obs["action_mask"])
        if len(legal_actions) == 0:
            return 0
        return random.choice(legal_actions)
    return env.action_space(agent).sample()

def profile(N, f, *params):
    total = 0
    for _ in range(N):
        start = time.time()
        x = f(*params)
        total += time.time() - start
    avg = total / N
    print(f'{N} loops, {avg} per loop')


def batched_env_obs(N):
    config = TFTConfig(observation_class=PoroXObservation)
    
    obs_list = []
    
    for _ in range(N):
        env = parallel_env(config)
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
        obs, rew, terminated, truncated, info = env.step(actions)
        
        obs_list.append(obs)

    return obs_list

import gymnasium as gym
import numpy as np
import random
from stable_baselines3.common.env_util import make_vec_env
from Simulator import game_round
from Simulator.observation import Observation
import time

# env_kwargs = {"visualize":True, "past_steps":1}
# env = make_vec_env("random1-v0", n_envs=1, env_kwargs=env_kwargs)

env = gym.make("Simulator/TFT-Set4")

env.reset()
terminated = False
truncated = False
print(env.observation_space, env.observation_space.shape)
start_time = time.time()
while (not terminated):
    action = np.zeros(env.action_space.sample().shape)
    # print("ACTION", action)
    observation, reward, terminated, truncated, info = env.step(action)
    # print(observation.shape, len(observation[0]))
    # print(reward)
    # print(info)

print("--- %s seconds ---" % (time.time() - start_time))
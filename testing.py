import gymnasium as gym
import numpy as np
import random
import Simulator
from Simulator import game_round
from Simulator.observation import Observation
import time
from ray.rllib.algorithms.ppo import PPOConfig

# env = make_vec_env("singleplayergame", n_envs=1)

config = (  # 1. Configure the algorithm,
    PPOConfig()
    .environment("tftSingle-v0")
    .rollouts(num_rollout_workers=2)
    .framework("torch")
    .training(model={"fcnet_hiddens": [64, 64]})
    .evaluation(evaluation_num_workers=1)
)

algo = config.build()  # 2. build the algorithm,

for _ in range(5):
    print(algo.train())  # 3. train it,

algo.evaluate()

exit()


env = gym.make("tftSingle-v0")
observation = env.reset()
terminated = False
truncated = False
print(env.observation_space, env.observation_space.shape)
start_time = time.time()
while (not terminated):
    # action = np.zeros(env.action_space.sample().shape)
    # action = env.action_space.sample()
    action = np.random.rand(1,10+5+9+10+7+4+7+4)
    observation, reward, terminated, truncated, info = env.step(action)
    # print(observation.shape, len(observation[0]))
    # print(reward)
    # print(info)

print("--- %s seconds ---" % (time.time() - start_time))
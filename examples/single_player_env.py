"""Single-player campaign environment (build a board vs a generated opponent)."""

import numpy as np

from Simulator.generators.episode_collector import random_policy
from Simulator.simulators.tft_simulator import TFTConfig
from Simulator.simulators.tft_single_player_simulator import TFT_Single_Player_Simulator


def main():
    env = TFT_Single_Player_Simulator(TFTConfig(num_players=1, max_actions_per_round=15))
    observation, info = env.reset()
    terminated = False
    steps = 0
    max_steps = 30
    reward = 0

    while not terminated and steps < max_steps:
        action = np.asarray(random_policy(observation, info, "player_0", env))
        observation, reward, terminated, truncated, info = env.step(action)
        steps += 1

    print("single-player steps", steps, "terminated", terminated, "reward", reward)
    env.close()


if __name__ == "__main__":
    main()

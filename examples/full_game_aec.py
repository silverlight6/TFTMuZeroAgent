"""Turn-based AEC loop for the same full-game environment."""

from Simulator.simulators.tft_simulator import TFTConfig, env as tft_env
from Simulator.generators.episode_collector import random_policy


def main():
    env = tft_env(TFTConfig(num_players=8))
    env.reset()

    for agent in env.agent_iter():
        observation, reward, terminated, truncated, info = env.last()
        if terminated or truncated:
            action = None
        else:
            action = random_policy(observation, info, agent, env)
        env.step(action)

    print("AEC game finished.")
    env.close()


if __name__ == "__main__":
    main()

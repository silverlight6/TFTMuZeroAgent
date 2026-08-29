"""Full 8-player game through the PettingZoo parallel API."""

from Simulator.simulators.tft_simulator import TFTConfig, parallel_env
from Simulator.generators.episode_collector import random_policy


def main():
    env = parallel_env(TFTConfig(num_players=8))
    observations, infos = env.reset()
    terminated = {agent: False for agent in env.possible_agents}

    while not all(terminated.values()):
        actions = {
            agent: random_policy(observations.get(agent), infos.get(agent, {}), agent, env)
            for agent in env.agents
            if not terminated.get(agent, False)
        }
        observations, rewards, terminated, truncated, infos = env.step(actions)

    print("Game finished. Final rewards:", rewards)
    env.close()


if __name__ == "__main__":
    main()

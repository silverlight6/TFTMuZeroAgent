"""Collect full-game episodes and save them to disk.

This is the slim, model-agnostic version of the old data worker:
- default policy is a random legal action (see random_policy)
- start_turn / save_battle / state_empty are recorded on each transition
- finishing place is assigned when a player is eliminated (8 -> 1)
- swap in your own model by passing policy_fn to collect_episodes

    def my_policy(observation, info, agent, env):
        # observation["observations"], observation["action_mask"]
        # info["start_turn"], info["save_battle"], info["game_round"], info["player"]
        return your_model.act(observation, info)

    collect_episodes(env, num_episodes=10, policy_fn=my_policy)
"""

from Simulator.generators.episode_collector import collect_episodes
from Simulator.simulators.tft_simulator import TFTConfig, parallel_env


def main():
    env = parallel_env(TFTConfig(num_players=8))
    paths = collect_episodes(
        env,
        num_episodes=1,
        output_dir="examples/output/episodes",
        prefix="random",
    )
    env.close()
    print("Wrote", paths)


if __name__ == "__main__":
    main()

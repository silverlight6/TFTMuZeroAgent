"""One seat uses the heuristic Default_Agent; everyone else plays random legal actions."""

from Simulator.generators.episode_collector import collect_episode, random_policy
from Simulator.simulators.tft_simulator import TFTConfig, parallel_env
from Simulator.utils import decode_action


def mixed_policy(observation, info, agent, env):
    if agent != "player_0":
        return random_policy(observation, info, agent, env)

    player = info.get("player")
    if player is None:
        return random_policy(observation, info, agent, env)

    mask = observation.get("action_mask") if isinstance(observation, dict) else None
    action_str = player.default_policy(info.get("game_round", 1), info.get("shop"), mask)
    return decode_action([action_str])[0]


def main():
    env = parallel_env(TFTConfig(num_players=8))
    recorder = collect_episode(env, policy_fn=mixed_policy)
    env.close()
    print("placements", recorder.placements)
    print("player_0 steps", len(recorder.transitions["player_0"]))


if __name__ == "__main__":
    main()

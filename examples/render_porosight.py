"""Dump an episode as JSON that PoroSight can open.

After this script finishes, start PoroSight
(`cd Simulator/PoroSight && npm install && npm run dev`)
and load the JSON file written under Games/.

Usage:
    python examples/render_porosight.py
    python examples/render_porosight.py full_game
    python examples/render_porosight.py single_player
    python examples/render_porosight.py position
    python examples/render_porosight.py item
"""

import argparse
import sys

import numpy as np

from Simulator.generators.episode_collector import random_policy
from Simulator.simulators.tft_item_simulator import TFT_Item_Simulator
from Simulator.simulators.tft_position_simulator import TFT_Position_Simulator
from Simulator.simulators.tft_simulator import TFTConfig, parallel_env
from Simulator.simulators.tft_single_player_simulator import TFT_Single_Player_Simulator


def dump_full_game(render_path):
    env = parallel_env(TFTConfig(num_players=8, render_mode="porosight", render_path=render_path))
    observations, infos = env.reset()
    terminated = {agent: False for agent in env.possible_agents}

    while not all(terminated.values()):
        actions = {
            agent: random_policy(observations.get(agent), infos.get(agent, {}), agent, env)
            for agent in env.agents
            if not terminated.get(agent, False)
        }
        observations, rewards, terminated, truncated, infos = env.step(actions)

    print("Wrote PoroSight JSON under", render_path, "Final rewards:", rewards)
    env.close()


def dump_single_player(render_path):
    env = TFT_Single_Player_Simulator(
        TFTConfig(num_players=1, max_actions_per_round=15, render_mode="porosight", render_path=render_path)
    )
    observation, info = env.reset()
    terminated = False
    reward = 0
    while not terminated:
        action = np.asarray(random_policy(observation, info, "player_0", env))
        observation, reward, terminated, truncated, info = env.step(action)
    print("Wrote PoroSight JSON under", render_path, "Final reward:", reward)
    env.close()


def dump_position(render_path):
    env = TFT_Position_Simulator(render_mode="porosight", render_path=render_path)
    observation, info = env.reset()
    mask = observation["action_mask"]
    legal = np.argwhere(mask > 0)
    action = np.zeros(env.max_action_count, dtype=np.int64)
    if len(legal):
        action[0] = legal[0][1]
    observation, reward, terminated, truncated, info = env.step(action)
    print("Wrote PoroSight JSON under", render_path, "position reward", reward)
    env.close()


def dump_item(render_path):
    env = TFT_Item_Simulator(render_mode="porosight", render_path=render_path)
    observation, info = env.reset()
    mask = observation["action_mask"]
    action = np.zeros(10, dtype=np.int64)
    legal = np.argwhere(mask > 0)
    if len(legal):
        action[legal[0][0]] = legal[0][1]
    observation, reward, terminated, truncated, info = env.step(action)
    print("Wrote PoroSight JSON under", render_path, "item reward", reward)
    env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Write a PoroSight JSON replay")
    parser.add_argument(
        "kind",
        nargs="?",
        default="full_game",
        choices=["full_game", "single_player", "position", "item"],
    )
    parser.add_argument("--render-path", default="Games")
    args = parser.parse_args(argv)

    dumps = {
        "full_game": dump_full_game,
        "single_player": dump_single_player,
        "position": dump_position,
        "item": dump_item,
    }
    dumps[args.kind](args.render_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

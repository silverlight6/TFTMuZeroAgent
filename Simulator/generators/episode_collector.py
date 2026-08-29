"""Model-agnostic episode collection for the TFT simulator.

This is a slim replacement for the old Ray/MuZero data worker. Plug in any
policy that maps (observation, info, agent, env) -> action. The default policy
samples a legal random action.

Info keys worth handling at each step:
    start_turn   – first action of a player round (snapshot board / shop here)
    save_battle  – env flagged this player's upcoming battle as worth saving
    state_empty  – skip storing empty terminal shells
    actions_taken, game_round, player, shop
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from Simulator.encoding.token.action import ActionToken

PolicyFn = Callable[[Any, Dict[str, Any], str, Any], Any]


def _as_mask(observation) -> Optional[np.ndarray]:
    if isinstance(observation, dict) and "action_mask" in observation:
        return np.asarray(observation["action_mask"])
    return None


def _action_class(env):
    unwrapped = getattr(env, "unwrapped", env)
    return getattr(unwrapped, "action_class", None) or ActionToken


def random_policy(observation, info, agent, env):
    """Sample a legal action from the action mask, or fall back to the action space."""
    action_cls = _action_class(env)
    mask = _as_mask(observation)
    space = env.action_space(agent) if callable(getattr(env, "action_space", None)) else env.action_space

    if mask is not None and np.any(mask > 0):
        if hasattr(space, "nvec") and hasattr(action_cls, "mask_to_sample_mask"):
            sample = space.sample(mask=action_cls.mask_to_sample_mask(mask))
            return action_cls.action_space_to_action(sample)
        if mask.ndim == 1:
            legal = np.flatnonzero(mask > 0)
            return action_cls.action_space_to_action(int(legal[np.random.randint(len(legal))]))
        legal = np.argwhere(mask > 0)
        row, col = legal[np.random.randint(len(legal))]
        return action_cls.action_space_to_action(int(row * mask.shape[1] + col))

    return action_cls.decode_env_action(space.sample())


def _player_snapshot(agent: str, info: Dict[str, Any], observation) -> Dict[str, Any]:
    player = info.get("player")
    opponent = getattr(player, "opponent", None) if player is not None else None
    return {
        "agent": agent,
        "game_round": info.get("game_round"),
        "player_num": getattr(player, "player_num", None),
        "opponent_num": getattr(opponent, "player_num", None),
        "health": getattr(player, "health", None),
        "level": getattr(player, "level", None),
        "gold": getattr(player, "gold", None),
        "observation": observation,
    }


@dataclass
class EpisodeRecorder:
    """Accumulates per-agent trajectories and optional battle snapshots."""

    agents: List[str]
    transitions: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    placements: Dict[str, Optional[int]] = field(default_factory=dict)
    battles: List[Dict[str, Any]] = field(default_factory=list)
    turn_starts: List[Dict[str, Any]] = field(default_factory=list)
    elapsed_ns: int = 0

    def __post_init__(self):
        self.transitions = {agent: [] for agent in self.agents}
        self.placements = {agent: None for agent in self.agents}

    def add_transition(
        self,
        agent: str,
        observation,
        action,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Dict[str, Any],
    ) -> None:
        # Skip dead-player shells, not early-game states with no legal econ actions.
        if info.get("player") is None and info.get("state_empty"):
            return
        record = {
            "observation": observation,
            "action": np.asarray(action),
            "reward": float(reward) if reward is not None else 0.0,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "start_turn": bool(info.get("start_turn", False)),
            "save_battle": bool(info.get("save_battle", False)),
            "actions_taken": info.get("actions_taken"),
            "game_round": info.get("game_round"),
        }
        self.transitions[agent].append(record)
        if record["start_turn"]:
            self.turn_starts.append(_player_snapshot(agent, info, observation))
        if record["save_battle"]:
            self.battles.append(_player_snapshot(agent, info, observation))

    def set_placement(self, agent: str, place: int) -> None:
        if self.placements.get(agent) is None:
            self.placements[agent] = place

    def finalize_placements(self, infos: Dict[str, Any]) -> None:
        remaining = [agent for agent in self.agents if self.placements[agent] is None]
        if not remaining:
            return

        def sort_key(agent):
            player = infos.get(agent, {}).get("player")
            return getattr(player, "health", 0) or 0

        remaining.sort(key=sort_key, reverse=True)
        for place, agent in enumerate(remaining, start=1):
            self.placements[agent] = place

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "agents": np.array(self.agents),
            "placements": np.array(
                [self.placements[agent] if self.placements[agent] is not None else -1
                 for agent in self.agents],
                dtype=np.int16,
            ),
            "elapsed_ns": np.array(self.elapsed_ns),
            "battles": np.array(self.battles, dtype=object),
            "turn_starts": np.array(self.turn_starts, dtype=object),
        }
        for agent in self.agents:
            payload[f"{agent}_transitions"] = np.array(self.transitions[agent], dtype=object)
        np.savez_compressed(path, **payload)
        return path


def collect_episode(
    env,
    policy_fn: Optional[PolicyFn] = None,
    max_steps: Optional[int] = None,
) -> EpisodeRecorder:
    """Play one full parallel game and record transitions.

    End-of-round behavior:
    - `start_turn` marks the first step of a new player round.
    - `save_battle` snapshots are stored separately for position/item datasets.
    - When an agent first becomes terminated, they receive a finishing place
      (worst remaining place, counting down from num_players to 1).
    """
    if policy_fn is None:
        policy_fn = random_policy

    observations, infos = env.reset()
    agents = list(env.possible_agents)
    recorder = EpisodeRecorder(agents)
    terminated = {agent: False for agent in agents}
    previously_done = set()
    next_place = len(agents)
    steps = 0
    started = time.time_ns()

    while not all(terminated.get(agent, False) for agent in agents):
        actions = {}
        living = getattr(env, "agents", None) or [
            agent for agent in agents if not terminated.get(agent, False)
        ]
        for agent in living:
            if terminated.get(agent, False):
                continue
            obs = observations.get(agent) if isinstance(observations, dict) else observations
            info = infos.get(agent, {}) if isinstance(infos, dict) else {}
            actions[agent] = np.asarray(policy_fn(obs, info, agent, env))

        if not actions:
            break

        observations, rewards, terminated, truncated, infos = env.step(actions)
        if not isinstance(terminated, dict):
            terminated = {agents[0]: bool(terminated)}
        if not isinstance(truncated, dict):
            truncated = {agent: False for agent in agents}
        if not isinstance(rewards, dict):
            rewards = {agent: rewards for agent in actions}
        if not isinstance(infos, dict):
            infos = {agent: infos for agent in actions}

        for agent, action in actions.items():
            info = infos.get(agent, {})
            obs = observations.get(agent) if isinstance(observations, dict) else observations
            recorder.add_transition(
                agent,
                obs,
                action,
                rewards.get(agent, 0.0),
                bool(terminated.get(agent, False)),
                bool(truncated.get(agent, False)),
                info,
            )

        for agent in agents:
            if terminated.get(agent, False) and agent not in previously_done:
                recorder.set_placement(agent, next_place)
                next_place -= 1
                previously_done.add(agent)

        steps += 1
        if max_steps is not None and steps >= max_steps:
            break

    recorder.finalize_placements(infos if isinstance(infos, dict) else {})
    recorder.elapsed_ns = time.time_ns() - started
    return recorder


def collect_episodes(
    env,
    num_episodes: int = 1,
    output_dir: str = "episodes",
    policy_fn: Optional[PolicyFn] = None,
    max_steps: Optional[int] = None,
    prefix: str = "episode",
) -> List[Path]:
    """Collect `num_episodes` games and write compressed npz files to `output_dir`."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    saved = []
    for episode_i in range(num_episodes):
        recorder = collect_episode(env, policy_fn=policy_fn, max_steps=max_steps)
        path = output / f"{prefix}_{episode_i:05d}.npz"
        recorder.save(path)
        saved.append(path)
        print(
            f"Episode {episode_i} finished in {recorder.elapsed_ns / 1e9:.2f}s "
            f"placements={recorder.placements} -> {path}"
        )
    return saved

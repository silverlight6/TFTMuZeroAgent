"""Two-tier seeding for training runs and deterministic env playouts.

Category 1 — run seed
    Trainer / experiment level.  Derive a distinct episode seed for every
    (env_index, episode_index) pair so a research run can start from the same
    conditions.  Does not by itself guarantee bit-identical playouts across
    different parallel layouts.

Category 2 — episode seed
    Per ``env.reset(seed=...)``.  Drives an isolated :class:`EnvRNG` so the
    same seed reproduces the same game (shop rolls, combat, rewards), including
    under in-process parallel execution once combat state is env-local.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

_MASK64 = (1 << 64) - 1
_MASK31 = (1 << 31) - 1


def _splitmix64(value: int) -> int:
    """Stable, platform-independent 64-bit mix (SplitMix64)."""
    value = (value + 0x9E3779B97F4A7C15) & _MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    return value ^ (value >> 31)


def derive_episode_seed(run_seed: int, env_index: int = 0, episode_index: int = 0) -> int:
    """Derive a Category-2 episode seed from a Category-1 run seed.

    The mix is deterministic and does not touch process-global RNG state.
    """
    mixed = _splitmix64(int(run_seed) & _MASK64)
    mixed = _splitmix64(mixed ^ (_splitmix64(int(env_index) + 1)))
    mixed = _splitmix64(mixed ^ (_splitmix64(int(episode_index) + 1) << 1))
    return int(mixed & _MASK31)


class NPRandomFacade:
    """Subset of ``numpy.random`` used by the simulator, backed by a Generator."""

    def __init__(self, generator: np.random.Generator):
        self._g = generator

    def randint(self, low, high=None, size=None):
        if high is None:
            high = low
            low = 0
        result = self._g.integers(low, high, size=size)
        if size is None:
            return int(result)
        return result

    def rand(self, *size):
        if not size:
            return float(self._g.random())
        if len(size) == 1:
            return self._g.random(size[0])
        return self._g.random(size)

    def choice(self, a, size=None, replace=True, p=None):
        if size is None and p is not None:
            index = int(self._g.choice(len(a), p=p))
            return a[index]
        result = self._g.choice(a, size=size, replace=replace, p=p)
        if size is None:
            return result.item() if hasattr(result, "item") else result
        return result

    def seed(self, *_args, **_kwargs):
        return None


@dataclass
class EnvRNG:
    """Isolated Python + NumPy generators created from an episode seed."""

    py: random.Random
    np: np.random.Generator
    episode_seed: Optional[int] = None
    np_api: NPRandomFacade = field(init=False)

    def __post_init__(self):
        self.np_api = NPRandomFacade(self.np)

    @classmethod
    def from_episode_seed(cls, episode_seed: Optional[int] = None) -> "EnvRNG":
        if episode_seed is None:
            episode_seed = int.from_bytes(os.urandom(8), "little") & _MASK31
        seed = int(episode_seed) & _MASK31
        return cls(
            py=random.Random(seed),
            np=np.random.default_rng(seed),
            episode_seed=seed,
        )


def seed_info(episode_seed, run_seed) -> dict:
    return {"episode_seed": episode_seed, "run_seed": run_seed}


def expand_vector_seeds(num_envs: int, seeds=None, options=None, run_seed=None):
    """Turn a run seed into per-env episode seeds for vector wrappers."""
    if isinstance(options, dict):
        if run_seed is None:
            run_seed = options.get("run_seed")
        options = [dict(options) for _ in range(num_envs)]
    options = list(options) if options is not None else [None] * num_envs
    if len(options) < num_envs:
        options = options + [None] * (num_envs - len(options))
    if run_seed is not None and (seeds is None or all(s is None for s in seeds)):
        seeds = [derive_episode_seed(run_seed, index, 0) for index in range(num_envs)]
        options = [dict(opt or {}, run_seed=run_seed) for opt in options]
    seeds = list(seeds) if seeds is not None else [None] * num_envs
    if len(seeds) < num_envs:
        seeds = seeds + [None] * (num_envs - len(seeds))
    return seeds, options

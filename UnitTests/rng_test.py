"""Unit tests for the two-tier seeding model."""

from Simulator.rng import EnvRNG, derive_episode_seed, expand_vector_seeds


def test_derive_episode_seed_is_stable():
    assert derive_episode_seed(42, 0, 0) == derive_episode_seed(42, 0, 0)
    assert derive_episode_seed(42, 0, 0) != derive_episode_seed(42, 1, 0)
    assert derive_episode_seed(42, 0, 0) != derive_episode_seed(42, 0, 1)
    assert derive_episode_seed(1, 0, 0) != derive_episode_seed(2, 0, 0)


def test_derive_episode_seed_is_platform_int():
    seed = derive_episode_seed(123456789, 7, 3)
    assert isinstance(seed, int)
    assert 0 <= seed < 2**31


def test_env_rng_from_same_episode_seed_matches():
    a = EnvRNG.from_episode_seed(7)
    b = EnvRNG.from_episode_seed(7)
    assert [a.py.random() for _ in range(8)] == [b.py.random() for _ in range(8)]
    assert list(a.np.integers(0, 100, size=8)) == list(b.np.integers(0, 100, size=8))


def test_env_rng_does_not_touch_process_global():
    import random

    import numpy as np

    random.seed(0)
    np.random.seed(0)
    before_py = random.random()
    before_np = float(np.random.random())

    random.seed(0)
    np.random.seed(0)
    EnvRNG.from_episode_seed(99).py.random()
    EnvRNG.from_episode_seed(99).np.random()

    after_py = random.random()
    after_np = float(np.random.random())
    assert before_py == after_py
    assert before_np == after_np


def test_expand_vector_seeds_from_run_seed():
    seeds, options = expand_vector_seeds(4, run_seed=42)
    assert seeds == [derive_episode_seed(42, i, 0) for i in range(4)]
    assert all(opt["run_seed"] == 42 for opt in options)

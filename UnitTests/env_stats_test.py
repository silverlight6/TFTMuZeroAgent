"""Smoke tests for the environment-statistics measurement helpers."""

from pathlib import Path
import sys

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "paper" / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from env_stats_lib import (  # noqa: E402
    observation_component_bytes,
    phase_of_round,
    quiet_runtime,
    reshape_shop_mask,
    run_position_episode,
    space_nbytes,
    tree_nbytes,
)


def test_phase_of_round_bins():
    assert phase_of_round(1) == "early"
    assert phase_of_round(10) == "early"
    assert phase_of_round(11) == "mid"
    assert phase_of_round(21) == "mid"
    assert phase_of_round(22) == "late"
    assert phase_of_round(40) == "late"


def test_reshape_shop_mask():
    flat = np.ones(55 * 38, dtype=np.int8)
    assert reshape_shop_mask(flat).shape == (55, 38)
    grid = np.ones((55, 38), dtype=np.int8)
    assert reshape_shop_mask(grid).shape == (55, 38)


def test_tree_and_space_nbytes():
    quiet_runtime()
    from gymnasium.spaces import Box, Dict

    payload = {
        "observations": {"board": np.zeros((8, 28, 5), dtype=np.int16)},
        "action_mask": np.zeros(2090, dtype=np.int8),
    }
    parts = observation_component_bytes(payload)
    assert parts["action_mask"] == 2090
    assert parts["observations_total"] == 8 * 28 * 5 * 2
    assert tree_nbytes(payload) == parts["obs_plus_mask"]

    space = Dict({
        "observations": Box(0, 1, (4,), np.float32),
        "action_mask": Box(0, 1, (8,), np.int8),
    })
    assert space_nbytes(space) == 4 * 4 + 8


def test_position_episode_records_combat():
    quiet_runtime()
    from env_stats_lib import install_combat_clock

    clock = install_combat_clock()
    result = run_position_episode("random", seed=0, level=2, clock=clock)
    assert result["steps"] == 1
    assert result["episode_s"] > 0
    assert result["combats"] >= 1
    assert "early" in result["phases"]


def test_full_game_episode_terminates():
    quiet_runtime()
    from env_stats_lib import install_combat_clock, run_full_game_episode

    clock = install_combat_clock()
    result = run_full_game_episode("random", seed=1, clock=clock)
    assert result["cycles"] < 2000
    assert result["actions"] > 100
    assert result["combats"] >= 1
    assert result["episode_s"] > 0.5

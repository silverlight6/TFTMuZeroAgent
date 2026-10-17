"""Seats are walked in a fixed order, so a seed reproduces the same game in any process.

PlayerManager kept the seat names in a set. Python randomizes str hashes per process
(PYTHONHASHSEED), so the set order, and with it the seat -> player_num mapping and the
order in which seats draw shops, PvE loot and carousel picks from the shared RNG, changed
from process to process for the same episode seed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from Simulator.battle.combat_context import CombatContext
from Simulator.game.player_manager import PlayerManager
from Simulator.game.pool import pool
from Simulator.rng import EnvRNG
from Simulator.simulators.tft_simulator import TFTConfig

REPO_ROOT = Path(__file__).resolve().parents[1]

_SCRIPT = r"""
import contextlib, io, json
from Simulator.simulators.tft_simulator import TFT_Simulator, TFTConfig
with contextlib.redirect_stdout(io.StringIO()):
    env = TFT_Simulator(TFTConfig())
    env.reset(seed=11)
states = env.player_manager.player_states
print(json.dumps({
    "order": list(env.player_manager.players),
    "player_num": {seat: states[seat].player_num for seat in sorted(states)},
    "shop": {seat: states[seat].shop for seat in sorted(states)},
    "bench": {seat: [c.name for c in states[seat].bench if c] for seat in sorted(states)},
}))
"""


def _reset_in_fresh_process(hash_seed, workdir):
    env = dict(os.environ, PYTHONHASHSEED=str(hash_seed), PYTHONPATH=str(REPO_ROOT))
    result = subprocess.run([sys.executable, "-c", _SCRIPT], cwd=workdir, env=env,
                            capture_output=True, text=True, timeout=600, check=True)
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_seat_names_map_to_their_player_num_in_seat_order():
    with CombatContext(rng=EnvRNG.from_episode_seed(1)).bind():
        manager = PlayerManager(8, pool(), TFTConfig())
    seats = [f"player_{i}" for i in range(8)]
    assert list(manager.players) == seats
    assert [manager.player_states[seat].player_num for seat in seats] == list(range(8))
    assert list(manager.player_states) == seats


def test_same_seed_same_game_under_any_hash_seed(tmp_path):
    runs = [_reset_in_fresh_process(hash_seed, tmp_path) for hash_seed in (0, 1, 2)]
    for run in runs:
        assert run["player_num"] == {f"player_{i}": i for i in range(8)}
    assert runs[0] == runs[1] == runs[2]

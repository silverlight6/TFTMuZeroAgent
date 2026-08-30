"""Episode-seed reproducibility under serial and concurrent env execution.

Two seed categories:

* **Run seed** (trainer / experiment): derive per-env episode seeds via
  ``derive_episode_seed``.  See ``Simulator.rng``.
* **Episode seed** (``env.reset(seed=...)``): bit-identical playout of one
  environment.  This file validates Category 2.

Environment variables
    TFT_CONCURRENCY_BATCH
        Number of threaded same-seed episodes (default 128; use 1024 for a
        heavier stress run).
    TFT_CONCURRENCY_WORKERS
        Thread pool size (default ``min(64, 2 * cpu_count)``).
"""

from __future__ import annotations

import hashlib
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

import numpy as np

from Simulator import config

SEED = 7
LEVEL = 2
# Default is large enough to expose races without making the suite glacial.
# Override for a heavier stress run, e.g. TFT_CONCURRENCY_BATCH=1024.
BATCH = int(os.environ.get("TFT_CONCURRENCY_BATCH", "128"))
MAX_WORKERS = int(os.environ.get("TFT_CONCURRENCY_WORKERS", "0")) or min(64, (os.cpu_count() or 8) * 2)
PASS_ACTION = np.full(12, 28, dtype=np.int64)


def _quiet():
    config.DEBUG = False
    config.PRINTMESSAGES = False
    config.LOGMESSAGES = False
    config.LOG_COMBAT = False
    import Simulator.game.player as player_mod

    player_mod.DEBUG = False


def _make_env():
    from Simulator.generators.battle_generator import BattleGenerator
    from Simulator.simulators.tft_position_simulator import TFT_Position_Simulator

    env = TFT_Position_Simulator()
    level = max(0, min(LEVEL, len(env.leveling_system.levels) - 1))
    env.leveling_system.level = level
    env.leveling_system.battle_generator = BattleGenerator(env.leveling_system.levels[level])
    return env


def _board_units(player) -> Tuple[Tuple[Any, ...], ...]:
    units = []
    for x in range(7):
        for y in range(4):
            unit = player.board[x][y]
            if unit is None:
                continue
            units.append(
                (
                    x,
                    y,
                    unit.name,
                    int(unit.stars),
                    bool(getattr(unit, "survive_combat", False)),
                    tuple(unit.items) if getattr(unit, "items", None) else (),
                )
            )
    return tuple(units)


def _obs_digest(observation: Dict[str, Any]) -> str:
    parts = []
    obs = observation["observations"]
    for key in sorted(obs.keys()):
        parts.append(key.encode())
        parts.append(np.ascontiguousarray(obs[key]).tobytes())
    return hashlib.sha1(b"".join(parts)).hexdigest()


def run_seeded_episode(worker_id: int = 0) -> Dict[str, Any]:
    """Reset with a fixed seed, take a no-op position action, fingerprint the result."""
    _quiet()
    env = None
    try:
        env = _make_env()
        observation, info = env.reset(seed=SEED)
        pre_digest = _obs_digest(observation)
        pre_units = _board_units(env.PLAYER)
        opponent_pre = _board_units(env.PLAYER.opponent)

        observation2, reward, terminated, truncated, _info2 = env.step(PASS_ACTION.copy())
        post_digest = _obs_digest(observation2)
        post_units = _board_units(env.PLAYER)
        opponent_post = _board_units(env.PLAYER.opponent)

        return {
            "ok": True,
            "worker": worker_id,
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "num_units": int(info.get("num_units", -1)),
            "pre_digest": pre_digest,
            "post_digest": post_digest,
            "pre_units": pre_units,
            "post_units": post_units,
            "opponent_pre": opponent_pre,
            "opponent_post": opponent_post,
            "fingerprint": (
                float(reward),
                bool(terminated),
                pre_digest,
                post_digest,
                pre_units,
                post_units,
                opponent_pre,
                opponent_post,
            ),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 — races often surface as exceptions
        return {
            "ok": False,
            "worker": worker_id,
            "reward": None,
            "fingerprint": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        if env is not None:
            env.close()


def _summarize(results: List[Dict[str, Any]]) -> str:
    errors = [r["error"] for r in results if not r["ok"]]
    fps = [r["fingerprint"] for r in results if r["ok"]]
    counts = Counter(fps)
    lines = [
        f"ok={len(fps)}/{len(results)} unique_fingerprints={len(counts)} errors={len(errors)}",
    ]
    for fp, n in counts.most_common(5):
        reward, terminated, pre, post, *_rest = fp
        lines.append(
            f"  n={n} reward={reward} term={terminated} pre={pre[:12]} post={post[:12]}"
        )
    err_counts = Counter(errors)
    for msg, n in err_counts.most_common(5):
        lines.append(f"  error n={n}: {msg}")
    return "\n".join(lines)


def test_same_seed_serial_reproducible():
    """Same episode seed, sequential runs must match."""
    _quiet()
    results = [run_seeded_episode(i) for i in range(8)]
    assert all(r["ok"] for r in results), _summarize(results)
    fingerprints = {r["fingerprint"] for r in results}
    assert len(fingerprints) == 1, _summarize(results)


def test_episode_seed_golden_snapshot():
    """Pinned episode seed=7 / level=2 playout used as a serial golden file."""
    _quiet()
    first = run_seeded_episode(0)
    second = run_seeded_episode(1)
    assert first["ok"] and second["ok"], _summarize([first, second])
    assert first["fingerprint"] == second["fingerprint"]
    assert first["terminated"] is True
    assert first["num_units"] >= 1
    assert first["pre_digest"]
    assert first["post_digest"]


def test_same_seed_threaded_batch_reproducible():
    """Same episode seed, many threads: identical playouts and no races."""
    _quiet()
    baseline = run_seeded_episode(-1)
    assert baseline["ok"], baseline["error"]

    workers = MAX_WORKERS
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_seeded_episode, i) for i in range(BATCH)]
        results = [fut.result() for fut in as_completed(futures)]

    summary = (
        f"batch={BATCH} workers={workers} baseline_reward={baseline['reward']}\n"
        + _summarize(results)
    )
    print(summary)

    failures = [r for r in results if not r["ok"]]
    assert not failures, summary

    fingerprints = {r["fingerprint"] for r in results}
    assert fingerprints == {baseline["fingerprint"]}, summary

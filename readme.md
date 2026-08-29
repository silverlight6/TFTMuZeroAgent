# TFT Set 4 Simulator

A Teamfight Tactics Set 4 battle simulator for reinforcement learning. Shops, combat, items, and traits live here. Bring your own model.

The full lobby is a multi-agent [PettingZoo](https://pettingzoo.farama.org/) environment. Positioning, items, and single-player are extra [Gymnasium](https://gymnasium.farama.org/) environments. Each env has its own page under [markdown/](markdown/README.md).

## Install

Python 3.10+. Dependencies: `numpy>=2.0`, `PettingZoo>=1.27`, `gymnasium>=1.3`.

```
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Quick start

```python
from Simulator import TFTConfig, parallel_env, ObservationToken

env = parallel_env(TFTConfig(observation_class=ObservationToken, num_players=8))
obs, infos = env.reset()
obs, rewards, terminations, truncations, infos = env.step(actions)
env.close()
```

Every env returns `{"observations": ..., "action_mask": ...}`. The default full-game action space is `Discrete(55 * 38)` so PettingZoo can sample with a 1D mask. Pass `action_class=ActionVector` (`Discrete(1296)`) or `action_class=ActionMultiDiscrete` (7-D) to use a compact space. `step` also accepts a `(from, to)` pair or a length-3 command (`pass` / `level` / `refresh` / `buy` / `sell` / `move` / `item`). Decode a sampled action with `action_class.action_space_to_action`.

Public imports (`from Simulator import ...`): `TFTConfig`, `env`, `parallel_env`, the Gym envs, `ActionToken` / `ActionVector` / `ActionMultiDiscrete`, `ObservationToken` / `ObservationVector`, `Default_Agent`, and `collect_episode` / `collect_episodes` / `random_policy`.

## Environments

| API | Kind | Docs |
|-----|------|------|
| `parallel_env` / `env` | Full 8-player game (PettingZoo) | [full_game.md](markdown/full_game.md) |
| `TFT_Position_Simulator` | Positioning before one fight | [position.md](markdown/position.md) |
| `TFT_Item_Simulator` | Item assignment before one fight | [item.md](markdown/item.md) |
| `TFT_Single_Player_Simulator` | Solo campaign | [single_player.md](markdown/single_player.md) |
| `TFT_Vector_Pos_Simulator` | Batched position envs | [vector_position.md](markdown/vector_position.md) |
| `TFT_Single_Player_Vector_Simulator` | Batched solo campaigns | [vector_single_player.md](markdown/vector_single_player.md) |

## Collecting episodes

`Simulator.generators.episode_collector` records full-game trajectories. The default policy is a random legal action. Pass `policy_fn(observation, info, agent, env)` to use your model. Finishing place is assigned when a player dies (8 down to 1). Files are `numpy.savez_compressed`.

```
python examples/collect_episodes.py
```

## Examples

| Script | What it shows |
|--------|----------------|
| `examples/full_game_parallel.py` | Parallel multi-agent loop |
| `examples/full_game_aec.py` | AEC turn-based loop |
| `examples/collect_episodes.py` | Save trajectories, turn/battle flags, placement |
| `examples/position_env.py` | Positioning env |
| `examples/item_env.py` | Item env |
| `examples/single_player_env.py` | Single-player env |
| `examples/vector_position_env.py` | Vectorized position envs |
| `examples/default_agent_vs_random.py` | Heuristic `Default_Agent` vs random |
| `examples/render_porosight.py` | JSON dump for PoroSight |

## Layout

| Folder | Contents |
|--------|----------|
| `Simulator/simulators/` | PettingZoo / Gymnasium env APIs |
| `Simulator/battle/` | Combat, champions, items, traits, field |
| `Simulator/game/` | Players, pool, rounds, shops, actions |
| `Simulator/generators/` | Battle generators, default agent, episode collector |
| `Simulator/encoding/` | Observation and action encodings |
| `Simulator/PoroSight/` | Optional Svelte replay viewer |
| `markdown/` | Per-env usage docs |
| `examples/` | Minimal loops for each env |
| `UnitTests/` | Simulator and official API checkers |

## PoroSight

Optional Svelte replay viewer. Set `render_mode="porosight"` on the full game, single-player, position, or item env. Writes JSON under `render_path` (default `Games/`). `None` does not record.

```
python examples/render_porosight.py              # full game
python examples/render_porosight.py position
python examples/render_porosight.py item
python examples/render_porosight.py single_player
cd Simulator/PoroSight
npm install
npm run dev
```

Load the JSON under `Games/`. The UI follows the dump's `kind`.

## Tests

```
pytest UnitTests
```

`UnitTests/api_compliance_test.py` runs PettingZoo `api_test` / `parallel_api_test` and Gymnasium `check_env` on the public envs.

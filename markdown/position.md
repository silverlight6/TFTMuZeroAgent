# Position — `TFT_Position_Simulator`

**Module:** `Simulator.simulators.tft_position_simulator`  
**Kind:** Gymnasium; usually one step  
**Example:** `examples/position_env.py`

You are handed two boards (yours and an opponent) plus the rest of the lobby. The action only rearranges **your** units. One combat then plays. Reward is that fight.

Use this to train or evaluate hex placement without shops, gold, or a full match. For item assignment see [item.md](item.md). For a full lobby see [full_game.md](full_game.md).

## Create the env

```python
from Simulator import TFT_Position_Simulator

env = TFT_Position_Simulator()
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(action)
env.close()
```

### Constructor

| Arg | Default | Meaning |
|-----|---------|---------|
| `data_generator` | `None` | Queue of saved `[player, opponent, other_players]`. If missing or empty, `PositionLevelingSystem` builds a fight |
| `index` | `None` | Id when this env is one of a vector batch |
| `multi_step` | `False` | Reserved; prefer `step_until_units_placed` |
| `preset_battle` | `False` | Use the leveling system's preset composition |
| `step_until_units_placed` | `False` | One unit move per `step` until the board is set, then combat |
| `single_player` | `False` | Leveling system uses the single-player battle table |
| `render_mode` | `None` | `"porosight"` writes a JSON replay after combat |
| `render_path` | `"Games"` | Output directory for PoroSight dumps |

This env does not take `TFTConfig`. Observation encoding is `ObservationToken`.

## Observation

```python
{"observations": ..., "action_mask": ...}
```

`observations` is `ObservationToken.position_observation_space()`:

- `board` — `(8, 28, 5)` int16, you first, then the other seven
- `traits` — `(8, 102)` float32
- `action_count` — `(1, 12)` one-hot of how many unit-steps have been taken

`action_mask` is `(12, 29)` float32. Row `i` is unit `i` on your board. Columns `0–27` are destination hexes; `28` is pass. Rows past `info["num_units"]` stay unused.

`info["num_units"]` is how many units you must place.

## Actions

`action_space` is `MultiDiscrete` of length 12, each entry `0–28` (hex or pass).

Index `i` is “move unit `i` to this hex.” Extra slots beyond the unit count are ignored. Destination `28` (or hex `7` after decode) is a pass for that unit.

Single-step (default): one `step` applies all 12 slots, then combat, then `terminated=True`.

`step_until_units_placed=True`: each `step` places one unit. Combat and `terminated=True` happen when `action_count` reaches `num_units`. Reward is `0` on intermediate steps.

A legal first-unit move:

```python
import numpy as np

legal = np.argwhere(obs["action_mask"] > 0)
action = np.zeros(env.max_action_count, dtype=np.int64)
if len(legal):
    action[0] = legal[0][1]
```

## Reward

The combat result for your player after the move (`player.reward`). There is no placement bonus and no shop economy.

## Saved battles vs generated ones

If you pass a `data_generator` with at least `MINIMUM_POP_AMOUNT` entries, `reset` pops `[player, opponent, other_players]` so you can replay real boards. Otherwise the leveling system samples a random (or preset) fight. That is how you mix self-play snapshots with synthetic data.

## Recording / PoroSight

Pass `render_mode="porosight"` (and optionally `render_path`) to write a position dump after combat. `None` does not record.

```python
env = TFT_Position_Simulator(render_mode="porosight", render_path="Games")
```

The JSON `kind` is `"position"`. PoroSight shows your board vs the opponent and hides shop/economy. See `examples/render_porosight.py position`.

To run many of these envs at once, see [vector_position.md](vector_position.md).

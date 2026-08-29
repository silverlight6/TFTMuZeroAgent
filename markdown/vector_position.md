# Vector position — `TFT_Vector_Pos_Simulator`

**Module:** `Simulator.simulators.tft_vector_simulator`  
**Kind:** In-process batch of Gym position envs (not PettingZoo, not `gymnasium.vector`)  
**Example:** `examples/vector_position_env.py`

Runs `num_envs` copies of `TFT_Position_Simulator` in one process. Use this when you want many positioning rollouts per call without Ray.

Each sub-env is documented in [position.md](position.md). This page is only the batch API.

## Create the wrapper

```python
from Simulator import TFT_Vector_Pos_Simulator

vec = TFT_Vector_Pos_Simulator(num_envs=4)
obs_list, info_list = vec.vector_reset()
obs_list, rewards, terminated, truncated, info_list = vec.vector_step(action_list)
vec.close()
```

### Constructor

| Arg | Default | Meaning |
|-----|---------|---------|
| `num_envs` | `1` | How many position envs to hold |
| `data_generator` | `None` | Shared replay queue, same as the single position env |
| `preset_battle` | `False` | Preset compositions in every sub-env |
| `step_until_units_placed` | `False` | One unit per step in every sub-env |
| `single_player` | `False` | Single-player leveling table |
| `render_mode` | `None` | `"porosight"` writes one JSON file per sub-env |
| `render_path` | `"Games"` | Output directory for PoroSight dumps |

`observation_space` and `action_space` are copied from sub-env `0` (the position env’s `Dict` + `MultiDiscrete(12)`).

## API

`vector_reset(seeds=None, options=None)` — lists of length `num_envs`. Returns `(obs_list, info_list)`.

`vector_step(actions)` — `actions` is a Python list, one length-12 integer array per sub-env. Returns `(obs_list, rewards, terminated, truncated, info_list)`.

`reset_at(index)` / per-index retry: if a sub-env raises, it is rebuilt when `restart_failed_sub_environments` is true.

A legal first-unit move in every env:

```python
import numpy as np

actions = []
for obs in obs_list:
    action = np.zeros(12, dtype=np.int64)
    legal = np.argwhere(obs["action_mask"] > 0)
    if len(legal):
        action[0] = legal[0][1]
    actions.append(action)
```

## Curriculum

When `preset_battle` is false, the wrapper tracks recent mean reward. If the last several windows stay strong, it calls `level_up()` on every sub-env so `PositionLevelingSystem` serves harder boards. That is specific to this wrapper; a single `TFT_Position_Simulator` does not auto-level unless you call `env.level_up()` yourself.

## What this is not

It is not a PettingZoo Parallel env and not `gymnasium.vector.VectorEnv`. Trainers that expect those APIs need a thin adapter, or use one `TFT_Position_Simulator` at a time.

## Recording / PoroSight

`render_mode="porosight"` is passed through to every sub-env. Each env writes `game_<timestamp>_env{i}.json` under `render_path`.

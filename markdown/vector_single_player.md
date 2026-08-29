# Vector single-player — `TFT_Single_Player_Vector_Simulator`

**Module:** `Simulator.simulators.tft_vector_simulator`  
**Kind:** In-process batch of Gym single-player campaigns (not PettingZoo, not `gymnasium.vector`)

Runs `num_envs` copies of `TFT_Single_Player_Simulator` in one process. Use this when you want many solo campaigns per call without Ray.

Each sub-env is documented in [single_player.md](single_player.md). This page is only the batch API. For batched **positioning** fights, see [vector_position.md](vector_position.md).

## Create the wrapper

```python
from Simulator import TFTConfig, TFT_Single_Player_Vector_Simulator

config = TFTConfig(num_players=1, max_actions_per_round=15)
vec = TFT_Single_Player_Vector_Simulator(config, num_envs=4)
obs_list, info_list = vec.vector_reset()
obs_list, rewards, terminated, truncated, info_list = vec.vector_step(action_list)
vec.close()
```

`tft_config` is required (unlike the position vector wrapper). Every sub-env gets the same config. `observation_space` / `action_space` come from sub-env `0`: player-only `ObservationToken` plus `Discrete(55 * 38)`.

Set `TFTConfig(render_mode="porosight")` to record. Each sub-env writes `game_<timestamp>_env{i}.json` under `render_path`.

## API

`vector_reset(seeds=None, options=None)` — lists of length `num_envs`. Returns `(obs_list, info_list)`.

`vector_step(actions)` — `actions` is a Python list, one full-game action per sub-env (scalar index, `(from, to)`, or `[type, x1, x2]`). Returns `(obs_list, rewards, terminated, truncated, info_list)`.

Failed sub-envs are rebuilt when `restart_failed_sub_environments` is true.

Random legal actions across the batch:

```python
from Simulator.generators.episode_collector import random_policy
import numpy as np

actions = [
    np.asarray(random_policy(obs, info, "player_0", vec.envs[i]))
    for i, (obs, info) in enumerate(zip(obs_list, info_list))
]
```

## Reward and episode length

Same as the single-player env: `+1` / `-1` per resolved round, accumulated until that campaign ends. Sub-envs do **not** share a global timestep. One campaign can finish while another is still shopping; `terminated[i]` is per env. Reset that index (or the whole vector) when you want a new campaign.

There is no auto-leveling curriculum here. That exists only on `TFT_Vector_Pos_Simulator`.

## What this is not

It is not a PettingZoo Parallel env and not `gymnasium.vector.VectorEnv`. There is no `examples/` script for this wrapper yet; copy the `vector_reset` / `vector_step` pattern from `examples/vector_position_env.py` and swap in `TFT_Single_Player_Vector_Simulator`.

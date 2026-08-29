# Single-player — `TFT_Single_Player_Simulator`

**Module:** `Simulator.simulators.tft_single_player_simulator`  
**Kind:** Gymnasium, multi-step campaign  
**Example:** `examples/single_player_env.py`

One controlled player against generated opponents. Same shop vocabulary as the full game (buy, sell, move, items, level, refresh), but you never act as the other seven seats. After `max_actions_per_round` actions, a round resolves (`+1` / `-1`). The episode ends when the campaign is over.

Use this for a solo policy or a simpler credit-assignment problem than 8-player self-play. The full lobby is [full_game.md](full_game.md). To batch many campaigns, see [vector_single_player.md](vector_single_player.md).

## Create the env

```python
from Simulator import TFTConfig, TFT_Single_Player_Simulator

env = TFT_Single_Player_Simulator(TFTConfig(num_players=1, max_actions_per_round=15))
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(action)
env.close()
```

### `TFTConfig` fields this env uses

| Field | Default | Meaning |
|-------|---------|---------|
| `num_players` | set to `1` if omitted | Still builds an 8-seat pool; you only act as `player_0` |
| `max_actions_per_round` | `15` | Shop actions before the round fights |
| `action_class` | `ActionToken` | Same discrete grid as the full game |
| `multi_step_position` | `False` | Stored; shop actions stay the full `ActionToken` space |
| `render_mode` | `None` | `"porosight"` writes a campaign replay |
| `render_path` | `"Games"` | Output directory for PoroSight dumps |

Observation encoding is always `ObservationToken` (player-only, not the stacked 8-player tensor).

## Observation

```python
{"observations": ..., "action_mask": ...}
```

`observations` is `ObservationToken.player_observation_space()` — your scalars, embedding scalars, board, bench, shop, items, and traits. There is no opponent tensor in the obs (opponents still exist in the sim for combat).

`action_mask` is length `2090` (`55 * 38`) `int8`, same layout as the full game.

`info` includes `game_round` and, after a resolved fight, a `player_0` block with `state_empty` / `game_round`.

## Actions

Same as the full game. `action_space` is `Discrete(55 * 38)`.

| From (55) | Meaning |
|-----------|---------|
| 0–27 | Board hex |
| 28–36 | Bench slot |
| 37–46 | Item bench slot |
| 47–51 | Shop slot |
| 52 | Pass |
| 53 | Level |
| 54 | Refresh |

| To (38) | Meaning |
|---------|---------|
| 0–27 | Board hex |
| 28–36 | Bench slot |
| 37 | Sell |

`step` accepts a scalar index, a length-2 `(from, to)`, or a length-3 `[type, x1, x2]`. Decode with `ActionToken.action_space_to_action`.

Typical loop:

```python
from Simulator.generators.episode_collector import random_policy
import numpy as np

obs, info = env.reset()
terminated = False
while not terminated:
    action = np.asarray(random_policy(obs, info, "player_0", env))
    obs, reward, terminated, truncated, info = env.step(action)
```

## Reward

`+1` when a resolved round is a win, `-1` on a loss. Those values accumulate on the env until the campaign ends. There is no 8-player placement table.

## How a campaign is built

`reset` creates a pool and `player_0`, plays the opening carousel / minion wave, refreshes the shop, then returns the first observation. Later rounds come from `single_player_game_round.Game_Round`, which uses `PositionLevelingSystem` for the other side of the fight.

## Recording / PoroSight

Set `render_mode="porosight"` on `TFTConfig`. Recording starts on `reset` and writes JSON when the campaign ends. `None` / `False` does not record.

```python
env = TFT_Single_Player_Simulator(TFTConfig(num_players=1, render_mode="porosight", render_path="Games"))
```

The JSON `kind` is `"single_player"`. See `examples/render_porosight.py single_player`.

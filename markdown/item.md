# Item — `TFT_Item_Simulator`

**Module:** `Simulator.simulators.tft_item_simulator`  
**Kind:** Gymnasium, one step  
**Example:** `examples/item_env.py`

You are handed a board, an item bench, and opponents. The action only assigns items from the bench onto units (or passes). Combat runs **before** and **after** the assignment. Reward is the difference, so a no-op scores about `0`.

Use this to train item slams without shops or a full match. Hex placement is [position.md](position.md). The full lobby is [full_game.md](full_game.md).

## Create the env

```python
from Simulator import TFT_Item_Simulator

env = TFT_Item_Simulator()
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(action)
env.close()
```

`data_generator` is optional. If it is omitted or empty, `BattleGenerator` builds the fight and the item guide is “slam everything” (`np.ones((10, 2))`). `render_mode="porosight"` and `render_path` are optional constructor args.

A generator pop is `[player, opponent, other_players, item_guide]`. `item_guide` marks which of the 10 bench slots the policy is allowed to spend.

This env does not take `TFTConfig`. Observation encoding is `ObservationVector` (flattened binary vectors), not the token class used by the full game.

## Observation

```python
{"observations": {"player": ..., "opponents": (...)}, "action_mask": ...}
```

- `player` — `ObservationVector.player_observation_space()`: scalars, board, bench, shop, items, traits
- `opponents` — tuple of 7 public observations (scalars, board, traits)
- `action_mask` — `(10, 38)` float32, the item-bench slice of the full-game `55 × 38` mask

## Actions

`action_space` is `MultiDiscrete` of length 10 (one entry per item-bench cell). Each entry is `0–28`: a destination hex / bench slot, or pass.

`step` applies `item_controller` using that vector and `item_guide`, then plays the second combat and returns. The episode always terminates.

A legal first-item move:

```python
import numpy as np

mask = obs["action_mask"]
action = np.zeros(10, dtype=np.int64)
legal = np.argwhere(mask > 0)
if len(legal):
    action[legal[0][0]] = legal[0][1]
```

## Reward

`reward_after - reward_before` for the same pairing. Improving the fight (or losing by less) is positive. Leaving items unused should stay near zero.

## Saved battles vs generated ones

Pass a `data_generator` if you want item decisions on boards taken from full games. Otherwise every `reset` is a fresh random composition from `BattleGenerator`.

## Recording / PoroSight

Pass `render_mode="porosight"` (and optionally `render_path`) to write an item dump with the before/after combats. `None` does not record.

```python
env = TFT_Item_Simulator(render_mode="porosight", render_path="Games")
```

The JSON `kind` is `"item"`. See `examples/render_porosight.py item`.

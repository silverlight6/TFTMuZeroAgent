# Full game — `parallel_env` / `env`

**Module:** `Simulator.simulators.tft_simulator`  
**Kind:** PettingZoo Parallel and AEC  
**Examples:** `examples/full_game_parallel.py`, `examples/full_game_aec.py`, `examples/collect_episodes.py`, `examples/render_porosight.py`, `examples/default_agent_vs_random.py`

An 8-player (configurable) Set 4 match: shop phase, carousel, minions, PvP, and placement rewards. Use this for self-play, full-game policies, or episode collection.

Other envs in this folder isolate one decision (hexes, items) or drop to a single player. This one is the complete lobby.

## Create the env

```python
from Simulator import TFTConfig, parallel_env
from Simulator.observation.token.basic_observation import ObservationToken

env = parallel_env(TFTConfig(observation_class=ObservationToken, num_players=8))
```

Or import the module directly:

```python
from Simulator.simulators.tft_simulator import TFTConfig, parallel_env, env as tft_env
```

### `TFTConfig`

| Field | Default | Meaning |
|-------|---------|---------|
| `num_players` | `8` | Agents (`player_0` …) |
| `max_actions_per_round` | `15` | Shop-phase actions before that player is truncated |
| `reward_type` | `"winloss"` | Placement-style reward |
| `render_mode` | `None` | `"porosight"` writes a PoroSight dump |
| `render_path` | `"Games"` | Directory for JSON dumps |
| `observation_class` | `ObservationToken` | Board encoding |
| `action_class` | `ActionToken` | Action encoding |

Observation classes you can pass in:

- `Simulator.observation.token.basic_observation.ObservationToken` (default)
- `Simulator.observation.vector.observation.ObservationVector`
- `Simulator.observation.vector.gemini_observation.GeminiObservation`

## Parallel loop

Every living player acts on each `step`. `actions` is `{agent_id: action}`. Dead players leave `env.agents`.

```python
obs, infos = env.reset()
obs, rewards, terminations, truncations, infos = env.step(actions)
env.close()
```

## AEC loop

One agent per `step`. Pass `None` for a dead or truncated agent.

```python
from Simulator import TFTConfig, env as tft_env

env = tft_env(TFTConfig(num_players=8))
env.reset()
for agent in env.agent_iter():
    observation, reward, terminated, truncated, info = env.last()
    env.step(None if terminated or truncated else action)
env.close()
```

## Observation

Every `observe` / `reset` / `step` payload is:

```python
{"observations": ..., "action_mask": ...}
```

`observations` matches `observation_class.observation_space(num_players)` (token boards, traits, bench, items, shop, scalars, embedding scalars for `ObservationToken`). `action_mask` is a length-`2090` `int8` vector (`55 * 38`), `1` where the action is legal.

### Info keys (per agent)

- `start_turn` — first action of that player's shop phase
- `save_battle` — this player's upcoming fight is flagged for saving
- `state_empty` — skip empty terminal shells
- `player`, `shop`, `game_round`, `actions_taken`

## Actions

`ActionToken.action_space()` is `Discrete(55 * 38)` so PettingZoo can `sample(mask=...)`.

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

`step` accepts:

- a scalar index into `55 * 38` (what `action_space.sample(mask=...)` returns)
- a length-2 pair `(from, to)`
- a length-3 command `[type, x1, x2]` — `0` pass, `1` level, `2` refresh, `3` buy, `4` sell, `5` move, `6` item

Decode a sampled index with `ActionToken.action_space_to_action(index)`.

A shop phase is up to `max_actions_per_round` actions. After every living player is done (or truncated), combat runs and the next round starts.

## Reward

Win/loss from finishing place (first is highest). Combat damage also adds `player.reward`. The episode ends when one player is left or the round counter passes 48.

## Recording and rendering

Collect trajectories with `Simulator.generators.episode_collector` (`examples/collect_episodes.py`). The default policy is a random legal action; pass your own `policy_fn(observation, info, agent, env)`.

JSON for PoroSight (`kind: "full_game"`). Recording runs only when `render_mode="porosight"`:

```python
parallel_env(TFTConfig(num_players=8, render_mode="porosight", render_path="Games"))
```

Then `cd Simulator/PoroSight && npm install && npm run dev` and load the file under `Games/`. See `examples/render_porosight.py`.

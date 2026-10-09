# TFT MCP server

The Python stdio server operates one seeded eight-player TFT Set 4 game. `player_0` is controlled through MCP; seven opponents own existing `Default_Agent(False)` policies. The server starts, inspects, and closes games and queries static champion and trait rules. Gameplay scheduling and action tools arrive in later slices. Read the shared [Spec](../SPEC.md).

Install the unchanged simulator first, then the extension in a CPU-only virtual environment. Run these commands from the repository checkout, replacing `/absolute/path/tft-mcp-venv` with your environment path:

```sh
python3 -m venv /absolute/path/tft-mcp-venv
/absolute/path/tft-mcp-venv/bin/python -m pip install .
/absolute/path/tft-mcp-venv/bin/python -m pip install './mcp/server[dev]'
```

No training, model-hosting or GPU packages are required. Both distributions are separate installations; root package discovery and dependencies stay unchanged.

Launch the installed entry point from any working directory:

```sh
TFT_MCP_AUDIT_PATH=/absolute/path/logs/tft-audit.jsonl \
TFT_MCP_NATIVE_LOG_DIR=/absolute/path/logs/native \
/absolute/path/tft-mcp-venv/bin/tft-mcp
```

`TFT_MCP_AUDIT_PATH` is required for tool execution. Its parent directory is created when writable. `TFT_MCP_NATIVE_LOG_DIR` is optional and defaults to `native` beside the audit file. Every game gets a separate native directory containing unchanged simulator `log.txt` output. stdout carries only MCP messages; simulator diagnostics go to stderr. Failed starts retain diagnostic directories identified by the audit's `failed_start` record or the error's details.

On a Linux host that exports `APPIMAGE`, launch through `env -u APPIMAGE`:

```sh
TFT_MCP_AUDIT_PATH=/absolute/path/logs/tft-audit.jsonl \
env -u APPIMAGE /absolute/path/tft-mcp-venv/bin/tft-mcp
```

The tested T3 Code AppImage host makes Python report the AppImage executable and ignore virtual-environment packages when that variable is inherited. Removing it from the server environment preserves the installed Python interpreter. Configure the MCP client command as `/usr/bin/env` with arguments `-u`, `APPIMAGE`, and the absolute `tft-mcp` path when needed.

The launcher re-executes Python with `PYTHONHASHSEED=0` before loading the SDK or simulator. Audit records include that setting and an actual interpreter hash probe, seed, baseline identity and seed, simulator source digest, distribution version, interpreter and dependency versions, and configuration. Source checkouts also record their Git revision. Set `TFT_MCP_SIMULATOR_REVISION` to the installed source revision for installations without Git metadata; otherwise the source SHA-256 identifies that revision. Paths and IDs do not affect gameplay equality.

`start_game` requires exactly `{"seed": integer}` with a value from 0 through 2147483647. Booleans, decimal JSON numbers, missing seeds and unknown arguments fail. `get_game_status` and `close_game` accept `{}` only. Status includes `state`, `game_id`, `controlled_player_id`, `round`, `planning_budget` and `outcome`. Idle game fields are null. Planning budget and terminal outcome remain null until the progression slice implements them.

Starting an active game returns `game_active` and preserves it. Closing a running game returns its ID, an outcome with `reason: "closed_incomplete"`, and idle status. Repeated close is harmless. Structured errors contain `code`, `message`, and `details`, with MCP `isError: true`. Codes are `invalid_input`, `game_active`, `log_unavailable` and `internal_error`. Malformed MCP request envelopes use native SDK protocol errors. Audit failures preserve active lifecycle state; failed initialization leaves idle. Process restarts begin idle and do not resume old games.

Run local checks from the repository checkout after installing the extension's `dev` extra:

```sh
/absolute/path/tft-mcp-venv/bin/python -m pytest -c mcp/server/pyproject.toml mcp/server/tests
/absolute/path/tft-mcp-venv/bin/python mcp/server/scripts/check_scope.py 54cbb8bb9e9933a80ea04bf99989bf001fe4774e
/absolute/path/tft-mcp-venv/bin/python -m pytest UnitTests/rng_test.py UnitTests/default_agent_test.py UnitTests/game_round_test.py UnitTests/simulator_test.py
```

When running directly against an uninstalled extension checkout, set `PYTHONPATH=mcp/server/src`. To verify the installed console entry point rather than source imports, set `TFT_MCP_TEST_COMMAND=/absolute/path/tft-mcp-venv/bin/tft-mcp` and run `mcp/server/tests/test_protocol.py`. These SDK tests launch the server outside the checkout and remove source-path imports for the installed command. On the affected AppImage host, prepend `env -u APPIMAGE` to Python commands too.

The [verification record](verification.md) distinguishes this slice's lifecycle evidence from later complete-game and real Codex client acceptance.

Champion and trait catalogs work in idle, running, and terminal states. `search_champions` accepts optional `query`, integer `cost` from 1 through 5, and exact `trait_id` filters. `search_traits` accepts optional `query`. Search queries match case-insensitive substrings of canonical IDs, filters combine with AND, and results sort by ID. Empty results are valid. IDs include `jarvaniv`, `leesin`, `tahmkench`, and `the_boss`; display-name aliases are unsupported.

`get_champion` requires `champion_id`. `get_trait` and `get_trait_champions` require `trait_id`. Exact IDs are case-sensitive. Unknown IDs return `unknown_champion` or `unknown_trait` and the requested ID in error details. Invalid filters, missing fields, nulls, types and unknown keys return `invalid_input` with `field` and `value` details. An unknown search trait filter is invalid input.

Champion details include raw base stats, other named rule parameters, intrinsic traits, star gold values and Chosen metadata. Trait details include thresholds, activation mode, raw effects and intrinsic membership. Ninja activates at exactly one or four; other thresholds are minimum counts. Values retain simulator table conventions and inactive array entries. Missing descriptions are null and explicitly listed in `unavailable_fields`. Responses do not calculate adjusted combat stats. Kayn form IDs describe supported item inputs; the source has inconsistent form names between assignment and combat, so the catalog does not establish transformation correctness.

Run catalog acceptance with the same environment as the local checks:

```sh
PYTHONPATH=mcp/server/src /absolute/path/tft-mcp-venv/bin/python -m pytest -c mcp/server/pyproject.toml mcp/server/tests/test_champion_catalog.py
```

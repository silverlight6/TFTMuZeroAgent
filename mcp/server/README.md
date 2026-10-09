# TFT MCP server

The Python stdio server operates one seeded eight-player game using the installed TFT simulator. `player_0` is controlled through MCP; seven opponents own existing `Default_Agent(False)` policies. The server starts, inspects, progresses, and closes games and queries static champion, trait and item rules. `end_turn` runs opponents and automated combat. `buy_unit` and `sell_unit` perform individual native purchases and sales. `refresh_shop` and `buy_xp` buy one native shop refresh or experience increment. Positioning and equipment tools arrive in later slices. Read the shared [Spec](../SPEC.md).

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

`TFT_MCP_AUDIT_PATH` is required for tool execution. Its parent directory is created when writable. `TFT_MCP_NATIVE_LOG_DIR` is optional and defaults to `native` beside the audit file. Every accepted gameplay transaction retains an isolated native directory containing unchanged simulator `log.txt` output. stdout carries only MCP messages; simulator diagnostics go to stderr. Failed starts retain diagnostic directories identified by the error's details. Failed candidate records are not published as accepted gameplay.

On a Linux host that exports `APPIMAGE`, launch through `env -u APPIMAGE`:

```sh
TFT_MCP_AUDIT_PATH=/absolute/path/logs/tft-audit.jsonl \
env -u APPIMAGE /absolute/path/tft-mcp-venv/bin/tft-mcp
```

The tested T3 Code AppImage host makes Python report the AppImage executable and ignore virtual-environment packages when that variable is inherited. Removing it from the server environment preserves the installed Python interpreter. Configure the MCP client command as `/usr/bin/env` with arguments `-u`, `APPIMAGE`, and the absolute `tft-mcp` path when needed.

The launcher re-executes Python with `PYTHONHASHSEED=0` before loading the SDK or simulator. Audit records include that setting and an actual interpreter hash probe, seed, baseline identity and seed, the installed simulator environment name and source digest, distribution version, interpreter and dependency versions, and configuration. Source checkouts also record their Git revision. Set `TFT_MCP_SIMULATOR_REVISION` to the installed source revision for installations without Git metadata; otherwise the source SHA-256 identifies that revision. Paths and IDs do not affect gameplay equality.

`start_game` requires exactly `{"seed": integer}` with a value from 0 through 2147483647. Booleans, decimal JSON numbers, missing seeds and unknown arguments fail. `get_game_status`, `end_turn`, and `close_game` accept `{}` only. Status includes `state`, `game_id`, `controlled_player_id`, `round`, `planning_budget` and `outcome`. Idle game fields are null. Running planning budget has capacity 14 and remaining slots. Individual actions reserve the fifteenth internal slot for `end_turn`; exhaustion never starts combat. `end_turn` drains planning and returns the next controlled decision. After controlled elimination, baselines finish the lobby before it returns terminal with null budget and the final controlled placement. Own terminal data is frozen until close.

Starting an active game returns `game_active` and preserves it. Closing a running game returns its ID, an outcome with `reason: "closed_incomplete"`, and idle status. Repeated close is harmless. Structured errors contain `code`, `message`, and `details`, with MCP `isError: true`. Codes include `invalid_input`, `game_active`, `no_game`, `game_terminal`, `budget_exhausted`, `log_unavailable` and `internal_error`. Malformed MCP request envelopes use native SDK protocol errors. Candidate gameplay, policies, RNG and native writes are published only after atomic audit replacement. Native, internal and audit failures preserve the committed game and accepted logs; failed initialization leaves idle. Process restarts begin idle and do not resume old games.

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

`search_items` and `get_item` query static installed simulator definitions even while idle. `search_items` accepts optional string `query` and optional `kind` of `component`, `equipment`, or `consumable`. Query uses case-insensitive identifier substring matching; both filters combine. Results are sorted by canonical `item_id` and include `kind` and `craftable`. `get_item` requires an exact case-sensitive `item_id`, for example `tear_of_the_goddess`, `guardian_angel`, or `kayn_rhast`. Unknown IDs return `unknown_item`. Extra arguments, wrong types, and explicit null filters return `invalid_input`.

Item detail contains raw `base_stats` and `effects`, a nullable two-component `recipe`, sorted `builds_into`, nullable `granted_trait`, and source-supported assignment `constraints`. Source parameter names, arrays and inverse values remain unchanged. Recipes retain duplicate ingredients and their source order. `description` is null with `unavailable_fields: ["description"]`, because official prose is unavailable. Consumables report their supported targets, capacity conditions and consumption behavior without executing mechanics or predicting random replacements.

Kayn form items have an unchanged simulator limitation. Assignment stores form-item IDs while combat checks different strings, and bench assignment writes a different attribute. The catalog reports these literal inputs and does not guarantee the intended combat transformation. Inspection never consumes planning budget or randomness.

Own inspections use seven separate tools: `get_board`, `get_bench`, `get_shop`, `get_items`, `get_economy`, `get_traits` and `get_round`. Only `get_board` and `get_traits` accept optional string `player_id`; omission selects `player_0`. Known living opponent selectors return their public board or stored traits in the same shape. Opponent board coordinates use that player's local frame. Explicit null, wrong types and extra arguments return `invalid_input`. Unknown selectors return `invalid_player` with supported IDs; idle or closed inspection returns `no_game`. Known removed opponents return `player_eliminated`, including a removed winner. Use `get_players` for their retained public status; historical opponent boards and traits are unavailable.

Player category responses contain `game_id`, `player_id` and `round`. Board has all 28 slots, ordered by x then y. x runs 0..6 left to right and y runs 0..3 bottom to top; the native flat index is x*4+y. Bench has nine slots with location kind and zero-based slot; shop has five slots with native purchase prices; loose inventory has ten slots. Empty slots contain explicit null unit/item/price values. Unit location comes from the container, rather than cached champion coordinates. `unit.champion` is the native identifier accepted by catalog `champion_id` inputs. Units expose stars, equipped items, actual Chosen trait or false, native cost, Kayn form, intrinsic traits, target-dummy flag and Azir sandguard coordinate pairs. Combat references are excluded.

Economy contains own gold, health, level, experience and the exact status planning budget. Traits contain sorted native stored trait IDs, counts and tiers without recalculation. These reads preserve gameplay, cached counts, randomness and planning budget. Terminal category data copies the retained own snapshot and reports its elimination or winner round. Terminal economy budget is null. `get_round` reports the final lobby round instead, which can be later. Closing clears retained inspection data; a new game begins with fresh data. See the [full schemas](../SPEC.md#own-inspection-contract).

`get_players` takes `{}` and returns `game_id` and all stable initial IDs sorted in `players`. Each record contains exactly `player_id`, `controlled`, `status`, `health`, `level` and `placement`. Status is `alive`, `eliminated` or `winner`; health and level are nullable public integers, and placement is the native recorded 1..8 value or null. Removed players retain only final public scalars. Public scouting excludes shops, bench, loose inventory, gold, experience, simulator caches and future outcomes. See the [public inspection contract](../SPEC.md#public-inspection-contract).

`buy_unit` requires exactly `{"shop_slot": integer}` with a zero-based slot from 0 through 4. `sell_unit` requires exactly `{"location": {"kind": "bench", "slot": integer}}` for slots 0 through 8, or `{"location": {"kind": "board", "x": integer, "y": integer}}` for the documented board coordinates. Actions accept no player selector. Each successful call consumes one planning slot and runs baseline opponents until the next controlled decision in the same round. Combat requires `end_turn`.

Buy receipts contain the detached original `purchased` unit, native `gold_spent`, consumed `shop_slot`, changed own unit and inventory slots, and status. Full-bench merges are supported even when the native action mask disables buying. Merge validation follows bench returns and drops before board returns through cascading promotions. Unsafe native merge capacity or copy preservation rejects atomically. Sale receipts contain the detached original `sold` unit, `location`, native `gold_gained`, changed own unit slots, status, and equipment `returned_items` or `dropped_items`. Board sales require space for all real equipment. Bench overflow drops the whole equipment set. Thieves gloves count as one real item; generated equipment disappears. Azir board sale includes removed sandguards.

Action errors include `empty_slot`, `insufficient_gold`, `capacity_exceeded` and `unsupported_action`, with location, slot, resource, capacity or reason details. Dummies, sandguards and unsupported native price or promotion ranges cannot be sold or purchased. Inconsistent native records or failed postconditions return `internal_error`. Rejected actions preserve gameplay, planning budget, baselines and RNG. See the [buy and sell contract](../SPEC.md#buy-and-sell-contract).


`refresh_shop` and `buy_xp` accept no arguments. Unknown keys, including `player_id`, return `invalid_input`. Each success spends the native instance cost and one shared planning action, then returns at the controlled decision in the same round. `refresh_shop` returns exactly `gold_spent`, five actual `slots` with native purchase prices, and `status`. Repeated visible offers are valid. `buy_xp` returns exactly `gold_spent`, `xp_before`, `xp`, `level_before`, `level`, `unit_capacity` and `status`. Costs, thresholds, experience increments and cap come from the installed simulator. Native recursive leveling and bonus capacity remain intact; reaching the cap clears residual experience.

Both tools validate lifecycle and budget before legality. XP checks `level_cap` before affordability and reports `level` and `max_level`. Unaffordable calls return `insufficient_gold` with `resource`, `required` and `available`. Failed native effects or inconsistent postconditions return `internal_error`. Rejections and failures preserve committed gameplay, budget, baseline policies, RNG and accepted logs through the shared transaction. See the [shop refresh and experience contract](../SPEC.md#shop-refresh-and-experience-contract).

Run the focused real adapter and SDK checks from the checkout:

```sh
env -u APPIMAGE PYTHONHASHSEED=0 PYTHONPATH=mcp/server/src /absolute/path/tft-mcp-venv/bin/python -m pytest -c mcp/server/pyproject.toml mcp/server/tests/test_shop_xp.py -q
```

The suite separates ordinary production stdio journeys from a test-local preconfigured real-session SDK memory fixture for rare level-cap boundaries.

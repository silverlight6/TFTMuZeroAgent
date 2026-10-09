# TFT MCP server

The Python stdio server operates one seeded eight-player game using the installed TFT simulator. `player_0` is controlled through MCP; seven opponents own existing `Default_Agent(False)` policies. The server starts, inspects, progresses, and closes games and queries static champion, trait and item rules. `end_turn` runs opponents and automated combat. `buy_unit` and `sell_unit` perform individual native purchases and sales. `refresh_shop` and `buy_xp` buy one native shop refresh or experience increment. `move_unit` positions owned units and performs supported native board/bench swaps. `equip_item` assigns inventory equipment and supported native consumables. Read the shared [Spec](../SPEC.md).

Use the shared [installation and native client guide](../README.md) for archived CPU-only installation, registration, installed protocol checks and removal. Both hosts use this package's absolute installed `tft-mcp` entry point. The simulator and extension remain separate noneditable installations. No training, model-hosting or GPU packages are required.

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


`move_unit` requires exactly `source` and `target`. Each is a strict board location `{"kind":"board","x":integer,"y":integer}` or bench location `{"kind":"bench","slot":integer}`. The documented board orientation and slot bounds apply. Unknown keys, nulls, booleans and decimal coordinates return `invalid_input`. No player selector is accepted. The source must hold an owned unit, even when the target is occupied. Same-location and native-disabled bench-to-bench requests return `unsupported_action`.

Board moves support empty destinations and native swaps in either coordinate direction. Bench entry into an empty board cell requires regular-unit capacity. Bench entry into an occupied cell displaces its unit to the native first bench vacancy after clearing the incoming slot. That vacancy can precede the requested source. Board moves to an empty bench slot use that exact slot. An occupied bench target supports a directed swap only when no earlier bench vacancy would send the outgoing board unit elsewhere. Full benches and full regular-unit capacity still permit supported swaps.

Board dummies and sandguards can reposition and swap on board but cannot leave to bench or be displaced there. Azir bench entry requires two free guard cells after displacement and outgoing Azir guard removal. Native guard creation, removal, board positioning, overlord state and linkage remain intact. Benched Azir retains its old coordinate list. Newly spawned guards retain native initial cached coordinates; receipts use their actual board storage. Glove tracking follows native supported swaps.

Each successful move consumes one shared planning action and returns in the same round. The receipt contains exactly detached `source`, `target`, observed `unit_changes` and `status`. Changes use the existing unit records in board x/y then bench-slot order, including displaced units and Azir guard/link changes. Distinct units with identical visible records can swap with an empty change list. Candidate identity checks still prove the move. Invalid legality, silent native failure, inconsistent metadata and corrupt postconditions discard the complete candidate, baselines, RNG and logs.

Run focused movement adapter and production SDK checks:

```sh
env -u APPIMAGE PYTHONHASHSEED=0 PYTHONPATH=mcp/server/src /absolute/path/tft-mcp-venv/bin/python -m pytest -c mcp/server/pyproject.toml mcp/server/tests/test_movement.py -q
```

Production stdio tests acquire units through gameplay and cover movement families, orientation, supported swaps, rejection, audit recovery and seeded replay with extra reads. Separate preconfigured real-session tests cover rare Azir, dummy, glove and capacity conditions. One rare Azir fixture also crosses official SDK memory streams through unchanged transport. See the [movement contract](../SPEC.md#movement-contract).


`equip_item` requires exactly `item_slot`, an integer from 0 through 9, and `target`, the strict owned board or bench Location used by movement. Empty inventory or target returns `empty_slot`. Ordinary equipment preserves native ordering: a complete item precedes the final component, and two final components combine through an installed recipe. Combination remains possible at the three-item limit. Duplicate trait grants and missing recipes return `incompatible_item`. The ordinary sparring-gloves pair returns `unsupported_action` because native `num_items` is stale. Direct `thieves_gloves` requires empty equipment, draws two distinct native items and tracks the board or bench location. Further ordinary equipment, remover and reforger on gloves are unsupported.

Duplicators require nonzero native champion cost and a genuine bench vacancy before any merge. They create a fresh default-star unit with the original Chosen and form arguments, without cloning stars, equipment or acquired attributes. Native constructor Chosen behavior can produce two stars. Supported bench-only cascades and cascades with board contributors only in the final merge phase preserve native item returns and bench whole-set drops. A board contributor in an earlier phase returns `unsupported_action` with reason `early_board_duplicate_cascade`; native outer repositioning can use an invalid intermediate bench slot. Unsafe board return capacity and unsupported promotion ranges reject before execution. Native copy weights, catalog, surviving units and the actual resulting unit establish success. There is no invented pool debit.

Remover and reforger require equipped items and vacancies for their complete count while the consumable still occupies its inventory slot. Remover returns items in equipment order. Trait removal requires intrinsic origins as the exact prefix and a suffix whose multiset equals every equipped trait grant. This supports safely assigned grants in either equipment order and rejects unsafe constructor or merged state. Reforger rejects all trait items, preserves spatula, and uses native category draws and exclusions for ordinary items. Neither consumable supports gloves.

Kayn tokens target board Kayn only. They consume every inventory copy of both form tokens and set the literal player and all board Kayn forms. Bench `kayn_form` and existing shop fields remain unchanged. Native writes bench `kaynform` instead, and stored token IDs differ from combat spellings. The receipt describes the literal result and does not establish working combat transformation. Reapplying a present token still consumes one action.

Every successful assignment returns exactly detached `item_slot`, original `item_id`, requested `target`, actual `unit_changes`, `item_changes`, nullable player `kayn_form` and `status`. All affected owned locations appear in board x/y then bench order, and inventory changes use slot order. Successful actions consume one shared planning slot without combat. Legality, native failures, corrupt results and receipt/audit failures discard the whole candidate, logs and RNG. Native masks and discarded wrapper booleans are insufficient evidence of success.

An installed observation limitation can still reject an otherwise legal native item operation. `ObservationToken` asserts inventory IDs below 58, so a remaining special consumable can make its post-action inventory update fail. The server returns atomic `internal_error` and retains the committed state. It does not change the simulator encoder or remove unrelated consumables. Native duplicator updates can also leave incremental observation and action-mask caches different from a freshly reconstructed encoder. Receipts and legality use actual owned state; equipment preserves the native update path.

Run focused equipment adapter and official SDK checks:

```sh
env -u APPIMAGE PYTHONHASHSEED=0 PYTHONPATH=mcp/server/src /absolute/path/tft-mcp-venv/bin/python -m pytest -c mcp/server/pyproject.toml mcp/server/tests/test_equipment.py mcp/server/tests/test_equipment_protocol.py -q
```

Production stdio tests naturally buy units, earn a sparring glove in combat, equip board and bench units, recover from audit failure and replay with extra reads and rejections. Combination and rare consumables cross the official SDK through test-local real-session memory fixtures. Focused native fixtures cover ordering, traits, capacities, fresh duplication, cascades, Azir, gloves, literal Kayn effects, observations, masks and random failure recovery. Whole-milestone replay and actual host gameplay remain later acceptance. See the [equipment contract](../SPEC.md#equipment-contract).

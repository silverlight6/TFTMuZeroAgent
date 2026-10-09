# MCP tools for the TFT Set 4 simulator

Status: Product scope and twelve-slice delivery breakdown accepted. Tickets are published in the fork. Shared scheduling and transaction contracts require review before action implementation.

## Problem statement

LLM agents need a documented set of MCP tools to inspect and play the existing TFT Set 4 simulator. They must choose what information to request and which individual game actions to perform.

## Solution

Provide a local stdio MCP server that owns one simulator game per server process. One player is controlled through MCP tools; seven opponents use an existing baseline policy. The client owns model inference and orchestration.

The MCP server and simulator adapter are implemented in Python.

Expose game information and rule knowledge as separate tools. Expose each supported game action through a singular tool. Preserve the existing simulator implementation and rules.

## User stories

1. As an agent, I can start a game with a seed and inspect its lifecycle status.
2. As an agent, I can request my board or a named opponent's publicly visible board.
3. As an agent, I can separately request my bench, shop, inventory, economy, active traits, round, and public player list.
4. As an agent, I can search champions and inspect their Set 4 rules.
5. As an agent, I can search traits, inspect thresholds, and find champions belonging to a trait.
6. As an agent, I can search items and inspect their effects and supported recipes.
7. As an agent, I can purchase a champion from a specific shop slot.
8. As an agent, I can sell one unit, move one unit, or assign one inventory item to a unit.
9. As an agent, I can refresh my shop or purchase experience through separate tools.
10. As an agent, I can explicitly end my planning phase and receive the resulting round or terminal status.
11. As an agent, I receive actionable errors for rejected requests without game-state changes or budget consumption.
12. As an operator, I can inspect a complete tool-call log and final placement.
13. As an operator, I can reproduce gameplay using the same seed, configuration, revision, baseline policies, and action sequence.
14. As an operator, I can install and connect the server on a CPU-only machine without hosting an LLM.
15. As an operator, I can close a game and subsequently start another without leaking its state.

## Implementation decisions

The MCP transport module owns tool registration, input schemas, descriptions, structured results, and protocol error translation. It depends on a game-session adapter. It contains no game-rule logic.

The game-session adapter owns lifecycle, scheduling, action validation, coordinate conversion, public information projection, baseline execution, and recording. It depends on the existing simulator. The simulator does not depend on the MCP server or adapter.

Rule-query tools read existing simulator definitions. They do not use current live-game metadata or duplicate game rules. Responses clearly distinguish data available from the simulator from descriptions that the simulator does not supply.

The extension is packaged separately from the simulator core. Existing simulator source, default behavior, and mandatory dependencies remain unchanged. Extension dependencies, installation instructions, tests, and entry point belong to the extension.

The controlled player defaults to player_0. Board queries accept an optional player identifier; omission selects the controlled player. Player identifiers are distinct from board coordinates.

Board locations use documented coordinates. Bench, shop, and inventory use documented zero-based slot indices. Unit responses include champion identity, star level, equipped items, and relevant Set 4 special attributes. The implementation must use the simulator's actual coordinate mapping consistently.

Inspection tools never advance the simulation, mutate gameplay state, or consume randomness. Hidden opponent shops, private simulator state, and future random outcomes are not exposed. Structured responses are limited to the requested information category. Action responses confirm the change and relevant status rather than returning the entire game.

Proposed tools are start_game, get_game_status, close_game, get_board, get_bench, get_shop, get_items, get_economy, get_traits, get_players, get_round, search_champions, get_champion, search_traits, get_trait, get_trait_champions, search_items, get_item, buy_unit, sell_unit, move_unit, equip_item, refresh_shop, buy_xp, and end_turn. Final schemas are documented and reviewed before their slice starts. No generic execute-action tool replaces the singular tools.

Actions are serialized within a session. The adapter validates schema, lifecycle, ownership, location, legality, and budget before committing an action. Rejected requests leave all gameplay state and RNG state unchanged. Unexpected internal failures must not leave a partially applied action; the chosen isolation or recovery strategy must be demonstrated with failure-path tests. Audit records may include rejected requests.

Valid action tools consume an explicit adapter planning budget. Information calls and rejected actions do not. At exhaustion, only end_turn can advance the planning phase. The adapter must reserve sufficient simulator capacity for explicit end-turn handling without altering core limits or monkey-patching the environment.

end_turn completes the planning phase, runs baseline opponents and automated combat, and returns at the next controlled-player decision or terminal state. Unsupported manual decisions, including simulator-automated carousel behavior and unimplemented shop locking, are documented rather than added to the core.

Logs contain the seed, revision, configuration, baseline identity, ordered tool requests and results, and final placement or incomplete-game reason. Internal baseline actions must also be recorded sufficiently to explain progression. Logs do not claim to contain client prompts, model identity, hidden reasoning, or all combat events. Wall-clock timestamps and record identifiers do not form part of gameplay determinism.

## Testing decisions

The principal acceptance seam is the MCP protocol: an official SDK client launches the production stdio server and calls its tools against the real simulator. Protocol tests prove discovery, schemas, structured results, and game behavior together.

Focused adapter tests cover cases that are difficult to produce through a full game, including atomic rejection, injected internal failures, budget exhaustion, terminal access, and lifecycle cleanup. These tests use the real simulator wherever practical.

Acceptance evidence includes a complete legal game through MCP, all action-tool families, rule-query consistency with simulator definitions, hidden-information exclusion, deterministic replay, and additional information calls without gameplay changes.

The existing simulator checks run without simulator modifications. A scope check verifies that the simulator source and mandatory dependency definitions are unchanged.

Automated MCP gameplay establishes tool operability. A real LLM client smoke test establishes client usability; it does not establish model skill. If that client or its model credentials are unavailable, report that evidence as unexecuted rather than claiming an LLM played successfully.

## Out of scope

Simulator or core modifications; new game mechanics; current TFT sets; MetaTFT integration; model hosting, provider adapters, and a custom LLM runner; training; comparative model benchmarking; remote MCP transport; concurrent games; restart persistence; strategic macro tools; automatic upstream submission, merging, or deployment.

## Delivery

Use a branch and pull request per useful slice in KyleDerZweite/TFTMuZeroAgent. Each slice has a short reviewed contract, explicit prerequisites, and acceptance evidence. A consolidated upstream pull request remains optional after the complete milestone is verified.

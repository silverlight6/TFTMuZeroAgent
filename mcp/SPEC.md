# MCP tools for the TFT Set 4 simulator

Status: Server product scope and revised twelve-slice dependency order confirmed on 2026-10-09. The owner subsequently accepted mcp/server/ with shared planning under mcp/ and requested explicit Codex CLI and Claude Code connection support. The owner confirmed prompt-driven setup, existing host permissions, and a complete Codex game as client acceptance; the server gameplay contract is unchanged. Lifecycle has a bounded concrete contract. AEC interleaving and automatic lobby completion after controlled-player elimination are accepted. The shared technical contracts in #7 still require review and real-simulator feasibility evidence before its implementation; dependent slices remain blocked.

## Problem statement

LLM agents need a documented set of MCP tools to inspect and play the existing TFT Set 4 simulator. They must choose what information to request and which individual game actions to perform.

## Solution

Provide a local stdio MCP server that owns one simulator game per server process. One player is controlled through MCP tools; seven opponents use an existing baseline policy. The client owns model inference and orchestration.

The MCP server and simulator adapter are implemented in Python.

Expose game information and rule knowledge as separate tools. Expose each supported game action through a singular tool. Preserve the existing simulator implementation and rules.

## User stories

1. As an agent, I want to start a seeded game and inspect its lifecycle status, so that I know which game I can operate.
2. As an agent, I want to request my board or a named opponent's publicly visible board, so that I can inspect positioning.
3. As an agent, I want to separately request my bench, shop, inventory, economy, active traits, round, and public player list, so that I can choose the information relevant to my next action.
4. As an agent, I want to search champions and inspect their Set 4 rules, so that I can evaluate available units.
5. As an agent, I want to search traits, inspect thresholds, and find their champions, so that I can understand team compositions.
6. As an agent, I want to search items and inspect effects and supported recipes, so that I can evaluate equipment choices.
7. As an agent, I want to purchase a champion from a specific shop slot, so that I can add that offer to my team.
8. As an agent, I want to sell one unit, move one unit, or assign one inventory item through singular tools, so that I retain control over each decision.
9. As an agent, I want separate shop refresh and experience purchase tools, so that I can choose each economy action explicitly.
10. As an agent, I want to explicitly end planning and receive the next decision or completed-lobby outcome, so that combat never starts because I purchased or moved a unit.
11. As an agent, I want actionable rejection errors without gameplay, RNG, or budget changes, so that I can correct a request without losing resources.
12. As an operator, I want complete tool-call and baseline-progression logs with final placement, so that I can inspect what happened.
13. As an operator, I want replay with the same seed, revision, configuration, baseline policies, and actions, so that I can reproduce gameplay.
14. As an operator, I want CPU-only installation and a local MCP connection without hosting an LLM, so that I can use an existing external client.
15. As an operator, I want to close a game and subsequently start another without state leakage, so that one server process can operate successive isolated games.
16. As an agent, I want my remaining planning budget exposed consistently, so that I know whether another action is available.
17. As an operator, I want the remaining lobby to finish automatically after the controlled player is eliminated, so that a full-game run still records the completed lobby and controlled placement.
18. As an agent, I want frozen terminal inspection data until close, so that I can inspect my final state even after core player removal.

## Implementation decisions

The MCP transport module owns tool registration, input schemas, descriptions, structured results, and protocol error translation. It depends on a game-session adapter. It contains no game-rule logic.

The game-session adapter owns lifecycle, scheduling, action validation, coordinate conversion, public information projection, baseline execution, and recording. It depends on the existing simulator. The simulator does not depend on the MCP server or adapter.

Rule-query tools read existing simulator definitions. They do not use current live-game metadata or duplicate game rules. Responses clearly distinguish data available from the simulator from descriptions that the simulator does not supply.

The MCP extension lives under mcp/. Shared planning and entry documentation belong at mcp/SPEC.md and mcp/README.md. Server implementation belongs in mcp/server/, including its Python package, packaging metadata, server dependencies, tests, developer commands, and entry point. A future custom client may live in mcp/client/ after its own scope and contracts are accepted; this milestone does not implement that client or create a placeholder package. The repository root packaging and the existing simulator source, rules, defaults, and mandatory dependencies remain unchanged. Install the unchanged simulator from the selected repository revision, then install the extension as its own package; the simulator distribution version alone does not identify the revision.

Keep implementation close to the existing Python style: ordinary functions, concrete classes where state is owned, straightforward snake_case names, and short explicit control flow. Use small typed records or dataclasses when they clarify an actual result or state shape. Beyond the MCP SDK needed for the accepted protocol, do not introduce a framework, dependency injection container, backend interface, plugin system, or generic action dispatcher for unrequested future use. Existing validation, failure atomicity, visibility, and determinism requirements remain mandatory.

The MCP transport delegates to a concrete session adapter. Simulator-specific field access, scheduling, validation, projections, and recovery stay in that adapter. Static catalog helpers read existing definitions and are called through the adapter. Share coordinate conversion and request/result conventions where the existing tools need them; avoid scattered rule duplication. Extract additional internal modules only when they have a distinct existing responsibility. Imports point from extension to simulator, never from simulator to extension.

Codex CLI and Claude Code are the intended existing MCP hosts. Their built-in MCP clients connect to the same local stdio server; no custom client application or model-provider adapter is needed. Deliver a single copyable setup prompt under mcp/SETUP_PROMPT.md with enough instructions for either host to install the server and register it using its existing native tools. The prompt refers to the verified revision and actual installation instructions rather than inventing commands. Include host-specific configuration examples, connection checks, removal instructions, and a separate playable-game prompt. No custom setup application is required.

Use the installed production launcher by absolute path and an explicit writable log destination; do not depend on launching from the repository or on an activated shell environment. Executing the copied setup prompt authorizes adding or updating the named TFT server entry in the operator's normal client configuration. Preserve unrelated settings and use existing authentication. Client configuration belongs to the operator and is not a new repository-root project file. Installation success, registration, an active MCP connection, and successful gameplay are separate evidence. If the host needs a new session to load the server, report that exact next step rather than claiming that registration proves connection.

The owner selected existing host tool permissions on 2026-10-09. Server tools have no additional authorization layer or approval dialogue. The setup prompt and instructions preserve Codex and Claude Code permission policies; they do not add TFT allowlists, change approval modes, or bypass host checks. Normal schema, lifecycle, game legality, ownership, budget, and error validation remain part of tool behavior. Existing host policy can still request approval or deny a call; the server does not override it.

The owner chose local checks only on 2026-10-09. Do not add GitHub Actions or other new project configuration outside mcp/. The existing root glossary remains the owner of domain vocabulary; the accepted planning-document consolidation and subsequent move to mcp/ are the documentation migrations in this change. If real simulator evidence demonstrates that a core change is unavoidable, record the limitation and proposed minimum change and obtain an explicit scope decision before altering the core.

The controlled player defaults to player_0. Board queries accept an optional player identifier; omission selects the controlled player. Player identifiers are distinct from board coordinates.

Board locations use documented coordinates. Bench, shop, and inventory use documented zero-based slot indices. Unit responses include champion identity, star level, equipped items, and relevant Set 4 special attributes. The implementation must use the simulator's actual coordinate mapping consistently.

Inspection tools never advance the simulation, mutate gameplay state, or consume randomness. Hidden opponent shops, private simulator state, and future random outcomes are not exposed. Structured responses are limited to the requested information category. Action responses confirm the change and relevant status rather than returning the entire game.

Proposed tools are start_game, get_game_status, close_game, get_board, get_bench, get_shop, get_items, get_economy, get_traits, get_players, get_round, search_champions, get_champion, search_traits, get_trait, get_trait_champions, search_items, get_item, buy_unit, sell_unit, move_unit, equip_item, refresh_shop, buy_xp, and end_turn. Final schemas are documented and reviewed before their slice starts. No generic execute-action tool replaces the singular tools.

Actions are serialized within a session. The adapter validates schema, lifecycle, ownership, location, legality, and budget before committing an action. Rejected requests leave all gameplay state and RNG state unchanged. Unexpected internal failures must not leave a partially applied action; the chosen isolation or recovery strategy must be demonstrated with failure-path tests. Audit records may include rejected requests.

Valid action tools consume an explicit adapter planning budget. Information calls and rejected actions do not. Reserve at least one internal controlled-player slot for explicit end_turn; with the unchanged default of 15 slots, the maximum adapter capacity is 14. At exhaustion, only end_turn can advance planning. No core limit changes or environment monkey-patching are permitted. #7 exposes capacity and remaining through get_game_status; #3 later projects those same values rather than introducing another counter.

The owner accepted the existing AEC interleaving on 2026-10-09. After a successful controlled action, the adapter runs baseline turns until the next controlled-player decision. The controlled action and its baseline progression are one transaction. Rejected actions and information calls do not run baselines. Reserving internal capacity prevents these individual actions from starting combat.

end_turn drains the remaining planning slots, runs baseline opponents and automated combat, and returns at the next controlled-player decision if that player remains alive. If the controlled player is eliminated, retain its final placement and own-state inspection snapshot before core removal, then automatically finish the remaining lobby with the existing baselines. The triggering end_turn returns terminal after lobby completion. Further controlled actions are rejected; frozen terminal inspection remains available until close. This owner decision was confirmed on 2026-10-09.

Unsupported manual decisions, including simulator-automated carousel behavior and unimplemented shop locking, are documented rather than added to the core.

Logs contain the seed, revision, configuration, baseline identity, ordered tool requests and results, and final placement or incomplete-game reason. Internal baseline actions must also be recorded sufficiently to explain progression. Logs do not claim to contain client prompts, model identity, hidden reasoning, or all combat events. Wall-clock timestamps and record identifiers do not form part of gameplay determinism. Native simulator log writes use an isolated writable game working directory. Both native and audit log failures are part of startup and action recovery acceptance; checking only the audit destination is insufficient.

## Lifecycle contract

The lifecycle interface is bounded as follows:

- start_game requires exactly one seed argument: an integer from 0 through 2147483647. Booleans, fractional values, omitted seeds, and unknown arguments are rejected. This canonical range avoids aliases caused by the simulator's 31-bit seed normalization.
- get_game_status and close_game take no arguments. Unknown arguments are rejected.
- Session state is idle, running, or terminal. Idle has no active game. Running owns one game. Terminal retains that game's outcome until close. Starting in running or terminal returns game_active without replacing the game; close returns to idle, and a subsequent start creates a fresh game.
- A status result contains state, nullable string game_id, nullable string controlled_player_id, nullable integer round, nullable object planning_budget, and nullable object outcome. The controlled player is player_0. Idle returns null game fields. start_game returns the running status. get_game_status returns the current status, including idle without raising no_game.
- planning_budget, when available after the progression slice, contains integer capacity and remaining, with 0 <= remaining <= capacity. outcome, when available, contains nullable integer controlled_placement from 1 through 8, boolean lobby_complete, and string reason. These fields are present and null before they are available; the shared progression contract establishes their terminal values. Executable terminal-path acceptance belongs to #7, so #2 has no dependency on later progression functionality.
- close_game returns nullable string closed_game_id, nullable object outcome, and the resulting idle status. Closing idle is idempotent and returns null closed-game fields. Closing a running game records an incomplete reason; closing terminal preserves its outcome in the close receipt and audit log.
- Tool failures use documented structured code, message, and details fields and the MCP tool-error indicator. Lifecycle codes include invalid_input, game_active, log_unavailable, and internal_error. Malformed protocol envelopes remain native MCP protocol errors. No live simulator objects are serialized.

The production launcher establishes a fixed recorded Python hash seed before simulator imports or protocol initialization, relaunching the interpreter when needed. Setting an environment variable after interpreter startup is insufficient. Episode and baseline RNG state are isolated and seeded. Simulator stdout must be redirected throughout construction and execution, not only after startup.

The adapter gives native simulator logging an isolated writable working directory for each game while preserving core source and defaults. The operator-configured audit destination and native log directory are checked before construction. Failed initialization does not publish a running game or leak candidate state/RNG; partial diagnostic files are identified as failed-start evidence. Paths and record identifiers are not gameplay equality inputs.

Repository documentation, issues, pull requests, and code comments are written in English. Discussion with the owner is in German.

## Testing decisions

The principal acceptance seam is the MCP protocol: an official SDK client launches the production stdio server and calls its tools against the real simulator. Protocol tests prove discovery, schemas, structured results, and game behavior together.

Focused adapter tests cover cases that are difficult to produce through a full game, including atomic rejection, injected internal failures, budget exhaustion, terminal access, and lifecycle cleanup. These tests use the real simulator wherever practical.

Acceptance evidence includes a complete legal game through MCP, all action-tool families, rule-query consistency with simulator definitions, hidden-information exclusion, deterministic replay, and additional information calls without gameplay changes.

The existing relevant simulator checks run without simulator modifications. Extension tests and their configuration live in mcp/server/ and are run explicitly, because the root test configuration discovers only simulator tests. Local installation acceptance uses a fresh CPU-only environment, installs the simulator before the extension, and launches the production entry point from a working directory outside the repository. This catches imports that succeed only because of the checkout directory.

A scope check compares added, modified, deleted, and untracked paths to a fixed reviewed base and the explicit extension/documentation allowlist. It verifies that simulator source, root packaging, and mandatory dependency definitions remain unchanged. It includes staged and unstaged work, not only commits. Any new outside-directory exception requires a recorded scope decision. No CI file is part of the current milestone.

Keep a short local verification record per slice: tested head/base/working diff, checks that passed or failed, unexecuted checks and their consequence, and the linked acceptance scenario. Provide the exact extension installation and check commands when #2 introduces them; do not document commands as available before their implementation.

Automated MCP gameplay establishes tool operability. Client acceptance additionally requires a complete game played by the Codex LLM through the installed production MCP tools in the owner's Codex environment. The owner authorized this test on 2026-10-09 once the server is implemented. Follow the documented setup and game prompt, retain the tested revision/configuration, seed, ordered calls, completed-lobby outcome, controlled placement, terminal inspection, close receipt, and log location. A scripted SDK game cannot substitute for this real Codex run, and client gameplay establishes usability rather than model skill.

Claude Code receives the same documented setup and game prompt. Verify its real connection and gameplay when an authenticated host session is available; the owner specifically selected Codex for the required LLM full-game evidence. Report unavailable Claude evidence separately without expanding this into a second mandatory full-game acceptance gate. A missing Codex connection or model access leaves the required Codex acceptance open, with the exact blocker or session-reload step recorded. Installed client executables and an existing conversation are not proof that new MCP tools are connected.

## Out of scope

Simulator or core modifications; new game mechanics; current TFT sets; MetaTFT integration; model hosting, provider adapters, and a custom LLM runner; training; comparative model benchmarking; remote MCP transport; concurrent games; restart persistence; strategic macro tools; automatic upstream submission, merging, or deployment.

## Further notes

The owner selected feat/mcp-server-main as the integration branch in KyleDerZweite/TFTMuZeroAgent on 2026-10-09. origin remains the fork remote; upstream remains the original repository remote. Each useful slice receives its own branch and pull request targeting feat/mcp-server-main. Start slice branches from the verified integrated prerequisite revision and integrate sequentially after review. The integration branch starts at the fork planning revision f79c3d1; its simulator baseline is upstream revision 33c2c6e.

After complete milestone verification, a separate integration PR to the fork main branch may be prepared. A later consolidated upstream contribution remains optional. The owner agreed to create the integration branch and commit the planning changes; it does not authorize merging any PR, changing the repository default branch, or submitting upstream. Each slice retains a short reviewed contract, explicit prerequisites, and acceptance evidence.

Execution order is #2, #5, #6, #7, #3, #4, #8, #9, #10, #11, #12, #13. #5, #6, and #7 may proceed independently after #2 and their own contract reviews. #7 no longer depends on #3: it is verifiable through lifecycle, status, and end_turn. #3 depends on #7, #4/#8/#9 depend on #3, #10 depends on #8, #11 depends on #6 and #8, #12 depends on #4/#5/#9/#10/#11, and #13 depends on #12. These minimal direct edges include all other prerequisites transitively. Display order does not impose dependencies between independent slices.

The milestone is not globally ready-for-agent while #7 needs technical design review. Only #2 is initially ready. A slice becomes ready after its blockers are verified and integrated and its concrete contract has been reviewed. Server gameplay and #13 client setup and acceptance decisions are settled. Engineering proof obligations must not be confused with owner decisions.

## Planning review and source evidence

The planning review on 2026-10-09 found four issues: #3 required an action budget defined later in #7; #4 required #7 terminal semantics; #2 lacked concrete lifecycle schemas despite its Ready label; and plan/spec/milestone status statements disagreed. The revised graph moves #7 before #3, makes #4 inherit its terminal contract through #3, and makes #8/#9 depend on own-state inspection. Lifecycle schemas and bootstrap rules are now concrete. Server gameplay decisions are settled; the technical feasibility and complete recovery proof remain work in #7. The subsequent owner interview settled #13 client setup and acceptance decisions.

Source review identified unordered player construction, baseline use of global NumPy randomness, action masks and discarded handler results that do not prove success, player-state removal at elimination, mutating observation helpers, simulator stdout, and native relative log writes. A Python hash-order difference was reproduced. This source review did not execute the simulator because its available Python lacked PettingZoo; it establishes no runtime acceptance.

The extension layout is supported by the existing packaging: the simulator is separately installable, its public imports include the environment/configuration and baseline policies, and no root package-finder change is required for a new extension package. The adapter still needs concrete simulator state for projections and recovery. That dependency is contained in the extension instead of redesigning the simulator.

GitHub readback on 2026-10-09 verified all twelve ticket bodies, labels, text/native blocking agreement, and actual sub-issue order. The sixteen direct edges are acyclic, every blocker precedes its dependent, no direct edge is transitively redundant, and every slice leads to #13. That verification used fork planning HEAD f79c3d1, upstream simulator base 33c2c6e, and the then-current planning diff. It is evidence for planning consistency, not implementation or runtime behavior.

The repository preparation review on 2026-10-09 found no material contract conflicts in the consolidated Spec, README, agent instructions, or twelve revised ticket drafts. A fixed-base path check against f79c3d1 passed for the extension files and the three accepted metadata paths, with no simulator/root-packaging changes or CI files. Local document links resolve, and the duplicate plan/spec files are removed. Runtime checks remain future acceptance work.

The owner accepted the mcp/ layout on 2026-10-09 to keep a future custom client possible. Current client connection research used local Codex CLI 0.161.0 and Claude Code 2.1.294 help, plus the official [Codex MCP documentation](https://developers.openai.com/codex/mcp/) and [Claude Code MCP documentation](https://code.claude.com/docs/en/mcp). Both support launching local stdio servers. Installed client executables establish neither model access nor a successful connection; server implementation had not begun at that research point.

On 2026-10-09 the owner chose one copyable prompt to let Codex or Claude Code perform setup, a required real Codex full-game test after implementation, and unchanged host permissions with no server-specific approval layer. These choices finalize #13 without changing the twelve-slice graph or authorizing a custom client. The setup prompt is a planning artifact until #13 verifies it against the implemented launcher and installation instructions.

## Lifecycle slice source review

Ticket #2 was reviewed against `TFT_Simulator.reset(seed)`, `CombatContext`, `EnvRNG`, `Default_Agent`, and native `log.txt` writes on 2026-10-09. The concrete session owns the raw eight-player simulator and seven `Default_Agent(champ_decider_action_format=False)` instances. It uses unchanged `TFTConfig` defaults. Scheduling and executable terminal transitions remain in #7.

The launcher fixes `PYTHONHASHSEED=0` through interpreter re-execution before SDK or simulator imports. The session records the simulator source digest, installed distribution version, Git revision when available, interpreter and dependency versions, episode seed, baseline seed and identity, hash configuration, and effective defaults. For source installations without Git metadata, operators can supply `TFT_MCP_SIMULATOR_REVISION`; the source digest remains recorded.

The concrete simulator scope redirects stdout to stderr, switches to the game's native log directory, binds its combat context when present, and saves/restores process Python and NumPy RNG state. Each game owns a seeded legacy NumPy stream for the existing baseline. Simulator module diagnostics and trait tiers are isolated with the game. This scope is process-local and serialized, matching the one-game stdio contract.

`TFT_MCP_AUDIT_PATH` selects the required JSONL audit file. `TFT_MCP_NATIVE_LOG_DIR` optionally selects the native log root; omission uses a sibling `native` directory beside the audit file. Startup probes both destinations through actual writes before simulator construction. Failed candidates remain unpublished; retained diagnostic paths are identified as failed-start evidence. Audit request/result recording is ordered. Lifecycle schema failures use extension errors, while malformed MCP envelopes remain SDK protocol errors.

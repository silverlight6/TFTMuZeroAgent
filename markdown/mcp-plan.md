# MCP simulator planning

Status: Product scope and delivery breakdown confirmed. Twelve implementation tickets have been published; implementation has not started.

## Accepted direction

Milestone 1 delivers a complete MCP toolset and an adapter for the existing TFT Set 4 simulator. The intended outcome is to let an LLM play through tools.

The MCP server and simulator adapter are implemented in Python.

The simulator's existing Set 4 rules are the initial game scope. The initial mode is a full lobby with one MCP-controlled player and seven existing baseline opponents.

Each information category has a separate MCP tool. The model chooses which information to request. Own-player state, publicly visible opponent information, and Set 4 rules are available. Hidden opponent shops and future random events are excluded.

An external MCP client owns the model interaction. This project exposes the simulator through an MCP server. A model provider adapter, inference hosting, and a custom LLM runner are outside Milestone 1.

The simulator, MCP server, and client run on the same machine. The initial transport is stdio. Automatic discovery by arbitrary clients is not assumed; clients must support MCP and be configured to launch the server.

One server session manages one game. Lifecycle tools start a seeded game, inspect its status, and close it.

Tools are game-oriented, clearly documented, and singular. Each action tool performs one game action rather than choosing or executing a strategy. Information tools do not advance the game. An explicit end-turn action advances baseline opponents and combat to the next controlled-player decision.

Rejected actions return complete, actionable errors without changing game state or consuming an action. Recording the rejected request in the audit log is permitted.

Milestone 1 includes game logs containing the seed, simulator revision, tool requests, results, and final placement. Completion means an MCP-capable agent can operate the simulator through a full game. Model strength and comparative benchmarks are later work.

Deterministic behavior means the same seed, simulator revision, configuration, baseline policies, and action sequence produce the same game states and outcomes. Information requests do not consume randomness or advance gameplay. Timestamps and log identifiers are excluded from gameplay equality.

Valid action tools consume the declared planning budget. Rejected actions and information tools do not. Exhausting the budget rejects further action tools while leaving end_turn available. No purchase or movement implicitly advances to the next round.

The simulator and core remain unchanged. Completeness covers decisions supported by the existing simulator; internally automated mechanics remain automated and their outcomes are inspectable.

Discussion takes place in German. Repository documentation, issues, pull requests, and code comments are written in English.

Useful implementation slices receive separate branches and pull requests in KyleDerZweite/TFTMuZeroAgent. A consolidated pull request to silverlight6/TFTMuZeroAgent may be prepared once the complete milestone is verified. Upstream submission is optional and is not yet authorized.

## Design obligations

- Validate and review the adapter's scheduling contract against the existing environment's action limit.
- Prove failure atomicity for supported action tools without core changes.
- Verify tool coverage against supported simulator decisions.
- Verify through a real MCP protocol client using the production stdio entry point.

Read-only contract review found no demonstrated need for core changes. Action slices are not Ready until scheduling, complete transaction isolation, terminal semantics, and process-level determinism have reviewed contracts.

The AEC environment rotates to an opponent after each controlled-player action; pass consumes an action slot rather than ending the phase. Whether baseline opponents may act between controlled action tools remains an open product decision. No scheduling alternative has been accepted yet.

Player construction uses an unordered set, and the heuristic baseline uses global NumPy randomness. The extension must control and record process hash configuration and baseline RNG state. Simulator masks and return values alone do not prove action success; rollback must cover shared state and random generators. Source review also identified simulator prints that must not contaminate stdio MCP output.

These findings are source-based, except a reproduced Python hash-order difference. The reviewer did not execute the simulator because its Python environment lacked PettingZoo.

## Repository evidence

The public environment APIs and usage documentation live in `Simulator/simulators/` and `markdown/`. The installed simulator dependencies are NumPy, PettingZoo, and Gymnasium. No GPU dependency is declared in `pyproject.toml`.

GitHub Issues were enabled in the fork during planning.

The milestone specification is tracked in [issue #1](https://github.com/KyleDerZweite/TFTMuZeroAgent/issues/1). The delivery breakdown is accepted. Individual slices become ready when their contracts are reviewed and their prerequisites are integrated.

The environment counts submitted actions and truncates a player when its internal limit is reached. The adapter must prevent invalid actions from reaching that interface and preserve an explicit end-turn operation. Shop locking is documented in the environment implementation as not implemented.

## Accepted delivery slices

1. Start, inspect, and close a seeded game through the Python stdio MCP server. Includes extension packaging, lifecycle, minimal request logging, and protocol tests. No blockers.
2. Inspect own board, bench, inventory, shop, economy, traits, and round through separate MCP tools with documented locations. Blocked by slice 1.
3. Inspect public opponent boards and traits and list lobby participants without exposing private state. Blocked by slice 2.
4. Search champions and traits, inspect their rules, and query trait membership through separate MCP tools. Blocked by slice 1.
5. Search items and inspect their rules and supported recipes through separate MCP tools. Blocked by slice 1.
6. Advance rounds using end_turn, seeded baseline opponents, and automated combat. Includes the shared action-budget and failure-atomicity contract, terminal handling, and progression logs. Blocked by slices 1 and 2.
7. Purchase and sell units through singular MCP tools with atomic validation and action-budget accounting. Blocked by slice 6.
8. Refresh the shop and purchase experience through singular MCP tools. Blocked by slice 6.
9. Move units between documented board and bench locations through a singular MCP tool, including supported swaps. Blocked by slice 7.
10. Equip an inventory item on a unit through a singular MCP tool with supported combination behavior. Blocked by slices 5 and 7.
11. Verify complete games through MCP, deterministic replay, complete logs, all action families, and unchanged simulator scope. Blocked by slices 3, 4, 5, 7, 8, 9, and 10.
12. Document installation and client setup and smoke-test the documented production entry point on a CPU-only setup. Blocked by slice 11. A real LLM-client smoke test is performed when a suitable configured client is available; unavailable evidence is explicitly reported.

Each slice is delivered as a branch and pull request in the fork. Slice 6 establishes the reviewed shared scheduling and action contract. Subsequent action slices reuse it and add acceptance coverage for their own behavior.

## Published implementation tickets

- Slice 1: [issue #2](https://github.com/KyleDerZweite/TFTMuZeroAgent/issues/2)
- Slice 2: [issue #3](https://github.com/KyleDerZweite/TFTMuZeroAgent/issues/3)
- Slice 3: [issue #4](https://github.com/KyleDerZweite/TFTMuZeroAgent/issues/4)
- Slice 4: [issue #5](https://github.com/KyleDerZweite/TFTMuZeroAgent/issues/5)
- Slice 5: [issue #6](https://github.com/KyleDerZweite/TFTMuZeroAgent/issues/6)
- Slice 6: [issue #7](https://github.com/KyleDerZweite/TFTMuZeroAgent/issues/7)
- Slice 7: [issue #8](https://github.com/KyleDerZweite/TFTMuZeroAgent/issues/8)
- Slice 8: [issue #9](https://github.com/KyleDerZweite/TFTMuZeroAgent/issues/9)
- Slice 9: [issue #10](https://github.com/KyleDerZweite/TFTMuZeroAgent/issues/10)
- Slice 10: [issue #11](https://github.com/KyleDerZweite/TFTMuZeroAgent/issues/11)
- Slice 11: [issue #12](https://github.com/KyleDerZweite/TFTMuZeroAgent/issues/12)
- Slice 12: [issue #13](https://github.com/KyleDerZweite/TFTMuZeroAgent/issues/13)

Tracker: GitHub Issues in KyleDerZweite/TFTMuZeroAgent. Use mcp for feature grouping, ready-for-agent for reviewed unblocked work, blocked for unmet prerequisites, and needs-design for unresolved design contracts. Only the lifecycle slice is initially ready. Dependencies use native GitHub blocking links and each ticket is a sub-issue of milestone issue #1.

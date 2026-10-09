# MCP implementation continuation

Paused again at the owner's explicit request on 2026-10-09 after verified integration of ticket #4. The owner resumed work for only that next ticket and requested another stop afterward. No implementation is running. Resume from the existing fork integration branch `feat/mcp-server-main`; do not restart completed slices.

## Accepted scope and workflow

The owner selected existing `mcp/server/` for implementation and `mcp/` for shared documentation. Use the installed simulator's definitions and rules without a fixed set number, set guard, copied catalog or runtime set selector. Actual source metadata may contain its own environment name. Simulator source, rules, defaults, root packaging and mandatory dependencies remain unchanged. No CI additions, main merge or upstream mutation is authorized.

Use GPT-6.1-Sol low or medium subagents after checking the live runtime catalog. Each writer uses an isolated worktree verified against the actual integration tip, branch and local changes. Start a dependent writer only after independently verified integration of its native blockers. Use TDD at the already accepted production MCP SDK and focused concrete adapter seams. Use Ponytail full, writing, unslop, relevant artifact-verification and final parallel Standards/Spec code-review. Keep ordinary Python functions and concrete state owners, without a framework or generic action dispatcher.

The owner authorized slice PRs targeting `feat/mcp-server-main` and merging them after successful checks. A separate merger reviews and integrates sequentially. Link every PR through T3 `link_pull_request`. Update tracker acceptance from actual integrated evidence. An optional future main or upstream contribution requires a separate instruction.

## Durable state

Repository: `KyleDerZweite/TFTMuZeroAgent`. Workspace: `/home/kyle/CodingProjects/TFTMuZeroAgent`. Remote `origin` is the fork; `upstream` is read-only for this task.

Fixed implementation/review base: `54cbb8bb9e9933a80ea04bf99989bf001fe4774e`. It already contains the accepted planning/layout and metadata preparation. Keep the implementation scope checker limited to `mcp/`. Simulator comparison base: `33c2c6e`.

Last functional merge: `f4a7de26e88b80468aa02ea45e1ad8c63fbdb617`. #4 started from the previous documentation checkpoint `bde19766296fe70b7ead45cd57057da7bc0dfd61`, after verified #3 integration. The later commit containing this record only saves documentation for the new pause. Use the live integration tip as the resume base, and preserve this functional evidence.

| Ticket | Integrated PR | Verified merge |
| --- | --- | --- |
| #2 lifecycle | [#14](https://github.com/KyleDerZweite/TFTMuZeroAgent/pull/14) | de46dd110f6a94229df4af5baa8ea619581138da |
| #6 items | [#15](https://github.com/KyleDerZweite/TFTMuZeroAgent/pull/15) | 24f8fb7a5498126f7cb608b61079b4c65edb88a6 |
| #5 champions/traits | [#16](https://github.com/KyleDerZweite/TFTMuZeroAgent/pull/16) | 497a426f50a16bc0b239f9fa843e38b24691ef59 |
| #7 atomic progression | [#17](https://github.com/KyleDerZweite/TFTMuZeroAgent/pull/17) | 7215c6a709d0075ed56408a8656865724aafe318 |
| #3 own inspection | [#18](https://github.com/KyleDerZweite/TFTMuZeroAgent/pull/18) | 684bbbf8e88be96890b0069c0bc7d4728ff0ceff |
| #4 public inspection | [#19](https://github.com/KyleDerZweite/TFTMuZeroAgent/pull/19) | f4a7de26e88b80468aa02ea45e1ad8c63fbdb617 |

These six tickets are completed. #6 was briefly reopened after the installed-simulator clarification and revalidated by #7. Milestone [#1](https://github.com/KyleDerZweite/TFTMuZeroAgent/issues/1) remains open and mirrors [SPEC.md](../SPEC.md). Tickets #8 through #13 remain open. #8/#9 are the next reviewed frontier; neither slice has started, and readiness does not revoke the owner's pause. The completed #4 writer worktree is removed after its clean head is verified as an ancestor of integration; branch history remains available.

Native direct blockers are `2:[]`, `5:[2]`, `6:[2]`, `7:[2]`, `3:[7]`, `4:[3]`, `8:[3]`, `9:[3]`, `10:[8]`, `11:[6,8]`, `12:[4,5,9,10,11]`, `13:[12]`. Display order adds no dependency.

## Integrated components and evidence

The installed console entry is `tft-mcp`. Launcher fixes `PYTHONHASHSEED=0` before imports through interpreter re-execution. `transport.py` owns strict MCP schemas, singular tool registration, manual argument validation, structured errors and request/result logging. `GameSession` owns one game, scheduling, concrete projections and transactions. Static catalog helpers read installed definitions. `freeze_unit` and `UNIT_SCHEMA` are the explicit shared visible-unit allowlist. `freeze_player` owns detached own records; it is inappropriate for public opponent projection.

`TFT_MCP_AUDIT_PATH` is required. `TFT_MCP_NATIVE_LOG_DIR` is optional and otherwise derives from the audit parent. There is no log CLI flag. Simulator stdout is redirected, and native relative logs run in isolated candidate directories. Startup checks actual audit and native writes before construction.

Progression reserves the fifteenth native controlled slot, exposing capacity 14. An accepted singular action interleaves seven existing `Default_Agent(False)` policies until the next controlled selection and cannot start combat. Only explicit `end_turn` drains planning. Controlled elimination freezes own state before core removal; baselines then finish the lobby. Terminal budget is null, outcome records placement and lobby completion, and further actions reject. Own category round records elimination or winner capture, while `get_round` returns final lobby round.

The transaction deepcopies a single aggregate preserving environment/pool/player/encoder/action-handler/RNG aliases, policy state, process RNG, module bindings and snapshots. Native candidate files are copied. Buffered JSONL audit replacement is the commit point, followed only by non-failing publication. Outer singular methods can call nested `controlled_action`, check postconditions and build a receipt before committing. Do not introduce another transaction, scheduler, budget or RNG store.

PR #17 exact head `abed92714d629faf9f80c1e1fe20d328a04fb066` passed all 40 extension tests, including complete deterministic lobbies. Merge tree matched the tested slice; integrated checks passed 38 with only two already-covered complete-lobby repetitions deselected. Fresh noneditable installed protocol and 11 unchanged simulator checks also passed. Independent atomicity review and merger found no material issue.

PR #18 implementation `d2cf5a93796e369212a0aba94431ea7d0df92a7d` passed all 17 affected inspection tests, including actual terminal inspection, and 54 combined quick checks with three complete-lobby scenarios deselected. Final slice `e865bba5613eec2eb05d7cad12ed055a4bc004ef` adds verification prose. Merger repeated 54 quick checks and proved tree identity `e206d95f2c2959b7bda37fd01c340b4f92e97938`; integrated inspection passed 16 with the already-tested terminal repetition deselected. Seven-path scope and whitespace checks passed; 11 simulator checks passed. Parent and merger found no material finding. Current terminal inspection observed own round 13, health 0 and final lobby round 30. Earlier probe values are observations, not fixed expected outcomes.

PR #19 functional head `7384dcb04a60ff733619df30e3634d590bbea37d` adds strict `get_players` and living-opponent board/trait reads. It resolves sorted initial dictionary keys, reuses public unit allowlists, and reads removed-player health/level and native placements from #7. Public status is alive/eliminated/winner. Private economy, bench, shop, loose inventory, caches, pools, policies, future matchups and RNG are excluded recursively. Removed opponents, including the winner, return `player_eliminated` for board/traits; own terminal projection remains unchanged. No opponent board-history store exists.

All four new SDK scenarios passed, including actual terminal removal and winner identity; three focused adapter checks cover distinctive private data, nested schemas, detached results and complete read purity. The exact functional terminal refresh passed separately. Final slice `22016669f3612ec5ead86a631a39b147e67204b6` adds verification prose only. Independent Standards and Spec reviews found zero material findings, and the Spec reviewer repeated all three focused adapter checks. The independent merger passed 60 quick checks with four documented full-lobby repetitions deselected, proved identical slice/merge tree `3453db57846b0ed32b9699192eb20f5c6630b0c1`, and passed six integrated public checks with the already-tested terminal repetition deselected. Eight-path scope and whitespace passed before and after integration; 11 unchanged simulator checks passed. No GitHub CI checks are configured under the owner's local-only choice.

Detailed commands, digests, failed fixture corrections and limits remain in [verification.md](verification.md). A complete whole-milestone suite at the eventual final functional head, all-family replay, final two-axis code-review and actual CLI LLM games remain unexecuted. Completed slice review is not final milestone review.

## Local environment and commands

`/tmp/tft-mcp-env` is a CPU-only Python 3.14.7 environment with MCP 1.30.0, NumPy 2.5.3, PettingZoo 1.27.0, Gymnasium 1.4.0 and pytest 9.1.1. Simulator is editable from the root checkout. Writers use their own extension through PYTHONPATH and do not reinstall shared packages. Existing fresh noneditable verification environments are `/tmp/tft-mcp-install-2` and `/tmp/tft-mcp-install-7`; they do not contain the #3/#4 extension and must not establish final-head installation evidence.

Inherited APPIMAGE from T3 can make Python report the AppImage executable and miss venv packages. All verification Python calls use `env -u APPIMAGE`. A production native-host registration can use `/usr/bin/env -u APPIMAGE /absolute/venv/bin/tft-mcp` when this inherited condition applies. Preserve the host's permissions.

From a writer worktree:

```sh
env -u APPIMAGE PYTHONHASHSEED=0 PYTHONPATH=mcp/server/src /tmp/tft-mcp-env/bin/python -m pytest -c mcp/server/pyproject.toml mcp/server/tests -q
env -u APPIMAGE /tmp/tft-mcp-env/bin/python mcp/server/scripts/check_scope.py <verified-slice-base>
git diff --check <verified-slice-base>
```

Use affected checks during slices. Do not repeat unchanged complete games after prose-only changes. Refresh required full-game evidence at #12 and final installed evidence at #13. If /tmp environments disappear, recreate from the actual documented packaging without changing root dependencies. Avoid leaving setuptools build/egg-info artifacts in root; build the unchanged simulator from an external archived source when appropriate.

Initial unchanged simulator baseline passed 36 tests covering RNG, default agent, buy masks, action space, bench capacity, steps and Player. Broader checks passed 8 and failed 2 under Gymnasium 1.4.0. Existing `test_gymnasium_item_env` and `test_gymnasium_single_player_env` return shared observation objects rejected by the newer checker. These are separate unmodified environments; relevant AEC/parallel/concurrency/round tests passed. Retain and report these pre-existing failures rather than repairing core in this milestone.

## Next slice contracts

Common action locations are strict `{kind:"board",x:0..6,y:0..3}` or `{kind:"bench",slot:0..8}`, with integer-only fields and no unknown keys. Shared native mapping is `x*4+y` or `28+slot`. All actions operate on the controlled player, with no player selector. Reuse #3 unit records and #7 status/budget. A unit delta is `{location,before:Unit|null,after:Unit|null}`; an item delta is `{slot,before:string|null,after:string|null}`. These are observed changes, without invented instance IDs. Validate before native execution and check actual postconditions before outer transaction commit. Action masks and discarded wrapper booleans are insufficient evidence.

| Tool | Inputs and native action | Receipt in addition to status |
| --- | --- | --- |
| buy_unit | shop_slot 0..4; `[3,slot,0]` | shop_slot, purchased Unit, gold_spent, unit_changes, item_changes |
| sell_unit | location; `[4,flat,0]` | location, sold Unit, gold_gained, returned_items, dropped_items, unit_changes |
| refresh_shop | no arguments; `[2,0,0]` | gold_spent, five actual shop records |
| buy_xp | no arguments; `[1,0,0]` | gold_spent, xp_before, xp, level_before, level, unit_capacity |
| move_unit | source,target; `[5,source_flat,target_flat]` | source, target, unit_changes |
| equip_item | item_slot 0..9,target; `[6,target_flat,item_slot]` | item_slot, item_id, target, unit_changes, item_changes, kayn_form |

#8 uses native `cost_star_values`, Chosen star prices, affordability and actual triple catalog. Full-bench merging is legal despite buy_mask=0; a full bench without a merge can native-autosell and must be rejected before execution. Board sale must prevalidate free inventory, as core can decrement catalog before failed item return. Bench sale can drop the entire equipment set on overflow; expose and document actual returned/dropped items. Generated glove equipment is not ordinary returnable sale equipment. Dummies/sandguards are unsupported sales. Preserve native cascading merges and Azir removal with actual deltas.

#9 uses player.refresh_cost/exp_cost/max_level. A refresh can validly repeat the same visible offers. Read actual recursive native XP/level/capacity changes instead of duplicating leveling rules. Reject cap and affordability errors without changes.

#10 starts after verified #8. Validate occupied source before core, because movement sorts endpoints and can otherwise move the target backwards. Reject same-location no-op and native-disabled bench-to-bench. Respect board capacity and dummy/sandguard restrictions. Native bench/board swaps place the displaced board unit in the first free bench slot, sometimes different from the named source. Expose that destination for bench-to-board. For directed board-to-occupied-bench, reject when an earlier vacancy would prevent the requested target from receiving the source unit. Preserve core behavior without symmetric swap fixes.

#11 starts after verified #6/#8. Ordinary recipes combine only the last equipped component; read recipes and trait grants from definitions and reject missing/incompatible recipes. Empty-equipped targets can receive thieves_gloves and its native random generated items. The ordinary sparring_gloves pair fails because native num_items is stale; reject/document it. Duplicator needs native nonzero cost and bench vacancy, constructs a native fresh unit with existing Chosen/form behavior and can merge. Ordinary remover/reforger needs free slots while the consumable still occupies inventory. Reject glove remover/reforger because core leaves stale tracking, and trait-item reforging because core retains origin. Bench Kayn form application writes the wrong attribute; reject it. Board form-token application can expose its literal native field/inventory effect, with all-board-Kayn changes, unchanged actual bench/shop fields and known mismatched combat spelling explicitly documented. Do not promise working combat transformation or repair core. #11 explicitly allows rejecting/documenting unsupported special cases.

Reuse errors invalid_input/no_game/game_terminal/budget_exhausted/log_unavailable/internal_error. Proposed action additions are empty_slot, insufficient_gold, capacity_exceeded, level_cap, incompatible_item and unsupported_action, with relevant slot/location/required/available/reason details. Keep strict validation order and verify baseline/pool/encoder/RNG rollback on injected failures.

## Final acceptance still required

#12 starts only after verified #4/#5/#9/#10/#11. Use actual SDK stdio tools and the real simulator. Exercise all 25 tools across meaningful scenarios. Reach a full lobby through legal singular actions, record the exact action tape, replay it in a second fresh process, then a third with repeated/reordered extra reads and a representative rejection. Compare action receipts, category checkpoints, baseline actions and outcome, not just endpoints. Keep rare real-simulator fixtures separate from naturally reached SDK gameplay. Audit ordered requests/results/errors, configuration, baselines, native paths and terminal/close evidence. Normalize only generated IDs, paths and timestamps, never gameplay order or values.

#13 starts after verified #12. Fresh CPU-only noneditable install of unchanged simulator plus separate extension must launch by absolute path outside checkout without PYTHONPATH. Document one copyable setup prompt and separate stated-seed playable-game prompt for native Codex CLI and Claude Code. Inspect existing named tft registration before mutation and preserve unrelated settings, authentication, permissions, approval/sandbox modes and allowlists. No custom LLM runner or provider adapter.

A full real Codex LLM game is required and explicitly authorized, using ordinary installed-host tools and existing authentication. Native `codex exec --model gpt-6.1-sol -c 'model_reasoning_effort="low"' --json` is a possible real host path, not a scripted SDK substitute. Record native session/model identity, revision/configuration, seed, ordered MCP calls, full outcome/placement, terminal inspection, close receipt and protected log location. Never bypass host permissions. Registration alone does not prove a loaded connection; use a fresh normal session. Claude private local project scope is supported; verify actual connection/gameplay if authenticated access remains available and report missing evidence separately. Initial read-only research found Codex 0.161.0 and Claude 2.1.294 with authentication, but no registration or LLM acceptance has run.

After all tickets complete, run parallel independent Standards and Spec reviews against fixed base 54cbb8b including staged, unstaged and untracked changes. Apply the code-review skill's full smell baseline with repository standards taking precedence. Fix material findings with one isolated implementer, integrate and refresh affected evidence. Then clean its worktree, verify all thread PR links, clean integration and current tracker status, and report implementation, passed/failed/unexecuted checks and remaining limits. Do not claim final review or milestone completion at this pause.

## External historical context

Additional exploration and orchestration files currently exist under `/tmp/tft-mcp-context/`: orchestration.json, issue snapshots, progression-design.md, catalog-contracts.md, inspection-contracts.md, action-contracts.md, acceptance-design.md, client-setup-research.md and review-7-atomicity.md. They are supplemental and may disappear; the Spec, tickets, this record, verification.md and retained Git commits are the durable resume owners. No secrets are stored in this record. Completed writer branch history remains in Git after worktree removal.

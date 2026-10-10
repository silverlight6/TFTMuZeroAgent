# Verification

Reviewed on 2026-10-10 for a pull request against `feat/mcp-server-main`. The starting head and PR base are `0baed806ff9dc48903becd3a018364b527102216`. Review covers the complete MCP milestone against upstream `33c2c6eb873da1aaf319dc0c36bcbec1ad20c48d`, plus staged, unstaged and untracked fixes. The functional checkpoint is `5f21ba153b4459c28bcca89d4c5eedd6c3ff49bd`; subsequent edits change only this verification record. Simulator source, rules, defaults, root packaging and root dependencies remain unchanged.

## Review findings and fixes

The Spec review found two correctness defects. Purchasing a cascading merge with board contributors before its final promotion could commit stale trait state. Purchase preflight now traces the native merge phases and rejects that unsafe case before execution. Safe bench cascades and final-phase board merges remain supported; independent real glove-contributor probes verified their traits, counts, returns and budget.

The SDK previously validated successful output schemas after gameplay committed. Transport now validates each result before audit success recording and publication. Malformed start, purchase, progression and close results return structured `internal_error`, preserve committed state and native logs, and append only the rejected audit record. Removing the fault permits a real successful retry.

The Standards review identified duplicated test infrastructure. Shared real-session fixtures, stdio and SDK memory clients, full graph/RNG/log snapshots and alias checks now live in `tests/support.py`. Scenario setup and fault injection remain explicit. Independent final Standards and Spec reviews report no remaining actionable findings.

Documentation now distinguishes setup from gameplay, checkout checks from installed acceptance, and current contracts from historical results. Setup discovers all 25 tools and verifies idle without starting a test game. Completed ticket chronology is retained in Git history rather than repeated in operational documentation.

## Current local checks

Python 3.14.7, MCP 1.30.0, NumPy 2.5.3, PettingZoo 1.27.0 and Gymnasium 1.4.0 were used. `APPIMAGE` was removed from Python child environments. `TFT_MCP_TEST_COMMAND` was absent during checkout checks, and source imports selected `mcp/server/src`.

| Check | Result |
| --- | --- |
| Reproductions before fixes | Unsafe purchase regression failed; all four malformed-output cases failed |
| Focused regressions after fixes | Five passed |
| Shared rollback and SDK memory scenarios | 81 passed |
| Final strengthened schema-failure/retry checks | Four passed |
| Refreshed shared inspection fixtures | Seven passed |
| Complete extension suite | 327 passed in 561.97 seconds, including all full-game checks |
| Final source/fixture refresh after the last test edits | 324 passed, three previously passed full-game checks deselected |
| Relevant unchanged simulator checks | 22 passed; three unrelated Gymnasium scenarios deselected |
| Fresh noneditable installation and installed SDK checks | Six passed; pip consistency passed; all seven server modules and 53 simulator files match |
| Scope, Markdown links/anchors, shell syntax and whitespace | Passed; all 27 PR paths are under mcp/ |

The fresh installation archives checkpoint tree `d572ed05436950384e740594416bb5a935a02204`, the exact tree of `5f21ba1`. It launches the absolute installed entry point from an external host directory without `PYTHONPATH`. Retained local SDK evidence includes `identity.json`, `check-results.json` and the protocol test audits/native logs. No native host configuration was changed.

The reviewed functional source and checks have SHA-256 `6e572e1937d5f6b52312c7cbbc3c93d790ad0bfcc08e6ebcbe4b84648d1e5358`. The digest covers sorted Python paths in `server/src` and `server/tests`, `server/pyproject.toml` and `server/scripts/check_scope.py`, with each repository-relative path and its contents separated by NUL bytes. Final documentation updates do not change those files.

The complete suite exercises all 25 tools, full-lobby completion, an exact action tape replayed across three production server processes, ordered audit/transcript agreement, terminal inspection, close/restart, privacy, strict schemas, budget, RNG isolation and failure recovery. SDK memory fixtures use the real simulator but remain distinct from production stdio journeys.

The complete and final refresh commands used the existing CPU-only venv with the `dev` extra. Substitute that environment path below. The final selector deselects exactly three full-game cases; the other exclusion terms do not match current test names. For routine developer checks, use [the source-check commands](README.md#developer-checks).

```sh
env -u APPIMAGE PYTHONHASHSEED=0 PYTHONPATH=mcp/server/src /absolute/path/venv/bin/python -m pytest -c mcp/server/pyproject.toml mcp/server/tests -q
env -u APPIMAGE -u TFT_MCP_TEST_COMMAND PYTHONHASHSEED=0 PYTHONPATH=mcp/server/src /absolute/path/venv/bin/python -m pytest -c mcp/server/pyproject.toml mcp/server/tests -k 'not full_game_exact_tape and not real_seed_zero_elimination and not full_lobby and not terminal_inspection_and_close and not actual_terminal' -q
env -u APPIMAGE /absolute/path/venv/bin/python -m pytest UnitTests/rng_test.py UnitTests/default_agent_test.py UnitTests/game_round_test.py UnitTests/simulator_test.py UnitTests/concurrency_repro_test.py UnitTests/bench_full_repro_test.py UnitTests/shop_buy_mask_test.py UnitTests/api_compliance_test.py -k 'not gymnasium' -q
env -u APPIMAGE /absolute/path/venv/bin/python mcp/server/scripts/check_scope.py 0baed806ff9dc48903becd3a018364b527102216
```

The native checks emit PettingZoo warnings about the existing dictionary observation shape. Installed checks use the separate [installation procedure](../README.md#install-a-pinned-revision) from outside the checkout without `PYTHONPATH`.

## Historical native-host acceptance

The previous noneditable installation at source `c4011d0ea32441b58827fb04b7a1720d8471a06c` passed a complete real Codex game on 2026-10-10 using the [game prompt](../GAME_PROMPT.md) and ordinary saved configuration. Seed 0 produced 129 individual MCP calls, including 25 explicit `end_turn` calls. There were 128 successes and one recovered full-bench rejection. The lobby completed at round 29 with controlled placement 4. All eight placements, frozen terminal reads, matching close receipt and subsequent idle were verified against the audit.

Codex's persistent `mcp_servers.tft.default_tools_approval_mode="approve"` applied only to the known local server. Global policies, per-tool restrictions, authentication, other server settings and Claude permissions were preserved. Independent review checked executed preservation assertions and retained proof; original private configuration baselines had been discarded and could not be reconstructed.

Raw native transcripts, audit/native logs, source-identity artifacts and configuration-preservation proof are maintainer-retained local evidence, not repository artifacts. Historical artifact names include `codex/session.jsonl`, `codex/audit.jsonl`, `codex/acceptance-summary.json`, `sdk/identity.json` and `config-preservation.json`. Detailed implementation chronology and older commands remain in [Git history](https://github.com/KyleDerZweite/TFTMuZeroAgent/blob/0baed806ff9dc48903becd3a018364b527102216/mcp/server/verification.md).

Unchanged simulator source digest is `43db488008ac7503201978bef81230635d65a7d15c37b82b6cd368d2fa669dcb`.

## Unexecuted checks and limits

A new native-host LLM game was not run for these fixes. The earlier Codex game is historical evidence; current automated acceptance exercises the production MCP protocol and real simulator separately. Host configuration was not changed during this review.

Claude previously connected and discovered all 25 tools, but its configured model returned HTTP 404 `model_not_found` before tool execution. Claude tool execution and gameplay remain unverified.

The complete repository test suite was not run. Relevant unchanged simulator tests passed; historical Gymnasium item/single-player checker failures remain outside the MCP scope. GitHub CI is not configured, following the selected local-only verification model.

Shop locking, unsafe native merge/equipment cases, encoder limits and literal Kayn transformation remain documented [server limitations](README.md). No merge, upstream submission or deployment is part of this review.

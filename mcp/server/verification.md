# Verification

Verified on 2026-10-10. Issues #2 through #13 and milestone #1 are complete on `feat/mcp-server-main`. [PR #25](https://github.com/KyleDerZweite/TFTMuZeroAgent/pull/25) merged the final slice. Simulator source and root dependencies are unchanged; rules come from the installed simulator without a fixed set number.

## Passed checks

| Check | Result | Tested source |
| --- | --- | --- |
| Full extension suite, including three-process replay | 322 passed, covering all 25 tools | `528ff4fee9827f699ac223045483df93f54e2781`, before final helper refactoring |
| Bounded extension suite after refactoring | 317 passed; five full-game scenarios deselected | `c4011d0ea32441b58827fb04b7a1720d8471a06c` |
| Independent affected fixtures | 39 passed | `c4011d0` |
| Fresh CPU-only noneditable installation | Six SDK checks and pip consistency passed | `c4011d0` |
| Independent merge checks | Six SDK checks before merge; two discovery/lifecycle checks afterward | Merge `0dc63e8f1532e7c23dff0d1dac575f7148a1b205` |
| Relevant unchanged simulator tests | 29 passed | #12 checkpoint `528ff4f` |
| Source identity and scope | Seven installed extension modules and 53 simulator files match; all 34 changed paths under mcp/; whitespace passed | Completion `dbefa604718ba26f6338c1814a06e4cf21cbf13f` |
| Final code review | Standards: zero findings. Spec: zero findings. | Completion `dbefa60` |

Replay compared the same action tape, receipts, snapshots, ordered progression and placements across three fresh processes, including extra/reordered reads and rejection. Tests also cover strict schemas, privacy, budget, failure atomicity, RNG preservation and restart.

## Real Codex game

The installed production server passed a complete game using the documented [game prompt](../GAME_PROMPT.md) and ordinary saved configuration:

```sh
codex exec --model gpt-6.1-sol -c 'model_reasoning_effort="low"' --json -
```

The prompt was supplied on stdin from `/home/kyle/.local/share/tft-mcp/issue-13-20261010/host`.

- Seed 0; 129 individual MCP calls, including 25 explicit `end_turn` calls.
- 128 successes and one correct full-bench `capacity_exceeded` rejection. Codex recovered and continued.
- Completed lobby at round 29, controlled placement 4, all eight placements recorded.
- Terminal board, bench, shop, items, economy, traits, round and players verified. Matching `close_game` and subsequent idle passed.
- Exit code 0. Independent review matched every call, argument and structured result to the server audit.

Thread: `01a12464-0496-7273-82bd-0ac6cffbecf0`. Installed source: `c4011d0ea32441b58827fb04b7a1720d8471a06c`.

Codex saves `mcp_servers.tft.default_tools_approval_mode="approve"` only for the known local server. Global `never`/`workspace-write`, other servers, per-tool restrictions, credentials and Claude policies are preserved. No session-only approval override was used. Full parsed preservation assertions used private baselines held in memory; independent reviewers could verify executed source, retained proof and current state, but could not reconstruct those discarded baselines.

## Evidence and reproduction

The retained installation is `/home/kyle/.local/share/tft-mcp/issue-13-fixed-20261010`. Evidence beneath its `logs/` directory:

- `codex/session.jsonl`, `codex/audit.jsonl`, `codex/native/`: actual host calls and simulator progression.
- `codex/acceptance-summary.json`, `codex/native-context.json`: run identity, results and artifact hashes.
- `sdk/protocol.log`, `sdk/identity.json`, `sdk/start-identity.json`: installed checks and source identity.
- `config-preservation.json`, `config-patch.executed-stdin-source.py`, `config-patch.source-provenance.txt`: configuration proof.
- `claude/session.jsonl`, `claude/session.stderr`: current connection and model failure.

Unchanged simulator digest: `43db488008ac7503201978bef81230635d65a7d15c37b82b6cd368d2fa669dcb`.

Use the [installation and check commands](../README.md) to reproduce verification. [CONTINUATION.md](CONTINUATION.md) records integrated slices and boundaries. Earlier per-slice commands and results remain in [Git history](https://github.com/KyleDerZweite/TFTMuZeroAgent/blob/dbefa604718ba26f6338c1814a06e4cf21cbf13f/mcp/server/verification.md).

## Unexecuted checks and limits

- Claude connected and discovered 25 tools, but `claude-opus-4-6` returned HTTP 404 `model_not_found` before tool execution. Claude gameplay remains unverified.
- Five complete-game SDK scenarios were not repeated after helper refactoring; their full #12 results are historical.
- GitHub CI was not configured or run, as requested.
- Historical Gymnasium item/single-player checker failures were neither rerun nor repaired.
- Shop locking and unsafe native item/merge cases remain limited. See [server limitations](README.md). No main/upstream merge or deployment occurred.

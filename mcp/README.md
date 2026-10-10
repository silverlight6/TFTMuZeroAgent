# TFT simulator MCP extension

The implemented local stdio server controls `player_0` in one seeded eight-player game. Seven opponents use unchanged `Default_Agent(False)` policies. Existing Codex CLI and Claude Code hosts own model inference. [SETUP_PROMPT.md](SETUP_PROMPT.md) is the single operational setup prompt; [GAME_PROMPT.md](GAME_PROMPT.md) starts a separate complete game.

[server/](server/) owns Python code, packaging, tests and tooling. [server/README.md](server/README.md) documents all 25 tools, schemas, coordinates, slots, budgets, privacy, errors and native equipment limitations. [SPEC.md](SPEC.md) owns accepted behavior and component boundaries; [the glossary](../GLOSSARY.md) owns vocabulary. GitHub owns ticket status and dependencies. Each useful slice has a PR targeting `feat/mcp-server-main` on the fork. No CI or custom client is added.

## Install archived source outside the checkout

Use Python 3.10 or newer and a fresh external directory. Select a verified full Git revision from the integration branch. Replace the three absolute paths and revision below. The archive makes the source immutable for this installation and avoids build artifacts in the checkout. These noneditable installations require only CPU dependencies NumPy, PettingZoo, Gymnasium and MCP; the `dev` extra adds pytest for checks.

```sh
TFT_REPO=/absolute/path/TFTMuZeroAgent
TFT_INSTALL=/absolute/path/operator-owned/tft-mcp
TFT_REVISION=FULL_VERIFIED_GIT_REVISION
mkdir -p "$TFT_INSTALL/source" "$TFT_INSTALL/host" "$TFT_INSTALL/logs/sdk" "$TFT_INSTALL/logs/codex" "$TFT_INSTALL/logs/claude"
chmod 700 "$TFT_INSTALL" "$TFT_INSTALL/logs"
git init -q "$TFT_INSTALL/host"
git -C "$TFT_REPO" archive --output="$TFT_INSTALL/source.tar" "$TFT_REVISION"
tar -xf "$TFT_INSTALL/source.tar" -C "$TFT_INSTALL/source"
env -u APPIMAGE python3 -m venv "$TFT_INSTALL/venv"
env -u APPIMAGE "$TFT_INSTALL/venv/bin/python" -m pip install "$TFT_INSTALL/source"
env -u APPIMAGE "$TFT_INSTALL/venv/bin/python" -m pip install "$TFT_INSTALL/source/mcp/server[dev]"
env -u APPIMAGE "$TFT_INSTALL/venv/bin/python" -m pip check
cd "$TFT_INSTALL/host"
env -u APPIMAGE -u PYTHONPATH "$TFT_INSTALL/venv/bin/python" -c 'import sys, Simulator, tft_mcp; from importlib.metadata import version; print(sys.executable, Simulator.__file__, tft_mcp.__file__); print({n:version(n) for n in ("tft-simulator","tft-mcp-server","numpy","PettingZoo","gymnasium","mcp")})'
env -u APPIMAGE -u PYTHONPATH TFT_MCP_SIMULATOR_REVISION="$TFT_REVISION" TFT_MCP_TEST_COMMAND="$TFT_INSTALL/venv/bin/tft-mcp" "$TFT_INSTALL/venv/bin/python" -m pytest -c "$TFT_INSTALL/source/mcp/server/pyproject.toml" "$TFT_INSTALL/source/mcp/server/tests/test_protocol.py" -k 'not full_lobby' -q --basetemp="$TFT_INSTALL/logs/sdk/protocol"
```

The SDK check initializes the installed launcher outside the checkout, discovers exactly 25 tools, checks idle, strict inputs, lifecycle, restart, deterministic bootstrap, failure recovery and clean JSON protocol stdout. It starts short test games under this explicit installation acceptance procedure. Setup without test-game authorization can select only `-k production_discovery_and_idle`. Imports must resolve inside the new venv. Record archive SHA-256 with `sha256sum "$TFT_INSTALL/source.tar"`. Record simulator and extension source revisions separately from their package versions, both currently `0.1.0`. Accepted game audit startup records include the simulator source digest, configuration and actual runtime dependencies. Set `TFT_MCP_SIMULATOR_REVISION` to the archive revision, not an older equivalent source revision.

## Register existing native hosts

Inspect `codex mcp get tft --json` or `claude mcp get tft` first. Compare command, args, TFT environment and the accepted Codex server default. Reuse a correct entry. For an existing different entry update only those fields, retaining unrelated environment fields and any per-tool restrictions. A remove/add cycle can discard custom policy. Persist `default_tools_approval_mode = "approve"` only for the inspected known local Codex `tft` server. Preserve global approval policy, sandbox, other servers, authentication, unrelated permissions and all existing per-tool restrictions. Claude policies and allowlists remain unchanged. These examples apply only when the named entry is absent.

Codex stores native registration in operator `~/.codex/config.toml`. Claude private local registration belongs to `~/.claude.json` under the exact external launch cwd. Use that same stable cwd in fresh Claude sessions; no repository `.mcp.json` is needed. The empty external Git project created above establishes the exact scope even when an ancestor has a `.git` marker. Check the actual stored project path, because Claude resolves local scope by project root.

```sh
cd "$TFT_INSTALL/host"
codex mcp add tft --env "TFT_MCP_AUDIT_PATH=$TFT_INSTALL/logs/codex/audit.jsonl" --env "TFT_MCP_NATIVE_LOG_DIR=$TFT_INSTALL/logs/codex/native" --env "TFT_MCP_SIMULATOR_REVISION=$TFT_REVISION" -- /usr/bin/env -u APPIMAGE "$TFT_INSTALL/venv/bin/tft-mcp"
claude mcp add --scope local --transport stdio tft -e "TFT_MCP_AUDIT_PATH=$TFT_INSTALL/logs/claude/audit.jsonl" -e "TFT_MCP_NATIVE_LOG_DIR=$TFT_INSTALL/logs/claude/native" -e "TFT_MCP_SIMULATOR_REVISION=$TFT_REVISION" -- /usr/bin/env -u APPIMAGE "$TFT_INSTALL/venv/bin/tft-mcp"
codex mcp get tft --json
claude mcp get tft
```

The `/usr/bin/env -u APPIMAGE` command preserves the venv interpreter on affected AppImage hosts. The launcher automatically fixes `PYTHONHASHSEED=0` before imports. Audit logs are mandatory; native logs default beside the audit when omitted. Simulator stdout is redirected to stderr. Keep audit, native and host transcripts protected and separate for each SDK/host run. Process restart starts idle; games do not resume. Close clears the current game while preserving logs.

For the known local server, add this saved Codex parent-table field before any `[mcp_servers.tft.env]` or per-tool child table:

```toml
[mcp_servers.tft]
default_tools_approval_mode = "approve"
```

Keep its existing command, args and unrelated parent fields. This persistent choice trusts current and future tools exposed by that installed server. Existing per-tool approval and enabled/disabled overrides retain precedence. It does not alter global `approval_policy` or sandbox settings, and the server adds no approval dialogue. Use saved configuration in the fresh native session, with no session-only approval override.

For an existing entry, parse the protected configuration in memory and patch only its exact approved leaves. Reparse the proposed text before replacing the original, check original bytes have not changed, and retain file ownership and mode. Compare full parsed configuration before/after while masking only named command, args, `TFT_MCP_AUDIT_PATH`, `TFT_MCP_NATIVE_LOG_DIR`, `TFT_MCP_SIMULATOR_REVISION` and Codex `default_tools_approval_mode`. For Claude mask only the corresponding command, args and three TFT environment fields at its exact project entry. Never exclude the whole named entry. Preserve all preexisting unrelated keys and per-tool policy. If Claude creates default project metadata, report its new key names separately. Do not copy operator configuration or secrets into repository evidence. Repeat native inspection on repeated setup and prove no rewrite when already correct.

Open a fresh native session and request an actual `tft.get_game_status` call. Registration and discovery do not prove tool execution. Existing host policy applies. If connection or model access fails, record the exact result and keep that acceptance open. Local help was checked with Codex 0.162.1 and Claude Code 2.1.294 on 2026-10-10. Official references are [Codex MCP](https://developers.openai.com/codex/mcp/) and [Claude MCP](https://code.claude.com/docs/en/mcp).

For the authorized complete-game acceptance, supply the text block from GAME_PROMPT.md to a normal fresh native session. Codex uses `codex exec --model gpt-6.1-sol -c 'model_reasoning_effort="low"' --json`; outside a Git repository add `--skip-git-repo-check`. Claude supports `claude --print --output-format stream-json --verbose`. Use the saved accepted `tft` default while retaining global policies, Claude permissions and authentication. Native transcripts plus correlated audit requests/results, game/server IDs, terminal placements, inspection and close receipt establish gameplay. SDK replay remains separate evidence.

## Gameplay and local checks

Reads do not consume budget or randomness. Individual actions consume one of 14 planning slots and interleave baseline decisions. Only explicit `end_turn` starts combat. Carousel, loot, baseline turns and remaining-lobby completion after controlled elimination are handled automatically. Terminal own categories retain the controlled snapshot; `get_round` returns the final lobby round. `get_players` exposes all eight final placements. Opponent privacy excludes shops, bench, inventory, gold and experience. Living opponent boards and stored traits are public; eliminated historical boards are unavailable.

Board coordinates are x=0..6 left to right and y=0..3 bottom to top. Bench slots are 0..8, shop 0..4 and loose inventory 0..9. Rule tools read installed definitions without a fixed set selector. Official prose, shop locking, unsafe native equipment combinations and intended Kayn combat transformation are unavailable; consult the precise limitations in server/README.md. Existing simulator limitations are retained.

Developer checks remain server-owned. From a checkout with the extension installed, run:

```sh
env -u APPIMAGE PYTHONHASHSEED=0 PYTHONPATH=mcp/server/src /absolute/path/venv/bin/python -m pytest -c mcp/server/pyproject.toml mcp/server/tests -q
env -u APPIMAGE /absolute/path/venv/bin/python mcp/server/scripts/check_scope.py 54cbb8bb9e9933a80ea04bf99989bf001fe4774e
env -u APPIMAGE /absolute/path/venv/bin/python -m pytest UnitTests/rng_test.py UnitTests/default_agent_test.py UnitTests/game_round_test.py UnitTests/simulator_test.py -q
```

[server/verification.md](server/verification.md) records current revision evidence, failures and unexecuted checks. Root Gymnasium-related historical failures do not authorize simulator changes.

## Remove the named entry

Run removal only when retiring this setup. Preserve unrelated configuration and logs. Claude removal must use the registration cwd.

```sh
codex mcp remove tft
cd "$TFT_INSTALL/host"
claude mcp remove --scope local tft
```

After both hosts stop using the entry, the operator can remove the dedicated environment/source directory and separately decide whether to retain audit/native/session logs. Removing an entry does not require deleting evidence.

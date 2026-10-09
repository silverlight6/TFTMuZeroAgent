# Set up the TFT MCP server

Copy the single prompt below into Codex or Claude Code. It authorizes installation and the inspected named `tft` configuration entry. Use [the separate game prompt](GAME_PROMPT.md) when gameplay is wanted.

```text
Install the TFT MCP server from https://github.com/KyleDerZweite/TFTMuZeroAgent
and connect it to my current Codex CLI or Claude Code host.

Read mcp/README.md and mcp/server/README.md at the selected verified revision
on feat/mcp-server-main. Reuse the correct checkout or obtain a separate one;
preserve unrelated work. Record the full source revision. Follow the actual
archived-source CPU-only instructions in mcp/README.md. Create a dedicated
external environment, install the unchanged simulator noneditably first, then
the extension separately. Record both source revisions, distribution versions
and simulator digest. Do not modify simulator, root dependencies or CI.

Use absolute installed paths and protected writable logs outside the checkout.
On an APPIMAGE host remove APPIMAGE only from Python/server child environments.
Do not use PYTHONPATH or an editable installation for acceptance. Run pip check
and the documented installed official SDK protocol checks from an external cwd.
Require exact discovery of 25 tools and idle get_game_status.

Inspect the existing named tft entry with the host's native MCP command before
registration, including repeated setup. Reuse an already correct entry without
rewriting. If different, update only its command, arguments and TFT environment
fields, preserving existing per-tool policy and unrelated environment fields.
Preserve all other configuration, authentication, permissions, approval modes,
sandbox and allowlists. Do not remove and recreate an entry with custom policy.
Use native registration when tft is absent. Codex uses operator config; Claude
uses private local scope in one stable external cwd for registration and play.
Do not create repository client configuration, an approval layer, a custom
client, provider adapter or LLM runner. Add no permission overrides.

Compare parsed configuration before and after, excluding only the named tft
entry. For Claude retain every preexisting project key and report any CLI-added
default metadata keys. Store no credentials or raw operator config in the repo
or shared temporary files. Keep SDK, Codex and Claude audit/native logs separate.

Start a fresh normal native host session from the registered external cwd and
verify an actual idle tft get_game_status call. Registration or discovery alone
is not active connection evidence. Report the exact reload/access blocker if
connection fails. Do not start a game during setup without explicit test scope.

Report revisions, package versions, interpreter/launcher paths, source digest,
log locations, host versions, registration preservation and checks actually
passed or still open. Provide mcp/GAME_PROMPT.md and the documented removal
commands. Inspect the named entry again on repeated setup and retain it unchanged
when correct. Complete authorized work without an extra approval gate.
```

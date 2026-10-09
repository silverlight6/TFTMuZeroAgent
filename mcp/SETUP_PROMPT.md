# Set up the TFT MCP server

Copy the prompt below into Codex or Claude Code. This is currently a planning artifact: the server has not been implemented, so setup cannot succeed yet. Issue #13 verifies this prompt against the implemented installation commands and production launcher.

```text
Install the local TFT MCP server from KyleDerZweite/TFTMuZeroAgent
and connect it to the Codex or Claude Code client I am using.

Read mcp/README.md, mcp/server/README.md, and the installation instructions
at the selected revision on feat/mcp-server-main. Reuse the correct existing
checkout when available; otherwise obtain a separate checkout from
https://github.com/KyleDerZweite/TFTMuZeroAgent. Preserve unrelated local work.
Record the exact revision. If the production launcher and installation
instructions are not implemented yet, report that and stop setup.

Use the documented CPU-only installation commands in a dedicated Python
environment. Install the unchanged simulator from the recorded revision
first, then the separate extension from mcp/server/. Use absolute paths
for the installed production launcher and a writable log destination.
Do not modify the simulator or add repository-root configuration.

Use the client's native MCP configuration tools to register the server
as tft. Inspect an existing tft entry before updating it and preserve
unrelated entries. For Claude Code, use private local scope for this
project rather than an uploaded or tracked repository configuration.
Keep the client's existing permission policy and authentication unchanged.
Do not add tool allowlists, change approval modes, or introduce extra
server-specific authorization. No custom client or LLM runner is needed.

Verify the installed launcher from outside the checkout with the documented
protocol check. Then verify the real client's active connection, tool
discovery, and an idle get_game_status call. If the client must restart
or open a new session to load the server, state that exact step and
leave the live connection check open until it has actually passed.

Report the installed revision, environment and launcher paths, log location,
client registration, checks actually performed, and any remaining step.
Show the documented game prompt and removal command. Do not start a game
until I give the game prompt or an explicit test instruction.
```

For implementers, #13 must add the verified host-specific configuration examples, exact installation/check/removal commands, and game prompt to the operational documentation. The game prompt must use a stated seed, individual tools, explicit end_turn, completed-lobby status and placement, terminal inspection, and close_game. It preserves the host's existing permissions.

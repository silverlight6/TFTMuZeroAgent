# Play a complete TFT game

Copy this prompt into a fresh Codex or Claude Code session after [setup](SETUP_PROMPT.md). Existing host permissions apply. Setup alone does not authorize gameplay.

```text
Play one complete TFT game through the connected tft MCP server. Use only
individual native MCP tools for gameplay, never a shell, SDK action loop,
custom client or scripted action tape. First call get_game_status and confirm
idle. Start with start_game {"seed":0}. Record the returned game_id.

Inspect shop, economy, bench, board and items as needed. Choose your own legal
purchases, positioning and equipment. Use buy_unit and move_unit at least once
when legal; equip a naturally available item when supported, or make another
legal individual choice. Keep each round to a few useful actions. Board x is
0..6 left to right, y is 0..3 bottom to top. Bench is 0..8, shop 0..4 and items
0..9. There are 14 planning actions; reads are free. Inspect tool schemas for
exact argument names. Rejected actions preserve state and budget.

Explicitly call end_turn after your choices each round. Continue through
controlled elimination until get_game_status reports state terminal,
outcome.lobby_complete true and controlled_placement between 1 and 8.
Use at most 40 end_turn calls. If a real error prevents completion, report
its exact result and preserve the evidence; never claim a completed game.

At terminal, call get_players and verify all eight placements, then inspect
get_board, get_bench, get_shop, get_items, get_economy, get_traits and get_round.
Own categories retain the controlled elimination or winning snapshot; final
get_round can be later. Save the completed outcome. Call close_game and verify
its outcome matches and its status is idle. Call get_game_status to confirm.

Report the native model/session identity when available, seed, game_id,
controlled placement, all eight placements, terminal inspections and matching
close receipt. Record installed source revisions, configuration and audit/native
log references supplied by setup. Actual tool results and logs are evidence;
do not invent revisions or paths missing from your session.
```

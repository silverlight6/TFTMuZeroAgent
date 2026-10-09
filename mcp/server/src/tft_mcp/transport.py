"""MCP schemas and error translation. Simulator access stays in GameSession."""

import json
import os

import anyio
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, TextContent, Tool

from tft_mcp.session import GameSession, SessionError


OUTCOME_SCHEMA = {
    "type": ["object", "null"],
    "properties": {
        "controlled_placement": {"type": ["integer", "null"], "minimum": 1, "maximum": 8},
        "lobby_complete": {"type": "boolean"}, "reason": {"type": "string"},
    },
    "required": ["controlled_placement", "lobby_complete", "reason"],
    "additionalProperties": False,
}
STATUS_SCHEMA = {
    "type": "object",
    "properties": {
        "state": {"type": "string", "enum": ["idle", "running", "terminal"]},
        "game_id": {"type": ["string", "null"]},
        "controlled_player_id": {"type": ["string", "null"]},
        "round": {"type": ["integer", "null"]},
        "planning_budget": {
            "type": ["object", "null"],
            "properties": {"capacity": {"type": "integer", "minimum": 0},
                           "remaining": {"type": "integer", "minimum": 0}},
            "required": ["capacity", "remaining"], "additionalProperties": False,
        },
        "outcome": OUTCOME_SCHEMA,
    },
    "required": ["state", "game_id", "controlled_player_id", "round", "planning_budget", "outcome"],
    "additionalProperties": False,
}
EMPTY_INPUT = {"type": "object", "properties": {}, "additionalProperties": False}
TOOLS = [
    Tool(name="start_game", description="Start one seeded eight-player game with player_0 controlled. Close an active game first.",
         inputSchema={"type": "object", "properties": {
             "seed": {"type": "integer", "minimum": 0, "maximum": 2147483647}},
             "required": ["seed"], "additionalProperties": False}, outputSchema=STATUS_SCHEMA),
    Tool(name="get_game_status", description="Inspect lifecycle status without advancing the game.",
         inputSchema=EMPTY_INPUT, outputSchema=STATUS_SCHEMA),
    Tool(name="close_game", description="Close the current game and return its outcome and idle status. Closing idle is idempotent.",
         inputSchema=EMPTY_INPUT, outputSchema={
             "type": "object", "properties": {
                 "closed_game_id": {"type": ["string", "null"]},
                 "outcome": OUTCOME_SCHEMA, "status": STATUS_SCHEMA},
             "required": ["closed_game_id", "outcome", "status"], "additionalProperties": False}),
]


def validate_arguments(name, arguments):
    if name == "start_game":
        if set(arguments) != {"seed"} or type(arguments.get("seed")) is not int or not 0 <= arguments["seed"] <= 2147483647:
            raise SessionError("invalid_input", "start_game requires only seed, an integer from 0 through 2147483647.",
                               {"tool": name})
    elif name in {"get_game_status", "close_game"}:
        if arguments:
            raise SessionError("invalid_input", f"{name} takes no arguments.", {"tool": name})
    else:
        raise SessionError("invalid_input", "Unknown tool.", {"tool": name})


async def serve():
    server = Server("tft-mcp-server")
    session = GameSession(os.environ.get("TFT_MCP_AUDIT_PATH"), os.environ.get("TFT_MCP_NATIVE_LOG_DIR"))
    lock = anyio.Lock()

    @server.list_tools()
    async def list_tools():
        return TOOLS

    @server.call_tool(validate_input=False)
    async def call_tool(name, arguments):
        async with lock:
            try:
                validate_arguments(name, arguments)
                with session.lifecycle_transaction():
                    session.record("tool_request", tool=name, arguments=arguments, game_id=session.game_id)
                    if name == "start_game":
                        result = session.start_game(arguments["seed"])
                    elif name == "get_game_status":
                        result = session.get_game_status()
                    else:
                        result = session.close_game()
                    session.record("tool_result", tool=name, result=result, is_error=False)
                return result
            except SessionError as error:
                result = error.result()
            except Exception as error:
                result = SessionError("internal_error", "Unexpected server failure.", {"error": str(error)}).result()
            try:
                session.record("tool_error", tool=name, arguments=arguments, result=result, is_error=True)
            except SessionError:
                pass
            return CallToolResult(content=[TextContent(type="text", text=json.dumps(result))],
                                  structuredContent=result, isError=True)

    async with stdio_server() as (read_stream, write_stream):
        try:
            await server.run(read_stream, write_stream, server.create_initialization_options())
        finally:
            try:
                session.close_game()
            except Exception:
                # Shutdown cannot write to protocol stdout or change a receipt already returned.
                pass

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
            "properties": {"capacity": {"type": "integer", "const": 14},
                           "remaining": {"type": "integer", "minimum": 0, "maximum": 14}},
            "required": ["capacity", "remaining"], "additionalProperties": False,
        },
        "outcome": OUTCOME_SCHEMA,
    },
    "required": ["state", "game_id", "controlled_player_id", "round", "planning_budget", "outcome"],
    "additionalProperties": False,
}
EMPTY_INPUT = {"type": "object", "properties": {}, "additionalProperties": False}
TOOLS = [
    Tool(name="end_turn", description="End planning explicitly and return the next decision or completed lobby.",
         inputSchema=EMPTY_INPUT, outputSchema=STATUS_SCHEMA),
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


CHAMPION_SUMMARY = {"type": "object", "properties": {
    "champion_id": {"type": "string"}, "cost": {"type": "integer"},
    "traits": {"type": "array", "items": {"type": "string"}}},
    "required": ["champion_id", "cost", "traits"], "additionalProperties": False}
TOOLS.append(Tool(name="search_champions", description="Search canonical installed simulator champion IDs with optional cost and intrinsic trait filters. Works without a game.",
    inputSchema={"type": "object", "properties": {"query": {"type": "string", "default": ""},
        "cost": {"type": "integer", "minimum": 1, "maximum": 5}, "trait_id": {"type": "string"}}, "additionalProperties": False},
    outputSchema={"type": "object", "properties": {"champions": {"type": "array", "items": CHAMPION_SUMMARY}},
        "required": ["champions"], "additionalProperties": False}))


CHAMPION_PROPERTIES = dict(CHAMPION_SUMMARY["properties"], **{
    "star_costs": {"type": "array", "items": {"type": "object", "properties": {
        "stars": {"type": "integer", "enum": [1, 2, 3]}, "gold": {"type": "integer"}},
        "required": ["stars", "gold"], "additionalProperties": False}},
    "base_stats": {"type": "object"}, "rule_parameters": {"type": "object"},
    "special_attributes": {"type": "object", "properties": {
        "chosen": {"type": "object", "properties": {
            "eligible_traits": {"type": "array", "items": {"type": "string"}},
            "bonus": {"type": ["object", "null"], "properties": {
                "stat": {"type": "string"}, "value": {"type": "number"}},
                "required": ["stat", "value"], "additionalProperties": False}},
            "required": ["eligible_traits", "bonus"], "additionalProperties": False},
        "kayn_forms": {"type": "array", "items": {"type": "string"}}},
        "required": ["chosen", "kayn_forms"], "additionalProperties": False},
    "description": {"type": "null"}, "ability_description": {"type": "null"},
    "unavailable_fields": {"type": "array", "items": {"type": "string"}},
})
TOOLS.append(Tool(name="get_champion", description="Read raw installed simulator champion definitions, star gold costs and Chosen metadata. Descriptions unavailable; stats are not dynamically adjusted.",
    inputSchema={"type": "object", "properties": {"champion_id": {"type": "string"}}, "required": ["champion_id"], "additionalProperties": False},
    outputSchema={"type": "object", "properties": CHAMPION_PROPERTIES, "required": list(CHAMPION_PROPERTIES), "additionalProperties": False}))


TRAIT_SUMMARY = {"type": "object", "properties": {
    "trait_id": {"type": "string"}, "thresholds": {"type": "array", "items": {"type": "integer"}}},
    "required": ["trait_id", "thresholds"], "additionalProperties": False}
TRAIT_PROPERTIES = dict(TRAIT_SUMMARY["properties"], **{
    "activation": {"type": "string", "enum": ["minimum", "exact"]}, "effects": {"type": "object"},
    "champion_ids": {"type": "array", "items": {"type": "string"}}, "chosen_eligible": {"type": "boolean"},
    "description": {"type": "null"}, "unavailable_fields": {"type": "array", "items": {"type": "string"}},
})
TOOLS.extend([
    Tool(name="search_traits", description="Search canonical installed simulator trait IDs and activation thresholds without a game.",
         inputSchema={"type": "object", "properties": {"query": {"type": "string", "default": ""}}, "additionalProperties": False},
         outputSchema={"type": "object", "properties": {"traits": {"type": "array", "items": TRAIT_SUMMARY}}, "required": ["traits"], "additionalProperties": False}),
    Tool(name="get_trait", description="Read installed simulator trait thresholds, raw effect parameters and intrinsic champion membership. Ninja activation is exact; descriptions unavailable.",
         inputSchema={"type": "object", "properties": {"trait_id": {"type": "string"}}, "required": ["trait_id"], "additionalProperties": False},
         outputSchema={"type": "object", "properties": TRAIT_PROPERTIES, "required": list(TRAIT_PROPERTIES), "additionalProperties": False}),
    Tool(name="get_trait_champions", description="List canonical champions with an intrinsic installed simulator trait, without item or Chosen counts.",
         inputSchema={"type": "object", "properties": {"trait_id": {"type": "string"}}, "required": ["trait_id"], "additionalProperties": False},
         outputSchema={"type": "object", "properties": {"trait_id": {"type": "string"}, "champions": {"type": "array", "items": CHAMPION_SUMMARY}}, "required": ["trait_id", "champions"], "additionalProperties": False}),
])


ITEM_SUMMARY_SCHEMA = {
    "type": "object", "properties": {
        "item_id": {"type": "string"},
        "kind": {"type": "string", "enum": ["component", "equipment", "consumable"]},
        "craftable": {"type": "boolean"}},
    "required": ["item_id", "kind", "craftable"], "additionalProperties": False,
}
COMPONENT_PAIR_SCHEMA = {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2}
ITEM_SCHEMA = {
    "type": "object", "properties": {
        **ITEM_SUMMARY_SCHEMA["properties"],
        "base_stats": {"type": "object"}, "effects": {"type": "object"},
        "recipe": {**COMPONENT_PAIR_SCHEMA, "type": ["array", "null"]},
        "builds_into": {"type": "array", "items": {
            "type": "object", "properties": {"item_id": {"type": "string"}, "components": COMPONENT_PAIR_SCHEMA},
            "required": ["item_id", "components"], "additionalProperties": False}},
        "granted_trait": {"type": ["string", "null"]},
        "constraints": {"type": "array", "items": {"type": "string"}},
        "description": {"type": "null"},
        "unavailable_fields": {"type": "array", "items": {"const": "description"}, "minItems": 1, "maxItems": 1}},
    "required": ["item_id", "kind", "craftable", "base_stats", "effects", "recipe", "builds_into",
                 "granted_trait", "constraints", "description", "unavailable_fields"],
    "additionalProperties": False,
}
TOOLS.extend([
    Tool(name="search_items", description="Search static installed simulator item IDs by case-insensitive substring and optional kind, without a game.",
         inputSchema={"type": "object", "properties": {
             "query": {"type": "string", "default": ""},
             "kind": {"type": "string", "enum": ["component", "equipment", "consumable"]}},
             "additionalProperties": False},
         outputSchema={"type": "object", "properties": {"items": {"type": "array", "items": ITEM_SUMMARY_SCHEMA}},
                       "required": ["items"], "additionalProperties": False}),
    Tool(name="get_item", description="Inspect an exact canonical installed simulator item ID, raw effects, recipes and simulator constraints. Official description is unavailable.",
         inputSchema={"type": "object", "properties": {"item_id": {"type": "string"}},
                      "required": ["item_id"], "additionalProperties": False}, outputSchema=ITEM_SCHEMA),
])


UNIT_SCHEMA = {
    "type": ["object", "null"], "properties": {
        "champion": {"type": "string"}, "stars": {"type": "integer", "minimum": 1, "maximum": 4},
        "items": {"type": "array", "items": {"type": "string"}},
        "chosen": {"anyOf": [{"const": False}, {"type": "string"}]},
        "cost": {"type": "integer"}, "kayn_form": {"type": ["string", "null"]},
        "traits": {"type": "array", "items": {"type": "string"}},
        "target_dummy": {"type": "boolean"},
        "sandguard_overlord_coordinates": {"type": "array", "items": {
            "type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2}}},
    "required": ["champion", "stars", "items", "chosen", "cost", "kayn_form", "traits", "target_dummy", "sandguard_overlord_coordinates"],
    "additionalProperties": False,
}
PLAYER_INPUT = {"type": "object", "properties": {"player_id": {"type": "string"}}, "additionalProperties": False}
PLAYER_PROPERTIES = {"game_id": {"type": "string"}, "player_id": {"type": "string"}, "round": {"type": "integer"}}
BOARD_PROPERTIES = {**PLAYER_PROPERTIES,
    "slots": {"type": "array", "minItems": 28, "maxItems": 28, "items": {
        "type": "object", "properties": {
            "location": {"type": "object", "properties": {
                "kind": {"const": "board"}, "x": {"type": "integer", "minimum": 0, "maximum": 6},
                "y": {"type": "integer", "minimum": 0, "maximum": 3}},
                "required": ["kind", "x", "y"], "additionalProperties": False}, "unit": UNIT_SCHEMA},
        "required": ["location", "unit"], "additionalProperties": False}},
    "num_units_in_play": {"type": "integer"}, "max_units": {"type": "integer"}}
TOOLS.append(Tool(name="get_board", description="Inspect own or a known living opponent board. Omitted player_id selects player_0. Removed opponents return player_eliminated. Boards use local x=0..6 left-to-right, y=0..3 bottom-to-top coordinates. Terminal round is the retained own snapshot round.",
    inputSchema=PLAYER_INPUT, outputSchema={"type": "object", "properties": BOARD_PROPERTIES,
    "required": list(BOARD_PROPERTIES), "additionalProperties": False}))


BENCH_PROPERTIES = {**PLAYER_PROPERTIES, "slots": {"type": "array", "minItems": 9, "maxItems": 9, "items": {
    "type": "object", "properties": {"location": {"type": "object", "properties": {
        "kind": {"const": "bench"}, "slot": {"type": "integer", "minimum": 0, "maximum": 8}},
        "required": ["kind", "slot"], "additionalProperties": False}, "unit": UNIT_SCHEMA},
    "required": ["location", "unit"], "additionalProperties": False}}}
TOOLS.append(Tool(name="get_bench", description="Inspect all own reserve slots, indexed 0..8. Terminal round is the retained own snapshot round.",
    inputSchema=EMPTY_INPUT, outputSchema={"type": "object", "properties": BENCH_PROPERTIES,
    "required": list(BENCH_PROPERTIES), "additionalProperties": False}))


SHOP_PROPERTIES = {**PLAYER_PROPERTIES, "slots": {"type": "array", "minItems": 5, "maxItems": 5, "items": {
    "type": "object", "properties": {"slot": {"type": "integer", "minimum": 0, "maximum": 4},
        "unit": UNIT_SCHEMA, "purchase_cost": {"type": ["integer", "null"]}},
    "required": ["slot", "unit", "purchase_cost"], "additionalProperties": False}}}
TOOLS.append(Tool(name="get_shop", description="Inspect all own shop offers and native purchase prices, indexed 0..4. Terminal round is the retained own snapshot round.",
    inputSchema=EMPTY_INPUT, outputSchema={"type": "object", "properties": SHOP_PROPERTIES,
    "required": list(SHOP_PROPERTIES), "additionalProperties": False}))


INVENTORY_PROPERTIES = {**PLAYER_PROPERTIES, "slots": {"type": "array", "minItems": 10, "maxItems": 10, "items": {
    "type": "object", "properties": {"slot": {"type": "integer", "minimum": 0, "maximum": 9},
        "item": {"type": ["string", "null"]}}, "required": ["slot", "item"], "additionalProperties": False}}}
TOOLS.append(Tool(name="get_items", description="Inspect all own loose inventory items, indexed 0..9. Terminal round is the retained own snapshot round.",
    inputSchema=EMPTY_INPUT, outputSchema={"type": "object", "properties": INVENTORY_PROPERTIES,
    "required": list(INVENTORY_PROPERTIES), "additionalProperties": False}))


ECONOMY_PROPERTIES = {**PLAYER_PROPERTIES, **{key: {"type": "integer"} for key in ("gold", "health", "level", "exp")},
    "planning_budget": STATUS_SCHEMA["properties"]["planning_budget"]}
TOOLS.append(Tool(name="get_economy", description="Inspect own gold, health, level, experience and the shared planning budget. Terminal budget is null and round is the retained own snapshot round.",
    inputSchema=EMPTY_INPUT, outputSchema={"type": "object", "properties": ECONOMY_PROPERTIES,
    "required": list(ECONOMY_PROPERTIES), "additionalProperties": False}))


OWN_TRAIT_PROPERTIES = {**PLAYER_PROPERTIES, "traits": {"type": "array", "items": {
    "type": "object", "properties": {"trait_id": {"type": "string"}, "count": {"type": "integer"}, "tier": {"type": "integer"}},
    "required": ["trait_id", "count", "tier"], "additionalProperties": False}}}
TOOLS.append(Tool(name="get_traits", description="Inspect own or a known living opponent stored trait counts and tiers without recomputation. Omitted player_id selects player_0; removed opponents return player_eliminated. Terminal round is the retained own snapshot round.",
    inputSchema=PLAYER_INPUT, outputSchema={"type": "object", "properties": OWN_TRAIT_PROPERTIES,
    "required": list(OWN_TRAIT_PROPERTIES), "additionalProperties": False}))


TOOLS.append(Tool(name="get_round", description="Inspect the current installed simulator round. Terminal returns the final completed-lobby round, which can differ from retained own category rounds.",
    inputSchema=EMPTY_INPUT, outputSchema={"type": "object", "properties": {
        "game_id": {"type": "string"}, "round": {"type": "integer"}},
        "required": ["game_id", "round"], "additionalProperties": False}))


PUBLIC_PLAYER_PROPERTIES = {
    "player_id": {"type": "string"}, "controlled": {"type": "boolean"},
    "status": {"enum": ["alive", "eliminated", "winner"]},
    "health": {"type": ["integer", "null"]}, "level": {"type": ["integer", "null"]},
    "placement": {"type": ["integer", "null"], "minimum": 1, "maximum": 8},
}
TOOLS.append(Tool(name="get_players", description="List stable initial player IDs and public health, level and native placement. Removed players retain public final status until close.",
    inputSchema=EMPTY_INPUT, outputSchema={"type": "object", "properties": {
        "game_id": {"type": "string"}, "players": {"type": "array", "items": {
            "type": "object", "properties": PUBLIC_PLAYER_PROPERTIES,
            "required": list(PUBLIC_PLAYER_PROPERTIES), "additionalProperties": False}}},
        "required": ["game_id", "players"], "additionalProperties": False}))


LOCATION_SCHEMA = {"oneOf": [
    BOARD_PROPERTIES["slots"]["items"]["properties"]["location"],
    BENCH_PROPERTIES["slots"]["items"]["properties"]["location"],
]}
UNIT_CHANGE_SCHEMA = {"type": "object", "properties": {
    "location": LOCATION_SCHEMA, "before": UNIT_SCHEMA, "after": UNIT_SCHEMA},
    "required": ["location", "before", "after"], "additionalProperties": False}
ITEM_CHANGE_SCHEMA = {"type": "object", "properties": {
    "slot": {"type": "integer", "minimum": 0, "maximum": 9},
    "before": {"type": ["string", "null"]}, "after": {"type": ["string", "null"]}},
    "required": ["slot", "before", "after"], "additionalProperties": False}
BUY_PROPERTIES = {
    "shop_slot": {"type": "integer", "minimum": 0, "maximum": 4},
    "purchased": {**UNIT_SCHEMA, "type": "object"},
    "gold_spent": {"type": "integer", "minimum": 0},
    "unit_changes": {"type": "array", "items": UNIT_CHANGE_SCHEMA},
    "item_changes": {"type": "array", "items": ITEM_CHANGE_SCHEMA}, "status": STATUS_SCHEMA,
}
SELL_PROPERTIES = {
    "location": LOCATION_SCHEMA, "sold": {**UNIT_SCHEMA, "type": "object"},
    "gold_gained": {"type": "integer", "minimum": 0},
    "returned_items": {"type": "array", "items": {"type": "string"}},
    "dropped_items": {"type": "array", "items": {"type": "string"}},
    "unit_changes": {"type": "array", "items": UNIT_CHANGE_SCHEMA}, "status": STATUS_SCHEMA,
}
TOOLS.extend([
    Tool(name="buy_unit", description="Buy one own shop offer at slot 0..4 using installed native prices and merges. Full-bench merges are supported. Consume one planning action and return at the same round without combat.",
         inputSchema={"type": "object", "properties": {"shop_slot": BUY_PROPERTIES["shop_slot"]},
                      "required": ["shop_slot"], "additionalProperties": False},
         outputSchema={"type": "object", "properties": BUY_PROPERTIES, "required": list(BUY_PROPERTIES), "additionalProperties": False}),
    Tool(name="sell_unit", description="Sell one owned board or bench unit. Board equipment needs inventory capacity; bench overflow drops the complete real equipment set. Thieves gloves return only the glove. Consume one planning action without combat.",
         inputSchema={"type": "object", "properties": {"location": LOCATION_SCHEMA},
                      "required": ["location"], "additionalProperties": False},
         outputSchema={"type": "object", "properties": SELL_PROPERTIES, "required": list(SELL_PROPERTIES), "additionalProperties": False}),
])


def validate_arguments(name, arguments):
    if name == "buy_unit":
        from tft_mcp.session import validate_buy_arguments
        validate_buy_arguments(arguments)
    elif name == "sell_unit":
        from tft_mcp.session import validate_sell_arguments
        validate_sell_arguments(arguments)
    elif name == "start_game":
        if set(arguments) != {"seed"} or type(arguments.get("seed")) is not int or not 0 <= arguments["seed"] <= 2147483647:
            raise SessionError("invalid_input", "start_game requires only seed, an integer from 0 through 2147483647.",
                               {"tool": name})
    elif name in {"get_game_status", "close_game", "end_turn", "get_bench", "get_shop", "get_items", "get_economy", "get_round", "get_players"}:
        if arguments:
            raise SessionError("invalid_input", f"{name} takes no arguments.", {"tool": name})
    elif name in {"get_board", "get_traits"}:
        validate_catalog_arguments(arguments, {"player_id"})
    elif name == "search_traits":
        validate_catalog_arguments(arguments, {"query"})
    elif name in {"get_trait", "get_trait_champions"}:
        validate_catalog_arguments(arguments, {"trait_id"}, {"trait_id"})
    elif name == "get_champion":
        validate_catalog_arguments(arguments, {"champion_id"}, {"champion_id"})
    elif name == "search_champions":
        validate_catalog_arguments(arguments, {"query", "cost", "trait_id"})

    elif name in {"search_items", "get_item"}:
        allowed = {"query", "kind"} if name == "search_items" else {"item_id"}
        for field in arguments:
            if field not in allowed:
                raise SessionError("invalid_input", "Unknown argument.", {"field": field, "value": arguments[field]})
        if name == "get_item" and "item_id" not in arguments:
            raise SessionError("invalid_input", "item_id is required.", {"field": "item_id", "value": None})
        for field, value in arguments.items():
            if type(value) is not str or (field == "kind" and value not in {"component", "equipment", "consumable"}):
                raise SessionError("invalid_input", f"Invalid {field}.", {"field": field, "value": value})
    else:
        raise SessionError("invalid_input", "Unknown tool.", {"tool": name})


def validate_catalog_arguments(arguments, allowed, required=()):
    for field in sorted(set(arguments) - allowed):
        raise SessionError("invalid_input", "Unknown argument.", {"field": field, "value": arguments[field]})
    for field in sorted(required):
        if field not in arguments:
            raise SessionError("invalid_input", "Missing required argument.", {"field": field, "value": None})
    for field, value in arguments.items():
        valid = type(value) is int and 1 <= value <= 5 if field == "cost" else type(value) is str
        if not valid:
            raise SessionError("invalid_input", "Invalid argument value.", {"field": field, "value": value})


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
                    elif name == "buy_unit":
                        result = session.buy_unit(**arguments)
                    elif name == "sell_unit":
                        result = session.sell_unit(**arguments)
                    elif name == "end_turn":
                        result = session.end_turn()

                    elif name == "get_players":
                        result = session.get_players()
                    elif name == "get_round":
                        result = session.get_round()
                    elif name == "get_traits":
                        result = session.get_traits(**arguments)
                    elif name == "get_economy":
                        result = session.get_economy()
                    elif name == "get_items":
                        result = session.get_items()
                    elif name == "get_shop":
                        result = session.get_shop()
                    elif name == "get_bench":
                        result = session.get_bench()
                    elif name == "get_board":
                        result = session.get_board(**arguments)
                    elif name == "search_traits":
                        result = session.search_traits(**arguments)
                    elif name == "get_trait":
                        result = session.get_trait(**arguments)
                    elif name == "get_trait_champions":
                        result = session.get_trait_champions(**arguments)
                    elif name == "get_champion":
                        result = session.get_champion(**arguments)
                    elif name == "search_champions":
                        result = session.search_champions(**arguments)

                    elif name == "search_items":
                        result = session.search_items(**arguments)
                    elif name == "get_item":
                        result = session.get_item(arguments["item_id"])
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
            except SessionError as log_error:
                result = log_error.result()
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

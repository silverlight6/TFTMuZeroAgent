"""Readable game state for a language-model agent.

The vector observation stays the default. This renderer is a second view of the
same player objects: text for a planning phase, and the legal command list taken
from the action mask.
"""

import numpy as np


def legal_commands(handler):
    """Decode every set mask bit into an ``[action_type, x1, x2]`` command."""
    mask = np.asarray(handler.fetch_action_mask())
    commands = []
    if mask.ndim == 2:
        width = mask.shape[1]
        rows, cols = np.where(mask > 0)
        for row, col in zip(rows.tolist(), cols.tolist()):
            commands.append(_command(handler.action_space_to_action(int(row) * width + int(col))))
    else:
        for index in np.flatnonzero(mask > 0).tolist():
            commands.append(_command(handler.action_space_to_action(int(index))))
    return commands


def _command(action):
    values = [int(part) for part in list(action)[:3]]
    while len(values) < 3:
        values.append(0)
    return values


def mask_allows(handler, action):
    """Return whether ``action`` is a set bit in the handler's mask."""
    kind, x1, x2 = (int(action[0]), int(action[1]), int(action[2]))
    mask = np.asarray(handler.fetch_action_mask())
    if mask.ndim != 2:
        return [kind, x1, x2] in legal_commands(handler)
    index = _mask_index(kind, x1, x2)
    if index is None:
        return False
    row, col = index
    return 0 <= row < mask.shape[0] and 0 <= col < mask.shape[1] and bool(mask[row, col])


def _mask_index(kind, x1, x2):
    if kind == 0:
        return 52, 0
    if kind == 1:
        return 53, 0
    if kind == 2:
        return 54, 0
    if kind == 3:
        return 47 + x1, 0
    if kind == 4:
        return x1, 37
    if kind == 5:
        return x1, x2
    if kind == 6:
        return 37 + x2, x1
    return None


# Sections an action can change. Pass refreshes opponents separately.
# The action counter is stored every time and shown when a phase opens.
_ACTION_SECTIONS = {
    0: frozenset(),
    1: frozenset({"gold", "exp", "level", "max_units"}),
    2: frozenset({"gold", "shop"}),
    3: frozenset({"gold", "shop", "bench", "board", "traits", "items", "units_in_play"}),
    4: frozenset({"gold", "board", "bench", "traits", "items", "units_in_play"}),
    5: frozenset({"board", "bench", "units_in_play"}),
    6: frozenset({"board", "bench", "items", "traits", "gold"}),
}
_ALL = (
    "round", "player", "health", "gold", "level", "exp", "win_streak", "loss_streak",
    "max_units", "units_in_play", "actions", "board", "bench", "shop", "items", "traits",
)


class _Cache:
    __slots__ = (
        "own", "opponents", "ready", "dirty", "failure", "outcome",
        "sections", "scout", "show_actions", "delivered",
    )

    def __init__(self):
        self.own = {}
        self.opponents = {}
        self.ready = False
        self.dirty = False
        self.failure = False
        self.outcome = None
        self.sections = frozenset()
        self.scout = False
        self.show_actions = False
        self.delivered = None


def remember_action(env, player_id, action, accepted, refresh_all):
    """Mark the next observation dirty. A refusal records failure and leaves the position cached."""
    cache = _cache(env, player_id)
    cache.dirty = True
    if not accepted:
        cache.failure = True
        cache.outcome = None
        return
    cache.failure = False
    cache.outcome = "succeeded"
    if refresh_all:
        cache.sections = frozenset(_ALL)
        cache.scout = True
        cache.show_actions = True
        return
    kind = int(action[0])
    cache.sections = _ACTION_SECTIONS.get(kind, frozenset(_ALL)) | {"actions"}
    cache.scout = kind == 0
    cache.show_actions = False


def render_llm_text(env, player_id):
    """Return the visible position, or only the sections changed since the last call.

    The first call has no previous position, so every section is included. Later
    calls rerender only the sections the last action could change. Opponent
    boards and traits are rerendered on the first call, on a pass, and when a
    round resolves.
    """
    cache = _cache(env, player_id)
    if cache.failure:
        cache.failure = False
        cache.dirty = False
        return "action failed"
    if not cache.dirty and cache.delivered is not None:
        return cache.delivered
    if not cache.ready:
        sections = frozenset(_ALL)
        scout = True
        show_actions = True
    else:
        sections = cache.sections
        scout = cache.scout
        show_actions = cache.show_actions
    text = _compose(env, player_id, cache, sections, scout, show_actions, cache.outcome)
    cache.ready = True
    cache.dirty = False
    cache.outcome = None
    cache.delivered = text
    return text


def _cache(env, player_id):
    caches = getattr(env, "_llm_caches", None)
    if caches is None:
        caches = {}
        env._llm_caches = caches
    cache = caches.get(player_id)
    if cache is None:
        cache = _Cache()
        caches[player_id] = cache
    return cache


def _compose(env, player_id, cache, sections, scout, show_actions, outcome):
    player = env.player_manager.player_states.get(player_id)
    if player is None:
        body = f"round {env.game_round.current_round}\n{player_id} eliminated"
        return f"action {outcome}\n{body}" if outcome else body
    parts = [f"action {outcome}"] if outcome else []
    render = set(sections)
    render.add("actions")
    for key in _ALL:
        if key not in render:
            continue
        current = _section(env, player, player_id, key)
        changed = cache.own.get(key) != current
        cache.own[key] = current
        if changed and (key != "actions" or show_actions):
            parts.append(current)
    if scout:
        changed_blocks = []
        opponents = {}
        for seat in env.possible_agents:
            if seat == player_id:
                continue
            block = _opponent_block(env, seat)
            opponents[seat] = block
            if cache.opponents.get(seat) != block:
                changed_blocks.append(block)
        cache.opponents = opponents
        if changed_blocks:
            parts.append("opponents\n" + "\n".join(changed_blocks))
    return "\n".join(parts)


def _section(env, player, player_id, key):
    if key == "round":
        return f"round {env.game_round.current_round}"
    if key == "player":
        return f"player {player_id}"
    if key == "health":
        return f"health {player.health}"
    if key == "gold":
        return f"gold {player.gold}"
    if key == "level":
        return f"level {player.level}"
    if key == "exp":
        return f"exp {player.exp}"
    if key == "win_streak":
        return f"win_streak {player.win_streak}"
    if key == "loss_streak":
        return f"loss_streak {player.loss_streak}"
    if key == "max_units":
        return f"max_units {player.max_units}"
    if key == "units_in_play":
        return f"units_in_play {player.num_units_in_play}"
    if key == "actions":
        return f"actions {env.actions_taken.get(player_id, 0)} of {env.max_actions_per_round}"
    if key == "board":
        return "\n".join(["board", *_board_lines(player, "  ")])
    if key == "bench":
        lines = ["bench"]
        for slot, unit in enumerate(player.bench):
            if unit is not None:
                lines.append(f"  {slot} {_unit_text(unit)}")
        return "\n".join(lines)
    if key == "shop":
        lines = ["shop"]
        for slot, name in enumerate(player.shop):
            lines.append(f"  {slot} {name if name else 'empty'}")
        return "\n".join(lines)
    if key == "items":
        lines = ["items"]
        for slot, item in enumerate(player.item_bench):
            if item is not None:
                lines.append(f"  {slot} {item}")
        return "\n".join(lines)
    lines = ["traits"]
    for trait, count in sorted(player.team_composition.items()):
        if count:
            lines.append(f"  {trait} {count} tier {player.team_tiers.get(trait, 0)}")
    return "\n".join(lines)


def _opponent_block(env, seat):
    other = env.player_manager.player_states.get(seat)
    if other is None:
        return f"  {seat} eliminated"
    lines = [f"  {seat} health {other.health} level {other.level}"]
    lines.extend(_board_lines(other, "    "))
    lines.append("    traits")
    for trait, count in sorted(other.team_composition.items()):
        if count:
            lines.append(f"      {trait} {count} tier {other.team_tiers.get(trait, 0)}")
    return "\n".join(lines)


def _board_lines(player, indent):
    lines = []
    for x, column in enumerate(player.board):
        for y, unit in enumerate(column):
            if unit is not None:
                lines.append(f"{indent}{x},{y} {_unit_text(unit)}")
    return lines


def _unit_text(unit):
    items = ",".join(unit.items) if unit.items else "-"
    chosen = unit.chosen if unit.chosen else "-"
    return f"{unit.name} stars {unit.stars} items {items} chosen {chosen}"

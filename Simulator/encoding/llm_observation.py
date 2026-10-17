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


def render_llm_text(env, player_id):
    """Describe the visible position. Occupied slots are listed; empty ones are omitted."""
    player = env.player_manager.player_states.get(player_id)
    lines = [f"round {env.game_round.current_round}"]
    if player is None:
        lines.append(f"{player_id} eliminated")
        return "\n".join(lines)
    taken = env.actions_taken.get(player_id, 0)
    lines.extend([
        f"player {player_id}",
        f"health {player.health}",
        f"gold {player.gold}",
        f"level {player.level}",
        f"exp {player.exp}",
        f"win_streak {player.win_streak}",
        f"loss_streak {player.loss_streak}",
        f"actions {taken} of {env.max_actions_per_round}",
        "board",
    ])
    lines.extend(_board_lines(player, "  "))
    lines.append("bench")
    for slot, unit in enumerate(player.bench):
        if unit is not None:
            lines.append(f"  {slot} {_unit_text(unit)}")
    lines.append("shop")
    for slot, name in enumerate(player.shop):
        lines.append(f"  {slot} {name if name else 'empty'}")
    lines.append("items")
    for slot, item in enumerate(player.item_bench):
        if item is not None:
            lines.append(f"  {slot} {item}")
    lines.append("traits")
    for trait, count in sorted(player.team_composition.items()):
        if count:
            lines.append(f"  {trait} {count}")
    lines.append("opponents")
    for seat in env.possible_agents:
        if seat == player_id:
            continue
        other = env.player_manager.player_states.get(seat)
        if other is None:
            lines.append(f"  {seat} eliminated")
            continue
        lines.append(f"  {seat} health {other.health} level {other.level}")
        lines.extend(_board_lines(other, "    "))
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

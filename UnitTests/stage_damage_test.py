"""Base player damage follows the stage, not the round index.

Round index r maps to stages as: r = 0..2 is stage 1 (1-2, 1-3, 1-4), then six indices per
stage from r = 3 (2-1 .. 2-7, carousel folded into x-5). Set 4 base damage per stage
(patch 10.24 notes, unchanged through 11.8) is 0/0/2/3/5/8/15 for stages 1-7.
The old table switched tiers at r = 3, 9, 15, ... (the first round of the next stage),
so x-2 .. x-7 used the next stage's damage.
"""

from __future__ import annotations

import pytest

from Simulator.battle import champion as champion_module
from Simulator.battle import minion
from Simulator.battle.combat_context import CombatContext
from Simulator.game.game_round import Game_Round
from Simulator.game.player import Player
from Simulator.game.pool import pool
from Simulator.rng import EnvRNG

# (round index, stage-round, base damage)
PVP_ROUNDS = [
    (3, "2-1", 0), (5, "2-3", 0), (7, "2-6", 0),
    (9, "3-1", 2), (11, "3-3", 2), (13, "3-6", 2),
    (15, "4-1", 3), (19, "4-6", 3),
    (21, "5-1", 5), (25, "5-6", 5),
    (27, "6-1", 8), (31, "6-6", 8),
    (33, "7-1", 15), (37, "7-6", 15),
    (39, "8-1", 15),
]

PVE_ROUNDS = [(8, "2-7", 0), (14, "3-7", 2), (20, "4-7", 3), (26, "5-7", 5), (38, "7-7", 15)]


def _record_base_damage(monkeypatch):
    seen = []

    def fake_run(champion_q, player_1, player_2, round_damage=0):
        seen.append(round_damage)
        return 1, round_damage

    monkeypatch.setattr(champion_module, "run", fake_run)
    return seen


@pytest.mark.parametrize("round_index,label,expected", PVP_ROUNDS, ids=[r[1] for r in PVP_ROUNDS])
def test_player_combat_base_damage_by_stage(monkeypatch, round_index, label, expected):
    seen = _record_base_damage(monkeypatch)
    with CombatContext(rng=EnvRNG.from_episode_seed(1)).bind():
        base_pool = pool()
        players = {"player_0": Player(base_pool, 0), "player_1": Player(base_pool, 1)}
        game_round = Game_Round(players, base_pool, None)
        game_round.matchups = [["player_0", "player_1"]]
        game_round.combat_phase(players, round_index)
    assert seen == [expected], label


@pytest.mark.parametrize("round_index,label,expected", PVE_ROUNDS, ids=[r[1] for r in PVE_ROUNDS])
def test_minion_combat_base_damage_by_stage(monkeypatch, round_index, label, expected):
    seen = _record_base_damage(monkeypatch)
    with CombatContext(rng=EnvRNG.from_episode_seed(1)).bind():
        player = Player(pool(), 0)
        minion.minion_round(player, round_index)
    assert seen == [expected], label

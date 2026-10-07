"""Passive gold and XP arrive from 1-2 on, as in the real game.

Real game: passive income is 2 / 2 / 3 / 4 gold at the start of 1-2 / 1-3 / 1-4 / 2-1
and 5 gold (plus interest and streak gold) from 2-2 on, and every round grants 2 XP.
In the simulator, round index 0 is the 1-1 carousel plus the 1-2 fight with no planning
step in between, and the env never calls start_round() before the 1-3 planning phase,
so players reached 1-3 with no passive gold or XP and the table [0, 2, 2, 3, 4] was
then paid at 1-4 / 2-1 / 2-2 instead: income ran two rounds late.
"""

from __future__ import annotations

from Simulator.battle import minion
from Simulator.battle.combat_context import CombatContext
from Simulator.game import single_player_game_round
from Simulator.game.game_round import Game_Round
from Simulator.game.player import Player
from Simulator.game.pool import pool
from Simulator.rng import EnvRNG


def _skip_pve(monkeypatch):
    # PvE loot can contain gold; keep the test to passive income only.
    monkeypatch.setattr(minion, "minion_round", lambda *args, **kwargs: True)


def test_first_planning_phase_has_income_of_1_2_and_1_3(monkeypatch):
    _skip_pve(monkeypatch)
    with CombatContext(rng=EnvRNG.from_episode_seed(2)).bind():
        base_pool = pool()
        players = {f"player_{i}": Player(base_pool, i) for i in range(8)}
        game_round = Game_Round(players, base_pool, None)
        game_round.play_game_round()  # 1-1 carousel and the 1-2 fight; next comes the 1-3 planning phase
    assert game_round.current_round == 1
    for player in players.values():
        assert (player.gold, player.level, player.exp) == (4, 3, 0)


def test_passive_income_schedule_through_2_2(monkeypatch):
    _skip_pve(monkeypatch)
    with CombatContext(rng=EnvRNG.from_episode_seed(2)).bind():
        base_pool = pool()
        player = Player(base_pool, 0)
        game_round = Game_Round({"player_0": player}, base_pool, None)
        game_round.play_game_round()
        seen = [(player.gold, player.level, player.exp)]
        # The env calls start_round(r) at the start of each later planning phase r.
        for round_index in (2, 3, 4):
            player.start_round(round_index)
            seen.append((player.gold, player.level, player.exp))
    # 1-3: 2+2, 1-4: +3, 2-1: +4, 2-2: +5 and 1 interest on 11 gold; level 4 at 2-2.
    assert seen == [(4, 3, 0), (7, 3, 2), (11, 3, 4), (17, 4, 0)]


def test_single_player_first_planning_phase_has_same_income(monkeypatch):
    _skip_pve(monkeypatch)
    with CombatContext(rng=EnvRNG.from_episode_seed(2)).bind():
        player = Player(pool(), 0)
        game_round = single_player_game_round.Game_Round(player, player.pool_obj, None)
        game_round.play_game_round()
    assert (player.gold, player.level, player.exp) == (4, 3, 0)


def test_gold_income_table():
    with CombatContext(rng=EnvRNG.from_episode_seed(2)).bind():
        base_pool = pool()
        paid = []
        for round_index in range(6):
            player = Player(base_pool, 0)
            player.gold_income(round_index)
            paid.append(player.gold)
    # rounds 0..3 are the planning phases of 1-2, 1-3, 1-4 and 2-1
    assert paid == [2, 2, 3, 4, 5, 5]

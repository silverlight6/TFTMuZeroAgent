"""Every round the schedule marks as a minion round actually fights monsters.

Game_Round.game_rounds puts minion_round at indices 1, 2 (1-3, 1-4) and 8, 14, 20, 26,
32, 38 (2-7 .. 7-7); round_1 plays the 1-2 fight at index 0. minion.minion_round only
started the late-game fight at index >= 33, so 6-7 (index 32) silently did nothing.
"""

from __future__ import annotations

from Simulator.battle import minion
from Simulator.battle.combat_context import CombatContext
from Simulator.game.game_round import Game_Round
from Simulator.game.player import Player
from Simulator.game.pool import pool
from Simulator.rng import EnvRNG


def _record_fights(monkeypatch):
    fights = []

    def fake_combat(player, enemy, round, others=None, other_rewards=True):
        fights.append((round, type(enemy).__name__))
        return True

    monkeypatch.setattr(minion, "minion_combat", fake_combat)
    return fights


def test_every_scheduled_minion_round_has_a_fight(monkeypatch):
    fights = _record_fights(monkeypatch)
    with CombatContext(rng=EnvRNG.from_episode_seed(1)).bind():
        base_pool = pool()
        player = Player(base_pool, 0)
        game_round = Game_Round({"player_0": player}, base_pool, None)
        scheduled = [i for i, phases in enumerate(game_round.game_rounds) if game_round.minion_round in phases]
        assert scheduled == [1, 2, 8, 14, 20, 26, 32, 38]
        for round_index in scheduled:
            minion.minion_round(player, round_index)
    assert [r for r, _ in fights] == scheduled


def test_stage_6_7_fights_the_late_game_monster(monkeypatch):
    fights = _record_fights(monkeypatch)
    with CombatContext(rng=EnvRNG.from_episode_seed(1)).bind():
        player = Player(pool(), 0)
        assert minion.minion_round(player, 32) is True
    assert fights == [(32, "Herald")]


def test_player_rounds_have_no_monster_fight(monkeypatch):
    fights = _record_fights(monkeypatch)
    with CombatContext(rng=EnvRNG.from_episode_seed(1)).bind():
        player = Player(pool(), 0)
        for round_index in (3, 9, 27, 31):
            assert minion.minion_round(player, round_index) is False
    assert fights == []


def test_game_round_loop_fights_at_6_7(monkeypatch):
    """The same check through Game_Round itself: play_game_round walks the schedule and passes
    its own current_round to minion.minion_round, so the index the loop is on at 6-7 is 32."""
    fights = _record_fights(monkeypatch)
    monkeypatch.setattr(Game_Round, "combat_phase", lambda self, players, player_round: None)  # nobody dies
    with CombatContext(rng=EnvRNG.from_episode_seed(1)).bind():
        base_pool = pool()
        player = Player(base_pool, 0)
        game_round = Game_Round({"player_0": player}, base_pool, None)
        while game_round.current_round < len(game_round.game_rounds):
            game_round.play_game_round()
    # round_1 plays the 1-2 fight with index 0, then the scheduled minion rounds follow
    assert [r for r, _ in fights] == [0, 1, 2, 8, 14, 20, 26, 32, 38]
    assert fights[-2:] == [(32, "Herald"), (38, "Herald")]

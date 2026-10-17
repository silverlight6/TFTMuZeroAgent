"""Scope of the matchmaking change in Game_Round.decide_player_combat.

What changes: which opponent the first player of the shuffled list gets.
- Weighted draw: r = randint(0, total - 1), where total is the sum of the eligible weights
  (weight >= MATCHMAKING_WEIGHTS), walked over the eligible opponents only. Every eligible
  opponent is drawn with probability weight / total; nobody below the threshold is drawn.
- Fallback (nobody eligible): the highest weight among the remaining players.

What does not change (pinned here): the shuffle, one randint per draw (none in the fallback),
the pairing structure (every living player in exactly one matchup, one ghost fight when the
count is odd), the weight bookkeeping after the draw, and the opponent flags written for the
observation.

Weights are keyed by "player_<player_num>" while the draw walks seat names, so the draw is
only unbiased when seat "player_i" holds player_num i (the seat-order fix). The last test checks
that against the env's own PlayerManager.
"""

from __future__ import annotations

import random as std_random

import pytest

from Simulator import config
from Simulator.battle.combat_context import CombatContext
from Simulator.game import game_round as game_round_module
from Simulator.game.game_round import Game_Round
from Simulator.game.player import Player
from Simulator.game.player_manager import PlayerManager
from Simulator.game.pool import pool
from Simulator.rng import EnvRNG
from Simulator.simulators.tft_simulator import TFTConfig

T = config.MATCHMAKING_WEIGHTS
INC = config.WEIGHTS_INCREMENT


class ScriptedRandom:
    """Stands in for game_round.random: no shuffle, randint returns scripted values."""

    def __init__(self, values=()):
        self.values = list(values)
        self.randint_calls = []
        self.shuffle_calls = 0

    def shuffle(self, x):
        self.shuffle_calls += 1

    def randint(self, a, b):
        self.randint_calls.append((a, b))
        return self.values.pop(0)


class CountingRandom:
    """Delegates to a seeded random.Random and counts the calls."""

    def __init__(self, seed):
        self.rng = std_random.Random(seed)
        self.randint_calls = []
        self.shuffle_calls = 0

    def shuffle(self, x):
        self.shuffle_calls += 1
        self.rng.shuffle(x)

    def randint(self, a, b):
        self.randint_calls.append((a, b))
        return self.rng.randint(a, b)


def _players(num_players, weights_of_first=None, weight=T):
    base_pool = pool()
    players = {f"player_{i}": Player(base_pool, i) for i in range(num_players)}
    for a, player in enumerate(players.values()):
        for b in range(num_players):
            if a != b:
                player.possible_opponents[f"player_{b}"] = weight
    if weights_of_first:
        players["player_0"].possible_opponents.update(weights_of_first)
    return base_pool, players


def _decide(players, base_pool, fake_random, monkeypatch):
    monkeypatch.setattr(game_round_module, "random", fake_random)
    game_round = Game_Round(players, base_pool, None)
    game_round.decide_player_combat()
    return game_round


# --- what changes ----------------------------------------------------------------------------

WEIGHTS = {"player_1": 10, "player_2": 21, "player_3": 9, "player_4": 0, "player_5": 13, "player_6": 3}


def test_every_draw_value_lands_on_an_eligible_opponent_in_proportion_to_its_weight(monkeypatch):
    """Enumerate every value randint can return: opponent i is hit exactly weight_i times."""
    eligible_total = sum(w for w in WEIGHTS.values() if w >= T)
    hits = {}
    with CombatContext(rng=EnvRNG.from_episode_seed(1)).bind():
        for r in range(eligible_total):
            base_pool, players = _players(7, WEIGHTS)
            fake = ScriptedRandom([r] + [0] * 10)
            game_round = _decide(players, base_pool, fake, monkeypatch)
            assert fake.randint_calls[0] == (0, eligible_total - 1)
            opponent = game_round.matchups[0][1]
            hits[opponent] = hits.get(opponent, 0) + 1
    assert hits == {key: w for key, w in WEIGHTS.items() if w >= T}


@pytest.mark.parametrize("weight,eligible", [(T - 1, False), (T, True), (T + INC, True)])
def test_threshold_is_inclusive(monkeypatch, weight, eligible):
    with CombatContext(rng=EnvRNG.from_episode_seed(2)).bind():
        base_pool, players = _players(3, {"player_1": weight, "player_2": T + 30})
        # r = 0 is the first eligible opponent in list order: player_1 if it counts, else player_2.
        game_round = _decide(players, base_pool, ScriptedRandom([0, 0]), monkeypatch)
    first = game_round.matchups[0]
    assert first[0] == "player_0"
    assert first[1] == ("player_1" if eligible else "player_2")


@pytest.mark.parametrize("weights,expected", [
    ({"player_1": 3, "player_2": 9, "player_3": 6}, "player_2"),
    ({"player_1": 0, "player_2": 0, "player_3": 3}, "player_3"),
    ({"player_1": 6, "player_2": 6, "player_3": 0}, "player_2"),   # tie: the later one (>=)
    ({"player_1": -1, "player_2": 0, "player_3": 0}, "player_3"),
])
def test_fallback_takes_the_highest_weight_without_drawing(monkeypatch, weights, expected):
    with CombatContext(rng=EnvRNG.from_episode_seed(3)).bind():
        base_pool, players = _players(4, weights, weight=0)
        fake = ScriptedRandom()
        game_round = _decide(players, base_pool, fake, monkeypatch)
    assert game_round.matchups[0] == ["player_0", expected]
    assert fake.randint_calls == []


# --- what does not change ----------------------------------------------------------------------

@pytest.mark.parametrize("num_players", [2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("seed", range(10))
def test_pairing_structure(monkeypatch, num_players, seed):
    """Every living player is in exactly one matchup; an odd count adds one ghost fight."""
    with CombatContext(rng=EnvRNG.from_episode_seed(seed)).bind():
        rng = std_random.Random(seed)
        base_pool, players = _players(num_players)
        for a, player in enumerate(players.values()):
            for b in range(num_players):
                if a != b:
                    player.possible_opponents[f"player_{b}"] = rng.choice([0, 3, 6, 9, 10, 13, 16, 19])
        game_round = _decide(players, base_pool, CountingRandom(seed), monkeypatch)
    seen = []
    ghosts = [m for m in game_round.matchups if m[1] == "ghost"]
    for match in game_round.matchups:
        seen.append(match[0])
        if match[1] != "ghost":
            seen.append(match[1])
    assert sorted(seen) == sorted(players)
    assert len(ghosts) == num_players % 2
    for match in ghosts:
        assert match[2] in players and match[2] != match[0]


@pytest.mark.parametrize("seed", range(20))
def test_one_shuffle_and_one_randint_per_weighted_draw(monkeypatch, seed):
    """The RNG is used as before: one shuffle, then one randint for each draw that has an eligible opponent."""
    with CombatContext(rng=EnvRNG.from_episode_seed(seed)).bind():
        rng = std_random.Random(seed)
        base_pool, players = _players(8)
        for a, player in enumerate(players.values()):
            for b in range(8):
                if a != b:
                    player.possible_opponents[f"player_{b}"] = rng.choice([0, 3, 6, 9, 10, 13])
        before = {k: dict(p.possible_opponents) for k, p in players.items()}
        fake = CountingRandom(seed)
        game_round = _decide(players, base_pool, fake, monkeypatch)
    assert fake.shuffle_calls == 1
    remaining = set(players)
    expected_draws = 0
    for first, second in game_round.matchups:
        if any(before[first][k] >= T for k in remaining if k != first):
            expected_draws += 1
        remaining -= {first, second}
    assert len(fake.randint_calls) == expected_draws
    assert all(a == 0 for a, _ in fake.randint_calls)


def test_weight_bookkeeping_after_the_draw(monkeypatch):
    """Paired players reset each other to 0, every other living seat gains WEIGHTS_INCREMENT, -1 stays."""
    with CombatContext(rng=EnvRNG.from_episode_seed(5)).bind():
        base_pool, players = _players(4, weight=T)
        before = {k: dict(p.possible_opponents) for k, p in players.items()}
        game_round = _decide(players, base_pool, CountingRandom(5), monkeypatch)
    for a, b in game_round.matchups:
        for me, other in ((a, b), (b, a)):
            weights = players[me].possible_opponents
            assert weights[other] == 0
            for key in players:
                if key not in (me, other):
                    expected = before[me][key] if before[me][key] == -1 else before[me][key] + INC
                    assert weights[key] == expected, (me, key)


def test_opponent_flags_for_the_observation(monkeypatch):
    """opponent_options marks the seats that were eligible before the draw, and the drawn opponent."""
    with CombatContext(rng=EnvRNG.from_episode_seed(6)).bind():
        base_pool, players = _players(4, {"player_1": T, "player_2": T - 1, "player_3": 0})
        before = dict(players["player_0"].possible_opponents)
        game_round = _decide(players, base_pool, ScriptedRandom([0, 0]), monkeypatch)
    assert game_round.matchups[0] == ["player_0", "player_1"]
    flags = players["player_0"].opponent_options
    for key in players:
        if key != "player_0" and before[key] >= T:
            assert flags[key] == 1
    assert flags["player_1"] == 1


# --- interaction with the seat order -----------------------------------------------------------

with CombatContext(rng=EnvRNG.from_episode_seed(0)).bind():
    _MANAGER = PlayerManager(8, pool(), TFTConfig())
_SEAT_IS_PLAYER_NUM = all(p.player_num == int(seat.split("_")[1]) for seat, p in _MANAGER.player_states.items())


@pytest.mark.xfail(condition=not _SEAT_IS_PLAYER_NUM, strict=True,
                   reason="seat -> player_num follows set order (PYTHONHASHSEED); needs the seat-order fix")
def test_env_players_can_draw_every_other_seat():
    """With the env's own PlayerManager, each player's -1 weight must be on its own seat.

    Otherwise it sits on another seat, which is then never eligible (-1 is never incremented)
    until the two meet in a forced or fallback pairing, and those seat pairs meet less often.
    """
    for seat, player in _MANAGER.player_states.items():
        blocked = [key for key, w in player.possible_opponents.items() if w < 0]
        assert blocked == [seat]

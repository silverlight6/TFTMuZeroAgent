"""Every living player gets one carousel pick, lowest HP first in pairs.

The old loop only inserted a player whose HP was <= the current front's, so most
players got nothing; at the first carousel Player.__eq__ made all (empty) players
compare equal and only the first one was added.
"""

from __future__ import annotations

from Simulator.battle.combat_context import CombatContext
from Simulator.game.carousel import carousel, carousel_order
from Simulator.game.player import Player
from Simulator.game.pool import pool
from Simulator.rng import EnvRNG


def _bench_count(player):
    return sum(champ is not None for champ in player.bench)


def test_first_carousel_gives_every_player_a_unit():
    with CombatContext(rng=EnvRNG.from_episode_seed(3)).bind():
        base_pool = pool()
        players = [Player(base_pool, i) for i in range(8)]
        carousel(players, 0, base_pool)
        assert [_bench_count(p) for p in players] == [1] * 8


def test_later_carousel_gives_every_living_player_a_unit():
    with CombatContext(rng=EnvRNG.from_episode_seed(5)).bind():
        base_pool = pool()
        players = [Player(base_pool, i) for i in range(8)]
        for player, hp in zip(players, [80, 90, 70, 60, 100, 50, 30, 95]):
            player.health = hp
        players[3].health = 0
        carousel(players + [None], 12, base_pool)
        assert [_bench_count(p) for p in players] == [1, 1, 1, 0, 1, 1, 1, 1]


def test_order_is_pairs_from_lowest_hp():
    for seed in range(10):
        with CombatContext(rng=EnvRNG.from_episode_seed(seed)).bind():
            base_pool = pool()
            players = [Player(base_pool, i) for i in range(8)]
            for player, hp in zip(players, [80, 90, 70, 60, 100, 50, 30, 95]):
                player.health = hp
            order = carousel_order(players, 18)
        pairs = [sorted(p.health for p in order[i:i + 2]) for i in range(0, 8, 2)]
        assert pairs == [[30, 50], [60, 70], [80, 90], [95, 100]]


def test_order_is_seeded():
    def names(seed):
        with CombatContext(rng=EnvRNG.from_episode_seed(seed)).bind():
            base_pool = pool()
            players = [Player(base_pool, i) for i in range(8)]
            return [p.player_num for p in carousel_order(players, 6)]

    assert names(1) == names(1)
    assert names(1) != names(2)

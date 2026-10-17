"""decide_player_combat picks opponents the way its comments describe.

Each player keeps a weight per opponent: 0 right after they fight, +WEIGHTS_INCREMENT each
round. Opponents at or above MATCHMAKING_WEIGHTS are eligible and are drawn in proportion
to their weight; if nobody is eligible the highest weight is taken. The old draw walked
over every opponent, eligible or not, so an opponent below the threshold could be drawn
(r = randint(0, weights) also had one value too many), and the fallback loop tested
`i < 0`, so it always took the next player in the shuffled list.
"""

from __future__ import annotations

from collections import Counter

from Simulator import config
from Simulator.battle.combat_context import CombatContext
from Simulator.game.game_round import Game_Round
from Simulator.game.player import Player
from Simulator.game.pool import pool
from Simulator.rng import EnvRNG


def _matchups(seed, num_players, weight_of):
    """Run one decide_player_combat with possible_opponents[a][b] = weight_of(a, b)."""
    with CombatContext(rng=EnvRNG.from_episode_seed(seed)).bind():
        base_pool = pool()
        players = {f"player_{i}": Player(base_pool, i) for i in range(num_players)}
        for a, player in enumerate(players.values()):
            for b in range(num_players):
                if a != b:
                    player.possible_opponents[f"player_{b}"] = weight_of(a, b)
        game_round = Game_Round(players, base_pool, None)
        game_round.decide_player_combat()
    return {tuple(sorted(match[:2])) for match in game_round.matchups}


PARTNERS = {("player_0", "player_1"), ("player_2", "player_3"), ("player_4", "player_5"), ("player_6", "player_7")}


def test_only_eligible_opponents_are_drawn():
    # Each player's only eligible opponent is its partner; every other weight is below the threshold.
    eligible = config.MATCHMAKING_WEIGHTS
    below = config.MATCHMAKING_WEIGHTS - 1
    for seed in range(30):
        matchups = _matchups(seed, 8, lambda a, b: eligible if a // 2 == b // 2 else below)
        assert matchups == PARTNERS, seed


def test_fallback_takes_the_highest_weight():
    # Nobody is eligible; each player's highest weight is its partner.
    high = config.MATCHMAKING_WEIGHTS - 1
    low = config.WEIGHTS_INCREMENT
    for seed in range(30):
        matchups = _matchups(seed, 8, lambda a, b: high if a // 2 == b // 2 else low)
        assert matchups == PARTNERS, seed


def test_draw_is_proportional_to_weight():
    # Two players: any draw is the other one. Three players: the first player of the shuffled
    # list draws between the other two, here weighted 10 against 30 for everyone.
    counts = Counter()
    for seed in range(400):
        weights = {(0, 1): 10, (0, 2): 30, (1, 0): 10, (1, 2): 30, (2, 0): 30, (2, 1): 10}
        matchups = _matchups(seed, 3, lambda a, b: weights[(a, b)])
        counts.update(matchups)
    # Pair (0, 2) has weight 30 from both sides and should be drawn far more often than (0, 1).
    assert counts[("player_0", "player_2")] > 2 * counts[("player_0", "player_1")]

"""A Chosen shop unit costs three times its 1-star price.

Patch 10.19 notes: Chosen champions "are already at 2-star level, so they cost three
times their normal 1-star price". The shop charged cost_star_values[cost - 1][1], the
2-star sell value (1-cost 3, otherwise 3 * cost - 1), so 2- to 5-cost Chosen were one
gold cheaper than in the game, and the buy mask used the same number.
"""

from __future__ import annotations

import pytest

from Simulator.battle.combat_context import CombatContext
from Simulator.encoding.token.action import ActionToken
from Simulator.game.player import Player
from Simulator.game.pool import pool
from Simulator.rng import EnvRNG

CHOSEN = [("diana_assassin_c", 1), ("annie_fortune_c", 2), ("akali_assassin_c", 3),
          ("ahri_spirit_c", 4), ("azir_keeper_c", 5)]


def _player_offered(shop_entry, gold):
    player = Player(pool(), 0)
    player.gold = gold
    player.shop = [shop_entry, "garen", "vayne", "nami", "elise"]
    player.shop_champions = player.create_shop_champions()
    return player


@pytest.mark.parametrize("shop_entry,cost", CHOSEN, ids=[entry for entry, _ in CHOSEN])
def test_chosen_costs_three_times_base(shop_entry, cost):
    with CombatContext(rng=EnvRNG.from_episode_seed(7)).bind():
        player = _player_offered(shop_entry, 50)
        assert player.shop_champions[0].stars == 2
        assert player.buy_shop_action(0)
        assert player.gold == 50 - 3 * cost


@pytest.mark.parametrize("shop_entry,cost", CHOSEN, ids=[entry for entry, _ in CHOSEN])
def test_buy_mask_needs_the_full_price(shop_entry, cost):
    with CombatContext(rng=EnvRNG.from_episode_seed(7)).bind():
        short = _player_offered(shop_entry, 3 * cost - 1)
        assert ActionToken(short).buy_mask[0] == 0
        assert not short.buy_shop_action(0)

        exact = _player_offered(shop_entry, 3 * cost)
        assert ActionToken(exact).buy_mask[0] == 1


def test_normal_units_keep_their_price():
    with CombatContext(rng=EnvRNG.from_episode_seed(7)).bind():
        player = _player_offered("annie", 10)
        assert player.buy_shop_action(0)
        assert player.gold == 8

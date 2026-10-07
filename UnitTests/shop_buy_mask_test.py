"""Buying one shop slot leaves the other slots buyable.

Player.shop_empty() returned `not all(self.shop)`, which is True as soon as one slot is
None. Buying a unit empties its slot, so after the first purchase the buy mask closed
every remaining slot until the next refresh.
"""

from __future__ import annotations

import numpy as np

from Simulator.battle.combat_context import CombatContext
from Simulator.encoding.token.action import ActionToken
from Simulator.encoding.token.action_vector import ActionVector, SHOP_START, PASS_INDEX
from Simulator.game.player import Player
from Simulator.game.pool import pool
from Simulator.rng import EnvRNG

SHOP = ["diana", "elise", "fiora", "garen", "nami"]


def _player_with_shop(gold=10):
    player = Player(pool(), 0)
    player.gold = gold
    player.shop = list(SHOP)
    player.shop_champions = player.create_shop_champions()
    return player


def test_shop_empty_only_when_every_slot_is_empty():
    with CombatContext(rng=EnvRNG.from_episode_seed(4)).bind():
        player = _player_with_shop()
        assert not player.shop_empty()
        player.shop[0] = None
        assert not player.shop_empty()
        player.shop = [None] * 5
        assert player.shop_empty()


def test_other_slots_stay_buyable_after_a_purchase():
    with CombatContext(rng=EnvRNG.from_episode_seed(4)).bind():
        player = _player_with_shop()
        actions = ActionToken(player)
        assert list(actions.buy_mask) == [1, 1, 1, 1, 1]

        assert player.buy_shop_action(0)
        actions.update_action_mask([3, 0, 0])
        assert list(actions.buy_mask) == [0, 1, 1, 1, 1]

        assert player.buy_shop_action(3)
        actions.update_action_mask([3, 3, 0])
        assert list(actions.buy_mask) == [0, 1, 1, 0, 1]


def test_vector_mask_follows():
    with CombatContext(rng=EnvRNG.from_episode_seed(4)).bind():
        player = _player_with_shop()
        actions = ActionVector(player)
        player.buy_shop_action(2)
        actions.update_action_mask([3, 2, 0])
        mask = actions.fetch_action_mask()
        assert np.array_equal(mask[SHOP_START:PASS_INDEX], [1, 1, 0, 1, 1])

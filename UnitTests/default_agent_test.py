"""Default_Agent fixes: the Katarina spelling and the bench-to-board swap.

- default_agent_stats spelled Katarina "katerina", so the fortune comp never counted
  her as a comp unit and the bot had no row to place her in.
- round_3_10 and round_11_end tested `board_unit in BASE_CHAMPION_LIST`, a champion
  object against a list of names, which is never true, so the bench swap never ran;
  round_3_10 also returned it as a buy ("3_...") instead of a move ("5_...").
"""

from __future__ import annotations

import numpy as np

from Simulator.battle.champion import champion
from Simulator.battle.combat_context import CombatContext
from Simulator.battle.stats import BASE_CHAMPION_LIST
from Simulator.generators import default_agent_stats as stats
from Simulator.generators.default_agent import Default_Agent
from Simulator.game.player import Player
from Simulator.game.pool import pool
from Simulator.rng import EnvRNG
from Simulator.utils import decode_action, x_y_to_1d_coord

EMPTY_SHOP = [None] * 5
NO_BUYS = np.zeros((55, 38))


def test_every_name_in_the_bot_tables_is_a_champion():
    tables = {
        "FRONT_LINE_UNITS": stats.FRONT_LINE_UNITS,
        "MIDDLE_LINE_UNITS": stats.MIDDLE_LINE_UNITS,
        "BACK_LINE_UNITS": stats.BACK_LINE_UNITS,
        "TEAM_COMPS": [name for comp in stats.TEAM_COMPS for name in comp],
        "TEAM_ITEM_HOLDER": [name for pair in stats.TEAM_ITEM_HOLDER for name in pair],
    }
    for table, names in tables.items():
        unknown = [name for name in names if name not in BASE_CHAMPION_LIST]
        assert unknown == [], table
    assert "katarina" in stats.TEAM_COMPS[stats.TEAM_COMP_TRAITS.index("fortune")]


def test_katarina_gets_a_board_slot():
    with CombatContext(rng=EnvRNG.from_episode_seed(6)).bind():
        player = Player(pool(), 0)
        assert Default_Agent().move_bench_to_empty_board(player, 28, "katarina").startswith("5_28_")


def _board_garen_bench_sett():
    player = Player(pool(), 0)
    player.gold = 0
    player.board[3][3] = champion("garen")
    player.num_units_in_play = 1
    player.max_units = 1
    player.bench[0] = champion("sett")
    return player


def test_round_3_10_swaps_a_better_bench_unit_in():
    with CombatContext(rng=EnvRNG.from_episode_seed(6)).bind():
        player = _board_garen_bench_sett()
        agent = Default_Agent()
        agent.next_round = 5
        action = agent.policy(player, EMPTY_SHOP, 5, NO_BUYS)
        assert action == "5_" + str(x_y_to_1d_coord(3, 3)) + "_28"

        action_type, x1, x2 = decode_action([action])[0]
        assert player.move_champ_action(int(x1), int(x2))
        assert player.board[3][3].name == "sett"
        assert player.bench[0].name == "garen"


def test_round_11_end_swaps_a_comp_unit_in():
    with CombatContext(rng=EnvRNG.from_episode_seed(6)).bind():
        player = _board_garen_bench_sett()
        agent = Default_Agent()
        agent.comp_number = stats.TEAM_COMP_TRAITS.index("fortune")  # sett is in this comp, garen is not
        agent.round_11_clean_up = False
        agent.next_round = 12
        action = agent.policy(player, EMPTY_SHOP, 12, NO_BUYS)
        assert action == "5_" + str(x_y_to_1d_coord(3, 3)) + "_28"

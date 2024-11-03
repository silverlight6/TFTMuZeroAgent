import time
import numpy as np
import copy
from Simulator.player import Player
from Simulator.pool import pool
from Simulator.champion import champion
from Simulator.observation.token.action import ActionToken
from Simulator.player_manager import PlayerManager
from Simulator.tft_simulator import TFTConfig
from Simulator.step_function import Step_Function


def setup() -> Player:
    """Creates fresh player and pool"""
    base_pool = pool()
    player1 = Player(base_pool, 0)
    return player1

def premade_warlord_test():
    # Azir, Garen, Jarvan, Katarina, Nidalee, Vi, XinZhou, add chosen and item (warlords_banner)
    p1 = setup()
    p1.max_units = 8
    p1.gold = 1000
    azir = champion('azir', chosen='warlord')
    p1.buy_champion(azir)
    p1.move_bench_to_board(0, 0, 0)
    garen = champion('garen')
    p1.buy_champion(garen)
    p1.move_bench_to_board(0, 2, 0)
    jarvaniv = champion('jarvaniv')
    p1.buy_champion(jarvaniv)
    p1.move_bench_to_board(0, 3, 0)
    katarina = champion('katarina')
    p1.buy_champion(katarina)
    p1.move_bench_to_board(0, 4, 0)
    nidalee = champion('nidalee')
    p1.buy_champion(nidalee)
    p1.move_bench_to_board(0, 5, 0)
    vi = champion('vi')
    p1.buy_champion(vi)
    p1.move_bench_to_board(0, 6, 0)
    xinzhao = champion('xinzhao')
    p1.buy_champion(xinzhao)
    p1.move_bench_to_board(0, 3, 1)
    yuumi = champion('yuumi', itemlist=['warlords_banner'])
    p1.buy_champion(yuumi)
    p1.move_bench_to_board(0, 2, 1)

    player_comp = p1.team_composition
    player_tiers = p1.team_tiers
    assert player_comp['warlord'] == 9
    assert player_tiers['warlord'] == 3

    tier_values = list(player_tiers.values())
    tier_keys = list(player_tiers.keys())
    for i in range(len(player_comp)):
        if tier_keys[i] != 'warlord' and tier_keys[i] != 'keeper' and tier_keys[i] != 'emperor':
            assert tier_values[i] == 0


def premade_cultist_test():
    ...

def premade_ninja_test():
    ...

def premade_adept_test():
    ...

def end_turn_comp_test():
    ...

def one_thousand_action_test():
    base_pool = pool()
    player_manager = PlayerManager(1, base_pool, TFTConfig)
    step_function = Step_Function(player_manager)
    p1 = Player(base_pool, 0)
    p1.max_units = 8
    p1.gold = 1000
    action = ActionToken(p1)

    # Take 1000 actions
    for _ in range(1000):
        mask = np.reshape(action.fetch_action_mask(), [-1])
        legal_actions = np.array([i for i, x in enumerate(mask) if x == 1.0])
        random_action = np.random.choice(legal_actions, 1)
        step_function.perform_1d_action('player_0', random_action)

    p2 = Player(base_pool, 1)
    p2.max_units = 8
    p2.gold = 1000
    for x in range(len(p1.board)):
        for y in range(len(p1.board[0])):
            if p1.board[x][y]:
                p2.buy_champion(copy.deepcopy(p1.board[x][y]))
                p2.move_bench_to_board(0, x, y)

    for p1_trait, p2_trait in zip(p1.team_composition.values, p2.team_composition.values):
        assert p1_trait == p2_trait


def test_list():
    premade_warlord_test()
    one_thousand_action_test()

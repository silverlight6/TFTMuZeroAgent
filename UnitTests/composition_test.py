import numpy as np
import copy
import config
import random
from Simulator.player import Player
from Simulator.pool import pool
from Simulator.champion import champion
from Simulator.player_manager import PlayerManager
from Simulator.tft_config import TFTConfig
from Simulator.step_function import Step_Function
from Simulator.tft_vector_simulator import TFT_Single_Player_Vector_Simulator
from Simulator.game_round import log_to_file, log_to_file_start


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
    log_to_file_start()
    base_pool = pool()
    player_manager = PlayerManager(2, base_pool, TFTConfig)
    step_function = Step_Function(player_manager)
    p1 = Player(base_pool, 0)
    p2 = Player(base_pool, 1)
    p1.gold = 2000
    player_manager.reinit_player_set([p1] + [p2])
    log_to_file_start()
    # Take 1000 actions
    for i in range(2000):
        mask = np.reshape(player_manager.action_handlers['player_0'].fetch_action_mask(), [-1])
        legal_actions = np.array([i for i, x in enumerate(mask) if x == 1.0])
        random_action = np.random.choice(legal_actions, 1)
        step_function.perform_1d_action('player_0', random_action[0])
        # print(f"action {player_manager.action_handlers['player_0'].action_space_to_action(random_action)}")
        if i % 15 == 0:
            p1.end_turn_actions()

    p1.end_turn_actions()
    p1.printComp(to_console=True)
    p1.printBench(log=False)
    p2 = Player(base_pool, 0)
    p2.max_units = 8
    p2.gold = 1000
    for x in range(len(p1.board)):
        for y in range(len(p1.board[0])):
            if p1.board[x][y]:
                p2.buy_champion(copy.deepcopy(p1.board[x][y]))
                p2.move_bench_to_board(0, x, y)

    print(f"This randomly fails p1 {p1.team_composition}, p2 {p2.team_composition}")
    p1.printComp(to_console=True)
    p2.printComp(to_console=True)
    log_to_file(p1)
    for key in p1.team_composition.keys():
        assert p1.team_composition[key] == p2.team_composition[key]

def multi_env_action_test():
    tftConfig = TFTConfig(max_actions_per_round=config.ACTIONS_PER_TURN, num_players=1)
    env = TFT_Single_Player_Vector_Simulator(tftConfig, num_envs=4)

    player_observation, info = env.vector_reset()
    masks = []
    for obs in player_observation:
        masks.append(np.reshape(np.array(obs["action_mask"]), [-1]))

    # Used to know when players die and which agent is currently acting
    terminated = [False for _ in range(env.num_envs)]
    storage_terminated = [False for _ in range(env.num_envs)]

    # While the game is still going on.
    while not all(terminated):
        # Ask our model for an action and policy. Use on normal case or if we only have current versions left
        legal_actions_batch = [[i for i, x in enumerate(mask) if x == 1.0] for mask in masks]
        random_action = []
        for legal_actions in legal_actions_batch:
            random_action.append(np.array(random.choice(legal_actions)))

        for i, terminate in enumerate(terminated):
            if terminate:
                storage_terminated[i] = True

        # Take that action within the environment and return all of our information for the next player
        next_observation, reward, terminated, _, info = env.vector_step(random_action, terminated)

        # Set up the observation for the next action
        masks = []
        for obs in next_observation:
            masks.append(np.reshape(np.array(obs["action_mask"]), [-1]))

    players_states = [env.envs[i].player_manager.player_states['player_0'] for i in range(env.num_envs)]
    boards = []
    comps = []
    tiers = []
    for z in range(env.num_envs):
        comps.append(players_states[z].team_composition)
        tiers.append(players_states[z].team_tiers)
        boards.append(players_states[z].board)

    print(comps)
    print(tiers)
    board_same = True
    for x in range(7):
        for y in range(4):
            test_unit = boards[0][x][y]
            for z in range(1, env.num_envs):
                if test_unit != boards[z][x][y]:
                    board_same = False
                if not board_same:
                    break
            if not board_same:
                break
        if not board_same:
            break
    assert not board_same

    comp_same = True
    tiers_same = True
    for key in comps[0].keys():
        comp_trait = comps[0][key]
        tier_trait = tiers[0][key]
        for z in range(1, env.num_envs):
            if comps[z][key] != comp_trait:
                comp_same = False
            if tiers[z][key] != tier_trait:
                tiers_same = False
            if not comp_same or not tiers_same:
                break
        if not comp_same or not tiers_same:
            break
    assert not comp_same or not tiers_same


def test_list():
    # premade_warlord_test()
    one_thousand_action_test()
    multi_env_action_test()

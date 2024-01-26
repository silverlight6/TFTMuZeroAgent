import Simulator.config as config
import numpy as np
import Simulator.champion as champion
from Simulator.stats import COST
import torch
import math

"""
Description - Object used for the simulation to interact with the environment. The agent passes in actions and those 
              actions take effect in this object.
Inputs      - pool_obj: Object pointer to the pool
                Pool object pointer used for refreshing shops and generating shop vectors.
              observation_obj: Object pointer to a Game_observation object
                Used for generating shop_vector and other player vectors on pass options.
"""


class Step_Function:
    def __init__(self, pool_obj, observation_objs):
        self.pool_obj = pool_obj
        self.shops = {"player_" + str(player_id): self.pool_obj.sample(None, 5) for player_id in
                      range(config.NUM_PLAYERS)}
        self.observation_objs = observation_objs

    """
    Description - Method used for generating a new shop for a given player
    Inputs      - player: Player object
                    Any alive player.
    """
    def generate_shop(self, key, player):
        self.shops[key] = self.pool_obj.sample(player, 5)
        self.observation_objs[key].generate_shop_vector(self.shops[key], player)


    """
    Description - Method used for generating a new shop for all players
    Inputs      - players: Dictionary of player objects
                    All of the players in the game. Currently both alive or dead. Used at the start of turn.
    """
    def generate_shops(self, players):
        for player_id, player in players.items():
            if player:
                self.shops[player_id] = self.pool_obj.sample(player, 5)
        self.generate_shop_vectors(players)

    """
    Description - Method used for generating a new shop vector for the observation for all players
    Inputs      - players: Dictionary of player objects
                    All of the players in the game. Currently both alive or dead.
    """
    def generate_shop_vectors(self, players):
        for player_id, player in players.items():
            if player:
                self.observation_objs[player_id].generate_shop_vector(self.shops[player_id], player)

    """
    Description - Calculates the 2 dimensional position in the board, from the 1 dimensional position on the list
    Inputs      - dcord: Int
                    For example, 27 -> 6 for x and 3 for y
    Outputs     - x: Int
                    x_coord
                  y: Int
                    y_coord
    """
    def dcord_to_2dcord(self, dcord):
        x = dcord % 7
        y = (dcord - x) // 7
        return x, y

    """
    Description - Method for taking an action in the environment when using a 2d action type. 
    Inputs      - action: List
                    Action in the form of 55d array. First 5 for decision. Next 28 for board. Next 9 for bench.
                    Position 43 for selling a champion, rest for item movements
                  player: Player Object
                    player whose turn it currently is. The None check is more of a safety. It should never be None.
                  players: Dictionary of Players
                    A dictionary containing all of the players, used when updating observation with other player info.
                  key: String
                    The key associated with the player. for example, "player_0"
                  game_observations: Dictionary of Game_Observations
                    Used for updating the observation after pass and shop actions.
    """
    def batch_2d_controller(self, action, player, players, key, game_observations):
        # single_step_action_controller took 0.0009961128234863281 seconds to finish
        if player:
            # action format = 0:7 (action_selector),
            # 7:44 (champ_loc_target), 44:54 (item_loc_target)
            action_selector = action[0]
            game_observations[key].generate_other_player_vectors(player, players)
            if action_selector == 0:
                # Pass
                player.print(f"pass action")
            elif action_selector == 1:
                # Roll to closest 10 multiple
                units = np.where(player.unit_directive > 0.5)[0]
                # cost_f = lambda t: COST[list(COST.items())[t][0]]
                # costs = ([cost_f(n) for n in units])
                mask = np.in1d(player.shop_elems, units)
                if not mask.any():
                    return
                min_arg = np.where(player.shop_costs > 0, np.where(mask, player.shop_costs, np.inf), np.inf).argmin()
                min_cost = player.shop_costs[min_arg]
                closest_ten = round_down(player.gold)
                while ((player.gold - min_cost > closest_ten and mask.any()) or player.gold - min_cost - 2 > closest_ten) and not player.bench_full():
                    if mask.any():
                        self.batch_shop(min_arg, player, game_observations[key], key)
                    elif player.gold - min_cost - 2 > closest_ten:
                        if player.refresh():
                            self.generate_shop(key, player)
                    mask = np.in1d(player.shop_elems, units)
                    if not mask.any():
                        continue
                    min_arg = np.where(player.shop_costs > 0, np.where(mask, player.shop_costs, np.inf), np.inf).argmin()
                    min_cost = player.shop_costs[min_arg]

                # champ_shop_target = action[1]
                # self.batch_shop(champ_shop_target, player, game_observations[key], key)
            # elif action_selector == 6:
            #     # Place item on champ
            #     item_selector = np.argmax(action[44:54])
            #     move_loc = np.argmax(action[7:43])
            #     if move_loc >= 28:
            #         move_loc -= 28
            #         player.move_item_to_bench(item_selector, move_loc)
            #     else:
            #         x, y = self.dcord_to_2dcord(move_loc)
            #         player.move_item_to_board(item_selector, x, y)
            elif action_selector == 2:
                # Level up
                current_level = player.level
                while player.level == current_level and player.level < 9 and player.decision_mask[4]:
                    player.buy_exp()
            elif action_selector == 3:
                # Sell Champions not in directive
                # TODO not take into account sanguard or dummy
                units = np.where(player.unit_directive > 0.5)[0]
                for x in player.board:
                    for champ in x:
                        if champ:
                            if list(COST.keys()).index(champ.name)-1 not in units and champ.name != "sandguard":
                                player.sell_champion(champ, field = True)
                for i in range(len(player.bench)):
                    if player.bench[i]:
                        if list(COST.keys()).index(player.bench[i].name)-1 not in units:
                            player.sell_from_bench(i)
                # target_1 = action[1]
                # if target_1 < 28:
                #     x, y = self.dcord_to_2dcord(target_1)
                #     if player.board[x][y]:
                #         player.sell_champion(player.board[x][y], field=True)
                # else:
                #     player.sell_from_bench(target_1 - 28)

    '''
    Description - Method used for buying a shop. Turns the string in the shop into a champion object to send to the 
                  player class. Also updates the observation.
    '''
    def batch_shop(self, shop_action, player, game_observation, player_id):
        if shop_action > 5:
            shop_action = int(np.floor(np.random.rand(1, 1) * 6))

        # name = list(COST.items())[shop_action+1][0]
        # champ_index = np.where(player.shop_elems == shop_action)
        # print(player_id, player.player_num, " Champ Index ", player.shop_elems[shop_action], shop_action," in shop ", self.shops[player_id], " elems ", player.shop_elems)

        # if len(champ_index[0]) == 0:
        #     player.reward += player.mistake_reward
        #     print("Champ not found, bug in mask")
        #     return
        # champ_index = champ_index[0][0]
        if self.shops[player_id][shop_action].endswith("_c"):
            c_shop = self.shops[player_id][shop_action].split('_')
            a_champion = champion.champion(c_shop[0], chosen=c_shop[1], itemlist=[])
        else:
            a_champion = champion.champion(self.shops[player_id][shop_action])
        success = player.buy_champion(a_champion)
        if success:
            self.shops[player_id][shop_action] = " "
            game_observation.generate_shop_vector(self.shops[player_id], player)
        else:
            # I get that this does nothing, but it tells whoever writes in this method next that there should be
            # Nothing that follows this line.
            return

def round_down(x):
    return math.floor(x / 10.0) * 10
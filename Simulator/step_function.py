import Simulator.config as config
import numpy as np
import Simulator.champion as champion
from Simulator.stats import COST

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
            # game_observations[key].generate_other_player_vectors(player, players)
            if action_selector == 0:
                player.print(f"pass action")
            elif action_selector == 1:
                # Swap champ place
                target_1 = action[1]
                # action[target_1 + 7] = 0
                target_2 = action[2]
                swap_loc_from = min(target_1, target_2)
                swap_loc_to = max(target_1, target_2)
                # Swap from swap_loc_from to swap_loc_to
                if swap_loc_from < 28:
                    if swap_loc_to < 28:
                        x1, y1 = self.dcord_to_2dcord(swap_loc_from)
                        x2, y2 = self.dcord_to_2dcord(swap_loc_to)
                        if player.board[x1][y1]:
                            player.move_board_to_board(x1, y1, x2, y2)
                        elif player.board[x2][y2]:
                            player.move_board_to_board(x2, y2, x1, y1)
                    else:
                        x1, y1 = self.dcord_to_2dcord(swap_loc_from)
                        bench_loc = swap_loc_to - 28
                        if player.bench[bench_loc]:
                            player.move_bench_to_board(bench_loc, x1, y1)
                        else:
                            player.move_board_to_bench(x1, y1)
            elif action_selector == 2:
                # Buy from shop
                champ_shop_target = action[1]
                self.batch_shop(champ_shop_target, player, game_observations[key], key)
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
            elif action_selector == 3:
                # Sell Champ
                target_1 = action[1]
                if target_1 < 28:
                    x, y = self.dcord_to_2dcord(target_1)
                    if player.board[x][y]:
                        player.sell_champion(player.board[x][y], field=True)
                else:
                    player.sell_from_bench(target_1 - 28)
            elif action_selector == 4:
                # Refresh shop
                if player.refresh():
                    self.generate_shop(key, player)
            elif action_selector == 5:
                # Buy EXP
                player.buy_exp()

    '''
    Description - Method used for buying a shop. Turns the string in the shop into a champion object to send to the 
                  player class. Also updates the observation.
    '''
    def batch_shop(self, shop_action, player, game_observation, player_id):
        if shop_action > 58:
            shop_action = int(np.floor(np.random.rand(1, 1) * 59))

        # name = list(COST.items())[shop_action+1][0]
        champ_index = np.where(player.shop_elems == shop_action)
        # print(player_id, player.player_num, " Champ Index ", champ_index, shop_action," in shop ", self.shops[player_id], " elems ", player.shop_elems)

        if len(champ_index[0]) == 0:
            player.reward += player.mistake_reward
            print("Champ not found, bug in mask")
            return
        champ_index = champ_index[0][0]
        if self.shops[player_id][champ_index].endswith("_c"):
            c_shop = self.shops[player_id][champ_index].split('_')
            a_champion = champion.champion(c_shop[0], chosen=c_shop[1], itemlist=[])
        else:
            a_champion = champion.champion(self.shops[player_id][champ_index])
        success = player.buy_champion(a_champion)
        if success:
            self.shops[player_id][champ_index] = " "
            game_observation.generate_shop_vector(self.shops[player_id], player)
        else:
            # I get that this does nothing, but it tells whoever writes in this method next that there should be
            # Nothing that follows this line.
            return

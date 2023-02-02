import config
import numpy as np
import Simulator.champion as champion


class Step_Function:
    def __init__(self, pool_obj, observation_objs):
        self.pool_obj = pool_obj
        self.shops = {player_id: self.pool_obj.sample(None, 5) for player_id in
                      range(config.NUM_PLAYERS)}
        self.observation_objs = observation_objs

    def generate_shop(self, player):
        self.shops[player.player_num] = self.pool_obj.sample(None, 5)

    def generate_shops(self, players):
        for player in players.values():
            if player:
                self.shops[player.player_num] = self.pool_obj.sample(player, 5)

    def batch_shop(self, shop_action, player, game_observation):
        if shop_action > 4:
            shop_action = int(np.floor(np.random.rand(1, 1) * 5))

        if self.shops[player.player_num][shop_action] == " ":
            player.reward += player.mistake_reward
            return
        if self.shops[player.player_num][shop_action].endswith("_c"):
            c_shop = self.shops[player.player_num][shop_action].split('_')
            a_champion = champion.champion(c_shop[0], chosen=c_shop[1], itemlist=[])
        else:
            a_champion = champion.champion(self.shops[player.player_num][shop_action])
        success = player.buy_champion(a_champion)
        if success:
            self.shops[player.player_num][shop_action] = " "
        else:
            return

    def dcord_to_2dcord(self, dcord):
        # Calculates the 2 dimensional position in the board, from the 1 dimensional position on the list
        x = dcord % 7
        y = (dcord - x ) // 7
        return x,y

    def single_step_action_controller(self, action, player, players, key, game_observations):
        # single_step_action_controller took 0.0009961128234863281 seconds to finish
        if player:
            # action format = 0:6 (action_selector), 6:43 (champ_loc_target), [43] sell "location", 44:54 (item_loc_target)
            # TODO(lobotuerk) Get rid of magic numbers like 36 (sell target wrt target vector) and 27 (board / bench division wrt target vector)
            action_selector = np.argmax(action[0:6])
            if action_selector == 0:
                # Pass action
                pass
            elif action_selector == 1:
                # Buy from shop
                champ_shop_target = np.argmax(action[6:11])
                self.batch_shop(champ_shop_target, player, game_observations[key])
            elif action_selector == 2:
                # Swap champ place
                target_1 = np.argmax(action[6:43])
                action[target_1 + 6] = 0
                target_2 = np.argmax(action[6:44])
                swap_loc_from = min(target_1, target_2)
                swap_loc_to = max(target_1, target_2)
                if swap_loc_to == 37:
                    # Sell Champ
                    if swap_loc_from < 28:
                        x,y = self.dcord_to_2dcord(swap_loc_from)
                        if player.board[x][y]:
                            player.sell_champion(player.board[x][y], field=True)
                    else:
                        player.sell_from_bench(swap_loc_from - 28)
                else:
                    # Swap from swap_loc_from to swap_loc_to
                    if swap_loc_from < 28:
                        if swap_loc_to < 28:
                            x1,y1 = self.dcord_to_2dcord(swap_loc_from)
                            x2,y2 = self.dcord_to_2dcord(swap_loc_to)
                            if player.board[x1][y1]:
                                player.move_board_to_board(x1, y1, x2, y2)
                            elif player.board[x2][y2]:
                                player.move_board_to_board(x2, y2, x1, y1)
                        else:
                            x1,y1 = self.dcord_to_2dcord(swap_loc_from)
                            bench_loc = swap_loc_to - 28
                            if player.bench[bench_loc]:
                                player.move_bench_to_board(bench_loc, x1, y1)
                            else:
                                player.move_board_to_bench(x1, y1)
            elif action_selector == 3:
                # Place item on champ
                item_selector = np.argmax(action[44:54])
                move_loc = np.argmax(action[6:43])
                if move_loc >= 28:
                    move_loc -= 28
                    player.move_item_to_bench(item_selector, move_loc)
                else:
                    x,y = self.dcord_to_2dcord(move_loc)
                    player.move_item_to_board(item_selector, x, y)
            elif action_selector == 4:
                # Buy EXP
                player.buy_exp()
            elif action_selector == 5:
                # Refresh shop
                if player.refresh():
                    self.shops[player.player_num] = self.pool_obj.sample(player, 5)

            
            observations = game_observations[key].get_lobo_observation(player, self.shops[player.player_num], players)
            return player.reward, observations
        return 0

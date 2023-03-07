import config
import numpy as np
import Simulator.champion as champion

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
            # action format = 0:6 (action_selector),
            # 6:43 (champ_loc_target), [43] sell "location", 44:54 (item_loc_target)
            action_selector = np.argmax(action[0:6])
            if action_selector == 0:
                game_observations[key].generate_game_comps_vector()
                game_observations[key].generate_other_player_vectors(player, players)
                player.print(f"pass action")
            elif action_selector == 1:
                # Buy from shop
                champ_shop_target = np.argmax(action[6:11])
                self.batch_shop(champ_shop_target, player, game_observations[key], key)
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
                        x, y = self.dcord_to_2dcord(swap_loc_from)
                        if player.board[x][y]:
                            player.sell_champion(player.board[x][y], field=True)
                    else:
                        player.sell_from_bench(swap_loc_from - 28)
                else:
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
            elif action_selector == 3:
                # Place item on champ
                item_selector = np.argmax(action[44:54])
                move_loc = np.argmax(action[6:43])
                if move_loc >= 28:
                    move_loc -= 28
                    player.move_item_to_bench(item_selector, move_loc)
                else:
                    x, y = self.dcord_to_2dcord(move_loc)
                    player.move_item_to_board(item_selector, x, y)
            elif action_selector == 4:
                # Buy EXP
                player.buy_exp()
            elif action_selector == 5:
                # Refresh shop
                if player.refresh():
                    self.shops[player.player_num] = self.pool_obj.sample(player, 5)

    # Leaving this method here to assist in setting up a human interface. Is not used in the environment
    # The return is the shop, boolean for end of turn, boolean for successful action, number of actions taken
    # TODO: Update if we're using this for UI Interface
    def multi_step(self, action, player, game_observation, agent, buffer, players):
        if action == 0:
            action_vector = np.array([0, 1, 0, 0, 0, 0, 0, 0])
            observation = game_observation.observation(player, buffer, action_vector)
            shop_action, policy = agent.policy(observation, player.player_num)

            if shop_action > 4:
                shop_action = int(np.floor(np.random.rand(1, 1) * 5))

            buffer.store_replay_buffer(observation, shop_action, 0, policy)

            if shop_action == 0:
                if self.shops[0] == " ":
                    player.reward += player.mistake_reward
                    return self.shops, False, False, 2
                if self.shops[0].endswith("_c"):
                    c_shop = self.shops[0].split('_')
                    a_champion = champion.champion(c_shop[0], chosen=c_shop[1], itemlist=[])
                else:
                    a_champion = champion.champion(self.shops[0])
                success = player.buy_champion(a_champion)
                if success:
                    self.shops[0] = " "
                    game_observation.generate_shop_vector(self.shops, player)
                else:
                    return self.shops, False, False, 2

            elif shop_action == 1:
                if self.shops[1] == " ":
                    player.reward += player.mistake_reward
                    return self.shops, False, False, 2
                if self.shops[1].endswith("_c"):
                    c_shop = self.shops[1].split('_')
                    a_champion = champion.champion(c_shop[0], chosen=c_shop[1], itemlist=[])
                else:
                    a_champion = champion.champion(self.shops[1])
                success = player.buy_champion(a_champion)
                if success:
                    self.shops[1] = " "
                    game_observation.generate_shop_vector(self.shops, player)
                else:
                    return self.shops, False, False, 2

            elif shop_action == 2:
                if self.shops[2] == " ":
                    player.reward += player.mistake_reward
                    return self.shops, False, False, 2
                if self.shops[2].endswith("_c"):
                    c_shop = self.shops[2].split('_')
                    a_champion = champion.champion(c_shop[0], chosen=c_shop[1], itemlist=[])
                else:
                    a_champion = champion.champion(self.shops[2])
                success = player.buy_champion(a_champion)
                if success:
                    self.shops[2] = " "
                    game_observation.generate_shop_vector(self.shops, player)
                else:
                    return self.shops, False, False, 2

            elif shop_action == 3:
                if self.shops[3] == " ":
                    player.reward += player.mistake_reward
                    return self.shops, False, False, 2
                if self.shops[3].endswith("_c"):
                    c_shop = self.shops[3].split('_')
                    a_champion = champion.champion(c_shop[0], chosen=c_shop[1], itemlist=[])
                else:
                    a_champion = champion.champion(self.shops[3])

                success = player.buy_champion(a_champion)
                if success:
                    self.shops[3] = " "
                    game_observation.generate_shop_vector(self.shops, player)
                else:
                    return self.shops, False, False, 2

            elif shop_action == 4:
                if self.shops[4] == " ":
                    player.reward += player.mistake_reward
                    return self.shops, False, False, 2
                if self.shops[4].endswith("_c"):
                    c_shop = self.shops[4].split('_')
                    a_champion = champion.champion(c_shop[0], chosen=c_shop[1], itemlist=[])
                else:
                    a_champion = champion.champion(self.shops[4])

                success = player.buy_champion(a_champion)
                if success:
                    self.shops[4] = " "
                    game_observation.generate_shop_vector(self.shops, player)
                else:
                    return self.shops, False, False, 2

        # Refresh
        elif action == 1:
            if player.refresh():
                self.shops = self.pool_obj.sample(player, 5)
                game_observation.generate_shop_vector(self.shops, player)
            else:
                return self.shops, False, False, 1

        # buy Exp
        elif action == 2:
            if player.buy_exp():
                pass
            else:
                return self.shops, False, False, 1

        # move Item
        elif action == 3:
            action_vector = np.array([0, 0, 0, 1, 0, 0, 0, 0])
            observation, _ = game_observation.observation(player, buffer, action_vector)
            item_action, policy = agent.policy(observation, player.player_num)

            # Ensure that the action is a legal action
            if item_action > 9:
                item_action = int(np.floor(np.random.rand(1, 1) * 10))

            buffer.store_replay_buffer(observation, item_action, 0, policy)

            action_vector = np.array([0, 0, 0, 0, 1, 0, 0, 0])
            observation, _ = game_observation.observation(player, buffer, action_vector)
            x_action, policy = agent.policy(observation, player.player_num)

            if x_action > 6:
                x_action = int(np.floor(np.random.rand(1, 1) * 7))

            buffer.store_replay_buffer(observation, x_action, 0, policy)

            action_vector = np.array([0, 0, 0, 0, 0, 1, 0, 0])
            observation, _ = game_observation.observation(player, buffer, action_vector)
            y_action, policy = agent.policy(observation, player.player_num)

            if y_action > 3:
                y_action = int(np.floor(np.random.rand(1, 1) * 4))

            buffer.store_replay_buffer(observation, y_action, 0, policy)

            # Call network to activate the move_item_agent
            if not player.move_item_to_board(item_action, x_action, y_action):
                return self.shops, False, False, 4
            else:
                return self.shops, False, True, 4

        # sell Unit
        elif action == 4:
            action_vector = np.array([0, 0, 1, 0, 0, 0, 0, 0])
            observation, _ = game_observation.observation(player, buffer, action_vector)
            bench_action, policy = agent.policy(observation, player.player_num)

            if bench_action > 8:
                bench_action = int(np.floor(np.random.rand(1, 1) * 9))

            buffer.store_replay_buffer(observation, bench_action, 0, policy)

            # Ensure that the action is a legal action
            if bench_action > 8:
                bench_action = int(np.floor(np.random.rand(1, 1) * 10))

            # Call network to activate the bench_agent
            if not player.sell_from_bench(bench_action):
                return self.shops, False, False, 2
            else:
                return self.shops, False, True, 2

        # move bench to Board
        elif action == 5:

            action_vector = np.array([0, 0, 1, 0, 0, 0, 0, 0])
            observation, _ = game_observation.observation(player, buffer, action_vector)
            bench_action, policy = agent.policy(observation, player.player_num)

            # Ensure that the action is a legal action
            if bench_action > 8:
                bench_action = int(np.floor(np.random.rand(1, 1) * 9))

            buffer.store_replay_buffer(observation, bench_action, 0, policy)

            action_vector = np.array([0, 0, 0, 0, 1, 0, 0, 0])
            observation, _ = game_observation.observation(player, buffer, action_vector)
            x_action, policy = agent.policy(observation, player.player_num)

            if x_action > 6:
                x_action = int(np.floor(np.random.rand(1, 1) * 7))

            buffer.store_replay_buffer(observation, x_action, 0, policy)

            action_vector = np.array([0, 0, 0, 0, 0, 1, 0, 0])
            observation, _ = game_observation.observation(player, buffer, action_vector)
            y_action, policy = agent.policy(observation, player.player_num)

            if y_action > 3:
                y_action = int(np.floor(np.random.rand(1, 1) * 4))

            buffer.store_replay_buffer(observation, y_action, 0, policy)

            # Call network to activate the bench and board agents
            if not player.move_bench_to_board(bench_action, x_action, y_action):
                return self.shops, False, False, 4
            else:
                return self.shops, False, True, 4

        # move board to bench
        elif action == 6:
            action_vector = np.array([0, 0, 0, 0, 1, 0, 0, 0])
            observation, _ = game_observation.observation(player, buffer, action_vector)
            x_action, policy = agent.policy(observation, player.player_num)

            if x_action > 6:
                x_action = int(np.floor(np.random.rand(1, 1) * 7))

            buffer.store_replay_buffer(observation, x_action, 0, policy)

            action_vector = np.array([0, 0, 0, 0, 0, 1, 0, 0])
            observation, _ = game_observation.observation(player, buffer, action_vector)
            y_action, policy = agent.policy(observation, player.player_num)

            if y_action > 3:
                y_action = int(np.floor(np.random.rand(1, 1) * 4))

            buffer.store_replay_buffer(observation, y_action, 0, policy)

            # Call network to activate the bench and board agents
            if not player.move_board_to_bench(x_action, y_action):
                return self.shops, False, False, 3
            else:
                return self.shops, False, True, 3

        # Move board to board
        elif action == 7:
            action_vector = np.array([0, 0, 0, 0, 1, 0, 0, 0])
            observation, _ = game_observation.observation(player, buffer, action_vector)
            x_action, policy = agent.policy(observation, player.player_num)

            if x_action > 6:
                x_action = int(np.floor(np.random.rand(1, 1) * 7))

            buffer.store_replay_buffer(observation, x_action, 0, policy)

            action_vector = np.array([0, 0, 0, 0, 0, 1, 0, 0])
            observation, _ = game_observation.observation(player, buffer, action_vector)
            y_action, policy = agent.policy(observation, player.player_num)

            if y_action > 3:
                y_action = int(np.floor(np.random.rand(1, 1) * 4))

            buffer.store_replay_buffer(observation, y_action, 0, policy)

            action_vector = np.array([0, 0, 0, 0, 1, 0, 0, 0])
            observation, _ = game_observation.observation(player, buffer, action_vector)
            x2_action, policy = agent.policy(observation, player.player_num)

            if x2_action > 6:
                x2_action = int(np.floor(np.random.rand(1, 1) * 7))

            buffer.store_replay_buffer(observation, x2_action, 0, policy)

            action_vector = np.array([0, 0, 0, 0, 0, 1, 0, 0])
            observation, _ = game_observation.observation(player, buffer, action_vector)
            y2_action, policy = agent.policy(observation, player.player_num)

            if y2_action > 3:
                y2_action = int(np.floor(np.random.rand(1, 1) * 4))

            buffer.store_replay_buffer(observation, y2_action, 0, policy)

            # Call network to activate the bench and board agents
            if not player.move_board_to_board(x_action, y_action, x2_action, y2_action):
                return self.shops, False, False, 5
            else:
                return self.shops, False, True, 5

        # Update all information in the observation relating to the other players.
        # Later in training, turn this number up to 7 due to how long it takes a normal player to execute
        elif action == 8:
            game_observation.generate_game_comps_vector()
            game_observation.generate_other_player_vectors(player, players)
            return self.shops, False, True, 1

        # end turn
        elif action == 9:
            # Testing a version that does not end the turn on this action.
            return self.shops, False, True, 1
            # return self.shops, True, True, 1

        # Possible to add another action here which is basically pass the action back.
        # Wait and do nothing. If anyone thinks that is beneficial, let me know.
        else:
            return self.shops, False, False, 1
        return self.shops, False, True, 1

    def action_controller(self, action, player, players, key, game_observations):
        if player:
            # Python doesn't allow comparisons between arrays,
            # so we're just checking if the nth value is 1 (true) or 0 (false)
            if player.action_vector[0]:
                self.batch_multi_step(action, player, players, game_observations[key], key)
            if player.action_vector[1]:
                self.batch_shop(action, player, game_observations[key], key)
                player.action_vector = np.array([1, 0, 0, 0, 0, 0, 0, 0])
            # Move item to board
            if player.current_action == 3:
                player.action_values.append(action)
                if player.action_vector[3]:
                    player.action_vector = np.array([0, 0, 0, 0, 1, 0, 0, 0])
                elif player.action_vector[4]:
                    player.action_vector = np.array([0, 0, 0, 0, 0, 1, 0, 0])
                else:
                    player.action_vector = np.array([1, 0, 0, 0, 0, 0, 0, 0])
                    if player.action_values[0] > 9:
                        player.action_values[0] = int(np.floor(np.random.rand(1, 1) * 10))
                    if player.action_values[1] > 6:
                        player.action_values[1] = int(np.floor(np.random.rand(1, 1) * 7))
                    if player.action_values[2] > 3:
                        player.action_values[2] = int(np.floor(np.random.rand(1, 1) * 4))
                    player.move_item_to_board(player.action_values[0], player.action_values[1],
                                              player.action_values[2])
                    player.action_values = []

            # Part 2 of selling unit from bench
            if player.current_action == 4:
                if action > 8:
                    action = int(np.floor(np.random.rand(1, 1) * 10))
                player.action_vector = np.array([1, 0, 0, 0, 0, 0, 0, 0])
                player.sell_from_bench(action)
            # Part 2 to 4 of moving bench to board
            if player.current_action == 5:
                player.action_values.append(action)
                if player.action_vector[2]:
                    player.action_vector = np.array([0, 0, 0, 0, 1, 0, 0, 0])
                elif player.action_vector[4]:
                    player.action_vector = np.array([0, 0, 0, 0, 0, 1, 0, 0])
                else:
                    player.action_vector = np.array([1, 0, 0, 0, 0, 0, 0, 0])
                    if player.action_values[0] > 8:
                        player.action_values[0] = int(np.floor(np.random.rand(1, 1) * 9))
                    if player.action_values[1] > 6:
                        player.action_values[1] = int(np.floor(np.random.rand(1, 1) * 7))
                    if player.action_values[2] > 3:
                        player.action_values[2] = int(np.floor(np.random.rand(1, 1) * 4))
                    player.move_bench_to_board(player.action_values[0], player.action_values[1],
                                               player.action_values[2])
                    player.action_values = []
            # Part 2 to 3 of moving board to bench
            if player.current_action == 6:
                player.action_values.append(action)
                if player.action_vector[4]:
                    player.action_vector = np.array([0, 0, 0, 0, 0, 1, 0, 0])
                else:
                    player.action_vector = np.array([1, 0, 0, 0, 0, 0, 0, 0])
                    if player.action_values[0] > 6:
                        player.action_values[0] = int(np.floor(np.random.rand(1, 1) * 7))
                    if player.action_values[1] > 3:
                        player.action_values[1] = int(np.floor(np.random.rand(1, 1) * 4))
                    player.move_board_to_bench(player.action_values[0], player.action_values[1])
                    player.action_values = []
            # Part 2 to 5 of moving board to board
            if player.current_action == 7:
                player.action_values.append(action)
                if player.action_vector[4]:
                    player.action_vector = np.array([0, 0, 0, 0, 0, 1, 0, 0])
                elif player.action_vector[5]:
                    player.action_vector = np.array([0, 0, 0, 0, 0, 0, 1, 0])
                elif player.action_vector[6]:
                    player.action_vector = np.array([0, 0, 0, 0, 0, 0, 0, 1])
                else:
                    player.action_vector = np.array([1, 0, 0, 0, 0, 0, 0, 0])
                    if player.action_values[0] > 6:
                        player.action_values[0] = int(np.floor(np.random.rand(1, 1) * 7))
                    if player.action_values[1] > 3:
                        player.action_values[1] = int(np.floor(np.random.rand(1, 1) * 4))
                    if player.action_values[2] > 6:
                        player.action_values[2] = int(np.floor(np.random.rand(1, 1) * 7))
                    if player.action_values[3] > 3:
                        player.action_values[3] = int(np.floor(np.random.rand(1, 1) * 4))
                    player.move_board_to_board(player.action_values[0], player.action_values[1],
                                               player.action_values[2], player.action_values[3])
                    player.action_values = []
            return player.reward
            # Some function that evens out rewards to all other players
        return 0

    def batch_multi_step(self, action, player, players, game_observation, player_id):
        player.current_action = action
        if action == 0:
            player.action_vector = np.array([0, 1, 0, 0, 0, 0, 0, 0])

        # action vector already == np.array([1, 0, 0, 0, 0, 0, 0, 0]) by this point
        elif action == 1:
            if player.refresh():
                self.shops[player_id] = self.pool_obj.sample(player, 5)
                game_observation.generate_shop_vector(self.shops[player_id], player)

        elif action == 2:
            player.buy_exp()

        elif action == 3:
            player.action_vector = np.array([0, 0, 0, 1, 0, 0, 0, 0])

        elif action == 4:
            player.action_vector = np.array([0, 0, 1, 0, 0, 0, 0, 0])

        elif action == 5:
            player.action_vector = np.array([0, 0, 1, 0, 0, 0, 0, 0])

        elif action == 6:
            player.action_vector = np.array([0, 0, 0, 0, 1, 0, 0, 0])

        elif action == 7:
            player.action_vector = np.array([0, 0, 0, 0, 1, 0, 0, 0])

        elif action == 8:
            game_observation.generate_game_comps_vector()
            game_observation.generate_other_player_vectors(player, players)

        elif action == 9:
            # This would normally be end turn but figure it out later
            pass

    '''
    Description - Method used for buying a shop. Turns the string in the shop into a champion object to send to the 
                  player class. Also updates the observation.
    '''
    def batch_shop(self, shop_action, player, game_observation, player_id):
        if shop_action > 4:
            shop_action = int(np.floor(np.random.rand(1, 1) * 5))

        if self.shops[player_id][shop_action] == " ":
            player.reward += player.mistake_reward
            return
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

import config
import numpy as np
import Simulator.champion as champion


class Step_Function:
    def __init__(self, pool_obj, observation_objs):
        self.pool_obj = pool_obj
        self.shops = [self.pool_obj.sample(None, 5) for _ in range(config.NUM_PLAYERS)]
        self.observation_objs = observation_objs

    def generate_shop(self, player):
        self.shops[player.player_num] = self.pool_obj.sample(None, 5)

    def generate_shops(self, players):
        for player in players:
            if player:
                self.shops[player.player_num] = self.pool_obj.sample(player, 5)

    def generate_shop_vectors(self, players):
        for player in players:
            if player:
                self.observation_objs[player.player_num].generate_shop_vector(self.shops[player.player_num])

    # The return is the shop, boolean for end of turn, boolean for successful action
    def step_5d(self, action, player):
        if action[0] == 0:
            if self.shops[0] == " ":
                player.reward += player.mistake_reward
                return self.shops, False, False
            if self.shops[0].endswith("_c"):
                c_shop = self.shops[0].split('_')
                a_champion = champion.champion(c_shop[0], chosen=c_shop[1], itemlist=[])
            else:
                a_champion = champion.champion(self.shops[0])
            success = player.buy_champion(a_champion)
            if success:
                self.shops[0] = " "
            else:
                return self.shops, False, False

        elif action[0] == 1:
            if self.shops[1] == " ":
                player.reward += player.mistake_reward
                return self.shops, False, False
            if self.shops[1].endswith("_c"):
                c_shop = self.shops[1].split('_')
                a_champion = champion.champion(c_shop[0], chosen=c_shop[1], itemlist=[])
            else:
                a_champion = champion.champion(self.shops[1])
            success = player.buy_champion(a_champion)
            if success:
                self.shops[1] = " "
            else:
                return self.shops, False, False

        elif action[0] == 2:
            if self.shops[2] == " ":
                player.reward += player.mistake_reward
                return self.shops, False, False
            if self.shops[2].endswith("_c"):
                c_shop = self.shops[2].split('_')
                a_champion = champion.champion(c_shop[0], chosen=c_shop[1], itemlist=[])
            else:
                a_champion = champion.champion(self.shops[2])
            success = player.buy_champion(a_champion)
            if success:
                self.shops[2] = " "
            else:
                return self.shops, False, False

        elif action[0] == 3:
            if self.shops[3] == " ":
                player.reward += player.mistake_reward
                return self.shops, False, False
            if self.shops[3].endswith("_c"):
                c_shop = self.shops[3].split('_')
                a_champion = champion.champion(c_shop[0], chosen=c_shop[1], itemlist=[])
            else:
                a_champion = champion.champion(self.shops[3])

            success = player.buy_champion(a_champion)
            if success:
                self.shops[3] = " "
            else:
                return self.shops, False, False

        elif action[0] == 4:
            if self.shops[4] == " ":
                player.reward += player.mistake_reward
                return self.shops, False, False
            if self.shops[4].endswith("_c"):
                c_shop = self.shops[4].split('_')
                a_champion = champion.champion(c_shop[0], chosen=c_shop[1], itemlist=[])
            else:
                a_champion = champion.champion(self.shops[4])

            success = player.buy_champion(a_champion)
            if success:
                self.shops[4] = " "
            else:
                return self.shops, False, False

        # Refresh
        elif action[0] == 5:
            if player.refresh():
                self.shops = self.pool_obj.sample(player, 5)
            else:
                return self.shops, False, False

        # buy Exp
        elif action[0] == 6:
            if player.buy_exp():
                pass
            else:
                return self.shops, False, False

        # end turn
        elif action[0] == 7:
            return self.shops, True, True

        # move Item
        elif action[0] == 8:
            # Call network to activate the move_item_agent
            if not player.move_item_to_board(action[1], action[3], action[4]):
                return self.shops, False, False

        # sell Unit
        elif action[0] == 9:
            # Call network to activate the bench_agent
            if not player.sell_from_bench(action[2]):
                return self.shops, False, False

        # move bench to Board
        elif action[0] == 10:
            # Call network to activate the bench and board agents
            if not player.move_bench_to_board(action[2], action[3], action[4]):
                return self.shops, False, False

        # move board to bench
        elif action[0] == 11:
            # Call network to activate the bench and board agents
            if not player.move_board_to_bench(action[3], action[4]):
                return self.shops, False, False

        else:
            return self.shops, False, False
        return self.shops, False, True

    # The return is the shop, boolean for end of turn, boolean for successful action, number of actions taken
    def multi_step(self, action, player, game_observation, agent, buffer):
        if action == 0:
            action_vector = np.array([0, 1, 0, 0, 0, 0, 0, 0])
            observation, _ = game_observation.observation(player, buffer, action_vector)
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
                    game_observation.generate_shop_vector(self.shops)
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
                    game_observation.generate_shop_vector(self.shops)
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
                    game_observation.generate_shop_vector(self.shops)
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
                    game_observation.generate_shop_vector(self.shops)
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
                    game_observation.generate_shop_vector(self.shops)
                else:
                    return self.shops, False, False, 2

        # Refresh
        elif action == 1:
            if player.refresh():
                self.shops = self.pool_obj.sample(player, 5)
                game_observation.generate_shop_vector(self.shops)
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

    def batch_controller(self, action, players, game_observations):
        rewards = []
        for i in range(config.NUM_PLAYERS):
            if players[i]:
                # Python doesn't allow comparisons between arrays,
                # so we're just checking if the nth value is 1 (true) or 0 (false)
                if players[i].action_vector[0]:
                    self.batch_multi_step(action[players[i].player_num], players[i],
                                          game_observations[players[i].player_num])
                if players[i].action_vector[1]:
                    self.batch_shop(action[players[i].player_num], players[i],
                                    game_observations[players[i].player_num])
                # Move item to board
                if players[i].current_action == 3:
                    players[i].action_values.append(action[players[i].player_num])
                    if players[i].action_vector[3]:
                        players[i].action_vector = np.array([0, 0, 0, 0, 1, 0, 0, 0])
                    elif players[i].action_vector[4]:
                        players[i].action_vector = np.array([0, 0, 0, 0, 0, 1, 0, 0])
                    else:
                        players[i].action_vector = np.array([1, 0, 0, 0, 0, 0, 0, 0])
                        if players[i].action_values[0] > 9:
                            players[i].action_values[0] = int(np.floor(np.random.rand(1, 1) * 10))
                        if players[i].action_values[1] > 6:
                            players[i].action_values[1] = int(np.floor(np.random.rand(1, 1) * 7))
                        if players[i].action_values[2] > 3:
                            players[i].action_values[2] = int(np.floor(np.random.rand(1, 1) * 4))
                        players[i].move_item_to_board(players[i].action_values[0], players[i].action_values[1],
                                                      players[i].action_values[2])
                        players[i].action_values = []

                # Part 2 of selling unit from bench
                if players[i].current_action == 4:
                    if action[players[i].player_num] > 8:
                        action[players[i].player_num] = int(np.floor(np.random.rand(1, 1) * 10))
                    players[i].action_vector = np.array([1, 0, 0, 0, 0, 0, 0, 0])
                    players[i].sell_from_bench(action[players[i].player_num])
                # Part 2 to 4 of moving bench to board
                if players[i].current_action == 5:
                    players[i].action_values.append(action[players[i].player_num])
                    if players[i].action_vector[2]:
                        players[i].action_vector = np.array([0, 0, 0, 0, 1, 0, 0, 0])
                    elif players[i].action_vector[4]:
                        players[i].action_vector = np.array([0, 0, 0, 0, 0, 1, 0, 0])
                    else:
                        players[i].action_vector = np.array([1, 0, 0, 0, 0, 0, 0, 0])
                        if players[i].action_values[0] > 8:
                            players[i].action_values[0] = int(np.floor(np.random.rand(1, 1) * 9))
                        if players[i].action_values[1] > 6:
                            players[i].action_values[1] = int(np.floor(np.random.rand(1, 1) * 7))
                        if players[i].action_values[2] > 3:
                            players[i].action_values[2] = int(np.floor(np.random.rand(1, 1) * 4))
                        players[i].move_bench_to_board(players[i].action_values[0], players[i].action_values[1],
                                                       players[i].action_values[2])
                        players[i].action_values = []
                # Part 2 to 3 of moving board to bench
                if players[i].current_action == 6:
                    players[i].action_values.append(action[players[i].player_num])
                    if players[i].action_vector[4]:
                        players[i].action_vector = np.array([0, 0, 0, 0, 0, 1, 0, 0])
                    else:
                        players[i].action_vector = np.array([1, 0, 0, 0, 0, 0, 0, 0])
                        if players[i].action_values[0] > 6:
                            players[i].action_values[0] = int(np.floor(np.random.rand(1, 1) * 7))
                        if players[i].action_values[1] > 3:
                            players[i].action_values[1] = int(np.floor(np.random.rand(1, 1) * 4))
                        players[i].move_board_to_bench(players[i].action_values[0], players[i].action_values[1])
                        players[i].action_values = []
                # Part 2 to 5 of moving board to board
                if players[i].current_action == 7:
                    players[i].action_values.append(action[players[i].player_num])
                    if players[i].action_vector[4]:
                        players[i].action_vector = np.array([0, 0, 0, 0, 0, 1, 0, 0])
                    elif players[i].action_vector[5]:
                        players[i].action_vector = np.array([0, 0, 0, 0, 0, 0, 1, 0])
                    elif players[i].action_vector[6]:
                        players[i].action_vector = np.array([0, 0, 0, 0, 0, 0, 0, 1])
                    else:
                        players[i].action_vector = np.array([1, 0, 0, 0, 0, 0, 0, 0])
                        if players[i].action_values[0] > 6:
                            players[i].action_values[0] = int(np.floor(np.random.rand(1, 1) * 7))
                        if players[i].action_values[1] > 3:
                            players[i].action_values[1] = int(np.floor(np.random.rand(1, 1) * 4))
                        if players[i].action_values[2] > 6:
                            players[i].action_values[2] = int(np.floor(np.random.rand(1, 1) * 7))
                        if players[i].action_values[3] > 3:
                            players[i].action_values[3] = int(np.floor(np.random.rand(1, 1) * 4))
                        players[i].move_board_to_board(players[i].action_values[0], players[i].action_values[1],
                                                       players[i].action_values[2], players[i].action_values[3])
                        players[i].action_values = []
                rewards.append(players[i].reward)
            else:
                rewards.append(0)
        # Some function that evens out rewards to all other players
        rewards = np.squeeze(np.asarray(rewards))
        return rewards

    def batch_multi_step(self, action, player, game_observation):
        player.current_action = action
        if action == 0:
            player.action_vector = np.array([0, 1, 0, 0, 0, 0, 0, 0])

        # action vector already == np.array([1, 0, 0, 0, 0, 0, 0, 0]) by this point
        elif action == 1:
            if player.refresh():
                self.shops[player.player_num] = self.pool_obj.sample(player, 5)
                game_observation.generate_shop_vector(self.shops[player.player_num])

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

        elif action == 9:
            # This would normally be end turn but figure it out later
            pass

    def batch_shop(self, shop_action, player, game_observation):
        if shop_action > 4:
            shop_action = int(np.floor(np.random.rand(1, 1) * 5))

        if shop_action == 0:
            if self.shops[player.player_num][0] == " ":
                player.reward += player.mistake_reward
                return
            if self.shops[player.player_num][0].endswith("_c"):
                c_shop = self.shops[player.player_num][0].split('_')
                a_champion = champion.champion(c_shop[0], chosen=c_shop[1], itemlist=[])
            else:
                a_champion = champion.champion(self.shops[player.player_num][0])
            success = player.buy_champion(a_champion)
            if success:
                self.shops[player.player_num][0] = " "
                game_observation.generate_shop_vector(self.shops[player.player_num])
            else:
                return

        elif shop_action == 1:
            if self.shops[player.player_num][1] == " ":
                player.reward += player.mistake_reward
                return
            if self.shops[player.player_num][1].endswith("_c"):
                c_shop = self.shops[player.player_num][1].split('_')
                a_champion = champion.champion(c_shop[0], chosen=c_shop[1], itemlist=[])
            else:
                a_champion = champion.champion(self.shops[player.player_num][1])
            success = player.buy_champion(a_champion)
            if success:
                self.shops[player.player_num][1] = " "
                game_observation.generate_shop_vector(self.shops[player.player_num])
            else:
                return

        elif shop_action == 2:
            if self.shops[player.player_num][2] == " ":
                player.reward += player.mistake_reward
                return
            if self.shops[player.player_num][2].endswith("_c"):
                c_shop = self.shops[player.player_num][2].split('_')
                a_champion = champion.champion(c_shop[0], chosen=c_shop[1], itemlist=[])
            else:
                a_champion = champion.champion(self.shops[player.player_num][2])
            success = player.buy_champion(a_champion)
            if success:
                self.shops[player.player_num][2] = " "
                game_observation.generate_shop_vector(self.shops[player.player_num])
            else:
                return

        elif shop_action == 3:
            if self.shops[player.player_num][3] == " ":
                player.reward += player.mistake_reward
                return
            if self.shops[player.player_num][3].endswith("_c"):
                c_shop = self.shops[player.player_num][3].split('_')
                a_champion = champion.champion(c_shop[0], chosen=c_shop[1], itemlist=[])
            else:
                a_champion = champion.champion(self.shops[player.player_num][3])

            success = player.buy_champion(a_champion)
            if success:
                self.shops[player.player_num][3] = " "
                game_observation.generate_shop_vector(self.shops[player.player_num])
            else:
                return

        elif shop_action == 4:
            if self.shops[player.player_num][4] == " ":
                player.reward += player.mistake_reward
                return
            if self.shops[player.player_num][4].endswith("_c"):
                c_shop = self.shops[player.player_num][4].split('_')
                a_champion = champion.champion(c_shop[0], chosen=c_shop[1], itemlist=[])
            else:
                a_champion = champion.champion(self.shops[player.player_num][4])

            success = player.buy_champion(a_champion)
            if success:
                self.shops[player.player_num][4] = " "
                game_observation.generate_shop_vector(self.shops[player.player_num])
            else:
                return


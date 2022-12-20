import time
import config
from Models import MuZero_trainer
from Simulator import champion, player as player_class, pool
import datetime
import game_round
import numpy as np
import tensorflow as tf
from Simulator.origin_class import team_traits, game_comp_tiers
from Simulator.stats import COST
from Models.MuZero_agent import MuZero_agent
from Models.MuZero_agent_2 import TFTNetwork, MCTSAgent, Batch_MCTSAgent
from Models.replay_muzero_buffer import ReplayBuffer
from global_buffer import GlobalBuffer

CURRENT_EPISODE = 0
previous_reward = [0 for _ in range(config.NUM_PLAYERS)]


def reset(sim):
    pool_obj = pool.pool()
    sim.PLAYERS = [player_class.player(pool_obj, i) for i in range(sim.num_players)]
    sim.NUM_DEAD = 0
    sim.player_rewards = [0 for i in range(sim.num_players)]
    # # This starts the game over from the beginning with a fresh set of players.
    # game_round.game_logic


# The return is the shop, boolean for end of turn, boolean for successful action
def step_5d(action, player, shop, pool_obj):
    if action[0] == 0:
        if shop[0] == " ":
            player.reward += player.mistake_reward
            return shop, False, False
        if shop[0].endswith("_c"):
            c_shop = shop[0].split('_')
            a_champion = champion.champion(c_shop[0], chosen=c_shop[1], itemlist=[])
        else:
            a_champion = champion.champion(shop[0])
        success = player.buy_champion(a_champion)
        if success:
            shop[0] = " "
        else:
            return shop, False, False

    elif action[0] == 1:
        if shop[1] == " ":
            player.reward += player.mistake_reward
            return shop, False, False
        if shop[1].endswith("_c"):
            c_shop = shop[1].split('_')
            a_champion = champion.champion(c_shop[0], chosen=c_shop[1], itemlist=[])
        else:
            a_champion = champion.champion(shop[1])
        success = player.buy_champion(a_champion)
        if success:
            shop[1] = " "
        else:
            return shop, False, False

    elif action[0] == 2:
        if shop[2] == " ":
            player.reward += player.mistake_reward
            return shop, False, False
        if shop[2].endswith("_c"):
            c_shop = shop[2].split('_')
            a_champion = champion.champion(c_shop[0], chosen=c_shop[1], itemlist=[])
        else:
            a_champion = champion.champion(shop[2])
        success = player.buy_champion(a_champion)
        if success:
            shop[2] = " "
        else:
            return shop, False, False

    elif action[0] == 3:
        if shop[3] == " ":
            player.reward += player.mistake_reward
            return shop, False, False
        if shop[3].endswith("_c"):
            c_shop = shop[3].split('_')
            a_champion = champion.champion(c_shop[0], chosen=c_shop[1], itemlist=[])
        else:
            a_champion = champion.champion(shop[3])

        success = player.buy_champion(a_champion)
        if success:
            shop[3] = " "
        else:
            return shop, False, False

    elif action[0] == 4:
        if shop[4] == " ":
            player.reward += player.mistake_reward
            return shop, False, False
        if shop[4].endswith("_c"):
            c_shop = shop[4].split('_')
            a_champion = champion.champion(c_shop[0], chosen=c_shop[1], itemlist=[])
        else:
            a_champion = champion.champion(shop[4])

        success = player.buy_champion(a_champion)
        if success:
            shop[4] = " "
        else:
            return shop, False, False

    # Refresh
    elif action[0] == 5:
        if player.refresh():
            shop = pool_obj.sample(player, 5)
        else:
            return shop, False, False

    # buy Exp
    elif action[0] == 6:
        if player.buy_exp():
            pass
        else:
            return shop, False, False

    # end turn 
    elif action[0] == 7:
        return shop, True, True

    # move Item
    elif action[0] == 8:
        # Call network to activate the move_item_agent
        if not player.move_item_to_board(action[1], action[3], action[4]):
            return shop, False, False

    # sell Unit
    elif action[0] == 9:
        # Call network to activate the bench_agent
        if not player.sell_from_bench(action[2]):
            return shop, False, False

    # move bench to Board
    elif action[0] == 10:
        # Call network to activate the bench and board agents
        if not player.move_bench_to_board(action[2], action[3], action[4]):
            return shop, False, False

    # move board to bench
    elif action[0] == 11:
        # Call network to activate the bench and board agents
        if not player.move_board_to_bench(action[3], action[4]):
            return shop, False, False

    else:
        return shop, False, False
    return shop, False, True


# The return is the shop, boolean for end of turn, boolean for successful action, number of actions taken
def multi_step(action, player, shop, pool_obj, game_observation, agent, buffer):
    if action == 0:
        action_vector = np.array([0, 1, 0, 0, 0, 0, 0, 0])
        observation, _ = game_observation.observation(player, buffer, action_vector)
        shop_action, policy = agent.policy(observation, player.player_num)

        if shop_action > 4:
            shop_action = int(np.floor(np.random.rand(1, 1) * 5))

        buffer.store_replay_buffer(observation, shop_action, 0, policy)

        if shop_action == 0:
            if shop[0] == " ":
                player.reward += player.mistake_reward
                return shop, False, False, 2
            if shop[0].endswith("_c"):
                c_shop = shop[0].split('_')
                a_champion = champion.champion(c_shop[0], chosen=c_shop[1], itemlist=[])
            else:
                a_champion = champion.champion(shop[0])
            success = player.buy_champion(a_champion)
            if success:
                shop[0] = " "
                game_observation.generate_shop_vector(shop)
            else:
                return shop, False, False, 2

        elif shop_action == 1:
            if shop[1] == " ":
                player.reward += player.mistake_reward
                return shop, False, False, 2
            if shop[1].endswith("_c"):
                c_shop = shop[1].split('_')
                a_champion = champion.champion(c_shop[0], chosen=c_shop[1], itemlist=[])
            else:
                a_champion = champion.champion(shop[1])
            success = player.buy_champion(a_champion)
            if success:
                shop[1] = " "
                game_observation.generate_shop_vector(shop)
            else:
                return shop, False, False, 2

        elif shop_action == 2:
            if shop[2] == " ":
                player.reward += player.mistake_reward
                return shop, False, False, 2
            if shop[2].endswith("_c"):
                c_shop = shop[2].split('_')
                a_champion = champion.champion(c_shop[0], chosen=c_shop[1], itemlist=[])
            else:
                a_champion = champion.champion(shop[2])
            success = player.buy_champion(a_champion)
            if success:
                shop[2] = " "
                game_observation.generate_shop_vector(shop)
            else:
                return shop, False, False, 2

        elif shop_action == 3:
            if shop[3] == " ":
                player.reward += player.mistake_reward
                return shop, False, False, 2
            if shop[3].endswith("_c"):
                c_shop = shop[3].split('_')
                a_champion = champion.champion(c_shop[0], chosen=c_shop[1], itemlist=[])
            else:
                a_champion = champion.champion(shop[3])

            success = player.buy_champion(a_champion)
            if success:
                shop[3] = " "
                game_observation.generate_shop_vector(shop)
            else:
                return shop, False, False, 2

        elif shop_action == 4:
            if shop[4] == " ":
                player.reward += player.mistake_reward
                return shop, False, False, 2
            if shop[4].endswith("_c"):
                c_shop = shop[4].split('_')
                a_champion = champion.champion(c_shop[0], chosen=c_shop[1], itemlist=[])
            else:
                a_champion = champion.champion(shop[4])

            success = player.buy_champion(a_champion)
            if success:
                shop[4] = " "
                game_observation.generate_shop_vector(shop)
            else:
                return shop, False, False, 2

    # Refresh
    elif action == 1:
        if player.refresh():
            shop = pool_obj.sample(player, 5)
            game_observation.generate_shop_vector(shop)
        else:
            return shop, False, False, 1

    # buy Exp
    elif action == 2:
        if player.buy_exp():
            pass
        else:
            return shop, False, False, 1

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
            return shop, False, False, 4
        else:
            return shop, False, True, 4

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
            return shop, False, False, 2
        else:
            return shop, False, True, 2

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
            return shop, False, False, 4
        else:
            return shop, False, True, 4

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
            return shop, False, False, 3
        else:
            return shop, False, True, 3

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
            return shop, False, False, 5
        else:
            return shop, False, True, 5

    # Update all information in the observation relating to the other players.
    # Later in training, turn this number up to 7 due to how long it takes a normal player to execute
    elif action == 8:
        game_observation.generate_game_comps_vector()
        return shop, False, True, 1

    # end turn
    elif action == 9:
        # Testing a version that does not end the turn on this action.
        return shop, False, True, 1
        # return shop, True, True, 1

    # Possible to add another action here which is basically pass the action back.
    # Wait and do nothing. If anyone thinks that is beneficial, let me know.
    else:
        return shop, False, False, 1
    return shop, False, True, 1


def batch_controller(action, players, shops, pool_obj, game_observations, agent, buffers):
    for player in players:
        if player:
            # Python doesn't allow comparisons between arrays, so we're just checking if the nth value is 1 (true) or 0 (false)
            if player.action_vector[0]:
                batch_multi_step(action[player.player_num], player, shops, pool_obj, game_observations[player.player_num])
            if player.action_vector[1]:
                batch_shop(action[player.player_num], player, shops, game_observations[player.player_num])
            # Move item to board
            if player.current_action == 3:
                player.action_values.append(action[player.player_num])
                if player.action_vector[3]:
                    player.action_vector[4]
                elif player.action_vector[4]:
                    player.action_vector[5]
                else:
                    player.action_vector = np.array([1, 0, 0, 0, 0, 0, 0, 0])
                    if player.action_values[0] > 9:
                        player.action_values[0] = int(np.floor(np.random.rand(1, 1) * 10))
                    if player.action_values[1] > 6:
                        player.action_values[1] = int(np.floor(np.random.rand(1, 1) * 7))
                    if player.action_values[2] > 3:
                        player.action_values[2] = int(np.floor(np.random.rand(1, 1) * 4))
                    player.move_item_to_board(player.action_values[0], player.action_values[1], player.action_values[2])
                    player.action_values = []

            # Part 2 of selling unit from bench
            if player.current_action == 4:
                if action[player.player_num] > 8:
                    action[player.player_num] = int(np.floor(np.random.rand(1, 1) * 10))
                player.action_vector = np.array([1, 0, 0, 0, 0, 0, 0, 0])
                player.sell_from_bench(action[player.player_num])
            # Part 2 to 4 of moving bench to board
            if player.current_action == 5:
                player.action_values.append(action[player.player_num])
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
                    player.move_bench_to_board(player.action_values[0], player.action_values[1], player.action_values[2])
                    player.action_values = []
            # Part 2 to 3 of moving board to bench
            if player.current_action == 6:
                player.action_values.append(action[player.player_num])
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
                player.action_values.append(action[player.player_num])
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


def batch_multi_step(action, player, shop, pool_obj, game_observation):

    player.current_action = action
    if action == 0:
        player.action_vector = np.array([0, 1, 0, 0, 0, 0, 0, 0])

    # action vector already == np.array([1, 0, 0, 0, 0, 0, 0, 0]) by this point
    elif action == 1:
        if player.refresh():
            shop[player.player_num] = pool_obj.sample(player, 5)
            game_observation.generate_shop_vector(shop[player.player_num])

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


def batch_shop(shop_action, player, shop, game_observation):
    if shop_action > 4:
        shop_action = int(np.floor(np.random.rand(1, 1) * 5))

    if shop_action == 0:
        if shop[player.player_num][0] == " ":
            player.reward += player.mistake_reward
            return
        if shop[player.player_num][0].endswith("_c"):
            c_shop = shop[player.player_num][0].split('_')
            a_champion = champion.champion(c_shop[0], chosen=c_shop[1], itemlist=[])
        else:
            a_champion = champion.champion(shop[player.player_num][0])
        success = player.buy_champion(a_champion)
        if success:
            shop[player.player_num][0] = " "
            game_observation.generate_shop_vector(shop[player.player_num])
        else:
            return

    elif shop_action == 1:
        if shop[player.player_num][1] == " ":
            player.reward += player.mistake_reward
            return
        if shop[player.player_num][1].endswith("_c"):
            c_shop = shop[player.player_num][1].split('_')
            a_champion = champion.champion(c_shop[0], chosen=c_shop[1], itemlist=[])
        else:
            a_champion = champion.champion(shop[player.player_num][1])
        success = player.buy_champion(a_champion)
        if success:
            shop[player.player_num][1] = " "
            game_observation.generate_shop_vector(shop[player.player_num])
        else:
            return

    elif shop_action == 2:
        if shop[player.player_num][2] == " ":
            player.reward += player.mistake_reward
            return
        if shop[player.player_num][2].endswith("_c"):
            c_shop = shop[player.player_num][2].split('_')
            a_champion = champion.champion(c_shop[0], chosen=c_shop[1], itemlist=[])
        else:
            a_champion = champion.champion(shop[player.player_num][2])
        success = player.buy_champion(a_champion)
        if success:
            shop[player.player_num][2] = " "
            game_observation.generate_shop_vector(shop[player.player_num])
        else:
            return

    elif shop_action == 3:
        if shop[player.player_num][3] == " ":
            player.reward += player.mistake_reward
            return
        if shop[player.player_num][3].endswith("_c"):
            c_shop = shop[player.player_num][3].split('_')
            a_champion = champion.champion(c_shop[0], chosen=c_shop[1], itemlist=[])
        else:
            a_champion = champion.champion(shop[player.player_num][3])

        success = player.buy_champion(a_champion)
        if success:
            shop[player.player_num][3] = " "
            game_observation.generate_shop_vector(shop[player.player_num])
        else:
            return

    elif shop_action == 4:
        if shop[player.player_num][4] == " ":
            player.reward += player.mistake_reward
            return
        if shop[player.player_num][4].endswith("_c"):
            c_shop = shop[player.player_num][4].split('_')
            a_champion = champion.champion(c_shop[0], chosen=c_shop[1], itemlist=[])
        else:
            a_champion = champion.champion(shop[player.player_num][4])

        success = player.buy_champion(a_champion)
        if success:
            shop[player.player_num][4] = " "
            game_observation.generate_shop_vector(shop[player.player_num])
        else:
            return


# Batch step
def batch_step(players, agent, buffers, pool_obj):
    shops = [pool_obj.sample(players[i], 5) for i in range(config.NUM_PLAYERS)]
    actions_taken = 0
    game_observations = [Observation() for _ in range(config.NUM_PLAYERS)]
    for i in range(config.NUM_PLAYERS):
        game_observations[i].generate_game_comps_vector()
        game_observations[i].generate_shop_vector(shops[i])
    t = time.time_ns()
    while actions_taken < 30:
        observation_list = []
        previous_action = []
        for player in players:
            if player:
                # TODO
                # store game state vector later
                observation, game_state_vector = game_observations[player.player_num]\
                    .observation(player, buffers[player.player_num], player.action_vector)
                observation_list.append(observation)
                buffers[player.player_num].store_observation(game_state_vector)
                previous_action.append(buffers[player.player_num].get_prev_action())

        observation_list = np.squeeze(np.array(observation_list))
        previous_action = np.array(previous_action)

        action, policy = agent.batch_policy(observation_list, previous_action)

        rewards = [player.reward - previous_reward[player.player_num] for player in players]
        batch_controller(action, players, shops, pool_obj, game_observations, agent, buffers)
        actions_taken += 1
        for player in players:
            buffers[player.player_num].store_replay_buffer(observation_list[player.player_num],
                                                           action[player.player_num], rewards[player.player_num],
                                                           policy[player.player_num])

            previous_reward[player.player_num] = player.reward
    print(time.time_ns() - t)


# Includes the vector of the shop, bench, board, and item list.
# Add a vector for each player composition makeup at the start of the round.
# action vector = [Decision, shop, champion_bench, item_bench, x_axis, y_axis, x_axis 2, y_axis 2]
class Observation:
    def __init__(self):
        self.shop_vector = np.zeros(45)
        self.game_comp_vector = np.zeros(208)

    def observation(self, player, buffer, action_vector):
        shop_vector = self.shop_vector
        game_state_vector = self.game_comp_vector
        complete_game_state_vector = np.concatenate([np.expand_dims(shop_vector, axis=0),
                                                     np.expand_dims(player.board_vector, axis=0),
                                                     np.expand_dims(player.bench_vector, axis=0),
                                                     np.expand_dims(player.chosen_vector, axis=0),
                                                     np.expand_dims(player.item_vector, axis=0),
                                                     np.expand_dims(player.player_vector, axis=0),
                                                     np.expand_dims(game_state_vector, axis=0),
                                                     np.expand_dims(action_vector, axis=0), ], axis=-1)
        i = 0
        input_vector = complete_game_state_vector
        while i < buffer.len_observation_buffer() and i < 0:
            i += 1
            input_vector = np.concatenate([input_vector, buffer.get_prev_observation(i)], axis=-1)

        while i < 0:
            i += 1
            input_vector = np.concatenate([input_vector, np.zeros(buffer.get_observation_shape())], axis=-1)
        # std = np.std(input_vector)
        # if std == 0:
        # input_vector = input_vector - np.mean(input_vector)
        # else:
        #     input_vector = (input_vector - np.mean(input_vector)) / std
        # print(input_vector.shape)
        return input_vector, complete_game_state_vector

    def generate_game_comps_vector(self):
        output = np.zeros(208)
        for i in range(len(game_comp_tiers)):
            tiers = np.array(list(game_comp_tiers[i].values()))
            tierMax = np.max(tiers)
            if tierMax != 0:
                tiers = tiers / tierMax
            output[i * 26: i * 26 + 26] = tiers
        self.game_comp_vector = output

    def generate_shop_vector(self, shop):
        # each champion has 6 bit for the name, 1 bit for the chosen.
        # 5 of them makes it 35.
        output_array = np.zeros(45)
        shop_chosen = False
        chosen_shop_index = -1
        chosen_shop = ''
        for x in range(0, len(shop)):
            input_array = np.zeros(8)
            if shop[x]:
                chosen = 0
                if shop[x].endswith("_c"):
                    chosen_shop_index = x
                    chosen_shop = shop[x]
                    c_shop = shop[x].split('_')
                    shop[x] = c_shop[0]
                    chosen = 1
                    shop_chosen = c_shop[1]
                i_index = list(COST.keys()).index(shop[x])
                # This should update the item name section of the vector
                for z in range(6, 0, -1):
                    if i_index > 2 ** (z - 1):
                        input_array[6 - z] = 1
                        i_index -= 2 ** (z - 1)
                input_array[7] = chosen
            # Input chosen mechanics once I go back and update the chosen mechanics.
            output_array[8 * x: 8 * (x + 1)] = input_array
        if shop_chosen:
            if shop_chosen == 'the':
                shop_chosen = 'the_boss'
            i_index = list(team_traits.keys()).index(shop_chosen)
            # This should update the item name section of the vector
            for z in range(5, 0, -1):
                if i_index > 2 * z:
                    output_array[45 - z] = 1
                    i_index -= 2 * z
            shop[chosen_shop_index] = chosen_shop
        self.shop_vector = output_array


def reward(player):
    return player.reward


# This is the main overarching gameplay method.
# This is going to be implemented mostly in the game_round file under the AI side of things. 
def collect_gameplay_experience(sim, agent, buffers, episode_cnt):
    reset(sim)
    sim.episode(agent, buffers, episode_cnt)


# TO DO: Implement evaluator
def train_model(max_episodes=10000):
    # # Uncomment if you change the size of the input array
    # pool_obj = pool.pool()
    # test_player = player_class.player(pool_obj, 0)
    # shop = pool_obj.sample(test_player, 5)
    # shape = np.array(observation(shop, test_player)).shape

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    global_agent = TFTNetwork()
    # global_agent = MuZero_agent()
    global_buffer = GlobalBuffer()
    trainer = MuZero_trainer.Trainer()
    game_sim = game_round.TFT_Simulation()
    # agents = [MuZero_agent() for _ in range(game_sim.num_players)]

    train_step = 0
    for episode_cnt in range(1, max_episodes):
        agent = Batch_MCTSAgent(network=global_agent)
        buffers = [ReplayBuffer(global_buffer) for _ in range(game_sim.num_players)]
        collect_gameplay_experience(game_sim, agent, buffers, episode_cnt)

        for i in range(game_sim.num_players):
            buffers[i].store_global_buffer()
        # Keeping this here in case I want to only update positive rewards
        # rewards = game_round.player_rewards
        while global_buffer.available_batch():
            gameplay_experience_batch = global_buffer.sample_batch()
            trainer.train_network(gameplay_experience_batch, global_agent, train_step, train_summary_writer)
            train_step += 1

        if episode_cnt % 5 == 0:
            game_round.log_to_file_start()
        # for i in range(game_sim.num_players):
        #     agents[i] = MCTSAgent(global_agent, agent_id=i)
        # if episode_cnt % 50 == 0:
        #     saveModel(agents[0].a3c_net, episode_cnt)
        print("Episode " + str(episode_cnt) + " Completed")


def saveModel(agent, epoch):
    checkpoint_path = "savedWeights/cp-{epoch:04d}.ckpt"
    agent.save_weights(checkpoint_path.format(epoch=epoch))


# TO DO: Has to run some episodes and return an average reward. Probably 5 games of 8 players.  
def evaluate(agent):
    return 0

import champion
import game_round
import player as player_class
import numpy as np
import pool
from stats import COST
from dqn_agent import DQN_Agent
from replay_buffer import ReplayBuffer

import tensorflow as tf
import numpy as np
from tf_agents.agents.dqn import dqn_agent


def reset():
    pool_obj = pool.pool()
    game_round.PLAYERS = [ player_class.player(pool_obj, i) for i in range( game_round.num_players ) ]
    game_round.NUM_DEAD = 0
    # # This starts the game over from the beginning with a fresh set of players.
    # game_round.game_logic


def start_game():
    game_round.game_logic


def check_dead():
    return game_round.check_dead()


# I'm having difficulty determining if 1 step is a game or one action.
# One step I'm 99% sure is one action or set of actions. 
# A set of actions would be a move command.
# I'm pretty sure this just means one action.
def step(action, player, shop, pool_obj):
    if (action[0] == 0):
        if shop[0] == " ": 
            player.reward -= 0.01
            return shop, False
        # print("shop at position 0 is " + shop[0] + " with gold amount " + str(player.gold))
        if shop[0].endswith("_c"):
            c_shop = shop[0].split('_')
            a_champion = champion.champion(c_shop[0], chosen = c_shop[1])
            print("buying chosen unit -> " + shop[0])
        else:
            a_champion = champion.champion(shop[0])
        success = player.buy_champion(a_champion)
        if (success): 
            shop[0] = " "
        player.print("buy option 0")
        

    elif (action[0] == 1): 
        if shop[1] == " ": 
            player.reward -= 0.01
            return shop, False
        if shop[1].endswith("_c"):
            c_shop = shop[1].split('_')
            a_champion = champion.champion(c_shop[0], chosen = c_shop[1])
            print("buying chosen unit -> " + shop[1])
        else:
            a_champion = champion.champion(shop[1])
        success = player.buy_champion(a_champion)
        if (success): 
            shop[1] = " "
        player.print("buy option 1")

    elif (action[0] == 2): 
        if shop[2] == " ": 
            player.reward -= 0.01
            return shop, False
        if shop[2].endswith("_c"):
            c_shop = shop[2].split('_')
            a_champion = champion.champion(c_shop[0], chosen = c_shop[1])
            print("buying chosen unit -> " + shop[2])
        else:
            a_champion = champion.champion(shop[2]) 
        success = player.buy_champion(a_champion)
        if (success): 
            shop[2] = " "
        player.print("buy option 2")

    elif (action[0] == 3): 
        if shop[3] == " ": 
            player.reward -= 0.01
            return shop, False
        if shop[3].endswith("_c"):
            c_shop = shop[3].split('_')
            a_champion = champion.champion(c_shop[0], chosen = c_shop[1])
            print("buying chosen unit -> " + shop[3])
        else:
            a_champion = champion.champion(shop[3])
        
        success = player.buy_champion(a_champion)
        if (success): 
            shop[3] = " "
        player.print("buy option 3")

    elif (action[0] == 4): 
        if shop[4] == " ": 
            player.reward -= 0.01
            return shop, False
        if shop[4].endswith("_c"):
            c_shop = shop[4].split('_')
            a_champion = champion.champion(c_shop[0], chosen = c_shop[1])
            print("buying chosen unit -> " + shop[4])
        else:
            a_champion = champion.champion(shop[4])
        
        success = player.buy_champion(a_champion)
        if (success): 
            shop[4] = " "

        player.print("buy option 4")

    # Refresh
    elif (action[0] == 5):
        if (player.refresh()): 
            shop = pool_obj.sample(player, 5)
            player.print("Refresh")
        else:       
            player.print('no gold, failed to refresh')

    # buy Exp
    elif (action[0] == 6):                
        if player.buy_exp(): 
            player.print("exp")
        else: 
            player.print('no enough gold, buy exp failed')

    # end turn 
    elif (action[0] == 7): 
        return shop, True

    # move Item
    elif (action[0] == 8): 
        # Call network to activate the move_item_agent
        player.printt("move item method")
        if not player.move_item_to_board(action[1], action[3], action[4]):
            player.print("Could not put item on {}, {}".format(action[3], action[4]))

    # sell Unit
    elif (action[0] == 9): 
        # Call network to activate the bench_agent
        player.printt("sell unit method")
        if(not player.sell_from_bench(action[2])):
            player.print("Unit sale failed")

    # move bench to Board
    elif (action[0] == 10): 
        # Call network to activate the bench and board agents
        player.print("move bench to board")
        if(not player.move_bench_to_board(action[2], action[3], action[4])):
            player.print("Move command failed")

    # move board to bench
    elif (action[0] == 11): 
        # Call network to activate the bench and board agents
        player.print("move board to bench")
        if(not player.move_board_to_bench(action[3], action[4])):
            player.print("Move command failed")
    
    else:
        player.print("wrong call")
    return shop, False


# Includes the vector of the shop, bench, board, and itemlist.
# Add a vector for each player composition makeup at the start of the round.
def observation(shop, player):
    input_vector = np.concatenate([np.expand_dims(player.board_vector, axis=0), \
                                np.expand_dims(player.bench_vector, axis=0), \
                                np.expand_dims(player.item_vector, axis=0)], axis=-1)
    shop_vector = generate_shop_vector(shop)
    input_vector = np.concatenate([np.expand_dims(shop_vector, axis=0), input_vector], axis=-1)
    return input_vector


# Includes the vector of the bench, board, and itemlist.
def board_observation(player):
    input_vector = np.concatenate([np.expand_dims(player.board_vector, axis=0), \
                                np.expand_dims(player.bench_vector, axis=0), \
                                np.expand_dims(player.item_vector, axis=0)], axis=-1)
    return input_vector


def generate_shop_vector(shop):
    # each champion has 6 bit for the name, 1 bit for the chosen.
    # 5 of them makes it 35.
    output_array = np.zeros(35)
    for x in range(0, len(shop)):
        input_array = np.zeros(7)
        if shop[x]:
            i_index = list(COST.keys()).index(shop[x])
            # This should update the item name section of the vector
            for z in range(0, 6, -1):
                if i_index > 2 * z:
                    input_array[6 - z] = 1
                    i_index -= 2 * z 
        # Input chosen mechanics once I go back and update the chosen mechanics. 
        output_array[7 * x: 7 * (x + 1)] = input_array
    return output_array


def reward(player):
    return player.reward


# This is the main overarching gameplay method.
# This is going to be implemented mostly in the game_round file under the AI side of things. 
def collect_gameplay_experience(agent, buffers):
    reset()
    game_round.episode(agent, buffers)


# TO DO: Implement decaying random policy
# TO DO: Implement evaluator
# TO DO: Implement an update on the target network
def train_model(max_episodes = 10000):
    # # Uncomment if you change the size of the input array
    # pool_obj = pool.pool()
    # test_player = player_class.player(pool_obj, 0)
    # shop = pool_obj.sample(test_player, 5)
    # shape = np.array(observation(shop, test_player)).shape
    shape = np.array([1, 1094])
    # print(shape)
    agent = DQN_Agent(shape)
    buffers = [ ReplayBuffer() for _ in range(game_round.num_players) ]
    for episode_cnt in range(1, max_episodes):
        collect_gameplay_experience(agent, buffers)

        # start here but change this up later. 
        # Idea is to only train on the winning network then copy the other agents
        # to the winner network and continue training on that.
        # Or to do the same thing with the top 2. 

        # Another idea is to train all of them until 1 network wins say 3 or 5 in a row. 
        # Then copy them all to that one network.
        print("Another episode COMPLETE")
        for i in range(game_round.num_players):
            gameplay_experience_batch = buffers[i].sample_gameplay_batch()
            loss = agent.train(gameplay_experience_batch)

        if episode_cnt % 50 == 0:
            # This is going to be changed later.
            agent.update_target_network()

# TO DO: Has to run some episodes and return an average reward. Probably 5 games of 8 players.  
def evaluate(agent):
    return
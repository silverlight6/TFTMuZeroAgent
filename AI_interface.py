from Models import MuZero_trainer
from Simulator import champion, player as player_class, pool
import datetime
import game_round
import numpy as np
import tensorflow as tf
from Simulator.origin_class import team_traits, game_comp_tiers
from Simulator.stats import COST
from Models.MuZero_agent import MuZero_agent
from Models.MuZero_agent_2 import TFTNetwork, MCTSAgent
from Models.replay_muzero_buffer import ReplayBuffer
from multiprocessing import Process
from global_buffer import GlobalBuffer

CURRENT_EPISODE = 0


def reset(sim):
    pool_obj = pool.pool()
    sim.PLAYERS = [player_class.player(pool_obj, i) for i in range(sim.num_players)]
    sim.NUM_DEAD = 0
    sim.player_rewards = [0 for i in range(sim.num_players)]
    # # This starts the game over from the beginning with a fresh set of players.
    # game_round.game_logic


# The return is the shop, boolean for end of turn, boolean for successful action
def step(action, player, shop, pool_obj):
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
        return shop, True, True, 1

    # Possible to add another action here which is basically pass the action back.
    # Wait and do nothing. If anyone thinks that is beneficial, let me know.
    else:
        return shop, False, False, 1
    return shop, False, True, 1


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

    # tf.debugging.set_log_device_placement(True)
    global_agent = TFTNetwork()
    bad_agent = TFTNetwork()
    
    # global_agent = MuZero_agent()
    global_buffer = GlobalBuffer()
    trainer = MuZero_trainer.Trainer()

    game_sim = game_round.TFT_Simulation()
    # agents = [MuZero_agent() for _ in range(game_sim.num_players)]
    TFTNetworks = [TFTNetwork() for _ in range(game_sim.num_players-1)]
    agents = [MCTSAgent(network=network, agent_id=i) for i, network in enumerate(TFTNetworks)]
    agents.append(MCTSAgent(network=bad_agent, agent_id=game_sim.num_players-1)) #add in the bad agent
    train_step = 0
    
    stats = [[0,0,0]] #stats list for data vis: [[agent tier, pos, episode]...] 
    tier = 0 
    beaten = False
    bad_agent_position = 0 

    for episode_cnt in range(1, max_episodes):

        buffers = [ReplayBuffer(global_buffer) for _ in range(game_sim.num_players)]
        collect_gameplay_experience(game_sim, agents, buffers, episode_cnt)

        for i in range(game_sim.num_players-1):
            buffers[i].store_global_buffer()
        # Keeping this here in case I want to only update positive rewards
        # rewards = game_round.player_rewards
        while global_buffer.available_batch():
            gameplay_experience_batch = global_buffer.sample_batch()
            trainer.train_network(gameplay_experience_batch, global_agent, train_step, train_summary_writer)
            train_step += 1
        bad_agent_position = agents[-1].game_pos
        if episode_cnt % 5 == 0:
            game_round.log_to_file_start()
            beaten = True
        
    
        for i in range(game_sim.num_players-1):
            agents[i] = MCTSAgent(global_agent, agent_id=i)
        
        # try: #If the bad agent has lost for the past 3 rounds, move up the agent. 
        #     if stats[-1][1] == 0 and stats[-2][1] == 0 and stats[-3][1] == 0:
        #         beaten = True
        # except:
        #     pass

        if beaten == True:
            global_agent.save_model(episode_cnt)
            bad_agent.load_model(episode_cnt)
            beaten = False 
            tier += 1 
        
        #add in the agent from x rounds previous, as a comparison to see training rate
        agents.append(MCTSAgent(network=bad_agent, agent_id=i+1))
        
        
        
        stats.append([tier, bad_agent_position, episode_cnt])
        print("Bad agent ID = "+str(i+1))
        print("Stats: ")
        print(stats[-1]) #[agent tier, agent pos, episode] (for agent pos, higher is a better place)
        print("Episode " + str(episode_cnt) + " Completed")

# TO DO: Has to run some episodes and return an average reward. Probably 5 games of 8 players.  
def evaluate(agent):
    return 0

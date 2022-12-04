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


# Includes the vector of the shop, bench, board, and itemlist.
# Add a vector for each player composition makeup at the start of the round.
def observation(shop, player, buffer):
    shop_vector = generate_shop_vector(shop)
    game_state_vector = generate_game_comps_vector()
    complete_game_state_vector = np.concatenate([np.expand_dims(shop_vector, axis=0),
                                                 np.expand_dims(player.board_vector, axis=0),
                                                 np.expand_dims(player.bench_vector, axis=0),
                                                 np.expand_dims(player.chosen_vector, axis=0),
                                                 np.expand_dims(player.item_vector, axis=0),
                                                 np.expand_dims(player.player_vector, axis=0),
                                                 np.expand_dims(game_state_vector, axis=0), ], axis=-1)
    i = 0
    input_vector = complete_game_state_vector
    while i < buffer.len_observation_buffer() and i < 4:
        i += 1
        input_vector = np.concatenate([input_vector, buffer.get_prev_observation(i)], axis=-1)

    while i < 4:
        i += 1
        input_vector = np.concatenate([input_vector, np.zeros(buffer.get_observation_shape())], axis=-1)
    # std = np.std(input_vector)
    # if std == 0:
    # input_vector = input_vector - np.mean(input_vector)
    # else:
    #     input_vector = (input_vector - np.mean(input_vector)) / std
    # print(input_vector.shape)
    return input_vector, complete_game_state_vector


# Includes the vector of the bench, board, and itemlist.
def board_observation(player):
    input_vector = np.concatenate([np.expand_dims(player.board_vector, axis=0),
                                   np.expand_dims(player.bench_vector, axis=0),
                                   np.expand_dims(player.chosen_vector, axis=0),
                                   np.expand_dims(player.item_vector, axis=0),
                                   np.expand_dims(player.player_vector, axis=0)], axis=-1)
    return input_vector


def generate_game_comps_vector():
    output = np.zeros(208)
    for i in range(len(game_comp_tiers)):
        tiers = np.array(list(game_comp_tiers[i].values()))
        tierMax = np.max(tiers)
        if tierMax != 0:
            tiers = tiers / tierMax
        output[i * 26: i * 26 + 26] = tiers
    return output


def generate_shop_vector(shop):
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
            for z in range(0, 6, -1):
                if i_index > 2 * z:
                    input_array[6 - z] = 1
                    i_index -= 2 * z
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
    return output_array


def reward(player):
    return player.reward


# This is the main overarching gameplay method.
# This is going to be implemented mostly in the game_round file under the AI side of things. 
def collect_gameplay_experience(sim, agent, buffers, episode_cnt):
    reset(sim)
    sim.episode(agent, buffers, episode_cnt)


class TFT_AI:
    def __init__(self):
        shape = np.array([1, 1382])
        self.global_agent = A3C_Agent(shape)
        self.num_workers = 8
        # self.num_workers = cpu_count()

    def train(self, max_episodes=10000):
        workers = []
        for i in range(self.num_workers):
            workers.append(WorkerAgent(self.global_agent, max_episodes))

        for worker in workers:
            worker.start()

        for worker in workers:
            worker.join()


class WorkerAgent(Process):
    def __init__(self, global_agent, max_episodes):
        Process.__init__(self)
        self.env = game_round.TFT_Simulation()

        shape = np.array([1, 1382])
        self.max_episodes = max_episodes
        self.global_agent = global_agent
        self.agents = [A3C_Agent(shape) for _ in range(8)]

        for agent in self.agents:
            agent.a3c_net.set_weights(self.global_agent.a3c_net.get_weights())

    def train(self):
        global CURRENT_EPISODE

        while self.max_episodes >= CURRENT_EPISODE:
            print(CURRENT_EPISODE)
            buffers = [ReplayBuffer() for _ in range(self.env.num_players)]
            collect_gameplay_experience(self.env, self.agents, buffers, CURRENT_EPISODE)
            for i in range(self.env.num_players):
                # if rewards[i] > 0:
                gameplay_experience_batch = buffers[i].sample_gameplay_batch()
                # lock.acquire()
                self.global_agent.train_step(gameplay_experience_batch, CURRENT_EPISODE)
                self.agents[i].a3c_net.set_weights(self.global_agent.a3c_net.get_weights())
                # lock.release()
            CURRENT_EPISODE += 1

    def run(self):
        self.train()


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
    # shape = np.array([1, 1382])
    global_agent = MuZero_agent()
    global_buffer = GlobalBuffer()
    trainer = MuZero_trainer.Trainer()
    # global_agent.a3c_net.save('~/A3C_net')
    game_sim = game_round.TFT_Simulation()
    # agents = [MuZero_agent() for _ in range(game_sim.num_players)]
    TFTNetworks = [TFTNetwork() for _ in range(game_sim.num_players)]
    agents = [MCTSAgent(network=network, agent_id=i) for i, network in enumerate(TFTNetworks)]
    train_step = 0
    for episode_cnt in range(1, max_episodes):
        buffers = [ReplayBuffer(global_buffer) for _ in range(game_sim.num_players)]
        collect_gameplay_experience(game_sim, agents, buffers, episode_cnt)

        for i in range(game_sim.num_players):
            buffers[i].store_global_buffer()
        # Keeping this here in case I want to only update positive rewards
        # rewards = game_round.player_rewards
        while global_buffer.available_batch():
            gameplay_experience_batch = global_buffer.sample_batch()
            trainer.train_network(gameplay_experience_batch, global_agent, train_step, train_summary_writer)
            train_step += 1

        game_round.log_to_file_start()
        for i in range(game_sim.num_players):
            agents[i] = global_agent
        # if episode_cnt % 50 == 0:
        #     saveModel(agents[0].a3c_net, episode_cnt)
        print("Episode " + str(episode_cnt) + " Completed")


def saveModel(agent, epoch):
    checkpoint_path = "savedWeights/cp-{epoch:04d}.ckpt"
    agent.save_weights(checkpoint_path.format(epoch=epoch))


# TO DO: Has to run some episodes and return an average reward. Probably 5 games of 8 players.  
def evaluate(agent):
    return 0

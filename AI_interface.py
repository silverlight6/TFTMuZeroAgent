import time
import config
import datetime
import numpy as np
import tensorflow as tf
import gymnasium as gym
from global_buffer import GlobalBuffer
from Models import MuZero_trainer
from Models.MuZero_agent_2 import TFTNetwork, Batch_MCTSAgent
from Models.replay_muzero_buffer import ReplayBuffer
from Simulator import player as player_class, pool, game_round
from Simulator.tft_simulator import TFT_Simulator
from Simulator.observation import Observation

CURRENT_EPISODE = 0
previous_reward = [0 for _ in range(config.NUM_PLAYERS)]


def reset(sim):
    pool_obj = pool.pool()
    sim.PLAYERS = [player_class.player(pool_obj, i) for i in range(sim.num_players)]
    sim.NUM_DEAD = 0
    sim.player_rewards = [0 for i in range(sim.num_players)]
    # # This starts the game over from the beginning with a fresh set of players.
    # game_round.game_logic


# Batch step
def batch_step(sim, agent, buffers):
    actions_taken = 0
    game_observations = [Observation() for _ in range(config.NUM_PLAYERS)]
    
    while actions_taken < 30:      
        observation_list, previous_action = sim.get_observation(buffers)

        action, policy = agent.batch_policy(observation_list, previous_action)

        rewards = sim.step_function.batch_controller(action, sim.PLAYERS, game_observations)

        for i in range(config.NUM_PLAYERS):
            if sim.PLAYERS[i]:
                local_reward = rewards[sim.PLAYERS[i].player_num] - previous_reward[sim.PLAYERS[i].player_num]
                buffers[sim.PLAYERS[i].player_num].store_replay_buffer(observation_list[sim.PLAYERS[i].player_num],
                                                                       action[sim.PLAYERS[i].player_num],
                                                                       local_reward, policy[sim.PLAYERS[i].player_num])
                previous_reward[sim.PLAYERS[i].player_num] = sim.PLAYERS[i].reward

        actions_taken += 1


# This is the main overarching gameplay method.
# This is going to be implemented mostly in the game_round file under the AI side of things. 
def collect_gameplay_experience(env, agent, buffers):
    observation, info = env.reset()
    terminated = False
    while ~terminated:
        action, policy = agent.batch_policy(observation)  # agent policy that uses the observation and info
        observation_list, rewards, terminated, truncated, info = env.step(action)
        for i in range(config.NUM_PLAYERS):
            if info["players"][i]:
                local_reward = rewards[info["players"][i].player_num] - previous_reward[info["players"][i].player_num]
                buffers[info["players"][i].player_num].\
                    store_replay_buffer(observation_list[info["players"][i].player_num],
                                        action[info["players"][i].player_num], local_reward,
                                        policy[info["players"][i].player_num])
                previous_reward[info["players"][i].player_num] = info["players"][i].reward

    env.close()


def train_model(max_episodes=10000):
    # # Uncomment if you change the size of the input array
    # pool_obj = pool.pool()
    # test_player = player_class.player(pool_obj, 0)
    # shop = pool_obj.sample(test_player, 5)
    # shape = np.array(observation(shop, test_player)).shape
    # register_env()

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    global_agent = TFTNetwork()
    global_buffer = GlobalBuffer()
    trainer = MuZero_trainer.Trainer()
    # agents = [MuZero_agent() for _ in range(game_sim.num_players)]
    train_step = 0
    # global_agent.load_model(0)
    env = gym.make("Simulator/TFT-Set4")

    for episode_cnt in range(1, max_episodes):
        agent = Batch_MCTSAgent(network=global_agent)
        buffers = [ReplayBuffer(global_buffer) for _ in range(config.NUM_PLAYERS)]
        collect_gameplay_experience(env, agent, buffers)

        for i in range(config.NUM_PLAYERS):
            buffers[i].store_global_buffer()
        # Keeping this here in case I want to only update positive rewards
        # rewards = game_round.player_rewards
        while global_buffer.available_batch():
            gameplay_experience_batch = global_buffer.sample_batch()
            trainer.train_network(gameplay_experience_batch, global_agent, train_step, train_summary_writer)
            train_step += 1
        global_agent.save_model(episode_cnt)
        if episode_cnt % 5 == 0:
            game_round.log_to_file_start()
        
        print("Episode " + str(episode_cnt) + " Completed")


def evaluate(agent):
    return 0


def register_env():
    gym.envs.registration.register(
        id="TFT-Set4",
        entry_point="Simulator:tft_simulator"
    )

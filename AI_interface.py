import time
import config
import datetime
import ray
import os
import tensorflow as tf
import gymnasium as gym
import numpy as np
from storage import Storage
from global_buffer import GlobalBuffer
from Models.replay_muzero_buffer import ReplayBuffer
from Simulator.tft_simulator import TFT_Simulator, parallel_env, env as tft_env
from ray.rllib.algorithms.ppo import PPOConfig
from Models import MuZero_trainer
from Models.replay_buffer_wrapper import BufferWrapper
from Models.MuZero_agent_2 import Batch_MCTSAgent, TFTNetwork
from ray.tune.registry import register_env
from ray.rllib.env import PettingZooEnv
from pettingzoo.test import parallel_api_test, api_test
import Simulator.utils as utils


# Can add scheduling_strategy="SPREAD" to ray.remote. Not sure if it makes any difference
@ray.remote(num_gpus=0.12)
class DataWorker(object):
    def __init__(self, rank):
        self.agent_network = TFTNetwork()
        self.prev_actions = [0 for _ in range(config.NUM_PLAYERS)]
        self.rank = rank

    # This is the main overarching gameplay method.
    # This is going to be implemented mostly in the game_round file under the AI side of things.
    def collect_gameplay_experience(self, env, buffers, global_buffer, storage, weights):
        # collect_gameplay_experience took 1635.8837022781372 seconds to finish one episode
        self.agent_network.set_weights(weights)
        agent = Batch_MCTSAgent(self.agent_network)
        while True:
            # Reset the environment
            player_observation = env.reset()
            # This is here to make the input (1, observation_size) for initial_inference
            player_observation = np.asarray(list(player_observation.values()))
            # Used to know when players die and which agent is currently acting
            terminated = {player_id: False for player_id in env.possible_agents}
            # Current action to help with MuZero
            self.prev_actions = ["0" for _ in range(config.NUM_PLAYERS)]

            # While the game is still going on.
            while not all(terminated.values()):
                # Ask our model for an action and policy
                actions, policy = agent.batch_policy(player_observation, list(self.prev_actions))
                step_actions = self.getStepActions(terminated, actions)
                if any(terminated.values()):
                    for player_id, terminate in terminated.items():
                        if terminate and player_id in env.agents:
                            # print("Deleting", player_id)
                            env.agents.remove(player_id)

                # Take that action within the environment and return all of our information for the next player
                next_observation, reward, terminated, _, info = env.step(step_actions)
                # store the action for MuZero
                concat_policy = np.concatenate([
                        policy[0],
                        policy[1],
                        policy[2]
                    ], axis=-1)
                for i, key in enumerate(terminated.keys()):
                    # Store the information in a buffer to train on later.
                    buffers.store_replay_buffer.remote(key, player_observation[i], actions[i], reward[key], concat_policy[i])
                # Set up the observation for the next action
                player_observation = np.asarray(list(next_observation.values()))
                self.prev_actions = actions

            # buffers.rewardNorm.remote()
            buffers.store_global_buffer.remote()
            buffers = BufferWrapper.remote(global_buffer)

            weights = ray.get(storage.get_model.remote())
            agent.network.set_weights(weights)
            self.rank += config.CONCURRENT_GAMES

    def getStepActions(self, terminated, actions):
        step_actions = {}
        i = 0
        for player_id, terminate in terminated.items():
            if not terminate:
                step_actions[player_id] = self.decode_action_to_one_hot(actions[i])
                i += 1
        return step_actions

    def collect_dummy_data(self):
        env = gym.make("TFT_Set4-v0", env_config={})
        while True:
            _, _ = env.reset()
            terminated = False
            t = time.time_ns()
            while not terminated:
                # agent policy that uses the observation and info
                action = np.random.randint(low=0, high=[10, 5, 9, 10, 7, 4, 7, 4], size=[8, 8])
                self.prev_actions = action
                observation_list, rewards, terminated, truncated, info = env.step(action)
            print("A game just finished in time {}".format(time.time_ns() - t))

    def decode_action_to_one_hot(self, str_action):
        num_items = str_action.count("_")
        split_action = str_action.split("_")
        element_list = [0,0,0]
        for i in range(num_items+1):
            element_list[i] = int(split_action[i])
        
        decoded_action = np.zeros(config.ACTION_DIM[0] + config.ACTION_DIM[1] + config.ACTION_DIM[2])
        decoded_action[0:6] = utils.one_hot_encode_number(element_list[0], 6)
        
        if element_list[0] == 1:
            decoded_action[6:11] = utils.one_hot_encode_number(element_list[1], 5)

        if element_list[0] == 2:
            decoded_action[6:44] = utils.one_hot_encode_number(element_list[1], 38) + utils.one_hot_encode_number(element_list[2], 38)
        
        if element_list[0] == 3:
            decoded_action[6:44] = utils.one_hot_encode_number(element_list[1], 38)
            decoded_action[44:54] = utils.one_hot_encode_number(element_list[2], 10)
        return decoded_action

    def evaluate_agents(self, env, storage):
        agents = {r: Batch_MCTSAgent(TFTNetwork())
                  for r in range(config.NUM_PLAYERS)}
        # agents["player_1"].network.tft_load_model(10)
        agents[2].network.tft_load_model(50)
        agents[3].network.tft_load_model(100)
        agents[4].network.tft_load_model(150)
        agents[5].network.tft_load_model(200)
        agents[6].network.tft_load_model(250)
        agents[7].network.tft_load_model(300)

        while True:
            # Reset the environment
            player_observation = env.reset()
            # This is here to make the input (1, observation_size) for initial_inference
            player_observation = np.asarray(
                list(player_observation.values()))
            # Used to know when players die and which agent is currently acting
            terminated = {
                player_id: False for player_id in env.possible_agents}
            # Current action to help with MuZero
            placements = {
                player_id: 0 for player_id in env.possible_agents}
            current_position = 7
            info = {player_id: {"player_won": False}
                    for player_id in env.possible_agents}
            # While the game is still going on.
            while not all(terminated.values()):
                # Ask our model for an action and policy
                actions = {agent: "0" for agent in env.possible_agents}
                for i, key, in enumerate(env.possible_agents):
                    action, _ = agents[key].policy(player_observation[i], actions[key])
                    actions[key] = action
                step_actions = self.getStepActions(terminated, actions)
                if any(terminated.values()):
                    for player_id, terminate in terminated.items():
                        if terminate and player_id in env.agents:
                            # print("Deleting", player_id)
                            env.agents.remove(player_id)
                            placements[player_id] = current_position
                            current_position -= 1


                # Take that action within the environment and return all of our information for the next player
                next_observation, reward, terminated, _, info = env.step(step_actions)

                # Set up the observation for the next action
                player_observation = np.asarray(list(next_observation.values()))

            for key, value in info.items():
                if value["player_won"]:
                    placements[key] = 0
            storage.record_placements.remote(placements)
            print("recorded places {}".format(placements))
            self.rank += config.CONCURRENT_GAMES


class AIInterface:

    def __init__(self):
        ...

    def train_model(self, starting_train_step=0):
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        gpus = tf.config.list_physical_devices('GPU')
        ray.init(num_gpus=len(gpus), num_cpus=20)

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        train_step = starting_train_step
        tf.config.optimizer.set_jit(True)

        global_buffer = GlobalBuffer.remote()

        trainer = MuZero_trainer.Trainer()
        storage = Storage.remote(train_step)

        env = parallel_env()

        buffers = [BufferWrapper.remote(global_buffer)
                   for _ in range(config.CONCURRENT_GAMES)]

        weights = ray.get(storage.get_target_model.remote())
        workers = []
        data_workers = [DataWorker.remote(rank) for rank in range(config.CONCURRENT_GAMES)]
        for i, worker in enumerate(data_workers):
            workers.append(worker.collect_gameplay_experience.remote(env, buffers[i], global_buffer,
                                                                     storage, weights))
            time.sleep(2)

        # ray.get(workers)
        global_agent = TFTNetwork()
        global_agent_weights = ray.get(storage.get_target_model.remote())
        global_agent.set_weights(global_agent_weights)

        while True:
            if ray.get(global_buffer.available_batch.remote()):
                gameplay_experience_batch = ray.get(global_buffer.sample_batch.remote())
                trainer.train_network(gameplay_experience_batch, global_agent, train_step, train_summary_writer)
                storage.set_target_model.remote(global_agent.get_weights())
                train_step += 1
                if train_step % 50 == 0:
                    storage.set_model.remote()
                    global_agent.tft_save_model(train_step)

    def collect_dummy_data(self):
        env = parallel_env()
        while True:
            _ = env.reset()
            terminated = {player_id: False for player_id in env.possible_agents}
            t = time.time_ns()
            while not all(terminated.values()):
                # agent policy that uses the observation and info
                action = np.random.randint(low=0, high=[10, 5, 9, 10, 7, 4, 7, 4], size=[8, 8])
                step_actions = {}
                i = 0
                for player_id, terminate in terminated.items():
                    if not terminate:
                        step_actions[player_id] = action[i]
                        i += 1
                observation_list, rewards, terminated, truncated, info = env.step(step_actions)
            print("A game just finished in time {}".format(time.time_ns() - t))

    def testEnv(self):
        local_env = parallel_env()
        parallel_api_test(local_env, num_cycles=100000)
        # second_env = tft_env()
        # api_test(second_env, num_cycles=100000)

    # function looks stupid as is right now, but should remain this way
    # for potential future abstractions

    def env_creator(self, cfg):
        return TFT_Simulator(cfg)

    def evaluate(self):
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        gpus = tf.config.list_physical_devices('GPU')
        ray.init(num_gpus=len(gpus), num_cpus=16)
        storage = Storage.remote(0)

        env = parallel_env()

        workers = []
        data_workers = [DataWorker.remote(rank) for rank in range(config.CONCURRENT_GAMES)]
        for i, worker in enumerate(data_workers):
            workers.append(worker.evaluate_agents.remote(env, storage))
            time.sleep(1)

        ray.get(workers)
        while True:
            time.sleep(10000)
            print("good luck getting past this")



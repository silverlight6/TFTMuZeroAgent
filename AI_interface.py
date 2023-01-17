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
from Models.A3C_Agent import A3C_Agent


# Can add scheduling_strategy="SPREAD" to ray.remote. Not sure if it makes any difference
@ray.remote(num_gpus=0.25)
class DataWorker(object):
    def __init__(self, rank):
        if config.MODEL == "MuZero":
            self.agent_network = TFTNetwork()
            self.prev_actions = [0 for _ in range(config.NUM_PLAYERS)]
        elif config.MODEL == "A3C":
            self.agent_network = A3C_Agent(config.INPUT_SHAPE)
            self.prev_actions = [[0 for _ in range(len(config.ACTION_8D_DIM))] for _ in range(config.NUM_PLAYERS)]
        else:
            self.agent_network = TFTNetwork()
            self.prev_actions = [0 for _ in range(config.NUM_PLAYERS)]

        self.rank = rank

    # This is the main overarching gameplay method.
    # This is going to be implemented mostly in the game_round file under the AI side of things.
    def collect_gameplay_experience(self, env, buffers, global_buffer, storage, weights):
        if config.MODEL == "MuZero":
            self.agent_network.set_weights(weights)
            agent = Batch_MCTSAgent(self.agent_network)
        elif config.MODEL == "A3C":
            self.agent_network.a3c_net.set_weights(weights)
            agent = self.agent_network
        else:
            self.agent_network.set_weights(weights)
            agent = Batch_MCTSAgent(self.agent_network)

        while True:
            # Reset the environment
            player_observation = env.reset()
            # This is here to make the input (1, observation_size) for initial_inference
            player_observation = np.asarray(list(player_observation.values()))
            # Used to know when players die and which agent is currently acting
            terminated = {player_id: False for player_id in env.possible_agents}

            # While the game is still going on.
            while not all(terminated.values()):
                # Ask our model for an action and policy
                actions, policy = agent.batch_policy(player_observation, self.prev_actions)
                step_actions = self.getStepActions(terminated, actions)

                # Take that action within the environment and return all of our information for the next player
                next_observation, reward, terminated, _, info = env.step(step_actions)
                # store the action for MuZero
                for i, key in enumerate(terminated.keys()):
                    # Store the information in a buffer to train on later.
                    buffers.store_replay_buffer.remote(key, player_observation[i], actions[i], reward[key], policy[i])

                # Set up the observation for the next action
                player_observation = np.asarray(list(next_observation.values()))
                # Update previous actions for the next round
                self.prev_actions = actions
                # Make sure it is of the same length as the observation when a player dies
                self.prev_actions = np.delete(self.prev_actions, np.argwhere(list(terminated.values())), axis=0)

            if config.MODEL == "MuZero":
                buffers.rewardNorm.remote()
                buffers.store_global_buffer.remote()
            buffers = BufferWrapper.remote(global_buffer)

            weights = ray.get(storage.get_model.remote())
            if config.MODEL == "MuZero":
                agent.network.set_weights(weights)
            elif config.MODEL == "A3C":
                agent.a3c_net.set_weights(weights)
            else:
                agent.network.set_weights(weights)

            # Current action to help with MuZero
            if config.MODEL == "MuZero":
                self.prev_actions = [0 for _ in range(config.NUM_PLAYERS)]
            elif config.MODEL == "A3C":
                self.prev_actions = [[0 for _ in range(len(config.ACTION_8D_DIM))] for _ in range(config.NUM_PLAYERS)]
            else:
                self.prev_actions = [0 for _ in range(config.NUM_PLAYERS)]
            self.rank += config.CONCURRENT_GAMES

    def test_collect_gameplay_experience(self, env, agent, buffers):
        observation, info = env.reset()
        terminated = False
        while not terminated:
            # agent policy that uses the observation and info
            action, policy = agent.batch_policy(observation, self.prev_actions)
            self.prev_actions = action
            observation_list, rewards, terminated, _, info = env.step(np.asarray(action))
            for i in range(config.NUM_PLAYERS):
                if info["players"][i]:
                    local_reward = rewards[info["players"][i].player_num] - \
                                   self.prev_reward[info["players"][i].player_num]
                    buffers[info["players"][i].player_num]. \
                        store_replay_buffer(observation_list[info["players"][i].player_num],
                                            action[info["players"][i].player_num], local_reward,
                                            policy[info["players"][i].player_num])
                    self.prev_reward[info["players"][i].player_num] = info["players"][i].reward

    def getStepActions(self, terminated, actions):
        step_actions = {}
        i = 0
        for player_id, terminate in terminated.items():
            if not terminate:
                step_actions[player_id] = actions[i]
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


class AIInterface:

    def __init__(self):
        self.env = parallel_env()

    def train_muzero_model(self, starting_train_step=0):
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        gpus = tf.config.list_physical_devices('GPU')
        ray.init(num_gpus=len(gpus), num_cpus=16)

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        global_buffer = GlobalBuffer.remote()

        trainer = MuZero_trainer.Trainer()
        storage = Storage.remote(starting_train_step)

        buffers = [BufferWrapper.remote(global_buffer) for _ in range(config.CONCURRENT_GAMES)]

        weights = ray.get(storage.get_target_model.remote())
        workers = []
        data_workers = [DataWorker.remote(rank) for rank in range(config.CONCURRENT_GAMES)]
        for i, worker in enumerate(data_workers):
            workers.append(worker.collect_gameplay_experience.remote(self.env, buffers[i], global_buffer,
                                                                     storage, weights))
            time.sleep(1)

        global_agent = TFTNetwork()
        global_agent_weights = ray.get(storage.get_target_model.remote())
        global_agent.set_weights(global_agent_weights)

        while True:
            if ray.get(global_buffer.available_batch.remote()):
                gameplay_experience_batch = ray.get(global_buffer.sample_batch.remote())
                trainer.train_network(gameplay_experience_batch, global_agent, train_step, train_summary_writer)
                storage.set_target_model.remote(global_agent.get_weights())
                train_step += 1
                if train_step % 100 == 0:
                    storage.set_model.remote()
                    global_agent.tft_save_model(train_step)

    def train_a3c_model(self):
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        gpus = tf.config.list_physical_devices('GPU')
        ray.init(num_gpus=len(gpus), num_cpus=16)

        train_step = 100

        global_buffer = GlobalBuffer.remote()

        storage = Storage.remote(train_step)

        global_agent = A3C_Agent(config.INPUT_SHAPE)
        global_agent_weights = ray.get(storage.get_target_model.remote())
        global_agent.a3c_net.set_weights(global_agent_weights)

        buffers = [BufferWrapper.remote(global_buffer) for _ in range(config.CONCURRENT_GAMES)]

        weights = ray.get(storage.get_target_model.remote())
        workers = []
        data_workers = [DataWorker.remote(rank) for rank in range(config.CONCURRENT_GAMES)]
        for i, worker in enumerate(data_workers):
            workers.append(worker.collect_gameplay_experience.remote(self.env, buffers[i], global_buffer,
                                                                     storage, weights))
            time.sleep(1)

        while True:
            if ray.get(global_buffer.available_batch.remote()):
                gameplay_experience_batch = ray.get(global_buffer.sample_a3c_batch.remote())
                global_agent.train_step(gameplay_experience_batch)
                storage.set_target_model.remote(global_agent.a3c_net.get_weights())
                train_step += 1
                if train_step % 100 == 0:
                    storage.set_model.remote()
                    global_agent.tft_save_model(train_step)

    def test_train_model(self, max_episodes=10000):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        global_agent = TFTNetwork()
        global_buffer = GlobalBuffer.remote()
        trainer = MuZero_trainer.Trainer()
        train_step = 0
        # global_agent.load_model(0)
        dataWorker = DataWorker().remote()

        for episode_cnt in range(1, max_episodes):
            agent = Batch_MCTSAgent(network=global_agent)
            buffers = [BufferWrapper.remote(global_buffer) for _ in range(config.NUM_PLAYERS)]
            dataWorker.test_collect_gameplay_experience(self.env, agent, buffers)

            for i in range(config.NUM_PLAYERS):
                buffers[i].store_global_buffer()
            while global_buffer.available_batch():
                gameplay_experience_batch = global_buffer.sample_batch()
                trainer.train_network(gameplay_experience_batch, global_agent, train_step, train_summary_writer)
                train_step += 1
            global_agent.save_model(episode_cnt)
            print("Episode " + str(episode_cnt) + " Completed")

    def collect_dummy_data(self):
        dataWorker = DataWorker()
        dataWorker.collect_dummy_data()

    def PPO_algorithm(self):
        # register our environment, we have no config parameters
        register_env('tft-set4-v0', lambda local_config: PettingZooEnv(self.env_creator(local_config)))

        # Create an RLlib Algorithm instance from a PPOConfig object.
        cfg = (
            PPOConfig().environment(
                # Env class to use (here: our gym.Env sub-class from above).
                env='tft-set4-v0',
                env_config={},
                observation_space=gym.spaces.Box(low=-5.0, high=5.0, shape=(config.OBSERVATION_SIZE,), dtype=np.float64),
                action_space=gym.spaces.Discrete(config.ACTION_DIM)
            )
            .rollouts(num_rollout_workers=1)
            .framework("tf2")
            .training(model={"fcnet_hiddens": [256, 256]})
            .evaluation(evaluation_num_workers=1, evaluation_interval=50)
        )
        # Construct the actual (PPO) algorithm object from the config.
        algo = cfg.build()

        for i in range(100):
            results = algo.train()
            print(f"Iter: {i}; avg. reward={results['episode_reward_mean']}")

        algo.evaluate()  # 4. and evaluate it.

    def evaluate(self, agent):
        return 0

    def testEnv(self):
        local_env = parallel_env()
        parallel_api_test(local_env, num_cycles=100000)
        second_env = tft_env()
        api_test(second_env, num_cycles=100000)

    # function looks stupid as is right now, but should remain this way
    # for potential future abstractions

    def env_creator(self, cfg):
        return TFT_Simulator(cfg)



import time
import config
import ray
import os
import gymnasium as gym
import numpy as np
import torch
from Concurrency.storage import Storage
from Simulator.tft_simulator import TFT_Simulator, parallel_env, env as tft_env
from ray.rllib.algorithms.ppo import PPOConfig
from Models.replay_buffer_wrapper import BufferWrapper
from ray.tune.registry import register_env
from ray.rllib.env import PettingZooEnv
from pettingzoo.test import parallel_api_test, api_test
from Models.MuZero_torch_agent import MuZeroNetwork as TFTNetwork
from Models.Muzero_default_agent import MuZeroDefaultNetwork as DefaultNetwork
from Concurrency.data_worker import DataWorker
from Concurrency.training_manager import TrainingManager


"""
Description - Highest level class for concurrent training. Called from main.py
"""
class AIInterface:

    def __init__(self):
        ...

    '''
    Description - Global train model method. This is what gets called from main.
    Inputs - starting_train_step: int
                Checkpoint number to load. If 0, a fresh model will be created.
    '''
    def train_torch_model(self, starting_train_step=0):
        gpus = torch.cuda.device_count()
        with ray.init(num_gpus=gpus, num_cpus=config.NUM_CPUS, namespace="TFT_AI"):
            train_step = starting_train_step

            workers = []
            data_workers = [DataWorker.remote(rank) for rank in range(config.CONCURRENT_GAMES)]
            storage = Storage.remote(train_step)
            if config.CHAMP_DECIDER:
                global_agent = DefaultNetwork()
            else:
                global_agent = TFTNetwork()

            global_agent_weights = ray.get(storage.get_target_model.remote())
            global_agent.set_weights(global_agent_weights)
            global_agent.to(config.DEVICE)

            training_manager = TrainingManager(global_agent, storage)

            # Keeping this line commented because this tells us the number of parameters that our current model has.
            # total_params = sum(p.numel() for p in global_agent.parameters())

            env = parallel_env()

            buffers = [BufferWrapper.remote()
                       for _ in range(config.CONCURRENT_GAMES)]

            weights = ray.get(storage.get_target_model.remote())

            for i, worker in enumerate(data_workers):
                if config.CHAMP_DECIDER:
                    workers.append(worker.collect_default_experience.remote(env, buffers[i], training_manager,
                                                                            storage, weights))
                else:
                    workers.append(worker.collect_gameplay_experience.remote(env, buffers[i], training_manager,
                                                                             storage, weights))
                time.sleep(0.5)

            training_manager.loop(storage, train_step)

            # This may be able to be ray.wait(workers). Here so we can keep all processes alive.
            # ray.get(storage)
            ray.get(workers)

    '''
    Description - Method used for testing the simulator. It does not call any AI and generates random actions from 
        numpy. Tests how fast the simulator is and if there are any bugs that can be caught via multiple runs.
    '''
    def collect_dummy_data(self):
        env = parallel_env()
        while True:
            _ = env.reset()
            terminated = {player_id: False for player_id in env.possible_agents}
            t = time.time_ns()
            while not all(terminated.values()):
                # agent policy that uses the observation and info
                action = {
                    agent: env.action_space(agent).sample()
                    for agent in env.agents
                    if (agent in terminated and not terminated[agent])
                }
                observation_list, rewards, terminated, truncated, info = env.step(action)
            print("A game just finished in time {}".format(time.time_ns() - t))

    '''
    Description - The PPO implementation for the TFT project. This is an alternative to our MuZero model.
    '''
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

    '''
    Description - The global side to the evaluator. Creates a set of workers to test a series of agents.
    '''
    def evaluate(self):
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        # gpus = tf.config.list_physical_devices('GPU')
        ray.init(num_gpus=4, num_cpus=16)
        storage = Storage.remote(0)

        env = parallel_env()

        workers = []
        data_workers = [DataWorker.remote(rank) for rank in range(config.CONCURRENT_GAMES)]
        for i, worker in enumerate(data_workers):
            workers.append(worker.evaluate_agents.remote(env, storage))
            time.sleep(1)

        ray.get(workers)

    '''
    Description - PettingZoo's api tests for the simulator.
    '''
    def testEnv(self):
        raw_env = tft_env()
        api_test(raw_env, num_cycles=100000)
        local_env = parallel_env()
        parallel_api_test(local_env, num_cycles=100000)

    '''
    Description - Creates the TFT environment for the PPO model.
    '''
    def env_creator(self, cfg):
        return TFT_Simulator(cfg)

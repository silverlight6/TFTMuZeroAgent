import time
import config
import ray
import os
import torch
from Concurrency.storage import Storage
from Simulator.tft_simulator import parallel_env
from Models.replay_buffer_wrapper import BufferWrapper
from Models.MuZero_torch_agent import MuZeroNetwork as TFTNetwork
from Models.Muzero_default_agent import MuZeroDefaultNetwork as DefaultNetwork
from Models.rllib_ppo import PPO_Models
from Concurrency.data_worker import DataWorker
from Concurrency.training_manager import TrainingManager
from Concurrency.queue_storage import QueueStorage



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
                workers.append(worker.collect_gameplay_experience.remote(env, buffers[i], training_manager,
                                                                         storage, weights))
                time.sleep(0.5)

            training_manager.loop(storage, train_step)

            # This may be able to be ray.wait(workers). Here so we can keep all processes alive.
            # ray.get(storage)
            ray.get(workers)

    def train_guide_model(self, starting_train_step=0):
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
            positioning_storage = QueueStorage()
            item_storage = QueueStorage()

            ppo_models = PPO_Models.remote()

            weights = ray.get(storage.get_target_model.remote())

            for i, worker in enumerate(data_workers):
                workers.append(worker.collect_default_experience.remote(env, buffers[i], training_manager, storage,
                                                                        weights, item_storage, positioning_storage))
                time.sleep(0.5)

            training_manager.loop(storage, train_step)

            # Tests for the position and item environment
            # test_envs = DataWorker.remote(0)
            # workers.append(test_envs.test_position_item_simulators.remote(positioning_storage, item_storage))

            workers.append(ppo_models.PPO_position_algorithm.remote(positioning_storage))
            workers.append(ppo_models.PPO_item_algorithm.remote(item_storage))

            # This may be able to be ray.wait(workers). Here so we can keep all processes alive.
            # ray.get(storage)
            ray.get(workers)

    def collect_dummy_data(self):
        """
        Method used for testing the simulator. It does not call any AI and generates random actions from numpy.
        Tests how fast the simulator is and if there are any bugs that can be caught via multiple runs.
        """
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

    def evaluate(self):
        """
        The global side to the evaluator. Creates a set of workers to test a series of agents.
        """
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

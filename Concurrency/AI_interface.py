import random
import time
import ray
import os
import torch
import config

from Concurrency.storage import Storage
from Simulator.tft_simulator import parallel_env, TFTConfig
from Simulator.observation.vector.observation import ObservationVector
from Models.replay_buffer_wrapper import BufferWrapper
from Models.MuZero_torch_agent import MuZeroNetwork as TFTNetwork
from Models.Muzero_default_agent import MuZeroDefaultNetwork as DefaultNetwork
from Models.rllib_ppo import PPO_Position_Model, PPO_Item_Model, Base_PPO_Position_Model
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
    def train_torch_model(self) -> None:
        gpus = torch.cuda.device_count()
        with ray.init(num_gpus=gpus, num_cpus=config.NUM_CPUS, namespace="TFT_AI"):
            train_step = config.STARTING_EPISODE

            workers = []
            model_config = config.ModelConfig()
            data_workers = [DataWorker.remote(rank, model_config) for rank in range(config.CONCURRENT_GAMES)]
            storage = Storage.remote(train_step)
            if config.CHAMP_DECIDER:
                global_agent = DefaultNetwork(model_config)
            else:
                global_agent = TFTNetwork(model_config)

            global_agent_weights = ray.get(storage.get_target_model.remote())
            global_agent.set_weights(global_agent_weights)
            global_agent.to(config.DEVICE)

            training_manager = TrainingManager(global_agent, storage)

            # Keeping this line commented because this tells us the number of parameters that our current model has.
            # total_params = sum(p.numel() for p in global_agent.parameters())

            tftConfig = TFTConfig(observation_class=ObservationVector)
            env = parallel_env(tftConfig)

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

    def train_guide_model(self) -> None:
        gpus = torch.cuda.device_count()
        with ray.init(num_gpus=gpus, num_cpus=config.NUM_CPUS, namespace="TFT_AI"):
            train_step = config.STARTING_EPISODE

            workers = []
            modelConfig = config.ModelConfig()
            data_workers = [DataWorker.remote(rank, modelConfig) for rank in range(config.CONCURRENT_GAMES)]
            storage = Storage.remote(train_step)

            if config.CHAMP_DECIDER:
                global_agent = DefaultNetwork(modelConfig)
            else:
                global_agent = TFTNetwork(modelConfig)

            global_agent_weights = ray.get(storage.get_target_model.remote())
            global_agent.set_weights(global_agent_weights)
            global_agent.to(config.DEVICE)

            training_manager = TrainingManager(global_agent, storage)

            # Keeping this line commented because this tells us the number of parameters that our current model has.
            # total_params = sum(p.numel() for p in global_agent.parameters())

            tftConfig = TFTConfig(observation_class=ObservationVector)
            env = parallel_env(tftConfig)

            buffers = [BufferWrapper.remote()
                       for _ in range(config.CONCURRENT_GAMES)]
            positioning_storage = QueueStorage(name="position")
            item_storage = QueueStorage(name="item")

            ppo_position_model = PPO_Position_Model.remote(positioning_storage)
            ppo_item_model = PPO_Item_Model.remote(item_storage)

            weights = ray.get(storage.get_target_model.remote())

            for i, worker in enumerate(data_workers):
                workers.append(worker.collect_default_experience.remote(env, buffers[i], training_manager, storage,
                                                                        weights, item_storage, positioning_storage,
                                                                        ppo_position_model, ppo_item_model))
                time.sleep(0.5)

            training_manager.loop(storage, train_step)

            # Tests for the position and item environment
            # test_envs = DataWorker.remote(0)
            # workers.append(test_envs.test_position_item_simulators.remote(positioning_storage, item_storage))

            ppo_item_model.train_item_model.remote()
            ppo_position_model.train_position_model.remote()

            # This may be able to be ray.wait(workers). Here so we can keep all processes alive.
            # ray.get(storage)
            ray.get(workers)

    def position_ppo_tune(self):
        from ray import train, tune
        from ray.tune.search.bohb.bohb_search import TuneBOHB
        from ray.tune.schedulers.hb_bohb import HyperBandForBOHB

        local_path = config.PATH
        print(f"PATH file://{local_path}/ray_results")

        positioning_storage = QueueStorage(name="position")
        ppo_position_model = Base_PPO_Position_Model(positioning_storage)

        search_space = {
            "lambda": tune.uniform(0.9, 1.0),
            "lr": tune.choice([5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6]),
            "clip_param": tune.uniform(0.1, 0.5),
            "num_sgd_iter": tune.randint(1, 30),
            "num_heads": tune.choice([1, 2, 4, 8, 16, 32]),
            "hidden_size": tune.choice([64, 128, 256, 512, 1024]),
            "entropy_coeff": tune.loguniform(1e-4, 1e-1)
        }

        ppo_position_config = ppo_position_model.PPO_position_algorithm(search_space)
        algo = TuneBOHB()
        bohb = HyperBandForBOHB(time_attr="training_iteration", max_t=100)

        stopping_criteria = {"training_iteration": 50, "episode_reward_mean": 1}

        tuner = tune.Tuner(
            ppo_position_config.algo_class,
            param_space=ppo_position_config,
            tune_config=tune.TuneConfig(
                metric="info/learner/default_policy/learner_stats/total_loss",
                mode="min",
                num_samples=8,
                max_concurrent_trials=1,
                search_alg=algo,
                scheduler=bohb,
            ),
            run_config=train.RunConfig(stop=stopping_criteria, storage_path=f"file://{local_path}/ray_results")
        )

        results = tuner.fit()

        # algo.save("./my-checkpoint.pkl")

        best_trial = results.get_best_result(metric="episode_reward_mean", mode="max")
        best_config = best_trial.config
        print("FINAL CONFIG")
        print(best_config)
        time.sleep(5)

    def position_ppo_testing(self):
        gpus = torch.cuda.device_count()
        print(f"gpu_count START = {gpus}")
        with ray.init(num_gpus=gpus, num_cpus=config.NUM_CPUS, dashboard_host="0.0.0.0"):

            positioning_storage = QueueStorage(name="position")
            ppo_position_model = Base_PPO_Position_Model(positioning_storage)

            search_space_start = {
                "lambda": 0.95,
                "gamma": 0.95,
                "lr": 5e-4,
                "clip_param": 0.2,
                "num_sgd_iter": 15,
            }

            ppo_position_config = ppo_position_model.PPO_position_algorithm(search_space_start)
            ppo_position_model.train_position_model(ppo_position_config)

    def ppo_checkpoint_test(self):
        save_path = "/home/silver/TFTacticsAI/TFTAI/ray_results"

        positioning_storage = QueueStorage(name="position")
        ppo_position_model = Base_PPO_Position_Model(positioning_storage)

        search_space = {
            "lambda": 1.0,
            "gamma": 1.0,
            "lr": 1e-3,
            "clip_param": 0.2,
            "num_sgd_iter": 5,
        }

        ppo_position_config = ppo_position_model.PPO_position_algorithm(search_space)
        PPO_Alg = ppo_position_config.algo_class(config=ppo_position_config)
        PPO_Alg.save_checkpoint(save_path)

        new_model = PPO_Alg.load_checkpoint(save_path)

        time.sleep(5)


    def collect_dummy_data(self) -> None:
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

    def evaluate(self, config) -> None:
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

import ray
import os
from ray import tune, train
from ray.rllib.algorithms.ppo import PPOConfig
from Simulator.tft_position_simulator import TFT_Position_Simulator
from Simulator.tft_item_simulator import TFT_Item_Simulator

@ray.remote(num_gpus=0.35)
class PPO_Position_Model:
    def __init__(self, position_buffer):
        self.position_model = self.PPO_position_algorithm(position_buffer)

    def PPO_position_algorithm(self, position_buffer):
        """
        The PPO implementation for the TFT project. This is an alternative to our MuZero model.
        """

        position_gym = self.position_env(position_buffer)
        # ray.rllib.utils.check_env(position_gym)
        # register our environment, we have no config parameters
        ray.tune.registry.register_env('TFT_Position_Simulator_s4_v0', lambda local_config: position_gym)

        # Create an RLlib Algorithm instance from a PPOConfig object.
        cfg = (
            PPOConfig().environment(
                # Env class to use (here: our gym.Env sub-class from above).
                env='TFT_Position_Simulator_s4_v0',
                env_config={},
                observation_space=position_gym.observation_space,
                action_space=position_gym.action_space
            )
            .resources(num_gpus=1)
            .rollouts(num_rollout_workers=1,
                      num_envs_per_worker=1)
            .framework("torch")
            .training(train_batch_size=4096,
                      lambda_=0.95,
                      gamma=0.95,
                      lr=0.001)
            .evaluation(evaluation_num_workers=1, evaluation_interval=5)
        )
        # Construct the actual (PPO) algorithm object from the config.
        # algo = cfg.build()

        print("RETURNING POSITION ALGORITHM")
        return cfg

    def fetch_model(self):
        return self.position_model

    def train_position_model(self):
        print("TRAINING POSITION ALGORITHM")
        i = 0
        while True:
            base_path = os.path.join("~/TFTacticsAI/TFTAI", "positionModel")
            if os.path.isdir(base_path):
                print("saved side")
                results = tune.Tuner.restore(base_path, trainable="PPO").fit()
            else:
                print("new model")
                results = tune.Tuner(
                    "PPO",
                    run_config=train.RunConfig(stop={"training_iteration": 100000},
                                               storage_path="~/TFTacticsAI/TFTAI/",
                                               name="positionModel",
                                               checkpoint_config=train.CheckpointConfig(checkpoint_frequency=2)
                                               ),
                    param_space=self.position_model.to_dict(),
                ).fit()
            i += 1
            print(f"Iter: {i}; avg. reward={results}")

    def position_env(self, buffer):
        """
        Creates the TFT environment for the PPO model.
        """
        return TFT_Position_Simulator(buffer)

@ray.remote(num_gpus=0.35)
class PPO_Item_Model:
    def __init__(self, item_buffer):
        self.item_model = self.PPO_item_algorithm(item_buffer)

    def PPO_item_algorithm(self, item_buffer):
        """
        The PPO implementation for the TFT project. This is an alternative to our MuZero model.
        """
        item_gymnasium = self.item_env(item_buffer)
        # register our environment, we have no config parameters
        ray.tune.registry.register_env('TFT_Item_Simulator_s4_v0', lambda local_config: item_gymnasium)

        # Create an RLlib Algorithm instance from a PPOConfig object.
        cfg = (
            PPOConfig().environment(
                # Env class to use (here: our gym.Env sub-class from above).
                env='TFT_Item_Simulator_s4_v0',
                env_config={},
                observation_space=item_gymnasium.observation_space,
                action_space=item_gymnasium.action_space
            )
            .rollouts(num_rollout_workers=6,
                      num_envs_per_worker=5)
            .framework("torch")
            .training(train_batch_size=4096,
                      lambda_=0.95,
                      gamma=0.95,
                      learning_rate=0.001)
            .evaluation(evaluation_num_workers=1, evaluation_interval=50)
        )
        cfg.environment(disable_env_checking=True)
        # Construct the actual (PPO) algorithm object from the config.
        algo = cfg.build()

        print("RETURNING ITEM ALGORITHM")

        return algo

    def fetch_model(self):
        return self.item_model

    def train_item_model(self):
        print("TRAINING ITEM ALGORITHM")
        i = 0
        while True:
            results = self.item_model.train()
            i += 1
            print(f"Iter: {i}; avg. reward={results['episode_reward_mean']}")

    def item_env(self, buffer):
        """
        Creates the TFT environment for the PPO model.
        """
        return TFT_Item_Simulator(buffer)

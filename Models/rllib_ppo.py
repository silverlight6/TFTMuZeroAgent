import ray
import config
import torch
import os

from Models.action_mask_catalog import ActionMaskCatalog
from Models.action_mask_env_runner import ActionMaskEnvRunner
from Models.action_mask_model import TorchActionMaskModel
from Models.action_mask_rlm import TorchActionMaskRLM
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
# from ray.rllib.agents.callbacks import DefaultCallbacks
# from ray.tune import logger
from ray.tune.logger import pretty_print
from ray.rllib.models import ModelCatalog
from Simulator.tft_item_simulator import TFT_Item_Simulator
from Simulator.tft_vector_simulator import TFT_Vector_Pos_Simulator as vector_env



@ray.remote(num_gpus=2)
class PPO_Position_Model:
    def __init__(self, position_buffer):
        self.position_model = self.PPO_position_algorithm(position_buffer)

    def PPO_position_algorithm(self, position_buffer):
        """
        The PPO implementation for the TFT project. This is an alternative to our MuZero model.
        """

        # Number of environments to sample in one thread in the vector gym.
        num_envs = 64

        # Register the environment into ray's memory so we can access it later.
        ModelCatalog.register_custom_model('action_mask_model', TorchActionMaskModel)

        # Build the config
        base_config = config.ModelConfig()

        # Create an RLlib Algorithm instance from a PPOConfig object.
        cfg = (
            PPOConfig().environment(
                # Env class to use (here: our gym.Env sub-class from above).
                env=vector_env,
                # Provide a data generator so that it can be run in parallel with the main game learner
                # It will not use the buffer if there is nothing in the buffer.
                env_config={"data_generator": position_buffer},
                disable_env_checking=True
            )
            # Custom environment runner. It is the equivalent of both the remote worker and singleAgentEnvRunner in Ray
            .rollouts(env_runner_cls=ActionMaskEnvRunner,
                      # Number of parallel threads.
                      num_rollout_workers=16,
                      num_envs_per_worker=num_envs,
                      remote_worker_envs=False)
            .experimental(_enable_new_api_stack=False,
                          _disable_preprocessor_api=True)
            # Custom RL_Module to allow for training and action masks
            .rl_module(rl_module_spec=SingleAgentRLModuleSpec(module_class=TorchActionMaskRLM,
                                                              catalog_class=ActionMaskCatalog))
            .resources(num_gpus=2,
                       # This number times the num_rollout_workers has to be less than the num_gpus
                       num_cpus_per_worker=1.5,
                       num_gpus_per_worker=0.1)

            .framework("torch")
            # Custom model to allow for multiple head observation space and action masks
            .training(model={"custom_model": "action_mask_model",
                             "custom_model_config": {"hidden_state_size": base_config.HIDDEN_STATE_SIZE // 2,
                                                     "num_hidden_layers": base_config.N_HEAD_HIDDEN_LAYERS}, },
                      train_batch_size=512,
                      sample_batch_size=128,
                      sgd_minibatch_size=512,
                      lambda_=0.95,
                      kl_coeff=0.5,
                      gamma=0.95,
                      observation_filter="NoFilter",
                      batch_mode="complete_episodes",
                      lr=0.001)
            .evaluation(evaluation_num_workers=1,
                        evaluation_interval=5,
                        enable_async_evaluation=True)
        )
        # Construct the actual (PPO) algorithm object from the config.
        self.position_model = cfg.build()
        return self.position_model

    def fetch_model(self):
        return self.position_model

    def train_position_model(self):
        while True:
            result = self.position_model.train()
            # This is what lets us know how our model is doing
            print(pretty_print(result))
            save_result = self.position_model.save()
            path_to_checkpoint = save_result.checkpoint.path
            print(
                "An Algorithm checkpoint has been created inside directory: "
                f"'{path_to_checkpoint}'."
            )


# This model is not use set up to use.
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

        ModelCatalog.register_custom_model('action_mask_model', TorchActionMaskModel)

        rlm_class = TorchActionMaskRLM
        rlm_spec = SingleAgentRLModuleSpec(module_class=rlm_class)

        base_config = config.ModelConfig()

        # Create an RLlib Algorithm instance from a PPOConfig object.
        cfg = (
            PPOConfig().environment(
                # Env class to use (here: our gym.Env sub-class from above).
                env='TFT_Item_Simulator_s4_v0',
                env_config={},
                disable_env_checking=True
            )
            .experimental(_enable_new_api_stack=False,
                          _disable_preprocessor_api=True)
            .resources(num_gpus=1,
                       num_gpus_per_worker=0.25)
            .rollouts(num_rollout_workers=1,
                      num_envs_per_worker=1)
            .framework("torch")
            .rl_module(rl_module_spec=rlm_spec)
            .training(model={"custom_model": "action_mask_model",
                             "custom_model_config": {"hidden_state_size": base_config.HIDDEN_STATE_SIZE // 2,
                                                     "num_hidden_layers": base_config.N_HEAD_HIDDEN_LAYERS}, },
                      train_batch_size=4096,
                      lambda_=0.95,
                      gamma=0.95,
                      lr=0.001)
            .evaluation(evaluation_num_workers=1, evaluation_interval=5)
        )
        # Construct the actual (PPO) algorithm object from the config.
        self.item_model = cfg.build()

        print("RETURNING POSITION ALGORITHM")
        return self.item_model

    def fetch_model(self):
        return self.item_model

    def train_item_model(self):
        while True:
            result = self.item_model.train()
            print(pretty_print(result))
            save_result = self.item_model.save()
            path_to_checkpoint = save_result.checkpoint.path
            print(
                "An Algorithm checkpoint has been created inside directory: "
                f"'{path_to_checkpoint}'."
            )

    def item_env(self, buffer):
        """
        Creates the TFT environment for the PPO model.
        """
        return TFT_Item_Simulator(buffer)


# This is the same as the other position model but not wrapped in Ray. Useful in some multi-processing situaiotns
class Base_PPO_Position_Model:
    def __init__(self, position_buffer):
        self.position_buffer = position_buffer

    def PPO_position_algorithm(self, ray_config):
        """
        The PPO implementation for the TFT project. This is an alternative to our MuZero model.
        """
        position_buffer = self.position_buffer

        # Number of environments to sample in one thread in the vector gym.
        num_envs = 64

        # Register the environment into ray's memory so we can access it later.
        ModelCatalog.register_custom_model('action_mask_model', TorchActionMaskModel)

        # class CustomCallbacks(DefaultCallbacks):
        #     def on_train_result(self, *, trainer, result: dict, **info):
        #         # Assuming you have a single policy named "default_policy"
        #         policy = trainer.get_policy("default_policy")
        #         entropy = policy.get_exploration_info()["entropy"]
        #         result["custom_metrics"] = {"entropy": entropy.item()}
        #         logger.info(f"Entropy: {entropy:.4f}")  # Optional: Print to console

        # Create an RLlib Algorithm instance from a PPOConfig object.
        cfg = (
            PPOConfig().environment(
                # Env class to use (here: our gym.Env sub-class from above).
                env=vector_env,
                # Provide a data generator so that it can be run in parallel with the main game learner
                # It will not use the buffer if there is nothing in the buffer.
                env_config={"data_generator": position_buffer},
                disable_env_checking=True
            )
            # Custom environment runner. It is the equivalent of both the remote worker and singleAgentEnvRunner in Ray
            .rollouts(env_runner_cls=ActionMaskEnvRunner,
                      # Number of parallel threads.
                      num_rollout_workers=16,
                      num_envs_per_worker=num_envs,
                      remote_worker_envs=False)
            .experimental(_enable_new_api_stack=False,
                          _disable_preprocessor_api=True)
            # Custom RL_Module to allow for training and action masks
            .rl_module(rl_module_spec=SingleAgentRLModuleSpec(module_class=TorchActionMaskRLM,
                                                              catalog_class=ActionMaskCatalog))
            # .resources(num_gpus=0.5,
            #            # This number times the num_rollout_workers has to be less than the num_gpus
            #            num_gpus_per_worker=0.1,
            #            num_cpus_per_worker=1.5)
            # In ray tune, the first num_gpus is how many gpus tune is allocated, the rest are for this algorithm
            .resources(num_gpus=0.5,
                       # This number times the num_rollout_workers has to be less than the num_gpus
                       num_gpus_per_worker=0.1,
                       num_cpus_per_worker=1)

            .framework("torch")
            # Custom model to allow for multiple head observation space and action masks
            .training(model={"custom_model": "action_mask_model",
                             "custom_model_config": {"hidden_state_size": ray_config["hidden_size"],
                                                     "num_hidden_layers": ray_config["num_heads"]}, },
                      train_batch_size=1024,
                      sgd_minibatch_size=128,
                      lambda_=ray_config["lambda"],
                      gamma=0.99,
                      lr=ray_config["lr"],
                      num_sgd_iter=ray_config["num_sgd_iter"],
                      clip_param=ray_config["clip_param"],
                      entropy_coeff=ray_config["entropy_coeff"],
                      grad_clip=1)
        )

        return cfg


    def fetch_model(self):
        return self.position_model

    def train_position_model(self, cfg):
        # Construct the actual (PPO) algorithm object from the config.
        import time
        self.position_model = cfg.build()

        # Training loop with Tune reporting
        for i in range(10000):
            result = self.position_model.train()
            printable_info = result['info']['learner']['default_policy']['learner_stats']
            print(f"episode reward mean {result['episode_reward_mean']} and total_loss {printable_info['total_loss']}")
            print(f"policy_loss {printable_info['policy_loss']}, entropy {printable_info['entropy']}")
            print(f"kl_divergence {printable_info['cur_kl_coeff']}, kl {printable_info['kl']}")
            ray.train.report(episode_reward_mean=result["episode_reward_mean"])  # report metrics

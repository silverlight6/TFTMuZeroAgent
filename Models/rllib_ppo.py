import ray
from ray.rllib.env import PettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig
from Simulator.tft_position_simulator import TFT_Position_Simulator
from Simulator.tft_item_simulator import TFT_Item_Simulator

@ray.remote(num_gpus=0.7)
class PPO_Models:
    def __init__(self):
        ...

    def PPO_position_algorithm(self, position_buffer):
        """
        The PPO implementation for the TFT project. This is an alternative to our MuZero model.
        """
        position_gym = self.position_env(position_buffer)
        # register our environment, we have no config parameters
        ray.tune.registry.register_env('TFT_Position_Simulator_s4_v0', lambda local_config: PettingZooEnv(position_gym))

        # Create an RLlib Algorithm instance from a PPOConfig object.
        cfg = (
            PPOConfig().environment(
                # Env class to use (here: our gym.Env sub-class from above).
                env='TFT_Position_Simulator_s4_v0',
                env_config={},
                observation_space=position_gym.observation_space,
                action_space=position_gym.action_space
            )
            .rollouts(num_rollout_workers=1)
            .framework("torch")
            .training(model={"fcnet_hiddens": [256, 256]})
            .evaluation(evaluation_num_workers=1, evaluation_interval=50)
        )
        # Construct the actual (PPO) algorithm object from the config.
        algo = cfg.build()

        for i in range(100):
            results = algo.train()
            print(f"Iter: {i}; avg. reward={results['episode_reward_mean']}")

        algo.evaluate()

    def PPO_item_algorithm(self, item_buffer):
        """
        The PPO implementation for the TFT project. This is an alternative to our MuZero model.
        """
        item_gymnasium = self.item_env(item_buffer)
        # register our environment, we have no config parameters
        ray.tune.registry.register_env('TFT_Item_Simulator_s4_v0', lambda local_config: PettingZooEnv(item_gymnasium))

        # Create an RLlib Algorithm instance from a PPOConfig object.
        cfg = (
            PPOConfig().environment(
                # Env class to use (here: our gym.Env sub-class from above).
                env='TFT_Item_Simulator_s4_v0',
                env_config={},
                observation_space=item_gymnasium.observation_space,
                action_space=item_gymnasium.action_space
            )
            .rollouts(num_rollout_workers=1)
            .framework("torch")
            .training(model={"fcnet_hiddens": [256, 256]})
            .evaluation(evaluation_num_workers=1, evaluation_interval=50)
        )
        # Construct the actual (PPO) algorithm object from the config.
        algo = cfg.build()

        for i in range(100):
            results = algo.train()
            print(f"Iter: {i}; avg. reward={results['episode_reward_mean']}")

        algo.evaluate()

    def position_env(self, buffer):
        """
        Creates the TFT environment for the PPO model.
        """
        return TFT_Position_Simulator(buffer)

    def item_env(self, buffer):
        """
        Creates the TFT environment for the PPO model.
        """
        return TFT_Item_Simulator(buffer)

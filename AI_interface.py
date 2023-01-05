import time
import config
import datetime
import tensorflow as tf
import gym
import numpy as np
from global_buffer import GlobalBuffer
from Models import MuZero_trainer
from Models.MuZero_agent_2 import TFTNetwork, Batch_MCTSAgent
from Models.replay_muzero_buffer import ReplayBuffer
from Simulator import game_round
from Simulator.observation import Observation
from Simulator.tft_simulator import TFT_Simulator
from ray.rllib.algorithms.ppo import PPOConfig


class AIInterface:

    def __init__(self):
        self.prev_actions = [[9] for _ in range(config.NUM_PLAYERS)]
        self.prev_reward = [0 for _ in range(config.NUM_PLAYERS)]

    # This is the main overarching gameplay method.
    # This is going to be implemented mostly in the game_round file under the AI side of things.
    def collect_gameplay_experience(self, env, agent, buffers):
        observation, info = env.reset()
        terminated = [False for _ in range(config.NUM_PLAYERS)]
        while not all(terminated):
            # agent policy that uses the observation and info
            actions, policy = agent.batch_policy(observation, self.prev_actions)
            self.prev_actions = actions
            observation = []
            rewards = []
            for i, action in enumerate(actions):
                player_observation, local_reward, local_terminated, info = env.step(np.asarray(action))
                observation.append(player_observation)
                rewards.append(local_reward)
                terminated[i] = local_terminated

            rewards = np.array(rewards)
            observation = np.array(observation)

            for i in range(config.NUM_PLAYERS):
                if info["players"][i]:
                    local_reward = rewards[info["players"][i].player_num] - \
                                   self.prev_reward[info["players"][i].player_num]
                    buffers[info["players"][i].player_num].\
                        store_replay_buffer(observation[info["players"][i].player_num],
                                            actions[info["players"][i].player_num], local_reward,
                                            policy[info["players"][i].player_num])
                    self.prev_reward[info["players"][i].player_num] = info["players"][i].reward

    def train_model(self, max_episodes=10000):
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
        env = gym.make("TFT_Set4-v0", env_config=None)

        for episode_cnt in range(1, max_episodes):
            agent = Batch_MCTSAgent(network=global_agent)
            buffers = [ReplayBuffer(global_buffer) for _ in range(config.NUM_PLAYERS)]
            self.collect_gameplay_experience(env, agent, buffers)

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
                observation_list, rewards, terminated, info = env.step(action)
            print("A game just finished in time {}".format(time.time_ns() - t))

    def PPO_algorithm(self):
        # Create an RLlib Algorithm instance from a PPOConfig object.
        cfg = (
            PPOConfig().environment(
                # Env class to use (here: our gym.Env sub-class from above).
                env=TFT_Simulator,
                env_config={},
            )
            .rollouts(num_rollout_workers=4)
            # .framework("tf2")
            # .training(model={"fcnet_hiddens": [256, 256]})
        )
        # Construct the actual (PPO) algorithm object from the config.
        algo = cfg.build()

        for i in range(100):
            results = algo.train()
            print(f"Iter: {i}; avg. reward={results['episode_reward_mean']}")

        algo.evaluate()  # 4. and evaluate it.

    def evaluate(self, agent):
        return 0


def env_creator(env_name):
    if env_name == 'TFT_Set4-v0':
        from Simulator.tft_simulator import TFT_Simulator as env
    else:
        raise NotImplementedError
    return env

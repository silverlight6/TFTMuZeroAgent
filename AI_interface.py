import time
import config
import datetime
import tensorflow as tf
import gym
import numpy as np
from global_buffer import GlobalBuffer
from Models import MuZero_trainer
from Models.MuZero_agent_2 import TFTNetwork, Batch_MCTSAgent, MCTSAgent
from Models.replay_muzero_buffer import ReplayBuffer
from Simulator import game_round
from Simulator.tft_simulator import TFT_Simulator, parallel_env
from ray.rllib.algorithms.ppo import PPOConfig
from Simulator.tft_simulator import env as global_env
from ray.tune.registry import register_env
from ray.rllib.env import ParallelPettingZooEnv
from pettingzoo.test import parallel_api_test



class AIInterface:

    def __init__(self):
        self.prev_actions = [[9] for _ in range(config.NUM_PLAYERS)]
        self.prev_reward = [0 for _ in range(config.NUM_PLAYERS)]

    # This is the main overarching gameplay method.
    # This is going to be implemented mostly in the game_round file under the AI side of things.
    def collect_gameplay_experience(self, env, agents, buffers):
        env.reset()
        terminated = {agent: False for agent in env.agents}
        while not all(terminated):
            # agent policy that uses the observation and info
            actions = []
            for key, terminate in terminated.items():
                if not terminate:
                    player_observation, local_reward, local_terminated, _, info = env.last()
                    player_observation = np.expand_dims(player_observation, axis=0)
                    local_action, local_policy = agents[key].policy(player_observation, self.prev_actions[i])
                    env.step(local_action)
                    actions.append(local_action)

                    terminated[key] = local_terminated
                    buffers[key].store_replay_buffer(player_observation, local_action, local_reward, local_policy)

            self.prev_actions = actions

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
        env = parallel_env()

        for episode_cnt in range(1, max_episodes):
            agents = { player_id : MCTSAgent(global_buffer, player_num = i) for i, player_id in enumerate(env.agents.keys()) } 
            buffers = { player_id: ReplayBuffer(global_buffer) for player_id in env.possible_agents }

            self.collect_gameplay_experience(env, agents, buffers)

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
        local_env_creator = lambda _: self.env_creator()

        register_env('tft-set4-v0', lambda local_config: ParallelPettingZooEnv(local_env_creator(local_config)))

        # Create an RLlib Algorithm instance from a PPOConfig object.
        cfg = (
            PPOConfig().environment(
                # Env class to use (here: our gym.Env sub-class from above).
                env='tft-set4-v0',
                env_config={},
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
        env = parallel_env()
        parallel_api_test(env, num_cycles=1000)

    def env_creator(self):
        env = parallel_env()
        self.testEnv(env)
        return env

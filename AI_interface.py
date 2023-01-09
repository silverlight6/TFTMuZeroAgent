import time
import config
import datetime
import tensorflow as tf
import gymnasium as gym
import numpy as np
from global_buffer import GlobalBuffer
from Models import MuZero_trainer
from Models.MuZero_agent_2 import TFTNetwork, Batch_MCTSAgent, MCTSAgent
from Models.replay_muzero_buffer import ReplayBuffer
from Simulator import game_round
from Simulator.tft_simulator import TFT_Simulator, parallel_env, env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.rllib.env import PettingZooEnv
from pettingzoo.test import parallel_api_test, api_test


class AIInterface:

    def __init__(self):
        self.prev_actions = [0 for _ in range(config.NUM_PLAYERS)]

    # This is the main overarching gameplay method.
    # This is going to be implemented mostly in the game_round file under the AI side of things.
    def collect_gameplay_experience(self, env, agent, buffers):
        # Reset the environment
        player_observation = env.reset()
        # This is here to make the input (1, observation_size) for initial_inference
        player_observation = np.asarray(list(player_observation.values()))
        # Used to know when players die and which agent is currently acting
        terminated = {player_id: False for player_id in env.possible_agents}
        # Current action to help with MuZero
        # While the game is still going on.
        while not all(terminated.values()):
            # Ask our model for an action and policy
            actions, policy = agent.batch_policy(player_observation, list(self.prev_actions))
            step_actions = self.getStepActions(terminated, actions)
            

            # Take that action within the environment and return all of our information for the next player
            next_observation, reward, terminated, _, info = env.step(step_actions)
            # store the action for MuZero
            for i, key in enumerate(terminated.keys()):
                # Store the information in a buffer to train on later.
                buffers[key].store_replay_buffer(player_observation, actions[i], reward[key], policy[i])
            # Set up the observation for the next action
            player_observation = np.asarray(list(next_observation.values()))
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
        tft_env = parallel_env()

        for episode_cnt in range(1, max_episodes):
            agent = Batch_MCTSAgent(global_agent)
            buffers = {player_id: ReplayBuffer(global_buffer) for player_id in tft_env.possible_agents}

            self.collect_gameplay_experience(tft_env, agent, buffers)

            for key in tft_env.possible_agents:
                buffers[key].store_global_buffer()
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
    
    '''
    rewards have to make sense
    clip the values
    normalize across all agents in a single game : mean of 1 SD of 1
    make sure non-para env passes test from PettingZoo
    '''

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

    def PPO_algorithm(self):
        #register our environment, we have no config parameters
        register_env('tft-set4-v0', lambda local_config: ParallelPettingZooEnv(self.env_creator(local_config)))

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
        second_env = env()
        api_test(second_env, num_cycles=100000)

    #function looks stupid as is right now, but should remain this way
    #for potential future abstractions
    def env_creator(self,config):
        return TFT_Simulator(config)
    
    def getStepActions(self, terminated, actions):
        step_actions = {}
        i = 0
        for player_id, terminate in terminated.items():
            if not terminate:
                step_actions[player_id] = actions[i]
                i += 1
        return step_actions

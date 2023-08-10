import time
import config
import datetime
import ray
import os
import copy
import gymnasium as gym
import numpy as np
from storage import Storage
from global_buffer import GlobalBuffer
from Simulator.tft_simulator import TFT_Simulator, parallel_env, env as tft_env
from ray.rllib.algorithms.ppo import PPOConfig
from Models.replay_buffer_wrapper import BufferWrapper
from ray.tune.registry import register_env
from ray.rllib.env import PettingZooEnv
from pettingzoo.test import parallel_api_test, api_test
from Simulator import utils

from Models.MCTS_torch import MCTS
from Models.MuZero_torch_agent import MuZeroNetwork as TFTNetwork
from Models.MuZero_torch_trainer import Trainer
from torch.utils.tensorboard import SummaryWriter
import torch

'''
Data workers are the "workers" or threads that collect game play experience. 
Can add scheduling_strategy="SPREAD" to ray.remote. Not sure if it makes any difference
'''
@ray.remote(num_gpus=config.GPU_SIZE_PER_WORKER)
class DataWorker(object):
    def __init__(self, rank):
        self.temp_model = TFTNetwork()
        self.agent_network = MCTS(self.temp_model)
        self.past_network = MCTS(self.temp_model)
        self.past_version = [False for _ in range(config.NUM_PLAYERS)]
        self.live_game = True
        self.rank = rank
        self.ckpt_time = time.time_ns()
        self.prob = 1
        self.past_episode = 0
        self.past_update = True

    '''
    Description -
        Each worker runs one full game and will restart after the game finishes. At the end of the game, 
        it will fetch a new agent. 
    Inputs -   
        env 
            A parallel environment of our tft simulator. This is the game that the worker interacts with.
        buffers
            One buffer wrapper that holds a series of buffers that we store all information required to train in.
        global_buffer
            A buffer that all the individual game buffers send their information to.
        storage
            An object that stores global information like the weights of the global model and current training progress
        weights
            Weights of the initial model for the agent to play the game with.
    '''
    def collect_gameplay_experience(self, env, buffers, global_buffer, storage, weights):
        self.agent_network.network.set_weights(weights)
        self.past_network.network.set_weights(weights)
        while True:
            # Reset the environment
            player_observation, info = env.reset()
            # This is here to make the input (1, observation_size) for initial_inference
            player_observation = self.observation_to_input(player_observation)

            # Used to know when players die and which agent is currently acting
            terminated = {player_id: False for player_id in env.possible_agents}
            position = 8

            # While the game is still going on.
            while not all(terminated.values()):
                # Ask our model for an action and policy. Use on normal case or if we only have current versions left
                actions, policy, string_samples = self.model_call(player_observation, terminated)

                storage_actions = utils.decode_action(actions)
                step_actions = self.getStepActions(terminated, storage_actions)

                # Take that action within the environment and return all of our information for the next player
                next_observation, reward, terminated, _, info = env.step(step_actions)
                # store the action for MuZero
                for i, key in enumerate(terminated.keys()):
                    if not info[key]["state_empty"]:
                        if not self.past_version[i]:
                            # Store the information in a buffer to train on later.
                            buffers.store_replay_buffer.remote(key, player_observation[0][i], storage_actions[i],
                                                               reward[key], policy[i], string_samples[i])

                offset = 0
                for i, [key, terminate] in enumerate(terminated.items()):
                    # Saying if that any of the 4 agents got first or second then we are saying we are not
                    # Currently beating that checkpoint
                    if terminate:
                        # print("player {} got position {} of game {}".format(i, position, self.rank))
                        buffers.set_ending_position.remote(key, position)
                        position -= 1
                        self.past_version.pop(i - offset)
                        offset += 1

                if not any(self.past_version) and len(terminated) == 2 and not self.past_update:
                    storage.update_checkpoint_score.remote(self.past_episode, self.prob)
                    self.past_update = True

                # Set up the observation for the next action
                player_observation = self.observation_to_input(next_observation)

            # buffers.rewardNorm.remote()
            buffers.store_global_buffer.remote()
            buffers = BufferWrapper.remote(global_buffer)

            # Might want to get rid of the hard constant 0.8 for something that can be adjusted in the future
            self.live_game = np.random.rand() <= 0.8
            self.past_version = [False for _ in range(config.NUM_PLAYERS)]
            if not self.live_game:
                past_weights, self.past_episode, self.prob = ray.get(storage.sample_past_model.remote())
                self.past_network = MCTS(self.temp_model)
                self.past_network.network.set_weights(past_weights)
                self.past_version[0:4] = [True, True, True, True]
                self.past_update = False
            # So if I do not have a live game, I need to sample a past model
            # Which means I need to create a list within the storage and sample from that.
            # All the probability distributions will be within the storage class as well.
            temp_weights = ray.get(storage.get_model.remote())
            weights = copy.deepcopy(temp_weights)
            self.agent_network = MCTS(self.temp_model)
            self.agent_network.network.set_weights(weights)
            self.rank += config.CONCURRENT_GAMES

    '''
    Description -
        Turns the actions from a format that is sent back from the model to a format that is usable by the environment.
        We check for any new dead agents in this method as well.
    Inputs
        terminated
            A dictionary of player_ids and booleans that tell us if a player can execute an action or not.
        actions
            The set of actions returned from the model in a list of strings format
    Returns
        step_actions
            A dictionary of player_ids and actions usable by the environment.
    '''
    def getStepActions(self, terminated, actions):
        step_actions = {}
        i = 0
        for player_id, terminate in terminated.items():
            if not terminate:
                step_actions[player_id] = actions[i]
                i += 1
        return step_actions

    '''
    Description -
        Turns a dictionary of player observations into a list of list format that the model can use.
        Adding key to the list to ensure the right values are attached in the right places in debugging.
    '''
    def observation_to_input(self, observation):
        tensors = []
        masks = []
        keys = []
        for key, obs in observation.items():
            tensors.append(obs["tensor"])
            masks.append(obs["mask"])
            keys.append(key)
        return [np.asarray(tensors), masks, keys]

    '''
    Description -
        Turns a string action into a series of one_hot lists that can be used in the step_function.
        More specifics on what every list means can be found in the step_function.
    '''

    def decode_action_to_one_hot(self, str_action):
        num_items = str_action.count("_")
        split_action = str_action.split("_")
        element_list = [0, 0, 0]
        for i in range(num_items + 1):
            element_list[i] = int(split_action[i])

        decoded_action = np.zeros(config.ACTION_DIM[0] + config.ACTION_DIM[1] + config.ACTION_DIM[2])
        decoded_action[0:7] = utils.one_hot_encode_number(element_list[0], 7)

        if element_list[0] == 1:
            decoded_action[7:12] = utils.one_hot_encode_number(element_list[1], 5)

        if element_list[0] == 2:
            decoded_action[7:44] = utils.one_hot_encode_number(element_list[1], 37) + \
                                   utils.one_hot_encode_number(element_list[2], 37)

        if element_list[0] == 3:
            decoded_action[7:44] = utils.one_hot_encode_number(element_list[1], 37)
            decoded_action[44:54] = utils.one_hot_encode_number(element_list[2], 10)

        if element_list[0] == 4:
            decoded_action[7:44] = utils.one_hot_encode_number(element_list[1], 37)
        return decoded_action

    def model_call(self, player_observation, terminated):
        if self.live_game or not any(self.past_version):
            actions, policy, string_samples = self.agent_network.policy(player_observation[:2])
        # if all of our agents are past versions. (Should exceedingly rarely come here)
        elif all(self.past_version):
            actions, policy, string_samples = self.past_network.policy(player_observation[:2])
        else:
            actions, policy, string_samples = self.mixed_model_call(player_observation, terminated)
        return actions, policy, string_samples

    def mixed_model_call(self, player_observation, terminated):
        # I need to send the observations that are part of the past players to one vector
        # and send the ones that are part of the live players to another vector
        # Problem comes if I want to do multiple different versions in a game.
        # I could limit it to one past verison. That is probably going to be fine for our purposes
        # Note for later that the current implementation is only going to be good for one past version
        # For our purposes, lets just start with half of the agents being false.

        # Now that I have the array for the past versions down to only the remaining players
        # Separate the observation.
        live_agent_observations = []
        past_agent_observations = []
        live_agent_masks = []
        past_agent_masks = []

        for i, past_version in enumerate(self.past_version):
            if not past_version:
                live_agent_observations.append(player_observation[0][i])
                live_agent_masks.append(player_observation[1][i])
            else:
                past_agent_observations.append(player_observation[0][i])
                past_agent_masks.append(player_observation[1][i])
        live_observation = [np.asarray(live_agent_observations), live_agent_masks]
        past_observation = [np.asarray(past_agent_observations), past_agent_masks]
        live_actions, live_policy, live_string_samples = self.agent_network.policy(live_observation)
        past_actions, past_policy, past_string_samples = self.past_network.policy(past_observation)
        actions = [None] * len(self.past_version)
        policy = [None] * len(self.past_version)
        string_samples = [None] * len(self.past_version)
        counter_live, counter_past = 0, 0
        for i, past_version in enumerate(self.past_version):
            if past_version:
                actions[i] = past_actions[counter_past]
                policy[i] = past_policy[counter_past]
                string_samples[i] = past_string_samples[counter_past]
                counter_past += 1
            else:
                actions[i] = live_actions[counter_live]
                policy[i] = live_policy[counter_live]
                string_samples[i] = live_string_samples[counter_live]
                counter_live += 1
        return actions, policy, string_samples

    '''
    Description -
        Loads in a set of agents from checkpoints of the users choice to play against each other. These agents all have
        their own policy function and are not required to be the same model. This method is used for evaluating the
        skill level of the current agent and to see how well our agents are training. Metrics from this method are 
        stored in the storage class because this is the data worker side so there are intended to be multiple copies
        of this method running at once. 
    '''
    def evaluate_agents(self, env, storage):
        agents = {"player_" + str(r): MCTS(TFTNetwork())
                  for r in range(config.NUM_PLAYERS)}
        agents["player_1"].network.tft_load_model(1000)
        agents["player_2"].network.tft_load_model(2000)
        agents["player_3"].network.tft_load_model(3000)
        agents["player_4"].network.tft_load_model(4000)
        agents["player_5"].network.tft_load_model(5000)
        agents["player_6"].network.tft_load_model(6000)
        agents["player_7"].network.tft_load_model(7000)

        while True:
            # Reset the environment
            player_observation = env.reset()
            # This is here to make the input (1, observation_size) for initial_inference
            player_observation = np.asarray(
                list(player_observation.values()))
            # Used to know when players die and which agent is currently acting
            terminated = {
                player_id: False for player_id in env.possible_agents}
            # Current action to help with MuZero
            placements = {
                player_id: 0 for player_id in env.possible_agents}
            current_position = 7
            info = {player_id: {"player_won": False}
                    for player_id in env.possible_agents}
            # While the game is still going on.
            while not all(terminated.values()):
                # Ask our model for an action and policy
                actions = {agent: 0 for agent in agents.keys()}
                for i, [key, agent] in enumerate(agents.items()):
                    action, _ = agent.policy(np.expand_dims(player_observation[i], axis=0))
                    actions[key] = action

                # step_actions = self.getStepActions(terminated, np.asarray(actions))

                # Take that action within the environment and return all of our information for the next player
                next_observation, reward, terminated, _, info = env.step(actions)

                # Set up the observation for the next action
                player_observation = np.asarray(list(next_observation.values()))

                for key, terminate in terminated.items():
                    if terminate:
                        placements[key] = current_position
                        current_position -= 1

            for key, value in info.items():
                if value["player_won"]:
                    placements[key] = 0
            storage.record_placements.remote(placements)
            print("recorded places {}".format(placements))
            self.rank += config.CONCURRENT_GAMES


class AIInterface:

    def __init__(self):
        ...

    '''
    Global train model method. This is what gets called from main.
    '''
    def train_torch_model(self, starting_train_step=0):
        gpus = torch.cuda.device_count()
        ray.init(num_gpus=gpus, num_cpus=config.NUM_CPUS)

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
        train_summary_writer = SummaryWriter(train_log_dir)
        train_step = starting_train_step

        storage = Storage.remote(train_step)
        global_buffer = GlobalBuffer.remote(storage)

        global_agent = TFTNetwork()
        global_agent_weights = ray.get(storage.get_target_model.remote())
        global_agent.set_weights(global_agent_weights)
        global_agent.to(config.DEVICE)

        trainer = Trainer(global_agent, train_summary_writer)

        env = parallel_env()

        buffers = [BufferWrapper.remote(global_buffer)
                   for _ in range(config.CONCURRENT_GAMES)]

        weights = ray.get(storage.get_target_model.remote())
        workers = []
        data_workers = [DataWorker.remote(rank) for rank in range(config.CONCURRENT_GAMES)]
        for i, worker in enumerate(data_workers):
            workers.append(worker.collect_gameplay_experience.remote(env, buffers[i], global_buffer,
                                                                     storage, weights))
            time.sleep(0.5)
        # ray.get(workers)

        while True:
            if ray.get(global_buffer.available_batch.remote()):
                gameplay_experience_batch = ray.get(global_buffer.sample_batch.remote())
                trainer.train_network(gameplay_experience_batch, train_step)
                storage.set_trainer_busy.remote(False)
                storage.set_target_model.remote(global_agent.get_weights())
                train_step += 1
                if train_step % config.CHECKPOINT_STEPS == 0:
                    storage.store_checkpoint.remote(train_step)
                    global_agent.tft_save_model(train_step)

    '''
    Method used for testing the simulator. It does not call any AI and generates random actions from numpy. Intended
    to test how fast the simulator is and if there are any bugs that can be caught via multiple runs.
    '''
    def collect_dummy_data(self):
        env = parallel_env()
        while True:
            _ = env.reset()
            terminated = {player_id: False for player_id in env.possible_agents}
            t = time.time_ns()
            rewards = None
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
    The PPO implementation for the TFT project. This is an alternative to our MuZero model.
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
    The global side to the evaluator. Creates a set of workers to test a series of agents.
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
    PettingZoo's api tests for the simulator.
    '''
    def testEnv(self):
        raw_env = tft_env()
        api_test(raw_env, num_cycles=100000)
        local_env = parallel_env()
        parallel_api_test(local_env, num_cycles=100000)

    '''
    Creates the TFT environment for the PPO model.
    '''
    def env_creator(self, cfg):
        return TFT_Simulator(cfg)

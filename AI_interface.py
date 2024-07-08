from collections import deque
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
from tqdm import tqdm

from Models.MCTS_torch import MCTS
from Models.MuZero_torch_agent import BaseMuZeroAgent, MuZeroAgent
import Models.MuZero_torch_trainer as MuZero_trainer
from torch.utils.tensorboard import SummaryWriter
import torch
from Models.Common_agents import CultistAgent, DivineAgent, RandomAgent, BuyingAgent

class Agregator:
    def __init__(self):
        self.agents = {}
        self.player_to_agents = {}

    def get_player_to_agent_mapping(self):
        return self.player_to_agents
    
    def set_agents(self, agents, agent_count):
        if sum(agent_count) > config.NUM_PLAYERS or len(agents) != len(agent_count):
            raise ValueError('More alocated agents than max players supported per config')
        for i, agent in enumerate(agents):
            for _ in range(agent_count[i]):
                self.add_agent(agent, f"player_{len(self.player_to_agents.keys())}")

    def add_agent(self, new_agent, player_name):
        if not new_agent.__class__ in self.agents.keys():
            self.agents[new_agent.__class__] = new_agent
        self.player_to_agents[player_name] = new_agent.__class__

    def get_actions(self, observations, rewards, terminated):
        mapped_obs = {}
        mapped_mask = {}
        mapped_reward = {}
        mapped_terminated = {}
        mapped_players = {}
        actions = {}
        for player in observations.keys():
            agent = self.player_to_agents[player]
            if not agent in mapped_obs.keys():
                mapped_obs[agent] = []
                mapped_mask[agent] = []
                mapped_reward[agent] = []
                mapped_players[agent] = []
                mapped_terminated[agent] = []
            mapped_obs[agent].append(observations[player]['tensor'])
            mapped_mask[agent].append(observations[player]['mask'])
            mapped_reward[agent].append(rewards[player])
            mapped_terminated[agent].append(terminated[player])
            mapped_players[agent].append(player)
        
        for key in mapped_obs.keys():
            # print(mapped_mask[key][0])
            # print("------------------------------")
            # print(mapped_mask[key][1])
            concat_actions = self.agents[key].select_action(np.array(mapped_obs[key]), 
                                                            np.array(mapped_mask[key], dtype="object"),
                                                            mapped_reward[key],
                                                            mapped_terminated[key])
            # print(concat_actions, key, mapped_players[key])
            for i, player in enumerate(mapped_players[key]):
                actions[player] = concat_actions[i]
        return actions

# Can add scheduling_strategy="SPREAD" to ray.remote. Not sure if it makes any difference
@ray.remote(num_gpus=0.001)
class DataWorker(object):
    def __init__(self, rank,):
        self.rank = rank

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
    def collect_gameplay_experience(self, agents, agents_count, placements=None):
        # self.agent_network.set_weights(weights)
        # agent = MCTS(self.agent_network
        self.ckpt_time = time.time()
        # Reset the environment
        self.env = parallel_env()
        players_observation = self.env.reset()
        # This is here to make the input (1, observation_size) for initial_inference
        # players_observation = self.observation_to_input(players_observation)
        # Used to know when players die and which agent is currently acting
        terminated = {player_id: False for player_id in self.env.possible_agents}
        reward = {player_id: 0.0 for player_id in self.env.possible_agents}
        scores = {player_id: 0 for player_id in self.env.possible_agents}
        agregator = Agregator()
        agregator.set_agents(agents, agents_count)

        # While the game is still going on.
        while not all(terminated.values()):
            # Ask our model for an action and policy
            actions = agregator.get_actions(players_observation, reward, terminated)
            # print(actions)

            # step_actions = self.getStepActions(terminated, actions)
            # storage_actions = utils.decode_action(actions)

            # Take that action within the environment and return all of our information for the next turn
            players_observation, reward, terminated, _, info = self.env.step(actions)
            #TODO move this to info from simulator
            for player in terminated.keys():
                if terminated[player]:
                    scores[player] = reward[player]
            # print(terminated)
            # store the action for MuZero
            # for i, key in enumerate(terminated.keys()):
            #     if not info[key]["state_empty"]:
            #         # Store the information in a buffer to train on later.
            #         self.buffer.store_replay_buffer(key, player_observation[0][i], storage_actions[i],
            #                                             reward[key], policy[i], string_samples[i])

            # Set up the observation for the next action
            # player_observation = self.observation_to_input(next_observation)

        # weights = copy.deepcopy(ray.get(storage).get_model())
        # agent.network.set_weights(weights)
        # self.rank += config.CONCURRENT_GAMES
        # self.buffer.store_global_buffer()
        # self.buffer.reset()
        
        if placements:
            placements = {i: None for i, _ in enumerate(self.env.possible_agents)}
            agent_mapping = agregator.get_player_to_agent_mapping()
            scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            for i, player in enumerate(scores):
                placements[i] = agent_mapping[player[0]]
            # print(f'Worker {self.rank} finished a game in {(time.time() - self.ckpt_time)/60} minutes. Max AVG traversed depth: {agent.max_depth_search / agent.runs}')
            if config.DEBUG:
                for x in placements.keys():
                    print(f'{x+1} place -> {placements[x]}')
        print(f'Worker {self.rank} finished a game in {(time.time() - self.ckpt_time)/60} minutes.')
        return self.rank, placements

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
                step_actions[player_id] = self.decode_action_to_one_hot(actions[i])
                i += 1
        return step_actions

    '''
    Description -
        Turns a dictionary of player observations into a list of list format that the model can use.
    '''
    def observation_to_input(self, observation):
        tensors = []
        masks = []
        for obs in observation.values():
            tensors.append(obs["tensor"])
            masks.append(obs["mask"])
        return [np.asarray(tensors), masks]

    '''
    Description -
        Turns a string action into a series of one_hot lists that can be used in the step_function.
        More specifics on what every list means can be found in the step_function.
    '''

    def decode_action_to_one_hot(self, str_action):
        split_action = str_action.split("_")[0]
        element_list = [0, 0, 0, 0]
        element_list[0] = int(split_action)

        # decoded_action = np.zeros(config.ACTION_DIM[0] + config.ACTION_DIM[1] + config.ACTION_DIM[2])
        # decoded_action[0:7] = utils.one_hot_encode_number(element_list[0], 7)

        # if element_list[0] == 1:
        #     decoded_action[7:12] = utils.one_hot_encode_number(element_list[1], 5)

        # if element_list[0] == 2:
        #     decoded_action[7:44] = utils.one_hot_encode_number(element_list[1], 37) + \
        #                            utils.one_hot_encode_number(element_list[2], 37)

        # if element_list[0] == 3:
        #     decoded_action[7:44] = utils.one_hot_encode_number(element_list[1], 37)
        #     decoded_action[44:54] = utils.one_hot_encode_number(element_list[2], 10)

        # if element_list[0] == 4:
        #     decoded_action[7:44] = utils.one_hot_encode_number(element_list[1], 37)
        return element_list


class AIInterface:

    def __init__(self):
        ...

    '''
    Global train model method. This is what gets called from main.
    '''
    def train_torch_model(self, starting_train_step=0, run_name=""):
        assert(config.EVALUATION_GAMES % config.EVALUATION_CONCURRENT_GAMES == 0, "Evaluation concurrency wrong")
        gpus = torch.cuda.device_count()
        ray.init(num_gpus=gpus, num_cpus=config.NUM_CPUS)
        # ray.init()

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/' + run_name + current_time
        train_summary_writer = SummaryWriter(train_log_dir)
        train_step = starting_train_step

        # storage = Storage.remote(train_step)
        # storage = Storage(train_step)
        # global_buffer = GlobalBuffer.remote(storage)

        # global_agent = TFTNetwork()
        # global_agent_weights = storage.get_target_model()
        # global_agent.set_weights(global_agent_weights)
        # global_agent.to("cuda")

        weights = np.array([0])
        # weights_ref = ray.put(weights)

        # Load model weights into shared memory

        # Create trainer
        trainer = MuZero_trainer.Trainer()

        # Create global buffer
        global_buffer = GlobalBuffer.remote()
        base_agent = BaseMuZeroAgent(3, [6, 37, 28], config.OBSERVATION_SIZE, config.NUM_SIMULATIONS, global_buffer)
        weights = copy.deepcopy(base_agent.get_weights())

        agent_1 = MuZeroAgent(3, [6, 37, 28], config.OBSERVATION_SIZE, config.NUM_SIMULATIONS, global_buffer, weights)
        agent_random = RandomAgent(3, [6, 37, 28], global_buffer)
        agent_buying_1 = CultistAgent(3, [6, 37, 28], ["elise", "twistedfate", "pyke", "evelynn", "aatrox", "zilean", "kalista", "jhin"], global_buffer)
        agent_buying_2 = DivineAgent(3, [6, 37, 28], ["wukong", "jax", "irelia", "lux", "warwick", "leesin", "ashe", "kindred", "teemo"], global_buffer)

        agents = [agent_1, agent_random, agent_buying_1, agent_buying_2]
        agents_count = [2, 4, 1, 1]

        eval_workers = [DataWorker.remote(agent_num) for agent_num in range(config.EVALUATION_CONCURRENT_GAMES)]
        data_workers = [DataWorker.remote(agent_num) for agent_num in range(config.CONCURRENT_GAMES)]
        # weights = storage.get_target_model()
        workers = [worker.collect_gameplay_experience.remote(agents, agents_count) for worker in data_workers]
        tpbar = tqdm(total=config.CHECKPOINT_STEPS, desc="Checkpoint Progress")
        while True:
            # while True:
                # with open("run.txt", "r") as file:
                #     if file.readline() == "0":
                #         input("run = 0, press enter to continue and change run to 1")
                #     else:
                #         print("Continuing training")
                #         break
            done, workers = ray.wait(workers)
            rank, _ = ray.get(done)[0]
            # print(f'Spawning agent {rank}')
            workers.extend([data_workers[rank].collect_gameplay_experience.remote(agents, agents_count)])
            while ray.get(global_buffer.available_gameplay_batch.remote(config.BATCH_SIZE)):
                # print("Starting training ")
                training_batch = ray.get(global_buffer.sample_gameplay_batch.remote(config.BATCH_SIZE))
                combat_batch = []
                if ray.get(global_buffer.available_combat_batch.remote(config.BATCH_SIZE//2)):
                    combat_batch = ray.get(global_buffer.sample_combat_batch.remote(config.BATCH_SIZE//2))
                trainer.train_network(training_batch, combat_batch, base_agent.model, train_step, train_summary_writer)
                train_step += 1
                tpbar.update(1)
                if train_step % config.CHECKPOINT_STEPS == 0:
                    agents = [base_agent, agent_1, agent_random, agent_buying_1, agent_buying_2]
                    agents_count = [1, 1, 4, 1, 1]
                    eval_steps = config.EVALUATION_GAMES // config.EVALUATION_CONCURRENT_GAMES
                    evalbar = tqdm(total=eval_steps, desc="Evaluation Progress")
                    base_placements = []
                    old_placements = []
                    for _ in range(eval_steps):
                        evaluator = [worker.collect_gameplay_experience.remote(agents, agents_count, True) for worker in eval_workers]
                        done, _ = ray.wait(evaluator)
                        _, placements = ray.get(done)[0]
                        base_placements.append(list(placements.keys())[list(placements.values()).index(base_agent.__class__)])
                        old_placements.append(list(placements.keys())[list(placements.values()).index(agent_1.__class__)])
                        evalbar.update(config.EVALUATION_CONCURRENT_GAMES)
                    evalbar.close()
                    if np.array(base_placements).mean() >= np.array(old_placements).mean():
                        weights = copy.deepcopy(base_agent.get_weights())
                        base_agent.model.tft_save_model(0)
                    train_summary_writer.add_scalar('evaluation/old', np.array(old_placements).mean(), train_step)
                    train_summary_writer.add_scalar('evaluation/new', np.array(base_placements).mean(), train_step)
                    tpbar.close()
                    tpbar = tqdm(total=config.CHECKPOINT_STEPS)
                
                # print("Finished training")
            

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
    PettingZoo's api tests for the simulator.
    '''
    def testEnv(self):
        raw_env = tft_env()
        api_test(raw_env, num_cycles=100000)
        local_env = parallel_env()
        parallel_api_test(local_env, num_cycles=100000)


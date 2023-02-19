import config
import datetime
import tensorflow as tf
import numpy as np
import os
from TestInterface.test_global_buffer import GlobalBuffer
from Simulator.tft_simulator import parallel_env
from Models import MuZero_trainer
from TestInterface.test_replay_wrapper import BufferWrapper
from Models.MuZero_agent_2 import TFTNetwork
from Models.MCTS import MCTS
from Simulator import utils


class DataWorker(object):
    def __init__(self, rank):
        self.agent_network = TFTNetwork()
        self.prev_actions = [0 for _ in range(config.NUM_PLAYERS)]
        self.rank = rank

    # This is the main overarching gameplay method.
    # This is going to be implemented mostly in the game_round file under the AI side of things.
    def collect_gameplay_experience(self, env, buffers, weights):

        self.agent_network.set_weights(weights)
        agent = MCTS(self.agent_network)
        # Reset the environment
        player_observation = env.reset()
        # This is here to make the input (1, observation_size) for initial_inference
        player_observation = self.observation_to_input(player_observation)

        # Used to know when players die and which agent is currently acting
        terminated = {player_id: False for player_id in env.possible_agents}

        # While the game is still going on.
        while not all(terminated.values()):
            # Ask our model for an action and policy
            actions, policy = agent.policy(player_observation)
            step_actions = self.getStepActions(terminated, actions)
            storage_actions = utils.decode_action(actions)

            # Take that action within the environment and return all of our information for the next player
            next_observation, reward, terminated, _, info = env.step(step_actions)
            # store the action for MuZero
            for i, key in enumerate(terminated.keys()):
                # Store the information in a buffer to train on later.
                buffers.store_replay_buffer(key, player_observation[0][i], storage_actions[i], reward[key], policy[i])
            # Set up the observation for the next action
            player_observation = self.observation_to_input(next_observation)

        # buffers.rewardNorm()
        buffers.store_global_buffer()

    def getStepActions(self, terminated, actions):
        step_actions = {}
        i = 0
        for player_id, terminate in terminated.items():
            if not terminate:
                step_actions[player_id] = self.decode_action_to_one_hot(actions[i])
                i += 1
        return step_actions

    def observation_to_input(self, observation):
        tensors = []
        masks = []
        for obs in observation.values():
            tensors.append(obs["tensor"])
            masks.append(obs["mask"])
        return [np.asarray(tensors), masks]

    def decode_action_to_one_hot(self, str_action):
        num_items = str_action.count("_")
        split_action = str_action.split("_")
        element_list = [0, 0, 0]
        for i in range(num_items + 1):
            element_list[i] = int(split_action[i])

        decoded_action = np.zeros(config.ACTION_DIM[0] + config.ACTION_DIM[1] + config.ACTION_DIM[2])
        decoded_action[0:6] = utils.one_hot_encode_number(element_list[0], 6)

        if element_list[0] == 1:
            decoded_action[6:11] = utils.one_hot_encode_number(element_list[1], 5)

        if element_list[0] == 2:
            decoded_action[6:44] = utils.one_hot_encode_number(element_list[1], 38) + utils.one_hot_encode_number(
                element_list[2], 38)

        if element_list[0] == 3:
            decoded_action[6:44] = utils.one_hot_encode_number(element_list[1], 38)
            decoded_action[44:54] = utils.one_hot_encode_number(element_list[2], 10)
        return decoded_action


class AIInterface:

    def __init__(self):
        ...

    def train_model(self, starting_train_step=0):
        tf.config.optimizer.set_jit(True)
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        train_step = starting_train_step

        global_buffer = GlobalBuffer()

        trainer = MuZero_trainer.Trainer()

        env = parallel_env()
        data_workers = DataWorker(0)
        global_agent = TFTNetwork()
        # global_agent.tft_load_model(train_step)

        while True:
            weights = global_agent.get_weights()
            buffers = BufferWrapper(global_buffer)
            data_workers.collect_gameplay_experience(env, buffers, weights)

            while global_buffer.available_batch():
                gameplay_experience_batch = global_buffer.sample_batch()
                trainer.train_network(gameplay_experience_batch, global_agent, train_step, train_summary_writer)
                train_step += 1
                if train_step % 10 == 0:
                    global_agent.tft_save_model(train_step)

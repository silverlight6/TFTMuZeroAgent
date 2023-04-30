import config
import datetime
import numpy as np
from TestInterface.test_global_buffer import GlobalBuffer
from Simulator.tft_simulator import parallel_env
import time

from TestInterface.test_replay_wrapper import BufferWrapper

from Simulator import utils

from Models.MCTS_torch import MCTS
from Models.MuZero_torch_agent import MuZeroNetwork as TFTNetwork
from Models import MuZero_torch_trainer as MuZero_trainer
from torch.utils.tensorboard import SummaryWriter


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
            actions, policy, string_samples = agent.policy(player_observation)
            step_actions = self.getStepActions(terminated, actions)
            storage_actions = utils.decode_action(actions)

            # Take that action within the environment and return all of our information for the next player
            next_observation, reward, terminated, _, info = env.step(step_actions)
            # store the action for MuZero
            for i, key in enumerate(terminated.keys()):
                if not info[key]["state_empty"]:
                    # Store the information in a buffer to train on later.
                    buffers.store_replay_buffer(key, player_observation[0][i], storage_actions[i], reward[key],
                                                policy[i], string_samples[i])

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

    def evaluate_agents(self, env ):
        info = {player_id: {"player_won": False}
                    for player_id in env.possible_agents}
        info2 = {i:{"traits used":0, "traits list":[],"xp bought":0, "champs bought":0, "2* champs":0, "2* champ list":[], "3* champs":0, "3* champ list":[]} for i in range(8)}
        info3 = {i:0 for i in range(8)}
        while True:
            agents = {"player_" + str(r): MCTS(TFTNetwork())
                  for r in range(config.NUM_PLAYERS)}
            agents["player_0"].network.tft_load_model(25)
            agents["player_1"].network.tft_load_model(0)
            agents["player_2"].network.tft_load_model(0)
            agents["player_3"].network.tft_load_model(0)
            agents["player_4"].network.tft_load_model(0)
            agents["player_5"].network.tft_load_model(0)
            agents["player_6"].network.tft_load_model(0)
            agents["player_7"].network.tft_load_model(7800)
            # Reset the environment
            player_observation = env.reset()
            # This is here to make the input (1, observation_size) for initial_inference
            player_observation = self.observation_to_input(player_observation)
            # Used to know when players die and which agent is currently acting
            terminated = {
                player_id: False for player_id in env.possible_agents}
            # Current action to help with MuZero
            self.placements = {
                player_id: 0 for player_id in env.possible_agents}
            current_position = 7
            
            #position in log file: 
            pos = 0
            # While the game is still going on.
            
            while not all(terminated.values()):
                # Ask our model for an action and policy
                actions = []
                for i,key in enumerate(agents):
                    action, _, _ = agents[key].policy([np.asarray([player_observation[0][i]]), [player_observation[1][i]]])
                    actions.append(action)
                actions = [i[0] for i in actions]

                step_actions = self.getStepActions(terminated, actions)

                # Take that action within the environment and return all of our information for the next player
                next_observation, reward, terminated, _, info = env.step(step_actions)
                #get info about actions
                log = open('log.txt','r')
                count = 1
                for line in log:
                    if count >= pos: #ensures code is ran only once
                        if line[0] != "E" and line[0] != "S": # Eliminate END ROUND and START GAME line s
                            player_num = int(line[0]) # first char is always player num
                            if "level = 2" in line: # Tier 2 champs 
                                champ = line[line.index("champion "):].split(" ")[1] #get actual champ name 

                                if champ not in info2[player_num]["2* champ list"]: #avoid duplicates 
                                    info2[player_num]["2* champs"] += 1 
                                    info2[player_num]["2* champ list"].append(champ)
                            if "level = 3" in line: # Tier 3 champs 
                                champ = line[line.index("champion "):].split(" ")[1] #get actual champ name 

                                if champ not in info2[player_num]["3* champ list"]: #avoid duplicates 
                                    info2[player_num]["3* champs"] += 1 
                                    info2[player_num]["3* champ list"].append(champ)
                            elif "Spending gold on champion" in line: 
                                info2[player_num]["champs bought"] += 1 
                            elif "exp" in line: # think this is when xp is bought 
                                info2[player_num]["xp bought"] += 1
                            if "tier: " in line and "tier: 0" not in line: # checks traits same as lvl 2 champs 
                                trait = line[:line.index("tier:")].split(" ")
                                trait = trait[-3]
                                if trait not in info2[player_num]["traits list"]:
                                    info2[player_num]["traits used"] += 1
                                    info2[player_num]["traits list"].append(trait)

                    count += 1 
                pos = count     

                # store the action for MuZero
                # Set up the observation for the next action
                player_observation = self.observation_to_input(next_observation)
                for key, terminate in terminated.items():
                    if terminate:
                        self.placements[key] = current_position
                        current_position -= 1
                        # print(key)
                        del agents[key]
            print(self.placements)
            for x,i in enumerate(self.placements):
                info3[x] += self.placements[i]
            print(info3)
        return(info2)


class AIInterface:

    def __init__(self):
        ...

    def train_model(self, starting_train_step=0):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
        train_step = starting_train_step

        global_buffer = GlobalBuffer()

        env = parallel_env()
        data_workers = DataWorker(0)
        global_agent = TFTNetwork()
        global_agent.tft_load_model(train_step)

        trainer = MuZero_trainer.Trainer(global_agent)
        train_summary_writer = SummaryWriter(train_log_dir)

        while True:
            weights = global_agent.get_weights()
            buffers = BufferWrapper(global_buffer)
            data_workers.collect_gameplay_experience(env, buffers, weights)

            while global_buffer.available_batch():
                gameplay_experience_batch = global_buffer.sample_batch()
                trainer.train_network(gameplay_experience_batch, global_agent, train_step, train_summary_writer)
                train_step += 1
                if train_step % 100 == 0:
                    global_agent.tft_save_model(train_step)
    
     
    def evaluate(self):
        env = parallel_env()
        data_workers = DataWorker(0)
        data_workers.evaluate_agents(env)

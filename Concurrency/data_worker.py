import time
import ray
import copy
import numpy as np
import config

from Simulator import utils
from Models.MCTS_torch import MCTS
from Models.MCTS_default_torch import Default_MCTS
from Models.MuZero_torch_agent import MuZeroNetwork as TFTNetwork
from Models.Muzero_default_agent import MuZeroDefaultNetwork as DefaultNetwork
from Simulator.tft_item_simulator import TFT_Item_Simulator
from Simulator.tft_position_simulator import TFT_Position_Simulator
from config import GPU_SIZE_PER_WORKER

# from pettingzoo.test import api_test, parallel_api_test as parallel_test

'''
Description - 
    Data workers are the "workers" or threads that collect game play experience. Can add scheduling_strategy="SPREAD" 
    to ray.remote. Not sure if it makes any difference
'''


@ray.remote(num_gpus=GPU_SIZE_PER_WORKER)
class DataWorker(object):
    def __init__(self, rank, model_config):
        if config.CHAMP_DECIDER:
            self.temp_model = DefaultNetwork(model_config)
            self.agent_network = Default_MCTS(self.temp_model, model_config)
            self.past_network = Default_MCTS(self.temp_model, model_config)
            self.default_agent = [False for _ in range(config.NUM_PLAYERS)]
        else:
            self.temp_model = TFTNetwork(model_config)
            self.agent_network = MCTS(self.temp_model, model_config)
            self.past_network = MCTS(self.temp_model, model_config)
            self.default_agent = [False for _ in range(config.NUM_PLAYERS)]
            # self.default_agent = [np.random.rand() < 0.5 for _ in range(config.NUM_PLAYERS)]
            # Ensure we have at least one model player and for testing
            # self.default_agent[0] = False

        self.past_version = [False for _ in range(config.NUM_PLAYERS)]

        # Testing purposes only
        self.live_game = True
        self.rank = rank
        self.ckpt_time = time.time_ns()
        self.prob = 1
        self.past_episode = 0
        self.past_update = True
        self.model_config = model_config

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
            player_observation, info = env.reset(options={"default_agent": self.default_agent})
            # This is here to make the input (1, observation_size) for initial_inference
            player_observation = self.observation_to_input(player_observation)

            # Used to know when players die and which agent is currently acting
            terminated = {player_id: False for player_id in env.possible_agents}
            position = 8

            # Doing this to try to get around the edge case of the very last time step for each player where
            # The player is null but we still need to collect a reward.
            current_comp = {
                key: info[key]["player"].get_tier_labels() for key in terminated.keys()
            }
            current_champs = {
                key: info[key]["player"].get_champion_labels() for key in terminated.keys()
            }

            # While the game is still going on.
            while not all(terminated.values()):
                # Ask our model for an action and policy. Use on normal case or if we only have current versions left
                actions, policy, string_samples, root_values = self.model_call(player_observation, info)
                storage_actions = utils.decode_action(actions)
                step_actions = self.getStepActions(terminated, storage_actions)

                # Take that action within the environment and return all of our information for the next player
                next_observation, reward, terminated, _, info = env.step(step_actions)
                # store the action for MuZero
                for i, key in enumerate(terminated.keys()):
                    if not info[key]["state_empty"]:
                        if not self.past_version[i] and (not self.default_agent[i] or config.IMITATION) \
                                and np.random.rand() <= config.CHANCE_BUFFER_SEND:
                            if info[key]["player"]:
                                current_comp[key] = info[key]["player"].get_tier_labels()
                                current_champs[key] = info[key]["player"].get_champion_labels()
                            # Store the information in a buffer to train on later.
                            buffers.store_replay_buffer.remote(key, self.get_obs_idx(player_observation[0], i),
                                                               storage_actions[i], reward[key], policy[i],
                                                               string_samples[i], root_values[i], current_comp[key],
                                                               current_champs[key])

                offset = 0
                for i, [key, terminate] in enumerate(terminated.items()):
                    # Saying if that any of the 4 agents got first or second then we are saying we are not
                    # Currently beating that checkpoint
                    if terminate:
                        # print("player {} got position {} of game {}".format(i, position, self.rank))
                        buffers.set_ending_position.remote(key, position)
                        position -= 1
                        self.past_version.pop(i - offset)
                        self.default_agent.pop(i - offset)
                        offset += 1

                if not any(self.past_version) and len(terminated) == 2 and not self.past_update:
                    storage.update_checkpoint_score.remote(self.past_episode, self.prob)
                    self.past_update = True

                # Set up the observation for the next action
                player_observation = self.observation_to_input(next_observation)

            # buffers.rewardNorm.remote()
            buffers.store_global_buffer.remote(global_buffer)

            buffers.reset_buffers.remote()

            # Might want to get rid of the hard constant 0.8 for something that can be adjusted in the future
            # Disabling to test out the default agent
            self.live_game = np.random.rand() <= 0.5
            self.past_version = [False for _ in range(config.NUM_PLAYERS)]
            if not self.live_game:
                [past_weights, self.past_episode, self.prob] = ray.get(storage.sample_past_model.remote())
                self.past_network = MCTS(self.temp_model, self.model_config)
                self.past_network.network.set_weights(past_weights)
                self.past_version[0:4] = [True, True, True, True]
                self.past_update = False

            # Reset the default agents for the next set of games.
            # self.default_agent = [np.random.rand() < 0.5 for _ in range(config.NUM_PLAYERS)]
            self.default_agent = [False for _ in range(config.NUM_PLAYERS)]
            # Ensure we have at least one model player
            # if all(self.default_agent):
            #     self.default_agent[0] = False

            # This is just to try to give the buffer some time to train on the data we have and not slow down our
            # buffers by sending thousands of store commands when the buffer is already full.
            while global_buffer.buffer_size() > config.GLOBAL_BUFFER_SIZE * 0.8:
                time.sleep(5)

            # So if I do not have a live game, I need to sample a past model
            # Which means I need to create a list within the storage and sample from that.
            # All the probability distributions will be within the storage class as well.
            temp_weights = ray.get(storage.get_model.remote())
            weights = copy.deepcopy(temp_weights)
            self.agent_network = MCTS(self.temp_model, self.model_config)
            self.agent_network.network.set_weights(weights)
            self.rank += config.CONCURRENT_GAMES

    '''
    Description -
        Each worker runs one full game and will restart after the game finishes. At the end of the game, 
        it will fetch a new agent. Same as above but runs the default model architecture instead of the model 
        architecture that decides every action. 
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

    def collect_default_experience(self, env, buffers, global_buffer, storage, weights,
                                   item_storage, positioning_storage, position_model_storage, item_model_storage):
        # print("Checkpoint 0")
        self.agent_network.network.set_weights(weights)
        while True:
            # Reset the environment
            next_observation, info = env.reset(options={"default_agent": self.default_agent})
            # This is here to make the input (1, observation_size) for initial_inference
            player_observation = self.observation_to_input(next_observation)

            # Used to know when players die and which agent is currently acting
            terminated = {player_id: False for player_id in env.possible_agents}
            reward = {player_id: 0 for player_id in env.possible_agents}
            position = 8

            # Doing this to try to get around the edge case of the very last time step for each player where
            # The player is null but we still need to collect a reward.
            current_comp = {
                key: info[key]["player"].get_tier_labels() for key in terminated.keys()
            }
            current_champs = {
                key: info[key]["player"].get_champion_labels() for key in terminated.keys()
            }

            # print("Checkpoint 1")
            # position_model = ray.get(position_model_storage.fetch_model.remote())
            # item_model = ray.get(item_model_storage.fetch_model.remote())

            # While the game is still going on.
            while not all(terminated.values()):
                info_values = list(info.values())
                if info_values[0]['start_turn']:
                    # Ask our model for an action and policy.
                    # Use on normal case or if we only have current versions left
                    c_actions, policy, string_samples, root_values = self.agent_network.policy(player_observation[:2])
                    # print("Checkpoint 2")
                    # item_position_observation = self.observation_to_pos_item_input(next_observation)
                    # print("Checkpoint 3")
                    # positioning_commands = position_model.evaluate(item_position_observation)
                    # print("Checkpoint 4")
                    # item_commands = item_model.evaluate(item_position_observation)
                    # print("Checkpoint 5")
                    storage_actions = utils.decode_action(c_actions)
                    step_actions = self.getStepActions(terminated, storage_actions)
                    for player_id in step_actions.keys():
                        info[player_id]['player'].default_guide(step_actions[player_id])

                    # store the action for MuZero
                    for i, key in enumerate(terminated.keys()):
                        if info[key]["player"]:
                            current_comp[key] = info[key]["player"].get_tier_labels()
                            current_champs[key] = info[key]["player"].get_champion_labels()
                        # Store the information in a buffer to train on later.
                        buffers.store_replay_buffer.remote(key, self.get_obs_idx(player_observation[0], i),
                                                           storage_actions[i], reward[key], policy[i],
                                                           string_samples[i], root_values[i], current_comp[key],
                                                           current_champs[key])

                actions = ["0"] * len(self.default_agent)
                for i, _ in enumerate(self.default_agent):
                    actions[i] = info_values[i]["player"].default_policy(info_values[i]["game_round"],
                                                                         info_values[i]["shop"],
                                                                         player_observation[1][i])
                storage_actions = utils.decode_action(actions)
                step_actions = self.getStepActions(terminated, storage_actions)
                # Take that action within the environment and return all of our information for the next player
                next_observation, reward, terminated, _, info = env.step(step_actions)
                player_observation = self.observation_to_input(next_observation)

                offset = 0
                for i, [key, terminate] in enumerate(terminated.items()):
                    # Saying if that any of the 4 agents got first or second then we are saying we are not
                    # Currently beating that checkpoint
                    if terminate:
                        item_storage.push((info[key]["player"], info[key]["player"].opponent, None,
                                           info[key]["player"].default_agent.item_guide))
                        # print("player {} got position {} of game {}".format(i, position, self.rank))
                        buffers.set_ending_position.remote(key, position)
                        position -= 1
                        self.default_agent.pop(i - offset)
                        offset += 1

                # only need to do this if it is the start of the turn
                if info_values[0]['start_turn']:
                    # Set up the observation for the next action
                    for i, key in enumerate(terminated.keys()):
                        if info[key]["save_battle"]:
                            if positioning_storage.q_size() < 10 or positioning_storage.q_size() > 990 or \
                                    item_storage.q_size() < 10 or item_storage.q_size() > 990:
                                time.sleep(2)
                            # all of the info needed for training the positioning model
                            positioning_storage.push((info[key]["player"], info[key]["player"].opponent,
                                                      {local_key: info[local_key]["player"]
                                                       for local_key in terminated.keys()}))

                            # Most of this information comes from the player class but I want to send it separately
                            # As to be clear what the item decider is training with.
                            item_storage.push((info[key]["player"], info[key]["player"].opponent,
                                               {local_key: info[local_key]["player"]
                                                for local_key in terminated.keys()},
                                               np.where(info[key]["player"].default_agent.item_guide == [0, 1, 0],
                                                        [0, 1, 0], [1, 0, 0])))

                            item_storage.push((info[key]["player"], info[key]["player"].opponent, None,
                                               np.where(info[key]["player"].default_agent.item_guide == [0, 0, 1],
                                                        [0, 1, 0], [1, 0, 0])))

            self.default_agent = [False for _ in range(config.NUM_PLAYERS)]

            # buffers.rewardNorm.remote()
            buffers.store_global_buffer.remote(global_buffer)
            buffers.reset_buffers.remote()

            # All the probability distributions will be within the storage class as well.
            temp_weights = ray.get(storage.get_model.remote())
            weights = copy.deepcopy(temp_weights)
            self.agent_network = Default_MCTS(self.temp_model, self.model_config)
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
            if not terminate and i < len(actions):
                step_actions[player_id] = actions[i]
            elif not terminate and i >= len(actions):
                # Some bug here but I'm not sure the source so sending a passing action. Happens once every 100 games
                step_actions[player_id] = np.asarray([0, 0, 0])
            i += 1
        return step_actions

    '''
    Description -
        Turns a dictionary of player observations into a list of list format that the model can use.
        Adding key to the list to ensure the right values are attached in the right places in debugging.
    '''

    def observation_to_input(self, observation):
        masks = []
        keys = []
        scalars = []
        shop = []
        board = []
        bench = []
        items = []
        traits = []
        other_players = []
        for key, obs in observation.items():
            scalars.append(obs["player"]["scalars"])
            shop.append(obs["player"]["shop"])
            board.append(obs["player"]["board"])
            bench.append(obs["player"]["bench"])
            items.append(obs["player"]["items"])
            traits.append(obs["player"]["traits"])
            local_other_players = np.concatenate([obs["opponents"][0]["board"], obs["opponents"][0]["scalars"],
                                                  obs["opponents"][0]["traits"]], axis=-1)
            # minus 1 because I filter out that player's observation in the player manager
            for x in range(1, config.NUM_PLAYERS - 1):
                local_other_players = np.concatenate(
                    [local_other_players, (np.concatenate([obs["opponents"][x]["board"],
                                                           obs["opponents"][x]["scalars"],
                                                           obs["opponents"][x]["traits"]], axis=-1))], )
            other_players.append(local_other_players)
            masks.append(obs["action_mask"])
            keys.append(key)
        tensors = {
            "scalars": np.array(scalars),
            "shop": np.array(shop),
            "board": np.array(board),
            "bench": np.array(bench),
            "items": np.array(items),
            "traits": np.array(traits),
            "other_players": np.array(other_players)
        }
        masks = np.array(masks)
        return [tensors, masks, keys]

    '''
        Description -
            Turns a dictionary of player observations into a list of list format that the model can use.
            Adding key to the list to ensure the right values are attached in the right places in debugging.
        '''

    def observation_to_pos_item_input(self, observation):
        observation = {player: {key: observation[player][key] for key in ["player", "opponents"]}
                       for player in range(observation.keys())}
        return observation

    def get_obs_idx(self, observation, idx):
        return {
            "scalars": observation["scalars"][idx],
            "shop": observation["shop"][idx],
            "board": observation["board"][idx],
            "bench": observation["bench"][idx],
            "items": observation["items"][idx],
            "traits": observation["traits"][idx],
            "other_players": observation["other_players"][idx]
        }

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

    """
    Description - 
        Determines which model call is needed for the current timestep.
    Inputs      -
        player_observation
            The dictionary of vectors that is the agents view of the world.
        info
            The dictionary of info that is returned from the simulator. Used for default agents.
    """

    def model_call(self, player_observation, info):
        if config.IMITATION:
            actions, policy, string_samples, root_values = self.imitation_learning(info, player_observation[1])
        # If all of our agents are current versions
        elif (self.live_game or not any(self.past_version)) and not any(self.default_agent):
            actions, policy, string_samples, root_values = self.agent_network.policy(player_observation[:2])
        # Ff all of our agents are past versions. (Should exceedingly rarely come here)
        elif all(self.past_version) and not any(self.default_agent):
            actions, policy, string_samples, root_values = self.past_network.policy(player_observation[:2])
        # If all of our versions are default agents
        elif all(self.default_agent):
            actions, policy, string_samples, root_values = self.default_model_call(info)
        # If there are no default agents but a mix of past and present
        elif not any(self.default_agent):
            actions, policy, string_samples, root_values = self.mixed_ai_model_call(player_observation[:2])
        # Implement the remaining mixes of agents here.
        elif not any(self.past_version):
            actions, policy, string_samples, root_values = self.live_default_model_call(player_observation[:2], info)
        # If we only have default_agents remaining.
        else:
            actions, policy, string_samples, root_values = self.default_model_call(info)
        return actions, policy, string_samples, root_values

    """
    Description - 
        Model call if some of the players are current agents and some of the players are past agents.
    """

    def mixed_ai_model_call(self, player_observation):
        live_observation, past_observation = self.split_live_past_observations(player_observation)
        live_actions, live_policy, live_string_samples, live_root_values = self.agent_network.policy(live_observation)
        past_actions, past_policy, past_string_samples, past_root_values = self.past_network.policy(past_observation)
        actions = [None] * len(self.past_version)
        policy = [None] * len(self.past_version)
        string_samples = [None] * len(self.past_version)
        root_values = [None] * len(self.past_version)
        counter_live, counter_past = 0, 0
        for i, past_version in enumerate(self.past_version):
            if past_version:
                actions[i] = past_actions[counter_past]
                policy[i] = past_policy[counter_past]
                string_samples[i] = past_string_samples[counter_past]
                root_values[i] = past_root_values[counter_past]
                counter_past += 1
            else:
                actions[i] = live_actions[counter_live]
                policy[i] = live_policy[counter_live]
                string_samples[i] = live_string_samples[counter_live]
                root_values[i] = live_root_values[counter_live]
                counter_live += 1
        return actions, policy, string_samples, root_values

    def split_live_past_observations(self, player_observation):
        live_agent_observations = {
            "scalars": [],
            "shop": [],
            "board": [],
            "bench": [],
            "items": [],
            "traits": [],
            "other_players": []
        }
        past_agent_observations = {
            "scalars": [],
            "shop": [],
            "board": [],
            "bench": [],
            "items": [],
            "traits": [],
            "other_players": []
        }
        live_agent_masks = []
        past_agent_masks = []

        for i, past_version in enumerate(self.past_version):
            local_obs = self.get_obs_idx(player_observation[0], i)
            local_mask = player_observation[1][i]
            if not past_version:
                for key in live_agent_observations.keys():
                    live_agent_observations[key].append(local_obs[key])
                live_agent_masks.append(local_mask)
            else:
                for key in past_agent_observations.keys():
                    past_agent_observations[key].append(local_obs[key])
                past_agent_masks.append(local_mask)

        for key in live_agent_observations.keys():
            live_agent_observations[key] = np.asarray(live_agent_observations[key])

        for key in past_agent_observations.keys():
            past_agent_observations[key] = np.asarray(past_agent_observations[key])

        live_observation = [live_agent_observations, live_agent_masks]
        past_observation = [past_agent_observations, past_agent_masks]
        return live_observation, past_observation

    """
    Description - 
        Model call if some of the agents are live and some of them are default agents.
    """

    def live_default_model_call(self, player_observation, info):
        actions = ["0"] * len(self.default_agent)
        policy = [None] * len(self.default_agent)
        string_samples = [None] * len(self.default_agent)
        root_values = [0] * len(self.default_agent)

        live_agent_observations = []
        live_agent_masks = []

        for i, default_agent in enumerate(self.default_agent):
            if not default_agent:
                live_agent_observations.append(self.get_obs_idx(player_observation[0], i))
                live_agent_masks.append(player_observation[1][i])

        live_observation = [live_agent_observations, live_agent_masks]
        if len(live_observation[0]) != 0:
            live_actions, live_policy, live_string_samples, live_root_values = \
                self.agent_network.policy(live_observation)

            counter_live, counter_default = 0, 0
            local_info = list(info.values())
            for i, default_agent in enumerate(self.default_agent):
                if default_agent:
                    actions[i] = local_info[i]["player"].default_policy(local_info[i]["game_round"],
                                                                        local_info[i]["shop"])
                    counter_default += 1
                else:
                    actions[i] = live_actions[counter_live]
                    policy[i] = live_policy[counter_live]
                    string_samples[i] = live_string_samples[counter_live]
                    root_values[i] = live_root_values[counter_live]
                    counter_live += 1
        else:
            counter_default = 0
            local_info = list(info.values())
            for i, default_agent in enumerate(self.default_agent):
                if default_agent:
                    print("{} and player_num {} also why".format(i, local_info[i]["player"].player_num))
                    actions[i] = local_info[i]["player"].default_policy(local_info[i]["game_round"],
                                                                        local_info[i]["shop"])
                    # Turn action into one hot policy
                    counter_default += 1
        return actions, policy, string_samples, root_values

    """
    Description - 
        Model call if all of the agents are a mix of default and past. Currently never called.
    """

    def past_default_model_call(self, info):
        actions = [None] * len(self.default_agent)
        policy = [None] * len(self.default_agent)
        string_samples = [None] * len(self.default_agent)
        root_values = [None] * len(self.default_agent)
        return actions, policy, string_samples, root_values

    """
    Description - 
        Model call if you have a mix of all 3 model types in a timestep. Currently never called.
    """

    def live_past_default_model_call(self, info):
        actions = [None] * len(self.default_agent)
        policy = [None] * len(self.default_agent)
        string_samples = [None] * len(self.default_agent)
        root_values = [None] * len(self.default_agent)
        return actions, policy, string_samples, root_values

    """
    Description - 
        Called if all players are default agents.
    """

    def default_model_call(self, info):
        actions = ["0"] * len(self.default_agent)
        policy = [None] * len(self.default_agent)
        string_samples = [None] * len(self.default_agent)
        root_values = [1] * len(self.default_agent)
        local_info = list(info.values())
        for i, default_agent in enumerate(self.default_agent):
            if default_agent:
                actions[i] = local_info[i]["player"].default_policy(local_info[i]["game_round"], local_info[i]["shop"])
        return actions, policy, string_samples, root_values

    """
    Description - 
        Model call if doing imitation learning. Only calls the default policy
    """

    def imitation_learning(self, info, mask):
        policy = [[1.0] for _ in range(len(self.default_agent))]
        string_samples = [[] for _ in range(len(self.default_agent))]
        root_values = [0 for _ in range(len(self.default_agent))]
        actions = ["0" for _ in range(len(self.default_agent))]

        local_info = list(info.values())
        for i, default_agent in enumerate(self.default_agent):
            string_samples[i] = \
                [local_info[i]["player"].default_policy(local_info[i]["game_round"], local_info[i]["shop"], mask[i])]
            actions[i] = string_samples[i][0]
        return actions, policy, string_samples, root_values

    def test_position_item_simulators(self, position_storage, item_storage):
        position_env = TFT_Position_Simulator(position_storage)
        item_env = TFT_Item_Simulator(item_storage)
        while True:
            position_player_observation = position_env.reset()
            item_player_observation = item_env.reset()

            # parallel_test(position_env, num_cycles=1)
            # parallel_test(item_env, num_cycles=1)

            _, reward, terminated, _, info = position_env.step({"player_0": np.random.randint(0, 29, 12)})
            _, reward, terminated, _, info = item_env.step({"player_0": np.random.randint(0, 29, 12)})

import config
import copy
import numpy as np
import time
from Core.TorchModels.MuZero_torch_agent import MuZeroNetwork as TFTNetwork
from Simulator import utils
from Core.MCTS_Trees.MCTS_torch import MCTS
from Core.MCTS_Trees.MCTS_default_torch import Default_MCTS
from Core.TorchModels.Muzero_default_agent import MuZeroDefaultNetwork as DefaultNetwork
from Core.TorchModels.MuZero_position_torch_agent import MuZero_Position_Network as PositionNetwork
from Core.MCTS_Trees.MCTS_position_torch import MCTS as Position_MCTS

class DataWorker(object):
    def __init__(self, rank, model_config):
        if config.CHAMP_DECIDER:
            self.temp_model = DefaultNetwork(model_config)
            self.agent_network = Default_MCTS(self.temp_model, model_config)
            self.past_network = Default_MCTS(self.temp_model, model_config)
            self.default_agent = [True for _ in range(config.NUM_PLAYERS)]
        elif config.MUZERO_POSITION:
            self.temp_model = PositionNetwork(model_config)
            self.agent_network = Position_MCTS(self.temp_model, model_config)
            self.past_network = Position_MCTS(self.temp_model, model_config)
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
    def collect_gameplay_experience(self, env, buffers, weights):
        self.agent_network.network.set_weights(weights)
        self.past_network.network.set_weights(weights)
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
                        buffers.store_replay_buffer(key, self.get_obs_idx(player_observation[0], i), storage_actions[i],
                                                    reward[key], policy[i], root_values[i],
                                                    string_samples=string_samples[i], team_tiers=current_comp[key],
                                                    team_champions=current_champs[key])

            offset = 0
            for i, [key, terminate] in enumerate(terminated.items()):
                # Saying if that any of the 4 agents got first or second then we are saying we are not
                # Currently beating that checkpoint
                if terminate:
                    # print("player {} got position {} of game {}".format(i, position, self.rank))
                    buffers.set_ending_position(key, position)
                    position -= 1
                    self.past_version.pop(i - offset)
                    self.default_agent.pop(i - offset)
                    offset += 1

            # Set up the observation for the next action
            player_observation = self.observation_to_input(next_observation)

        # buffers.rewardNorm.remote()
        buffers.store_global_buffer()
        self.default_agent = [False for _ in range(config.NUM_PLAYERS)]
        self.past_version = [False for _ in range(config.NUM_PLAYERS)]

    def collect_default_experience(self, env, buffers, weights):
        self.agent_network.network.set_weights(weights)
        while True:
            # Reset the environment
            player_observation, info = env.reset(options={"default_agent": self.default_agent})
            # This is here to make the input (1, observation_size) for initial_inference
            player_observation = self.observation_to_input(player_observation)

            # Used to know when players die and which agent is currently acting
            terminated = {player_id: False for player_id in env.possible_agents}
            reward = {player_id: 0 for player_id in env.possible_agents}
            position = 8

            # While the game is still going on.
            while not all(terminated.values()):
                info_values = list(info.values())
                if info_values[0]['start_turn']:
                    # Ask our model for an action and policy.
                    # Use on normal case or if we only have current versions left
                    c_actions, policy, string_samples, root_values = self.agent_network.policy(player_observation[:2])
                    storage_actions = utils.decode_action(c_actions)
                    step_actions = self.getStepActions(terminated, storage_actions)
                    for player_id in step_actions.keys():
                        info[player_id]['player'].default_champion_list(step_actions[player_id])

                    # store the action for MuZero
                    for i, key in enumerate(terminated.keys()):
                        # Store the information in a buffer to train on later.
                        buffers.store_replay_buffer(key, player_observation[0][i], storage_actions[i], reward[key],
                                                    policy[i], root_values[i], string_samples=string_samples[i])

                actions = ["0"] * len(self.default_agent)
                for i, default_agent in enumerate(self.default_agent):
                    actions[i] = info_values[i]["player"].default_policy(info_values[i]["game_round"],
                                                                         info_values[i]["shop"])
                storage_actions = utils.decode_action(actions)
                step_actions = self.getStepActions(terminated, storage_actions)
                # Take that action within the environment and return all of our information for the next player
                next_observation, reward, terminated, _, info = env.step(step_actions)

                offset = 0
                for i, [key, terminate] in enumerate(terminated.items()):
                    # Saying if that any of the 4 agents got first or second then we are saying we are not
                    # Currently beating that checkpoint
                    if terminate:
                        # print("player {} got position {} of game {}".format(i, position, self.rank))
                        buffers.set_ending_position(key, position)
                        position -= 1
                        self.default_agent.pop(i - offset)
                        offset += 1

                # Set up the observation for the next action
                player_observation = self.observation_to_input(next_observation)

            self.default_agent = [True for _ in range(config.NUM_PLAYERS)]

            print("SENDING TO BUFFER AT TIME {}".format(time.time_ns() - self.ckpt_time))
            buffers.store_global_buffer()

    def collect_position_experience(self, env, buffers, weights):
        self.temp_model = PositionNetwork(self.model_config)
        self.agent_network = Position_MCTS(self.temp_model, self.model_config)
        self.past_network = Position_MCTS(self.temp_model, self.model_config)
        self.default_agent = [False for _ in range(config.NUM_PLAYERS)]
        self.agent_network.network.set_weights(weights)
        # episode_rewards = [[0] for _ in range(env.num_envs)]
        # Reset the environment
        player_observation, info = env.vector_reset()
        player_observation = env.list_to_dict([player_observation])

        # Used to know when players die and which agent is currently acting
        terminated = [False for _ in range(env.num_envs)]
        storage_terminated = [False for _ in range(env.num_envs)]
        action_count = [0 for _ in range(env.num_envs)]
        action_limits = [info[i]["num_units"] + 1 for i in range(len(info))]
        rewards = []

        # While the game is still going on.
        while not all(terminated):
            simulator_envs = copy.deepcopy(env.envs)
            # Ask our model for an action and policy. Use on normal case or if we only have current versions left
            actions, policy, root_values = self.agent_network.policy(player_observation, simulator_envs,
                                                                     action_count, action_limits)
            step_actions = np.array(actions)

            # Take that action within the environment and return all of our information for the next player
            next_observation, reward, terminated, _, info = env.vector_step(step_actions)
            # store the action for MuZero
            # Using i for all possible players and alive_i for all alive players
            alive_i = 0
            action_limits = []
            for i in range(len(terminated)):
                if not storage_terminated[i]:
                    # Store the information in a buffer to train on later.
                    buffers.store_replay_buffer(f"player_{i}", self.get_position_obs_idx(
                        player_observation["observations"], alive_i), step_actions[alive_i], reward[alive_i],
                                                       policy[alive_i], root_values[alive_i])
                    if terminated[i]:
                        storage_terminated[i] = True
                        rewards.append(reward[alive_i])
                        # print(f"game {i} ended with reward {reward[alive_i]}")
                        # episode_rewards[i].append(reward[alive_i])
                        #
                        # if self.check_greater_than_last_five(episode_rewards[i]):
                        #     episode_rewards[i] = []
                            # env.envs[i].level_up()
                        #
                        # if len(episode_rewards[i]) > 100:
                        #     episode_rewards[i] = episode_rewards[i][-95:]

                    action_limits.append(info[alive_i]["num_units"] + 1)

                    alive_i += 1
                    action_count[i] += 1

            # Set up the observation for the next action
            player_observation = env.list_to_dict([next_observation])

        buffers.store_global_buffer()

        return sum(rewards) / len(rewards)

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
            for x in range(1, config.NUM_PLAYERS):
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

    def get_position_obs_idx(self, observation, idx):
        return {
            "board": observation["board"][idx],
            "traits": observation["traits"][idx],
            "action_count": observation["action_count"][idx],
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

    def model_call(self, player_observation, info):
        if config.IMITATION:
            actions, policy, string_samples, root_values = self.imitation_learning(player_observation[:2], info)
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

    def mixed_ai_model_call(self, player_observation):
        # I need to send the observations that are part of the past players to one token
        # and send the ones that are part of the live players to another token
        # Problem comes if I want to do multiple different versions in a game.
        # I could limit it to one past verison. That is probably going to be fine for our purposes
        # Note for later that the current implementation is only going to be good for one past version
        # For our purposes, lets just start with half of the agents being false.

        # Now that I have the array for the past versions down to only the remaining players
        # Separate the observation.

        print("In the mixed model call")
        live_agent_observations = []
        past_agent_observations = []
        live_agent_masks = []
        past_agent_masks = []

        for i, past_version in enumerate(self.past_version):
            if not past_version:
                live_agent_observations.append(self.get_obs_idx(player_observation[0], i))
                live_agent_masks.append(player_observation[1][i])
            else:
                past_agent_observations.append(player_observation[0][i])
                past_agent_masks.append(player_observation[1][i])
        live_observation = [np.asarray(live_agent_observations), live_agent_masks]
        past_observation = [np.asarray(past_agent_observations), past_agent_masks]
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

    def past_default_model_call(self, info):
        actions = [None] * len(self.default_agent)
        policy = [None] * len(self.default_agent)
        string_samples = [None] * len(self.default_agent)
        root_values = [None] * len(self.default_agent)
        return actions, policy, string_samples, root_values

    def live_past_default_model_call(self, info):
        actions = [None] * len(self.default_agent)
        policy = [None] * len(self.default_agent)
        string_samples = [None] * len(self.default_agent)
        root_values = [None] * len(self.default_agent)
        return actions, policy, string_samples, root_values

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

    def imitation_learning(self, player_observation, info):
        policy = [[1.0] for _ in range(len(self.default_agent))]
        string_samples = [[] for _ in range(len(self.default_agent))]
        root_values = [0 for _ in range(len(self.default_agent))]
        actions = ["0" for _ in range(len(self.default_agent))]

        local_info = list(info.values())
        for i, default_agent in enumerate(self.default_agent):
            string_samples[i] = [local_info[i]["player"].default_policy(local_info[i]["game_round"],
                                                                        local_info[i]["shop"])]
            actions[i] = string_samples[i][0]
        return actions, policy, string_samples, root_values

import config
import torch
import numpy as np
import Core.CplusplusTrees.ctree.cytree as tree
from Core.MCTS_Trees.MCTS_torch import MCTS

class Default_MCTS(MCTS):
    def __init__(self, network, model_config):
        super().__init__(network, model_config)
        self.default_string_mapping = []
        self.champ_decider_action_dim = config.CHAMP_DECIDER_ACTION_DIM
        self.model_config = model_config

    def policy(self, observation):
        with torch.no_grad():
            self.NUM_ALIVE = observation[0]["shop"].shape[0]

            # 0.02 seconds
            network_output = self.network.initial_inference(observation[0])

            reward_pool = np.array(network_output["reward"]).reshape(-1).tolist()

            policy_logits = [output_head.cpu().numpy() for output_head in network_output["policy_logits"]]

            # 0.01 seconds
            policy_logits_pool, string_mapping = self.encode_action_to_str(policy_logits, observation[1])

            noises = [
                [
                    np.random.dirichlet([self.model_config.ROOT_DIRICHLET_ALPHA] * len(policy_logits_pool[i][j])
                                        ).astype(np.float32).tolist()
                    for j in range(self.NUM_ALIVE)
                ]
                for i in range(len(policy_logits_pool))
            ]

            # Policy Logits -> [ [], [], [], [], [], [], [], [],]

            policy_logits_pool = self.add_exploration_noise(policy_logits_pool, noises)

            # 0.003 seconds
            policy_logits_pool, string_mapping, mappings, policy_sizes = \
                self.sample(policy_logits_pool, string_mapping, self.model_config.NUM_SAMPLES)

            # less than 0.0001 seconds
            # Setup specialised roots datastructures, format: env_nums, action_space_size, num_simulations
            # Number of agents, previous action, number of simulations for memory purposes
            roots_cpp = tree.Roots(self.NUM_ALIVE, self.model_config.NUM_SIMULATIONS, self.model_config.NUM_SAMPLES)

            # 0.0002 seconds
            # prepare the nodes to feed them into batch_mcts,
            # for statement to deal with different lengths due to masking.
            roots_cpp.prepare_no_noise(reward_pool, policy_logits_pool, mappings, policy_sizes)

            # Output for root node
            hidden_state_pool = network_output["hidden_state"]

            # set up nodes to be able to find and select actions
            self.run_batch_mcts(roots_cpp, hidden_state_pool)
            roots_distributions = roots_cpp.get_distributions()

            root_values = roots_cpp.get_values()

            actions = []
            target_policy = []
            temp = self.visit_softmax_temperature()  # controls the way actions are chosen
            deterministic = False  # False = sample distribution, True = argmax
            for i in range(self.NUM_ALIVE):
                distributions = roots_distributions[i]
                action = self.select_action(distributions, temperature=temp, deterministic=deterministic)
                actions.append(string_mapping[i][action])
                target_policy.append([x / self.model_config.NUM_SIMULATIONS for x in distributions])

            # Notes on possibilities for other dimensions at the bottom
            self.num_actions += 1
            return actions, target_policy, string_mapping, root_values

    def run_batch_mcts(self, roots_cpp, hidden_state_pool):
        # preparation
        num = roots_cpp.num
        # config variables
        discount = config.DISCOUNT
        pb_c_init = self.model_config.PB_C_INIT
        pb_c_base = self.model_config.PB_C_BASE
        hidden_state_index_x = 0

        # minimax value storage data structure
        min_max_stats_lst = tree.MinMaxStatsList(num)
        hidden_state_pool = [hidden_state_pool]
        # go through the tree NUM_SIMULATIONS times
        for _ in range(self.model_config.NUM_SIMULATIONS):
            # prepare a result wrapper to transport results between python and c++ parts
            results = tree.ResultsWrapper(num)
            # 0.001 seconds
            # evaluation for leaf nodes, traversing across the tree and updating values
            hidden_state_index_x_lst, hidden_state_index_y_lst, last_action = \
                tree.batch_traverse(roots_cpp, pb_c_base, pb_c_init, discount, min_max_stats_lst, results)

            self.max_depth_search = sum(results.get_search_len()) / len(results.get_search_len())
            num_states = len(hidden_state_index_x_lst)
            tensors_states = torch.empty((num_states, self.model_config.HIDDEN_STATE_SIZE)).to(config.DEVICE)

            # obtain the states for leaf nodes
            for ix, iy, idx in zip(hidden_state_index_x_lst, hidden_state_index_y_lst, range(num_states)):
                tensors_states[idx] = hidden_state_pool[ix][iy]

            # Inside the search tree we use the dynamics function to obtain the next
            # hidden state given an action and the previous hidden state.
            last_action = np.asarray(last_action)

            # 0.003 seconds
            network_output = self.network.recurrent_inference(tensors_states, last_action)

            reward_pool = network_output["reward"].reshape(-1).tolist()
            value_pool = network_output["value"].reshape(-1).tolist()
            diff = max(value_pool) - min(value_pool)
            if diff > 150.:
                print(f"EUREKA, VALUES MAX: {max(value_pool)}, AND MIN: {min(value_pool)}, RANGE {diff}")

            policy_logits = [output_head.cpu().numpy() for output_head in network_output["policy_logits"]]

            # 0.014 seconds
            policy_logits, _, mappings, policy_sizes = \
                self.sample(policy_logits, self.default_string_mapping, self.model_config.NUM_SAMPLES)

            # These assignments take 0.0001 > time
            # add nodes to the pool after each search
            hidden_states_nodes = network_output["hidden_state"]
            hidden_state_pool.append(hidden_states_nodes)

            hidden_state_index_x += 1

            # 0.001 seconds
            # backpropagation along the search path to update the attributes
            tree.batch_back_propagate(hidden_state_index_x, discount, reward_pool, value_pool, policy_logits,
                                      min_max_stats_lst, results, mappings, policy_sizes)

    def add_exploration_noise(self, policy_logits, noises):
        exploration_fraction = self.model_config.ROOT_EXPLORATION_FRACTION
        for i in range(len(noises)):  # Batch
            for j in range(len(noises[i])):  # Policy Dims
                for k in range(len(noises[i][j])):
                    policy_logits[i][j][k] = policy_logits[i][j][k] * (1 - exploration_fraction) + \
                                             noises[i][j][k] * exploration_fraction
        return policy_logits

    def encode_action_to_str(self, policy_logits, mask):
        return policy_logits, []

    # TODO: Duplication value and shink size of array with duplicates.
    # I don't expect duplicates too often with an action space size of 2^57 but it's possible.
    def sample(self, policy_logits, string_mapping, num_samples):
        batch_size = len(policy_logits[0])  # 8

        output_logits = []
        output_string_mapping = []
        output_byte_mapping = []
        policy_sizes = []

        for idx in range(batch_size):
            local_byte = []
            sampled_action = []
            probs = self.softmax_stable(policy_logits[0][idx])
            policy_range = np.arange(stop=len(policy_logits[0][idx]))

            samples = np.random.choice(a=policy_range, p=probs, size=num_samples)
            for sample in samples:
                sampled_action.append(str(sample))

            for i in range(1, len(self.champ_decider_action_dim)):
                probs = self.softmax_stable(policy_logits[i][idx])
                policy_range = np.arange(stop=len(policy_logits[i][idx]))

                samples = np.random.choice(a=policy_range, p=probs, size=num_samples)

                for k, sample in enumerate(samples):
                    sampled_action[k] = sampled_action[k] + "_" + str(sample)

            for sample in enumerate(sampled_action):
                local_byte.append(bytes(str(sample[1]), "utf-8"))

            output_logits.append([((1 / num_samples) * num_samples)])
            output_string_mapping.append(sampled_action)
            output_byte_mapping.append(local_byte)
            policy_sizes.append(num_samples)

        return output_logits, output_string_mapping, output_byte_mapping, policy_sizes

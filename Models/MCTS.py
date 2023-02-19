import time
import config
import numpy as np
import core.ctree.cytree as tree
from typing import Dict
from scipy.stats import entropy


# EXPLANATION OF MCTS:
"""
1. select leaf node with maximum value using method called UCB1 
2. expand the leaf node, adding children for each possible action
3. Update leaf node and ancestor values using the values learnt from the children
 - values for the children are generated using neural network 
4. Repeat above steps a given number of times
5. Select path with highest value
"""


class MCTS:
    def __init__(self, network):
        self.network = network
        self.times = [0] * 6
        self.NUM_ALIVE = config.NUM_PLAYERS
        self.num_actions = 0
        self.ckpt_time = time.time_ns()
        self.default_byte_mapping, self.default_string_mapping = self.create_default_mapping()

    def policy(self, observation):
        self.NUM_ALIVE = observation[0].shape[0]

        # 0.02 seconds
        network_output = self.network.initial_inference(observation[0])

        value_prefix_pool = np.array(network_output["value_logits"]).reshape(-1).tolist()
        policy_logits = network_output["policy_logits"].numpy()

        # 0.01 seconds
        policy_logits_pool, mappings, string_mapping = self.encode_action_to_str(policy_logits, observation[1])

        # 0.003 seconds
        policy_logits_pool, string_mapping, mapping = self.sample(policy_logits_pool, string_mapping,
                                                                   mappings, config.NUM_SAMPLES)

        # less than 0.0001 seconds
        # Setup specialised roots datastructures, format: env_nums, action_space_size, num_simulations
        # Number of agents, previous action, number of simulations for memory purposes
        roots_cpp = tree.Roots(self.NUM_ALIVE, config.NUM_SAMPLES, config.NUM_SIMULATIONS)

        # 0.0002 seconds
        # prepare the nodes to feed them into batch_mcts, for statement to deal with different lengths due to masking.
        noises = [np.random.dirichlet([config.ROOT_DIRICHLET_ALPHA] *
                                      len(policy_logits_pool[i])).astype(np.float32).tolist()
                  for i in range(self.NUM_ALIVE)]

        # 0.01 seconds
        roots_cpp.prepare(config.ROOT_EXPLORATION_FRACTION, noises, value_prefix_pool, policy_logits_pool, mappings)

        # Output for root node
        hidden_state_pool = network_output["hidden_state"]

        # set up nodes to be able to find and select actions
        self.run_batch_mcts(roots_cpp, hidden_state_pool)
        roots_distributions = roots_cpp.get_distributions()

        actions = []
        target_policy = []
        temp = self.visit_softmax_temperature()  # controls the way actions are chosen
        for i in range(self.NUM_ALIVE):
            deterministic = False  # False = sample distribution, True = argmax
            distributions = roots_distributions[i]
            action, _ = self.select_action(distributions, temperature=temp, deterministic=deterministic)
            actions.append(string_mapping[i][action])
            output_policy = self.map_sample_to_distribution(string_mapping[i],
                                                            [x / config.NUM_SIMULATIONS for x in distributions])
            target_policy.append(output_policy)

        # Notes on possibilities for other dimensions at the bottom
        self.num_actions += 1
        return actions, target_policy

    def run_batch_mcts(self, roots_cpp, hidden_state_pool):
        # preparation
        num = roots_cpp.num
        # config variables
        discount = config.DISCOUNT
        pb_c_init = config.PB_C_INIT
        pb_c_base = config.PB_C_BASE
        hidden_state_index_x = 0

        # minimax value storage data structure
        min_max_stats_lst = tree.MinMaxStatsList(num)
        min_max_stats_lst.set_delta(config.MAXIMUM_REWARD * 2)  # config.MINIMUM_REWARD *2
        # self.config.lstm_horizon_len, seems to be the number of timesteps predicted in the future
        horizons = 1
        hidden_state_pool = [hidden_state_pool]
        # go through the tree NUM_SIMULATIONS times
        for _ in range(config.NUM_SIMULATIONS):
            # prepare a result wrapper to transport results between python and c++ parts
            hidden_states = []
            results = tree.ResultsWrapper(num)

            # 0.001 seconds
            # evaluation for leaf nodes, traversing across the tree and updating values
            hidden_state_index_x_lst, hidden_state_index_y_lst, last_action = \
                tree.batch_traverse(roots_cpp, pb_c_base, pb_c_init, discount, min_max_stats_lst, results)
            search_lens = results.get_search_len()

            # obtain the states for leaf nodes
            for ix, iy in zip(hidden_state_index_x_lst, hidden_state_index_y_lst):
                hidden_states.append(hidden_state_pool[ix][iy])

            # Inside the search tree we use the dynamics function to obtain the next
            # hidden state given an action and the previous hidden state.

            last_action = np.asarray(last_action)

            # 0.026 to 0.064 seconds
            network_output = self.network.recurrent_inference(np.asarray(hidden_states), last_action)

            value_prefix_pool = np.array(network_output["value_logits"]).reshape(-1).tolist()
            value_pool = np.array(network_output["value"]).reshape(-1).tolist()

            # 0.002 seconds
            policy_logits, _, mappings = self.sample(network_output["policy_logits"].numpy(),
                                                    self.default_string_mapping,self.default_byte_mapping,
                                                    config.NUM_SAMPLES)
            # These assignments take 0.0001 > time
            # add nodes to the pool after each search
            hidden_states_nodes = network_output["hidden_state"]
            hidden_state_pool.append(hidden_states_nodes)

            reset_idx = (np.array(search_lens) % horizons == 0)
            is_reset_lst = reset_idx.astype(np.int32).tolist()
            # tree node.isreset = is_reset_list[node]
            hidden_state_index_x += 1

            # 0.001 seconds
            # backpropagation along the search path to update the attributes
            tree.batch_back_propagate(hidden_state_index_x, discount, value_prefix_pool, value_pool, policy_logits,
                                      min_max_stats_lst, results, is_reset_lst, mappings)

    @staticmethod
    def select_action(visit_counts, temperature=1, deterministic=True):
        """select action from the root visit counts.
        Parameters
        ----------
        visit_counts: list
            visit counts for each child
        temperature: float
            the temperature for the distribution
        deterministic: bool
            True -> select the argmax
            False -> sample from the distribution
        """
        action_probs = [visit_count_i ** (1 / temperature) for visit_count_i in visit_counts]
        total_count = sum(action_probs)
        action_probs = [x / total_count for x in action_probs]
        if deterministic:
            action_pos = np.argmax([v for v in visit_counts])
        else:
            action_pos = np.random.choice(len(visit_counts), p=action_probs)

        count_entropy = entropy(action_probs, base=2)
        return action_pos, count_entropy

    @staticmethod
    def encode_action_to_str(policy_logits, mask):
        actions = []
        mappings = []
        second_mappings = []
        for idx in range(len(policy_logits)):
            local_counter = 0
            local_action = [policy_logits[idx][local_counter]]
            local_mappings = [bytes("0", "utf-8")]
            second_local_mappings = ["0"]
            local_counter += 1
            for i in range(5):
                if mask[idx][1][i]:
                    local_action.append(policy_logits[idx][local_counter])
                    local_mappings.append(bytes(f"1_{i}", "utf-8"))
                    second_local_mappings.append(f"1_{i}")
                local_counter += 1
            for a in range(37):
                for b in range(a, 38):
                    if a == b:
                        continue
                    # This does not account for max units yet
                    if not ((a < 28 and mask[idx][2][a]) or (a > 27 and mask[idx][3][a - 28])):
                        local_counter += 1
                        continue
                    local_action.append(policy_logits[idx][local_counter])
                    local_mappings.append(bytes(f"2_{a}_{b}", "utf-8"))
                    second_local_mappings.append(f"2_{a}_{b}")
                    local_counter += 1
            for a in range(37):
                for b in range(10):
                    if not ((a < 28 and mask[idx][2][a]) or (a > 27 and mask[idx][3][a - 28]) and mask[idx][4][b]):
                        local_counter += 1
                        continue
                    local_action.append(policy_logits[idx][local_counter])
                    local_mappings.append(bytes(f"3_{a}_{b}", "utf-8"))
                    second_local_mappings.append(f"3_{a}_{b}")
                    local_counter += 1
            if mask[idx][0][4]:
                local_action.append(policy_logits[idx][local_counter])
                local_mappings.append(bytes("4", "utf-8"))
                second_local_mappings.append("4")
            local_counter += 1
            if mask[idx][0][5]:
                local_action.append(policy_logits[idx][local_counter])
                local_mappings.append(bytes("5", "utf-8"))
                second_local_mappings.append("5")

            actions.append(local_action)
            mappings.append(local_mappings)
            second_mappings.append(second_local_mappings)
        return actions, mappings, second_mappings

    @staticmethod
    def create_default_mapping():
        mappings = [bytes("0", "utf-8")]
        second_mappings = ["0"]
        for i in range(5):
            mappings.append(bytes(f"1_{i}", "utf-8"))
            second_mappings.append(f"1_{i}")
        for a in range(37):
            for b in range(a, 38):
                if a == b:
                    continue
                mappings.append(bytes(f"2_{a}_{b}", "utf-8"))
                second_mappings.append(f"2_{a}_{b}")
        for a in range(37):
            for b in range(10):
                mappings.append(bytes(f"3_{a}_{b}", "utf-8"))
                second_mappings.append(f"3_{a}_{b}")
        mappings.append(bytes("4", "utf-8"))
        second_mappings.append("4")
        mappings.append(bytes("5", "utf-8"))
        second_mappings.append("5")
        mappings = [mappings for _ in range(config.NUM_PLAYERS)]
        second_mappings = [second_mappings for _ in range(config.NUM_PLAYERS)]
        return mappings, second_mappings

    def sample(self, policy_logits, string_mapping, byte_mapping, num_samples):
        output_logits = []
        output_string_mapping = []
        output_byte_mapping = []
        for i in range(len(policy_logits)):
            local_logits = []
            local_string = []
            local_byte = []
            num_pass_shop_actions = 6
            refresh_level_actions = 2
            # Add samples for pass and the 5 shop options
            # Note that if there are not 5 available shop options, the sample here will be move options
            for fixed_sample in range(0, 6):
                if string_mapping[i][fixed_sample][0] == "0" or string_mapping[i][fixed_sample][0] == "1":
                    local_logits.append(policy_logits[i][fixed_sample])
                    local_string.append(string_mapping[i][fixed_sample])
                    local_byte.append(byte_mapping[i][fixed_sample])
                else:
                    num_pass_shop_actions -= 1
            # Add samples for refresh and level
            # Note if either refresh or level is not available, the samples here will be move options
            for last_sample in range(len(policy_logits[i]) - 2, len(policy_logits[i])):
                if string_mapping[i][last_sample][0] == "4" or string_mapping[i][last_sample][0] == "5":
                    local_logits.append(policy_logits[i][last_sample])
                    local_string.append(string_mapping[i][last_sample])
                    local_byte.append(byte_mapping[i][last_sample])
                else:
                    refresh_level_actions -= 1
            num_core_actions = num_pass_shop_actions + refresh_level_actions
            # Get the softmax of the policy output
            probs = self.softmax_stable(policy_logits[i][num_pass_shop_actions:
                                                         len(policy_logits[i]) - refresh_level_actions])
            # array of size [action_dim] with [0, 1, 2, 3... action_dim - 8]
            # We are removing 8 samples initially because those are the most important actions
            policy_range = np.arange(stop=len(policy_logits[i]) - num_core_actions)
            # sample values, can break if less than 20 values in the policy_logits. Working on fix.
            # -8 here because we already added 8 samples above
            samples = np.random.choice(a=policy_range, size=num_samples - num_core_actions, replace=False, p=probs)
            # Sort now so the mapping back to 1081 later is much faster
            samples.sort()
            for sample in samples:
                # Add the base value for the sample
                # +6 because we have to skip the first 6 values but never want to hit the last 2
                local_logits.append(policy_logits[i][sample + num_pass_shop_actions])
                # Add the name of the string action
                local_string.append(string_mapping[i][sample + num_pass_shop_actions])
                # Same but for the c++ side
                local_byte.append(byte_mapping[i][sample + num_pass_shop_actions])
            # Need to separate to keep batch_dim the same
            output_logits.append(local_logits)
            output_string_mapping.append(local_string)
            output_byte_mapping.append(local_byte)
        return output_logits, output_string_mapping, output_byte_mapping

    def map_sample_to_distribution(self, mapping, sample_dist):
        local_counter = 0
        output_policy = []
        # If 0 is part of our sampling (will always be true if using specified sampling)
        if mapping[local_counter] == "0":
            output_policy.append(sample_dist[local_counter])
            local_counter += 1
        # else add 0 to indicate probability 0.
        else:
            output_policy.append(0)
        for i in range(5):
            if mapping[local_counter] == f"1_{i}":
                output_policy.append(sample_dist[local_counter])
                local_counter += 1
            else:
                output_policy.append(0)
        for a in range(37):
            for b in range(a, 38):
                if a == b:
                    continue
                if mapping[local_counter] == f"2_{a}_{b}":
                    output_policy.append(sample_dist[local_counter])
                    local_counter += 1
                    if local_counter == len(mapping):
                        local_counter -= 1
                else:
                    output_policy.append(0)
        for a in range(37):
            for b in range(10):
                if mapping[local_counter] == f"3_{a}_{b}":
                    output_policy.append(sample_dist[local_counter])
                    local_counter += 1
                    if local_counter == len(mapping):
                        local_counter -= 1
                else:
                    output_policy.append(0)
        if mapping[local_counter] == "4":
            output_policy.append(sample_dist[local_counter])
            local_counter += 1
            if local_counter == len(mapping):
                local_counter -= 1
        else:
            output_policy.append(0)
        if mapping[local_counter] == "5":
            output_policy.append(sample_dist[local_counter])
        else:
            output_policy.append(0)
        return output_policy

    @staticmethod
    def softmax_stable(x):
        return np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum()

    def fill_metadata(self) -> Dict[str, str]:
        return {'network_id': str(self.network.training_steps())}

    @staticmethod
    def histogram_sample(distribution, temperature, use_softmax=False, mask=None):
        actions = [d[1] for d in distribution]
        visit_counts = np.array([d[0] for d in distribution], dtype=np.float64)
        if temperature == 0.:
            probs = masked_count_distribution(visit_counts, mask=mask)
            return actions[np.argmax(probs)]
        if use_softmax:
            logits = visit_counts / temperature
            probs = masked_softmax(logits, mask)
        else:
            logits = visit_counts ** (1. / temperature)
            probs = masked_count_distribution(logits, mask)
        return np.random.choice(actions, p=probs)

    @staticmethod
    def visit_softmax_temperature():
        return 0.9

def masked_distribution(x, use_exp, mask=None):
    if mask is None:
        mask = [1] * len(x)
    assert sum(mask) > 0, 'Not all values can be masked.'
    assert len(mask) == len(x), (
        'The dimensions of the mask and x need to be the same.')
    x = np.exp(x) if use_exp else np.array(x, dtype=np.float64)
    mask = np.array(mask, dtype=np.float64)
    x *= mask
    if sum(x) == 0:
        # No unmasked value has any weight. Use uniform distribution over unmasked
        # tokens.
        x = mask
    return x / np.sum(x, keepdims=True)


def masked_softmax(x, mask=None):
    x = np.array(x) - np.max(x, axis=-1)  # to avoid overflow
    return masked_distribution(x, use_exp=True, mask=mask)


def masked_count_distribution(x, mask=None):
    return masked_distribution(x, use_exp=False, mask=mask)

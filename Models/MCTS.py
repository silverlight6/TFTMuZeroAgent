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

            # 0.0001 seconds
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
            # 0.03 seconds
            self.ckpt_time = time.time_ns()
            network_output = self.network.recurrent_inference(np.asarray(hidden_states), last_action)
            print("recurrent_inference takes {} time".format(time.time_ns() - self.ckpt_time))
            value_prefix_pool = np.array(network_output["value_logits"]).reshape(-1).tolist()
            value_pool = np.array(network_output["value"]).reshape(-1).tolist()
            policy_action = network_output["policy_logits"][0].numpy()
            policy_target = network_output["policy_logits"][1].numpy()
            policy_item = network_output["policy_logits"][2].numpy()

            # .015 to 0.045 seconds
            policy_logits_pool, _, _, _, _ = self.encode_action_to_str(policy_action, policy_target, policy_item)

            # These assignments take 0.0001 > time
            # add nodes to the pool after each search
            hidden_states_nodes = network_output["hidden_state"]
            hidden_state_pool.append(hidden_states_nodes)

            reset_idx = (np.array(search_lens) % horizons == 0)
            is_reset_lst = reset_idx.astype(np.int32).tolist()
            # tree node.isreset = is_reset_list[node]
            hidden_state_index_x += 1

            # 0.001 to 0.006 seconds
            # backpropagation along the search path to update the attributes
            tree.batch_back_propagate(hidden_state_index_x, discount, value_prefix_pool, value_pool, policy_logits_pool,
                                      min_max_stats_lst, results, is_reset_lst)

    def policy(self, observation):
        self.NUM_ALIVE = observation[0].shape[0]

        # 0.97 seconds for first action to 0.04 after a few actions
        self.ckpt_time = time.time_ns()
        network_output = self.network.initial_inference([observation[0], observation[1]])
        print("initial inference takes {} time".format(time.time_ns() - self.ckpt_time))
        value_prefix_pool = np.array(network_output["value_logits"]).reshape(-1).tolist()
        policy_action = network_output["policy_logits"][0].numpy()
        policy_target = network_output["policy_logits"][1].numpy()
        policy_item = network_output["policy_logits"][2].numpy()
        # 0.015 seconds
        policy_logits_pool, mappings, action_sizes, max_length, string_mapping = \
            self.encode_action_to_str(policy_action, policy_target, policy_item, observation[2])

        # less than 0.0001 seconds
        # Setup specialised roots datastructures, format: env_nums, action_space_size, num_simulations
        # Number of agents, previous action, number of simulations for memory purposes
        roots_cpp = tree.Roots(self.NUM_ALIVE, action_sizes, config.NUM_SIMULATIONS, max_length)

        # 0.0002 seconds
        # prepare the nodes to feed them into batch_mcts
        noises = [np.random.dirichlet([config.ROOT_DIRICHLET_ALPHA] *
                                      len(policy_logits_pool[0])).astype(np.float32).tolist()
                  for _ in range(self.NUM_ALIVE)]

        # 0.002 seconds
        roots_cpp.prepare(config.ROOT_EXPLORATION_FRACTION, noises, value_prefix_pool, policy_logits_pool, mappings)

        # Output for root node
        hidden_state_pool = network_output["hidden_state"]

        # set up nodes to be able to find and select actions
        self.run_batch_mcts(roots_cpp, hidden_state_pool)
        roots_distributions = roots_cpp.get_distributions()

        actions = []
        temp = self.visit_softmax_temperature()  # controls the way actions are chosen
        for i in range(self.NUM_ALIVE):
            deterministic = False  # False = sample distribution, True = argmax
            distributions = roots_distributions[i]
            action, _ = self.select_action(distributions, temperature=temp, deterministic=deterministic)
            actions.append(action)

        # Notes on possibilities for other dimensions at the bottom
        self.num_actions += 1
        str_action = []
        for i, act in enumerate(actions):
            str_action.append(string_mapping[i][act])
        return str_action, network_output["policy_logits"]

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
    def encode_action_to_str(action, target, item, mask=None):
        actions = []
        mappings = []
        second_mappings = []
        action_sizes = []
        max_length = 0
        for idx in range(len(action)):
            local_action = [action[idx][0]]
            local_mappings = [bytes("0", "utf-8")]
            second_local_mappings = ["0"]
            shop_sum = sum(list(target[idx])[0:5])
            board_sum_a = sum(list(target[idx])[0:37])
            board_sum_b = sum(list(target[idx])[0:38])
            item_sum = sum(list(item[idx]))
            if mask is not None:
                for i in range(5):
                    if mask[idx][1][i]:
                        local_action.append(action[idx][1] * target[idx][i] / shop_sum)
                        local_mappings.append(bytes(f"1_{i}", "utf-8"))
                        second_local_mappings.append(f"1_{i}")
                for a in range(37):
                    for b in range(a, 38):
                        if a == b:
                            continue
                        # This does not account for max units yet
                        if not ((a < 28 and mask[idx][2][a]) or (a > 27 and mask[idx][3][a - 28])):
                            continue
                        local_action.append(action[idx][2] * (target[idx][a] / board_sum_a *
                                                              (target[idx][b] / (board_sum_b - target[idx][a]))))
                        local_mappings.append(bytes(f"2_{a}_{b}", "utf-8"))
                        second_local_mappings.append(f"2_{a}_{b}")
                for a in range(37):
                    for b in range(10):
                        if not ((a < 28 and mask[idx][2][a]) or (a > 27 and mask[idx][3][a - 28]) and mask[idx][4][b]):
                            continue
                        local_action.append(action[idx][3] * (target[idx][a] / board_sum_a) *
                                            (item[idx][b] / item_sum))
                        local_mappings.append(bytes(f"3_{a}_{b}", "utf-8"))
                        second_local_mappings.append(f"3_{a}_{b}")
                if mask[idx][0][4]:
                    local_action.append(action[idx][4])
                    local_mappings.append(bytes("4", "utf-8"))
                    second_local_mappings.append("4")
                if mask[idx][0][5]:
                    local_action.append(action[idx][5])
                    second_local_mappings.append("5")
                second_mappings.append(second_local_mappings)
            else:
                for i in range(5):
                    local_action.append(action[idx][1] * target[idx][i] / shop_sum)
                    local_mappings.append(bytes(f"1_{i}", "utf-8"))
                for a in range(37):
                    for b in range(a, 38):
                        if a == b:
                            continue
                        local_action.append(action[idx][2] * (target[idx][a] / board_sum_a *
                                                              (target[idx][b] / (board_sum_b - target[idx][a]))))
                        local_mappings.append(bytes(f"2_{a}_{b}", "utf-8"))
                for a in range(37):
                    for b in range(10):
                        local_action.append(action[idx][3] * (target[idx][a] / board_sum_a) *
                                            (item[idx][b] / item_sum))
                        local_mappings.append(bytes(f"3_{a}_{b}", "utf-8"))
                local_action.append(action[idx][4])
                local_mappings.append(bytes("4", "utf-8"))
                local_action.append(action[idx][5])
                local_mappings.append(bytes("5", "utf-8"))
            actions.append(local_action)
            mappings.append(local_mappings)
            action_sizes.append(len(local_action))
            if len(local_action) > max_length:
                max_length = len(local_action)
        return actions, mappings, action_sizes, max_length, second_mappings

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
        return 1


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

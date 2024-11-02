import time
import numpy as np
import core.muzero_ctree.cytree as tree
import torch
import Models.MCTS_Util as util
import config
from typing import Dict

"""
TODO: Update this
EXPLANATION OF MCTS:
1. select leaf node with maximum value using method called UCB1 
2. expand the leaf node, adding children for each possible action
3. Update leaf node and ancestor values using the values learnt from the children
 - values for the children are generated using neural network 
4. Repeat above steps a given number of times
5. Select path with highest value
"""


class MCTS:
    def __init__(self, network, model_config):
        self.network = network
        self.times = [0] * 6
        self.batch_size = config.NUM_ENVS
        self.num_actions = 0
        self.ckpt_time = time.time_ns()
        self.max_depth_search = 0
        self.model_config = model_config

    def policy(self, observation, simulators, action_count):
        with torch.no_grad():
            self.batch_size = observation["observations"]["board"].shape[0]

            # 0.013 seconds
            network_output = self.network.initial_inference(observation["observations"])

            reward_pool = [0] * self.batch_size

            policy_logits = network_output["policy_logits"]
            # Mask illegal actions

            # is it quicker to do this as a tensor or as a numpy array?
            flat_mask = torch.tensor(observation["action_mask"][np.arange(self.batch_size),
                                     np.array(action_count).flatten(), :]).to(config.DEVICE)
            inf_mask = torch.clamp(torch.log(flat_mask), min=-3.4e38)
            policy_logits = policy_logits + inf_mask
            masked_policy_logits = policy_logits.cpu().numpy()

            noises = [
                    np.random.dirichlet([self.model_config.ROOT_DIRICHLET_ALPHA] * len(masked_policy_logits[j])
                                        ).astype(np.float32).tolist()
                    for j in range(self.batch_size)
                ]

            policy_logits_pool = self.add_exploration_noise(masked_policy_logits, noises)

            # less than 0.0001 seconds
            # Setup specialised roots datastructures, format: env_nums, action_space_size, num_simulations
            # Number of agents, previous action, number of simulations for memory purposes, keep above number of actions
            roots_cpp = tree.Roots(self.batch_size, self.model_config.NUM_SIMULATIONS, 30)

            # 0.0002 seconds
            # prepare the nodes to feed them into batch_mcts,
            # for statement to deal with different lengths due to masking.
            roots_cpp.prepare_no_noise(reward_pool, policy_logits_pool.tolist(), [29] * self.batch_size, action_count)

            # Output for root node
            hidden_state_pool = network_output["hidden_state"]

            # set up nodes to be able to find and select actions
            self.run_batch_mcts(roots_cpp, hidden_state_pool, observation["action_mask"], simulators, action_count)
            roots_distributions = roots_cpp.get_distributions()

            root_values = roots_cpp.get_values()

            actions = []
            target_policy = []
            temp = self.visit_softmax_temperature()  # controls the way actions are chosen
            deterministic = False  # False = sample distribution, True = argmax
            for i in range(self.batch_size):
                distributions = roots_distributions[i]
                action = self.select_action(distributions, temperature=temp, deterministic=deterministic)
                actions.append(action)
                # sum(x) since x might add up to a number different from the number of simulations
                # due to terminal states.
                # This can be less than the number of actions which is a bit concerning but I think it's fine
                dist_sum = sum(distributions)
                # print(f"dist_sum -> {dist_sum}")
                if dist_sum > 1000:
                    print(f"dist_sum -> {dist_sum} with action_count {action_count}")
                target_policy.append([x / dist_sum for x in distributions])
            # Notes on possibilities for other dimensions at the bottom
            self.num_actions += 1
            return actions, target_policy, root_values

    def run_batch_mcts(self, roots_cpp, hidden_state_pool, action_mask, simulators, action_count):
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
            # basically 0 seconds (3e-5)
            # evaluation for leaf nodes, traversing across the tree and updating values
            hidden_state_index_x_lst, hidden_state_index_y_lst, last_action, search_lens = \
                tree.batch_traverse(roots_cpp, pb_c_base, pb_c_init, discount, min_max_stats_lst, results)

            self.max_depth_search = sum(results.get_search_len()) / len(results.get_search_len())
            num_states = len(hidden_state_index_x_lst)
            tensors_states = torch.empty((num_states, self.model_config.HIDDEN_STATE_SIZE)).to(config.DEVICE)

            # obtain the states for leaf nodes
            for ix, iy, idx in zip(hidden_state_index_x_lst, hidden_state_index_y_lst, range(num_states)):
                tensors_states[idx] = hidden_state_pool[ix][iy]

            # Inside the search tree we use the dynamics function to obtain the next
            # hidden state given an action and the previous hidden state.
            # TODO: Turn last_action into an array. The previous action goes to the recurrent_inference,
            #  rest for simulator
            last_action_input = []
            for action in last_action:
                last_action_input.append(action[-1])

            # 0.005 to 0.01 seconds
            network_output = self.network.recurrent_inference(tensors_states, np.array(last_action_input))

            reward_pool = []
            step_action_counts = []
            for i, simulator in enumerate(simulators):
                # - 1 for the indexing for the mask
                step_action_counts.append(action_count[i] + search_lens[i] - 1)
                if step_action_counts[i] == 11:
                    reward_pool.append(simulator.fake_step(last_action[i], action_count[i]))
                else:
                    reward_pool.append(0)

            value_pool = network_output["value"].reshape(-1).tolist()

            policy_logits = network_output["policy_logits"]

            # Mask illegal actions
            flat_mask = torch.tensor(action_mask[np.arange(self.batch_size),
                                     np.array(step_action_counts).flatten(), :]).to(config.DEVICE)
            inf_mask = torch.clamp(torch.log(flat_mask), min=-3.4e38)
            policy_logits = policy_logits + inf_mask
            masked_policy_logits = policy_logits.cpu().numpy()

            # These assignments take 0.0001 > time
            # add nodes to the pool after each search
            hidden_states_nodes = network_output["hidden_state"]
            hidden_state_pool.append(hidden_states_nodes)

            hidden_state_index_x += 1

            # 0.001 seconds
            # backpropagation along the search path to update the attributes
            tree.batch_back_propagate(hidden_state_index_x, discount, reward_pool, value_pool,
                                      masked_policy_logits.tolist(), min_max_stats_lst, results,
                                      [29] * self.batch_size, step_action_counts)

    def add_exploration_noise(self, policy_logits, noises):
        exploration_fraction = self.model_config.ROOT_EXPLORATION_FRACTION
        for i in range(len(noises)):  # Batch
            for k in range(len(noises[i])):
                policy_logits[i][k] = policy_logits[i][k] * (1 - exploration_fraction) + \
                                         noises[i][k] * exploration_fraction
        return policy_logits

    """
    Description - select action from the root visit counts.
    Inputs      - visit_counts: list
                    visit counts for each child
                  temperature: float
                    the temperature for the distribution
                  deterministic: bool
                    True -> select the argmax
                    False -> sample from the distribution
    Outputs     - action_pos
                    position of the action in the policy and string array.
                  count_entropy
                    entropy of the improved policy.
    """
    @staticmethod
    def select_action(visit_counts, temperature=1.0, deterministic=True):
        action_probs = [visit_count_i ** (1 / temperature)
                        for visit_count_i in visit_counts]
        total_count = sum(action_probs)
        action_probs = [x / total_count for x in action_probs]
        if deterministic:
            action_pos = np.argmax([v for v in visit_counts])
        else:
            action_pos = np.random.choice(len(visit_counts), p=action_probs)

        return action_pos

    @staticmethod
    def softmax_stable(x):
        top_value = x - np.max(x)
        top_value = np.exp(top_value)
        bottom_value = np.exp(x - np.max(x)).sum()
        return top_value / bottom_value

    def fill_metadata(self) -> Dict[str, str]:
        return {'network_id': str(self.network.training_steps())}

    def visit_softmax_temperature(self):
        return self.model_config.VISIT_TEMPERATURE


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

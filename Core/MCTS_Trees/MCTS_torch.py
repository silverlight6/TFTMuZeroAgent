import time
import numpy as np
import Core.CplusplusTrees.ctree.cytree as tree
import torch
import Core.MCTS_Trees.MCTS_Util as util
import config
from typing import Dict

"""
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
        self.NUM_ALIVE = config.NUM_PLAYERS
        self.num_actions = 0
        self.ckpt_time = time.time_ns()
        self.default_string_mapping, self.default_mask = util.create_default_mapping()
        self.max_depth_search = 0
        self.model_config = model_config
        self.default_mask = torch.tensor(self.default_mask).to(config.DEVICE)

    def policy(self, observation):
        with torch.no_grad():
            self.NUM_ALIVE = observation["observations"]["shop"].shape[0]

            # 0.013 seconds
            network_output = self.network.initial_inference(observation["observations"])

            reward_pool = np.array(network_output["reward"]).reshape(-1).tolist()

            policy_logits = network_output["policy_logits"]

            # Mask illegal actions
            # is it quicker to do this as a tensor or as a numpy array?
            flat_mask = torch.tensor(np.reshape(observation["action_mask"], (self.NUM_ALIVE, -1))).to(config.DEVICE)
            inf_mask = torch.clamp(torch.log(flat_mask), min=-3.4e38)
            policy_logits = policy_logits + inf_mask
            masked_policy_logits = policy_logits.cpu().numpy()

            noises = [
                    np.random.dirichlet([self.model_config.ROOT_DIRICHLET_ALPHA] * len(masked_policy_logits[j])
                                        ).astype(np.float32).tolist()
                    for j in range(self.NUM_ALIVE)
                ]

            policy_logits_pool = self.add_exploration_noise(masked_policy_logits, noises)

            # 0.001 seconds
            policy_logits_pool, string_mapping, mappings, policy_sizes = \
                self.sample(policy_logits_pool, self.default_string_mapping, self.model_config.NUM_SAMPLES)

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
            # basically 0 seconds (3e-5)
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

            # 0.005 to 0.01 seconds
            network_output = self.network.recurrent_inference(tensors_states, last_action)

            reward_pool = network_output["reward"].reshape(-1).tolist()
            value_pool = network_output["value"].reshape(-1).tolist()
            diff = max(value_pool) - min(value_pool)
            if diff > 150.:
                print(f"EUREKA, VALUES MAX: {max(value_pool)}, AND MIN: {min(value_pool)}, RANGE {diff}")

            policy_logits = network_output["policy_logits"]
            # Mask illegal actions

            default_mask = self.default_mask.repeat(self.NUM_ALIVE, 1)
            inf_mask = torch.clamp(torch.log(default_mask), min=-3.4e38)
            policy_logits = policy_logits + inf_mask
            masked_policy_logits = policy_logits.cpu().numpy()

            # 0.003 to 0.01 seconds
            policy_logits, _, mappings, policy_sizes = \
                self.sample(masked_policy_logits, self.default_string_mapping, self.model_config.NUM_SAMPLES)

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


    """
    Description - This is the CplusplusTrees to the Complex Action Spaces paper. We take a set number of sample actions from the 
                  total number of actions based off of the current policy to expand on each turn. There are two options
                  as to how the samples are chosen. You can either set num_pass_shop_actions and refresh_level_actions
                  to 0 and comment out the following for loops or keep those variables at 6 and 2 and leave the for
                  loops in. The first option is a pure sample with no specific CplusplusTrees actions. The second option gives 
                  you a set of CplusplusTrees options to use. 
    Inputs      - policy_logits - List
                      Output to either initial_inference or recurrent_inference for policy
                  string_mapping - List
                      A map that is equal to policy_logits.shape[-1] in size to map to the specified action
                  byte_mapping - List
                      Same as string mapping but used on the c++ side of the code
                  num_samples - Int
                      Typically set to config.NUM_SAMPLES. Number of samples to use per expansion of the tree
    Outputs     - output_logits - List
                      The sampled policy logits 
                  output_string_mapping - List
                      The sampled string mapping. Size = output.logits.shape
                  output_byte_mapping - List
                      Same as output_string_mapping but for c++ side
                  policy_sizes - List
                      Number of samples per player, can change if legal actions < num_samples
    """
    def sample(self, policy_logits, string_mapping, num_samples):
        batch_size = len(policy_logits)  # 8

        output_logits = []
        output_string_mapping = []
        output_byte_mapping = []
        policy_sizes = []

        for idx in range(batch_size):
            local_logits = []
            local_string = []
            local_byte = []

            probs = self.softmax_stable(policy_logits[idx])
            policy_range = np.arange(stop=len(policy_logits[idx]))

            samples = np.random.choice(a=policy_range, p=probs, size=num_samples)  # size 25
            counts = np.bincount(samples, minlength=len(policy_logits[idx]))

            for i, count in enumerate(counts):
                if count > 0:
                    dim_base_string = string_mapping[idx][i]
                    local_logits.append(((1 / num_samples) * count))
                    local_string.append(dim_base_string)
                    local_byte.append(bytes(dim_base_string, "utf-8"))

            output_logits.append(local_logits)
            output_string_mapping.append(local_string)
            output_byte_mapping.append(local_byte)
            policy_sizes.append(len(local_logits))

        return output_logits, output_string_mapping, output_byte_mapping, policy_sizes

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

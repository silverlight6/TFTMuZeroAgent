import time
import config
import numpy as np
import core.ctree.cytree as tree
import torch
import Models.MCTS_Util as util
from typing import Dict
from scipy.stats import entropy

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
    def __init__(self, network):
        self.network = network
        self.times = [0] * 6
        self.NUM_ALIVE = config.NUM_PLAYERS
        self.num_actions = 0
        self.ckpt_time = time.time_ns()
        self.default_byte_mapping, self.default_string_mapping = util.create_default_mapping()

    def policy(self, observation):
        with torch.no_grad():
            self.NUM_ALIVE = observation[0].shape[0]

            # 0.02 seconds
            network_output = self.network.initial_inference(observation[0])

            reward_pool = np.array(network_output["reward"]).reshape(-1).tolist()
            policy_logits = network_output["policy_logits"].detach().cpu().numpy()

            # 0.01 seconds
            policy_logits_pool, mappings, string_mapping = self.encode_action_to_str(policy_logits, observation[1])

            noises = [np.random.dirichlet([config.ROOT_DIRICHLET_ALPHA] *
                                          len(policy_logits_pool[i])).astype(np.float32).tolist()
                      for i in range(self.NUM_ALIVE)]

            policy_logits_pool = self.add_exploration_noise(policy_logits_pool, noises)
            # time.sleep(0.5)
            # 0.003 seconds
            policy_logits_pool, string_mapping, mappings, policy_sizes = \
                self.sample(policy_logits_pool, string_mapping, mappings, config.NUM_SAMPLES)

            # less than 0.0001 seconds
            # Setup specialised roots datastructures, format: env_nums, action_space_size, num_simulations
            # Number of agents, previous action, number of simulations for memory purposes
            roots_cpp = tree.Roots(self.NUM_ALIVE, config.NUM_SIMULATIONS, config.NUM_SAMPLES)

            # 0.0002 seconds
            # prepare the nodes to feed them into batch_mcts,
            # for statement to deal with different lengths due to masking.
            roots_cpp.prepare_no_noise(reward_pool, policy_logits_pool, mappings, policy_sizes)

            # Output for root node
            hidden_state_pool = network_output["hidden_state"]

            # set up nodes to be able to find and select actions
            self.run_batch_mcts(roots_cpp, hidden_state_pool)
            roots_distributions = roots_cpp.get_distributions()

            actions = []
            target_policy = []
            temp = self.visit_softmax_temperature()  # controls the way actions are chosen
            deterministic = False  # False = sample distribution, True = argmax
            for i in range(self.NUM_ALIVE):
                distributions = roots_distributions[i]
                action, _ = self.select_action(distributions, temperature=temp, deterministic=deterministic)
                actions.append(string_mapping[i][action])
                target_policy.append([x / config.NUM_SIMULATIONS for x in distributions])

            # Notes on possibilities for other dimensions at the bottom
            self.num_actions += 1
            return actions, target_policy, string_mapping

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
        hidden_state_pool = [hidden_state_pool]
        # go through the tree NUM_SIMULATIONS times
        for _ in range(config.NUM_SIMULATIONS):
            # prepare a result wrapper to transport results between python and c++ parts
            results = tree.ResultsWrapper(num)
            # 0.001 seconds
            # evaluation for leaf nodes, traversing across the tree and updating values
            hidden_state_index_x_lst, hidden_state_index_y_lst, last_action = \
                tree.batch_traverse(roots_cpp, pb_c_base, pb_c_init, discount, min_max_stats_lst, results)
            num_states = len(hidden_state_index_x_lst)
            tensors_states = torch.empty((num_states, config.LAYER_HIDDEN_SIZE)).to('cuda')

            # obtain the states for leaf nodes
            for ix, iy, idx in zip(hidden_state_index_x_lst, hidden_state_index_y_lst, range(num_states)):
                tensors_states[idx] = hidden_state_pool[ix][iy]

            # Inside the search tree we use the dynamics function to obtain the next
            # hidden state given an action and the previous hidden state.
            last_action = np.asarray(last_action)

            # 0.003 seconds
            network_output = self.network.recurrent_inference(tensors_states, last_action)

            reward_pool = np.array(network_output["reward"]).reshape(-1).tolist()
            value_pool = np.array(network_output["value"]).reshape(-1).tolist()

            # 0.002 seconds
            policy_logits, _, mappings, policy_sizes = \
                self.sample(network_output["policy_logits"].cpu().numpy(), self.default_string_mapping,
                            self.default_byte_mapping, config.NUM_SAMPLES)

            # These assignments take 0.0001 > time
            # add nodes to the pool after each search
            hidden_states_nodes = network_output["hidden_state"]
            hidden_state_pool.append(hidden_states_nodes)

            hidden_state_index_x += 1

            # 0.001 seconds
            # backpropagation along the search path to update the attributes
            tree.batch_back_propagate(hidden_state_index_x, discount, reward_pool, value_pool, policy_logits,
                                      min_max_stats_lst, results, mappings, policy_sizes)

    def add_exploration_noise(self, noise, policy_logits):
        exploration_fraction = config.ROOT_EXPLORATION_FRACTION
        for i in range(len(noise)):
            for j in range(len(noise[i])):
                policy_logits[i][j] = policy_logits[i][j] * (1 - exploration_fraction) + \
                                      noise[i][j] * exploration_fraction
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
        action_probs = [visit_count_i ** (1 / temperature) for visit_count_i in visit_counts]
        total_count = sum(action_probs)
        action_probs = [x / total_count for x in action_probs]
        if deterministic:
            action_pos = np.argmax([v for v in visit_counts])
        else:
            action_pos = np.random.choice(len(visit_counts), p=action_probs)

        count_entropy = entropy(action_probs, base=2)
        return action_pos, count_entropy

    """
    Description - Turns a 1081 action into a policy that includes only actions that are legal in the current state
                  This also creates a mask for both the c++ side and python side to convert the legal action set into
                  a single action that we can give to the buffers and the trainer.
                  Masks for this method are generated in the player and observation classes.
                  This is only called by the root node since that is the only node that has access to the observation
    Inputs      - Policy logits: List
                      output of the prediction network, initial_inference in this case
                  Mappings: List
                      A mask of binary values that tell the policy what actions are legal and what actions are not.
    Outputs     - Actions: List
                      A policy including actions that are legal in the field.
                  Mappings: List
                      A byte mapping that maps those actions to a single 3 dimensional action that can be used in the 
                      simulator as well as in the recurrent inference. This gets sent to the c++ side
                  Seconds Mappings: List
                      A string mapping that is used in the same way but for the python side. This gets used on the 
                      values that get sent back to the AI_Interface
    """
    @staticmethod
    def encode_action_to_str(policy_logits, mask):
        # mask[0] = decision mask
        # mask[1] = shop mask - 1 if can buy champ, 0 if can't
        # mask[2] = board mask - 1 if slot is occupied, 0 if not
        # mask[3] = bench mask - 1 if slot is occupied, 0 if not
        # mask[4] = item mask - 1 if slot is occupied, 0 if not
        # mask[5] = util mask
        # mask[5][0] = board mask, mask[5][1] = bench mask, mask[5][2] item_bench mask
        # mask[6] = thieves glove mask - 1 if slot has a thieves glove, 0 if not
        # mask[7] = sparring glove + item mask
        # mask[8] = glove mask
        # TODO: add 7 more masks for:
        # 1. Kayn items on bench
        # 2. Kayn champions on board
        # 3. Reforger on bench
        # 4. thieves glove on bench
        # 5. if champion has items
        # 6. if champion has FULL items
        # 7. if champion is azir sandguard
        # 
        actions = []
        mappings = []
        second_mappings = []
        for idx in range(len(policy_logits)):
            local_counter = 0
            local_action = [policy_logits[idx][local_counter]]
            local_mappings = [bytes("0", "utf-8")]
            # do nothing
            second_local_mappings = ["0"]
            local_counter += 1
            # for every shop index...
            for i in range(5):
                if mask[idx][1][i] and mask[idx][5][1]:
                    local_action.append(policy_logits[idx][local_counter])
                    local_mappings.append(bytes(f"1_{i}", "utf-8"))
                    second_local_mappings.append(f"1_{i}")
                local_counter += 1
            # for all board + bench slots...
            for a in range(37):
                # rest of board slot locs for moving, last for sale
                for b in range(a, 38):
                    if a == b:
                        continue
                    if a > 27 and b != 37:
                        continue
                    # if we are trying to move a non-existent champion, skip
                    if not (((a < 28 and mask[idx][2][a]) or (a > 27 and mask[idx][3][a - 28])) or
                            ((b < 28 and mask[idx][2][b]) or (b > 27 and b != 37 and mask[idx][3][b - 28]))):
                        local_counter += 1
                        continue
                    # if we're doing a bench to board move and board is full and there is no champ at destination, skip
                    if a < 28 and b > 27 and b != 37 and not mask[idx][5][0] and not mask[idx][2][a]:
                        local_counter += 1
                        continue
                    local_action.append(policy_logits[idx][local_counter])
                    local_mappings.append(bytes(f"2_{a}_{b}", "utf-8"))
                    second_local_mappings.append(f"2_{a}_{b}")
                    local_counter += 1
            # for all board + bench slots...
            for a in range(37):
                # for every item slot...
                for b in range(10):
                    # if there is a unit and there is an item
                    if not (((a < 28 and mask[idx][2][a]) or (a > 27 and mask[idx][3][a - 28])) and mask[idx][4][b]):
                        local_counter += 1
                        continue
                    # if it is a legal action to put that item on the unit
                    if (mask[idx][7][a] and mask[idx][8][b]) or mask[idx][6][a]:
                        local_counter += 1
                        continue
                    local_action.append(policy_logits[idx][local_counter])
                    local_mappings.append(bytes(f"3_{a}_{b}", "utf-8"))
                    second_local_mappings.append(f"3_{a}_{b}")
                    local_counter += 1
            # level
            if mask[idx][0][4]:
                local_action.append(policy_logits[idx][local_counter])
                local_mappings.append(bytes("4", "utf-8"))
                second_local_mappings.append("4")
            local_counter += 1
            # roll
            if mask[idx][0][5]:
                local_action.append(policy_logits[idx][local_counter])
                local_mappings.append(bytes("5", "utf-8"))
                second_local_mappings.append("5")

            actions.append(local_action)
            mappings.append(local_mappings)
            second_mappings.append(second_local_mappings)
        return actions, mappings, second_mappings

    """
    Description - This is the core to the Complex Action Spaces paper. We take a set number of sample actions from the 
                  total number of actions based off of the current policy to expand on each turn. There are two options
                  as to how the samples are chosen. You can either set num_pass_shop_actions and refresh_level_actions
                  to 0 and comment out the following for loops or keep those variables at 6 and 2 and leave the for
                  loops in. The first option is a pure sample with no specific core actions. The second option gives 
                  you a set of core options to use. 
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
    def sample(self, policy_logits, string_mapping, byte_mapping, num_samples):
        output_logits = []
        output_string_mapping = []
        output_byte_mapping = []
        policy_sizes = []
        for i in range(len(policy_logits)):
            local_logits = []
            local_string = []
            local_byte = []
            # Switch this to 6 and refresh to 2 if using specified sampling.
            num_pass_shop_actions = 0
            refresh_level_actions = 0
            # Add samples for pass and the 5 shop options
            # Note that if there are not 5 available shop options, the sample here will be move options

            # for fixed_sample in range(0, 6):
            #     if (string_mapping[i][fixed_sample][0] == "0" or string_mapping[i][fixed_sample][0] == "1") \
            #             and config.SELECTED_SAMPLES:
            #         local_logits.append(policy_logits[i][fixed_sample])
            #         local_string.append(string_mapping[i][fixed_sample])
            #         local_byte.append(byte_mapping[i][fixed_sample])
            #     else:
            #         num_pass_shop_actions -= 1
            # # Add samples for refresh and level
            # # Note if either refresh or level is not available, the samples here will be move options
            # for last_sample in range(len(policy_logits[i]) - 2, len(policy_logits[i])):
            #     if (string_mapping[i][last_sample][0] == "4" or string_mapping[i][last_sample][0] == "5") \
            #             and config.SELECTED_SAMPLES:
            #         local_logits.append(policy_logits[i][last_sample])
            #         local_string.append(string_mapping[i][last_sample])
            #         local_byte.append(byte_mapping[i][last_sample])
            #     else:
            #         refresh_level_actions -= 1
            num_core_actions = num_pass_shop_actions + refresh_level_actions
            # Get the softmax of the policy output
            probs = self.softmax_stable(policy_logits[i][num_pass_shop_actions:
                                                         len(policy_logits[i]) - refresh_level_actions])
            # array of size [action_dim] with [0, 1, 2, 3... action_dim - 8]
            # We are removing 8 samples initially because those are the most important actions
            policy_range = np.arange(stop=len(policy_logits[i]) - num_core_actions)
            samples = np.random.choice(a=policy_range, size=num_samples - num_core_actions, p=probs)
            # Sort now so the mapping back to 1081 later is much faster
            samples.sort()
            prev_sample = -1
            for sample in samples:
                if sample == prev_sample:
                    local_logits[-1] += 1 / (num_samples - num_core_actions)
                else:
                    # Add the base value for the sample
                    local_logits.append(1 / (num_samples - num_core_actions))
                    # Add the name of the string action
                    local_string.append(string_mapping[i][sample + num_pass_shop_actions])
                    # Same but for the c++ side
                    local_byte.append(byte_mapping[i][sample + num_pass_shop_actions])
                prev_sample = sample
            output_logits.append(local_logits)
            output_string_mapping.append(local_string)
            output_byte_mapping.append(local_byte)
            policy_sizes.append(len(local_logits))
        return output_logits, output_string_mapping, output_byte_mapping, policy_sizes

    @staticmethod
    def softmax_stable(x):
        return np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum()

    def fill_metadata(self) -> Dict[str, str]:
        return {'network_id': str(self.network.training_steps())}

    @staticmethod
    def visit_softmax_temperature():
        return 1.0


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

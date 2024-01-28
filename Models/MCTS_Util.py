import config
import torch
import numpy as np


def create_default_mapping():
    local = []

    # Shop masking
    # for i in range(58):
    #     local.append(f"2_{i}_0_{378+252+1+1+37+i}")

    # Board masking
    # move_index = -1
    # for pos_1 in range(27):
    #     for pos_2 in range(pos_1 + 1, 28):
    #         move_index += 1
    #         local.append(f"1_{pos_1}_{pos_2}_{move_index}")

    # move_index = -1
    # for bench in range(9):
    #     for pos in range(28):
    #         move_index += 1
    #         local.append(f"1_{bench}_{pos}_{378 + move_index}")
    # Item masking
    # For all board + bench slots...
    # TODO
    # for a in range(37):
    #     # For every item slot...
    #     for b in range(10):
    #         # if there is a unit and there is an item
    #         local.append(f"_{a}_{b}")

    # Sell unit masking
    # for pos in range(37):
    #     local.append(f"3_{pos}_0_{378+252+1+1+pos}")

    # local.append(f"0_0_0_{378+252+1+1+37+58}")
    # local.append(f"4_0_0_{378+252+1}")
    # local.append(f"5_0_0_{378+252}")
    local.append("0_0_0_0")
    local.append("1_1_1_1")
    local.append("2_2_2_2")
    local.append("3_3_3_3")

    mappings = [local] * config.NUM_PLAYERS

    return mappings


# ["0", "1_0", "2_20_20", "3_4_20", "4_10", "5", "6"] ->
# [["0", "1", "2", "3", "4", "5", "6"], ["_0"], ["_20_20"], ["_4_20"], ["_10"]]
def split_sample_set(sample_mapping, target_policy):
    new_policy = np.zeros((4))

    for index, str_action in enumerate(sample_mapping):
        action = str_action.split("_")[0]
        new_policy[int(action)] = target_policy[index]

    return sample_mapping, new_policy

# [batch_size, unroll_steps, num_samples] to [unroll_steps, num dims, (batch_size, dim)]
def split_batch(mapping_batch, policy_batch):
    batch_size = len(mapping_batch)  # 256
    unroll_steps = len(mapping_batch[0])  # 6

    mapping = []
    policy = []

    for unroll_idx in range(unroll_steps):
        unroll_mapping = []
        unroll_policy = []

        for batch_idx in range(batch_size):
            local_mapping = mapping_batch[batch_idx][unroll_idx]
            local_policy = policy_batch[batch_idx][unroll_idx]
            unroll_mapping.append(local_mapping)
            unroll_policy.append(local_policy)

        mapping.append(unroll_mapping)
        policy.append(unroll_policy)

    return mapping, policy

def action_to_idx(action, dim):
    mapped_idx = None

    if dim == 0:  # type dim; 7; "0", "1", ... "6"
        mapped_idx = int(action)

    elif dim == 1:  # shop dim; 5; "_0", "_1", ... "_4"
        mapped_idx = int(action[1])  # "_1" -> "1"

    elif dim == 2:  # board dim; 630; "_0_1", "_0_2", ... "_37_28"
        action = action.split('_')  # "_20_21" -> ["", "20", "21"]
        from_loc = int(action[1])  # ["", "20", "21"] -> "20"
        to_loc = int(action[2])  # ["", "20", "21"] -> "21"
        mapped_idx = sum([35 - i for i in range(from_loc)]) + (to_loc - 1)

    elif dim == 3:  # item dim; 370; "_0_0", "_0_1", ... "_9_36"
        action = action.split('_')  # "_10_9" -> ["", "10", "9"]
        item_loc = int(action[1])  # "_0_20" -> "0"
        champ_loc = int(action[2])  # "_0_20" -> "20"
        mapped_idx = (10 * item_loc) + champ_loc

    elif dim == 4:  # sell dim; 37; "_0", "_1", "_36"
        mapped_idx = int(action[1:])  # "_15" -> "15"

    return mapped_idx


# both are (num_dims, [(batch_size, dim) ...] )
# TODO: Add a description of what this does
def map_output_to_distribution(mapping, policy_logits):
    num_dims = len(config.POLICY_HEAD_SIZES)
    batch_size = len(mapping[0])

    sampled_policy = []

    for batch_idx in range(batch_size):
        softmax_op = torch.nn.LogSoftmax(dim=-1)  # This may be able to be called outside this loop
        local_dim = softmax_op(policy_logits[batch_idx])
        local_mapping = mapping[batch_idx]

        sampled_dim = []

        # for action in local_mapping:
            # mapped_idx = action_to_idx(action, dim)
            # sampled_dim.append(local_dim[mapped_idx])

        # sampled_policy.append(sampled_dim)

    return sampled_policy


"""
Description - Turns the output_policy from shape [batch, num_samples] to [batch, encoding_size] to allow the trainer
              to train on the improved policy. 0s for everywhere that was not sampled.
Inputs      - mapping - List
                  A string mapping to know which values were sampled and which ones were not
              sample_dist - List
                  The improved policy output of the MCTS with size [batch, num_samples]
Outputs     - output_policy - List
                  The improved policy output of the MCTS with size [batch, encoding_size]
"""


def map_distribution_to_sample(mapping, policy_logits):
    output_policy = []
    for i in range(policy_logits.shape[0]):
        sampled_policy = torch.index_select(policy_logits[i], -1, torch.tensor(mapping[i]).cuda())
        output_policy.append(sampled_policy)
    return output_policy

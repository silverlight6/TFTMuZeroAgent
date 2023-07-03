import config
import torch
import numpy as np


def create_default_mapping():
    local_type = []
    local_shop = []
    local_board = []
    local_item = []
    local_sell = []

    # Shop masking
    for i in range(5):
        local_shop.append(f"_{i}")

    # Board masking
    # For all board + bench slots...
    for a in range(37):
        # rest of board slot locs for moving, last for sale
        for b in range(a, 37):
            if a == b:
                continue
            if a > 27:
                continue
            local_board.append(f"_{a}_{b}")
    # Item masking
    # For all board + bench slots...
    for a in range(37):
        # For every item slot...
        for b in range(10):
            # if there is a unit and there is an item
            local_item.append(f"_{a}_{b}")
    # Sell unit masking
    for a in range(9):
        local_sell.append(f"_{a}")

    # All Type mappings
    for i in range(7):
        local_type.append(f"{i}")

    mappings = [[local_type] * config.NUM_PLAYERS, [local_shop] * config.NUM_PLAYERS,
                [local_board] * config.NUM_PLAYERS, [local_item] * config.NUM_PLAYERS,
                [local_sell] * config.NUM_PLAYERS]

    return mappings


# ["0", "1_0", "2_20_20", "3_4_20", "4_10", "5", "6"] ->
# [["0", "1", "2", "3", "4", "5", "6"], ["_0"], ["_20_20"], ["_4_20"], ["_10"]]
def split_sample_set(sample_mapping, target_policy):
    split_sample = [[], [], [], [], []]
    split_policy = [[], [], [], [], []]

    # split_policy[0] = [0] * config.POLICY_HEAD_SIZES[0]

    for i, sample in enumerate(sample_mapping):
        base = sample[0]
        idx = int(base)

        if idx in config.NEEDS_2ND_DIM:
            location = sample[1:]

            if base not in split_sample[0]:
                split_sample[0].append(base)
                split_policy[0].append(0)

            split_sample[idx].append(location)
            split_policy[idx].append(target_policy[i])
            # split_policy[0][idx] += target_policy[i]
        else:
            split_sample[0].append(sample)
            split_policy[0].append(target_policy[i])

    # Accumulate the policy for each multidim action
    for i, base in enumerate(split_sample[0]):
        idx = int(base)

        if idx in config.NEEDS_2ND_DIM:
            policy_sum = sum(split_policy[idx])
            split_policy[0][i] += policy_sum

    return split_sample, split_policy

# [batch_size, unroll_steps, num_samples] to [unroll_steps, num dims, (batch_size, dim)]
def split_batch(mapping_batch, policy_batch):
    batch_size = len(mapping_batch)  # 256
    unroll_steps = len(mapping_batch[0])  # 6

    mapping = []
    policy = []

    for unroll_idx in range(unroll_steps):
        unroll_mapping = [[], [], [], [], []]
        unroll_policy = [[], [], [], [], []]

        for batch_idx in range(batch_size):
            local_mapping = mapping_batch[batch_idx][unroll_idx]
            local_policy = policy_batch[batch_idx][unroll_idx]

            for dim_idx in range(len(local_mapping)):
                unroll_mapping[dim_idx].append(local_mapping[dim_idx])
                unroll_policy[dim_idx].append(local_policy[dim_idx])

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

# Sample set is [ [batch_size, dim], [batch_size, dim] ...]
# Convert the sample set string to the corresponding idx
def sample_set_to_idx(sample_set):
    idx_set = []
    
    for dim, batch in enumerate(sample_set): # [batch_size, dim]
        dim_idx = []
        for sample in batch: # [dim]
            local_idx = []
            for action in sample: # single action
                local_idx.append(action_to_idx(action, dim))
            dim_idx.append(local_idx)
        idx_set.append(dim_idx)
                
    return idx_set

# target [ [batch_size, sampled_dim], [batch_size, sampled_dim] ...]
# idx_set [ [batch_size, sampled_dim], [batch_size, sampled_dim] ...]
# Create filled target with zeros as well as mask
def create_target_and_mask(target, idx_set):
    batch_size = config.BATCH_SIZE
    dim_sizes = config.POLICY_HEAD_SIZES
    
    target_filled = [
        np.zeros((batch_size, dim), dtype=np.float32) for dim in dim_sizes
    ]

    mask = [
        np.zeros((batch_size, dim), dtype=np.float32) for dim in dim_sizes
    ]
    
    # TODO: Find a native numpy function to do this
    for dim, batch in enumerate(idx_set): # [batch_size, dim]
        for batch_idx, sample in enumerate(batch):
            for target_idx, mapped_idx in enumerate(sample):
                target_filled[dim][batch_idx][mapped_idx] = target[dim][batch_idx][target_idx]
                mask[dim][batch_idx][mapped_idx] = 1.0
                
    return target_filled, mask


# both are (num_dims, [(batch_size, dim) ...] )
# TODO: Add a description of what this does
def map_output_to_distribution(mapping, policy_logits):
    num_dims = len(config.POLICY_HEAD_SIZES)
    batch_size = len(mapping[0])

    sampled_policy = []

    for dim in range(num_dims):
        batch_sampled_dim = []
        for batch_idx in range(batch_size):
            softmax_op = torch.nn.LogSoftmax(dim=-1)  # This may be able to be called outside this loop
            local_dim = softmax_op(policy_logits[dim][batch_idx])
            local_mapping = mapping[dim][batch_idx]

            sampled_dim = []

            for action in local_mapping:
                mapped_idx = action_to_idx(action, dim)
                sampled_dim.append(local_dim[mapped_idx])

            batch_sampled_dim.append(sampled_dim)

        sampled_policy.append(batch_sampled_dim)

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
        sampled_policy = torch.index_select(policy_logits[i], -1, torch.tensor(mapping[i]).to(config.DEVICE))
        output_policy.append(sampled_policy)
    return output_policy

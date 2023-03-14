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
    for a in range(37):
        local_sell.append(f"_{a}")

    # All Type mappings
    for i in range(7):
        local_type.append(f"{i}")

    mappings = []

    mappings.append([local_type] * config.NUM_PLAYERS)
    mappings.append([local_shop] * config.NUM_PLAYERS)
    mappings.append([local_board] * config.NUM_PLAYERS)
    mappings.append([local_item] * config.NUM_PLAYERS)
    mappings.append([local_sell] * config.NUM_PLAYERS)

    return mappings


# _, default_mapping = create_default_mapping()
# default_mapping = default_mapping[0]

# Takes in an action in string format "0" or "2_0_1" and outputs the default mapping index
action_dimensions = [1, 5, 667, 370, 1, 1]


def flatten_action(str_action):
    # Decode action
    num_items = str_action.count("_")  # 1
    split_action = str_action.split("_")  # [1, 0]

    action = [0, 0, 0]
    for i in range(num_items + 1):
        action[i] = int(split_action[i])

    # To index
    action_type = action[0]
    index = sum(action_dimensions[:action_type])
    if action_type == 1:
        index += action[1]
    elif action_type == 2:
        a = action[1]
        b = action[2]
        if a < 28:
            prev = sum([37 - i for i in range(a)])
            index += prev + (b - a - 1)
        else:
            index += action_dimensions[2] - (37 - a)
    elif action_type == 3:
        a = action[1]
        b = action[2]
        index += (10 * a) + b
    # No change needed for 4 and 5
    return index


def action_str_to_idx(sample_set):
    return [[[flatten_action(m) for m in batch] for batch in player] for player in sample_set]


# God forgive me
def flatten_sample_set(sample_set):
    return [item for player in sample_set for batch in player for item in batch]


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

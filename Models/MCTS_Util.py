import config
import torch
from Simulator import utils
import numpy as np
import time

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
            if a > 27 and b != 37:
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
    # converting mappings to batch size for all players in a game
    mappings = [mappings for _ in range(config.NUM_PLAYERS)]
    second_mappings = [second_mappings for _ in range(config.NUM_PLAYERS)]
    return mappings, second_mappings


# _, default_mapping = create_default_mapping()
# default_mapping = default_mapping[0]

# Takes in an action in string format "0" or "2_0_1" and outputs the default mapping index
action_dimensions = [1, 5, 667, 370, 1, 1]
def flatten_action(str_action):
    # Decode action
    num_items = str_action.count("_") # 1
    split_action = str_action.split("_") # [1, 0]
    
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
    # print(policy_logits.shape)
    output_policy = []
    mapping_indicies = [np.asarray([flatten_action(m) for m in batch]) for batch in mapping]
    for i in range(policy_logits.shape[0]):
        sampled_policy = torch.index_select(policy_logits[i], -1, torch.from_numpy(mapping_indicies[i]).cuda())
        output_policy.append(sampled_policy)
    return output_policy
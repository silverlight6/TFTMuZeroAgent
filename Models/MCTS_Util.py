import config
import torch

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


_, default_mapping = create_default_mapping()
default_mapping = default_mapping[0]

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
    for i in range(policy_logits.shape[0]):
        local_counter = 0
        local_policy = torch.empty(len(mapping[i]))
        # local_policy_original = []
        for j in range(len(default_mapping)):
            if default_mapping[j] == mapping[i][local_counter]:
                local_policy[local_counter] = policy_logits[i][j]
                # local_policy_original.append(policy_logits[i][j])
                local_counter += 1
                if local_counter == len(mapping[i]):
                    break
        output_policy.append(local_policy)

    return output_policy

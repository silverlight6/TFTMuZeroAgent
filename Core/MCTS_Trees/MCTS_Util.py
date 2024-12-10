import config
import torch
import numpy as np
import time


def create_default_mapping():
    mappings = []
    mask = []

    for b in range(37):
        for a in range(38):
            if a == 37:
                mappings.append(f"4_{b}")
                mask.append(1)
            else:
                mappings.append(f"5_{a}_{b}")
                if a == b:
                    mask.append(0)
                else:
                    mask.append(1)

    for b in range(10):
        for a in range(38):
            mappings.append(f"6_{a}_{b}")
            mask.append(1)

    for b in range(5):
        for a in range(38):
            if a == 0:
                mappings.append(f"3_{b}")
                mask.append(1)
            else:
                mappings.append(f"3_{b}_{a}")
                mask.append(0)

    for a in range(38):
        if a == 0:
            mappings.append(f"0")
            mask.append(1)
        else:
            mappings.append(f"0_{a}")
            mask.append(0)

    for a in range(38):
        if a == 0:
            mappings.append(f"1")
            mask.append(1)
        else:
            mappings.append(f"1_{a}")
            mask.append(0)

    for a in range(38):
        if a == 0:
            mappings.append(f"2")
            mask.append(1)
        else:
            mappings.append(f"2_{a}")
            mask.append(0)

    mappings = [mappings] * config.NUM_PLAYERS

    return mappings, mask

def create_position_default_mapping():
    mappings = []
    mask = []

    for a in range(29):
        mappings.append(f"{a}")
        mask.append(1)

    return mappings, mask

def split_sample_decide(sample_mapping, target_policy):
    if config.CHAMP_DECIDER:
        return split_sample_set_champ_decider(sample_mapping, target_policy)
    else:
        return split_sample_set(sample_mapping, target_policy)

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

# Size is [# of samples, len of champion list]
def split_sample_set_champ_decider(sample_mapping, target_policy):
    split_sample = [["0", "1", "2", "3", "4"] for _ in range(len(config.CHAMPION_ACTION_DIM))] + [["0", "1"]] + \
                   [["0", "1", "2"] for _ in range(len(config.ITEM_CHOICE_DIM))]
    split_policy = [[0, 0, 0, 0, 0] for _ in range(len(config.CHAMPION_ACTION_DIM))] + [[0, 0]] + \
                   [[0, 0, 0] for _ in range(len(config.ITEM_CHOICE_DIM))]
    for i, sample in enumerate(sample_mapping):
        for k in range(0, len(sample), 2):
            base = sample[k]
            idx = int(base)
            split_policy[int(k/2)][idx] += target_policy[i]
    return split_sample, split_policy

# [batch_size, unroll_steps, num_samples] to [unroll_steps, num dims, (batch_size, dim)]
def split_batch(mapping_batch, policy_batch):
    batch_size = len(mapping_batch)  # config.BATCH_SIZE
    unroll_steps = len(mapping_batch[0])  # config.UNROLL_STEPS

    mapping = []
    policy = []

    for unroll_idx in range(unroll_steps):
        if config.CHAMP_DECIDER:
            unroll_mapping = [[] for _ in range(len(config.CHAMP_DECIDER_ACTION_DIM))]
            unroll_policy = [[] for _ in range(len(config.CHAMP_DECIDER_ACTION_DIM))]

            for batch_idx in range(batch_size):
                local_mapping = mapping_batch[batch_idx][unroll_idx]
                local_policy = policy_batch[batch_idx][unroll_idx]

                for dim_idx in range(len(local_mapping)):
                    unroll_mapping[dim_idx].append(local_mapping[dim_idx])
                    unroll_policy[dim_idx].append(local_policy[dim_idx])

        else:
            unroll_mapping = []
            unroll_policy = []

            for batch_idx in range(batch_size):
                unroll_mapping.append(mapping_batch[batch_idx][unroll_idx])
                unroll_policy.append(policy_batch[batch_idx][unroll_idx])

        mapping.append(unroll_mapping)
        policy.append(unroll_policy)

    return mapping, policy

def action_to_idx(sample):
    mapped_idx = None
    split_action = sample.split("_")
    first_idx = int(split_action[0])

    if first_idx < 3:  # type dim; 7; "0", "1", ... "6"
        mapped_idx = 1977 + 38 * first_idx

    elif first_idx == 3:  # shop dim; 5; "_0", "_1", ... "_4"
        mapped_idx = 1939 + int(split_action[1])  # "_1" -> "1"

    elif first_idx == 4:  # board dim; 630; "_0_1", "_0_2", ... "_37_28"
        mapped_idx = 37 + 38 * int(split_action[1])

    elif first_idx == 5:  # item dim; 370; "_0_0", "_0_1", ... "_9_36"
        mapped_idx = 38 * int(split_action[1]) + int(split_action[2])

    elif first_idx == 6:  # sell dim; 37; "_0", "_1", "_36"
        mapped_idx = 1559 + 38 * int(split_action[2]) + int(split_action[1])

    return mapped_idx

# Sample set is [ [batch_size, dim], [batch_size, dim] ...]
# Convert the sample set string to the corresponding idx
def sample_set_to_idx(sample_set):
    idx_set = []

    for batch in sample_set:  # [batch_size, dim]
        dim_idx = []
        for sample in batch:  # [dim]
            dim_idx.append(action_to_idx(sample))
        idx_set.append(dim_idx)

    return idx_set

# target [ [batch_size, sampled_dim], [batch_size, sampled_dim] ...]
# idx_set [ [batch_size, sampled_dim], [batch_size, sampled_dim] ...]
# Create filled target with zeros as well as mask
def create_target_and_mask(target, idx_set):
    batch_size = config.BATCH_SIZE

    # TODO: If you see this code again, find a way to bring the config into this section of the code
    target_filled = np.zeros((batch_size, 2070), dtype=np.float32)

    # TODO: Find a native numpy function to do this
    for batch_idx, batch in enumerate(idx_set):  # [batch_size, dim]
        for sample_idx, sample in enumerate(batch):
            target_filled[batch_idx][sample] = target[batch_idx][sample_idx]

    return target_filled


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

class ValueEncoder:
    """Encoder for reward and value targets from Appendix of MuZero Paper."""

    def __init__(self,
                 min_value,
                 max_value,
                 num_steps,
                 use_contractive_mapping=True):

        if not max_value > min_value:
            raise ValueError('max_value must be > min_value')
        if use_contractive_mapping:
            max_value = contractive_mapping(max_value)
            min_value = contractive_mapping(min_value)
        if num_steps <= 0:
            num_steps = torch.ceil(max_value) + 1 - torch.floor(min_value)
        self.min_value = min_value
        self.max_value = max_value
        self.value_range = max_value - min_value
        self.num_steps = num_steps
        self.step_size = self.value_range / (num_steps - 1)
        self.step_range_int = torch.arange(
            0, self.num_steps, dtype=torch.int64)
        self.step_range_float = self.step_range_int.type(
            torch.float32).to(config.DEVICE)
        self.use_contractive_mapping = use_contractive_mapping

    def encode(self, value):  # not worth optimizing
        if len(value.shape) != 1:
            raise ValueError(
                'Expected value to be 1D Tensor [batch_size], but got {}.'.format(
                    value.shape))
        if self.use_contractive_mapping:
            value = contractive_mapping(value)
        value = torch.unsqueeze(value, -1)
        clipped_value = torch.clip(value, self.min_value, self.max_value)
        above_min = clipped_value - self.min_value
        num_steps = above_min / self.step_size
        lower_step = torch.floor(num_steps)
        upper_mod = num_steps - lower_step
        lower_step = lower_step.type(torch.int64)
        upper_step = lower_step + 1
        lower_mod = 1.0 - upper_mod
        lower_encoding, upper_encoding = (
            torch.eq(step, self.step_range_int).type(torch.float32) * mod
            for step, mod in (
                (lower_step, lower_mod),
                (upper_step, upper_mod),)
        )
        return lower_encoding + upper_encoding

    def decode(self, logits):  # not worth optimizing
        if len(logits.shape) != 2:
            raise ValueError(
                'Expected logits to be 2D Tensor [batch_size, steps], but got {}.'
                .format(logits.shape))
        num_steps = torch.sum(logits * self.step_range_float, -1)
        above_min = num_steps * self.step_size
        value = above_min + self.min_value
        if self.use_contractive_mapping:
            value = inverse_contractive_mapping(value)
        return value

    def decode_softmax(self, logits):
        return self.decode(torch.softmax(logits, dim=-1))

# From the MuZero paper.


def contractive_mapping(x, eps=0.001):
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1.) - 1.) + eps * x


# From the MuZero paper.
def inverse_contractive_mapping(x, eps=0.001):
    return torch.sign(x) * (torch.square((torch.sqrt(4 * eps * (torch.abs(x) + 1. + eps) + 1.) - 1.) / (2. * eps)) - 1.)

# Softmax function in np because we're converting it anyway


def softmax_stable(x):
    return np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum()

def dict_to_cpu(dictionary):
    cpu_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, torch.Tensor):
            cpu_dict[key] = value.cpu()
        elif isinstance(value, dict):
            cpu_dict[key] = dict_to_cpu(value)
        else:
            cpu_dict[key] = value
    return cpu_dict

def dcord_to_2dcord(dcord):
    x = dcord % 7
    y = (dcord - x) // 7
    return x, y


def action_to_3d(action):
    cube_action = np.zeros((action.shape[0], 7, 4, 7))
    for i in range(action.shape[0]):
        action_selector = np.argmax(action[i][0])
        if action_selector == 0:
            cube_action[0, :, :] = np.ones((1, 4, 7))
        elif action_selector == 1:
            champ_shop_target = np.argmax(action[i][2])
            if champ_shop_target < 5:
                cube_action[1, champ_shop_target, 9] = 1
        elif action_selector == 2:
            champ1 = dcord_to_2dcord(action[i][1])
            cube_action[2, champ1[0], champ1[1]] = 1
            champ2 = dcord_to_2dcord(action[i][2])
            cube_action[2, champ2[0], champ2[1]] = 1
        elif action_selector == 3:
            champ1 = dcord_to_2dcord(action[i][1])
            cube_action[3, champ1[0], champ1[1]] = 1
            cube_action[3, 5, action[i][2]] = 1
        elif action_selector == 4:
            champ1 = dcord_to_2dcord(action[i][1])
            cube_action[4, champ1[0], champ1[1]] = 1
        elif action_selector == 5:
            cube_action[5, :, :] = np.ones((1, 4, 7))
        elif action_selector == 6:
            cube_action[6, :, :] = np.ones((1, 4, 7))
    return cube_action
import torch
import config
import collections
import numpy as np
import time
import os
import torch.nn as nn

NetworkOutput = collections.namedtuple(
    'NetworkOutput',
    'value reward policy_logits hidden_state')


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


class AbstractNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def initial_inference(self, observation):
        pass

    def recurrent_inference(self, encoded_state, action):
        pass

    def get_weights(self):
        return dict_to_cpu(self.state_dict())

    def set_weights(self, weights):
        self.load_state_dict(weights)
        self.eval()

    # Renaming as to not override built-in functions
    def tft_save_model(self, episode):
        if not os.path.exists("./Checkpoints"):
            os.makedirs("./Checkpoints")

        path = f'./Checkpoints/checkpoint_{episode}'
        torch.save(self.state_dict(), path)

    # Renaming as to not override built-in functions
    def tft_load_model(self, episode):
        path = f'./Checkpoints/checkpoint_{episode}'
        if os.path.isfile(path):
            self.load_state_dict(torch.load(path))
            self.eval()
        else:
            # print("Initializing model with new weights.")
            pass


class MuZeroNetwork(AbstractNetwork):
    def __init__(self):
        super().__init__()
        self.full_support_size = config.ENCODER_NUM_STEPS

        self.representation_network = mlp(config.OBSERVATION_SIZE, [config.LAYER_HIDDEN_SIZE] *
                                          config.N_HEAD_HIDDEN_LAYERS, config.HIDDEN_STATE_SIZE)

        self.action_encodings = mlp(config.ACTION_CONCAT_SIZE, [
                                    config.LAYER_HIDDEN_SIZE] * 0, config.HIDDEN_STATE_SIZE)

        self.dynamics_hidden_state_network = torch.nn.LSTM(input_size=config.HIDDEN_STATE_SIZE,
                                                           num_layers=config.NUM_RNN_CELLS,
                                                           hidden_size=config.LSTM_SIZE, batch_first=True).to(config.DEVICE)

        self.dynamics_reward_network = mlp(config.HIDDEN_STATE_SIZE, [
                                           1] * 1, self.full_support_size)

        self.prediction_policy_network = MultiMlp(config.HIDDEN_STATE_SIZE, [config.LAYER_HIDDEN_SIZE] *
                                                  config.N_HEAD_HIDDEN_LAYERS, config.POLICY_HEAD_SIZES)

        self.prediction_value_network = mlp(config.HIDDEN_STATE_SIZE, [config.LAYER_HIDDEN_SIZE] *
                                            config.N_HEAD_HIDDEN_LAYERS, self.full_support_size)

        min = torch.tensor(-300., dtype=torch.float32)
        max = torch.tensor(300., dtype=torch.float32)
        self.value_encoder = ValueEncoder(
            *tuple(map(inverse_contractive_mapping, (min, max))), 0)
        self.reward_encoder = ValueEncoder(
            *tuple(map(inverse_contractive_mapping, (min, max))), 0)

    def prediction(self, encoded_state):
        policy_logits = self.prediction_policy_network(encoded_state)
        value = self.prediction_value_network(encoded_state)
        return policy_logits, value

    def representation(self, observation):
        observation = torch.from_numpy(observation).float().to(config.DEVICE)
        encoded_state = self.representation_network(observation)
        # Scale encoded state between [0, 1] (See appendix paper Training)
        min_encoded_state = encoded_state.min(1, keepdim=True)[0]
        max_encoded_state = encoded_state.max(1, keepdim=True)[0]
        scale_encoded_state = max_encoded_state - min_encoded_state
        scale_encoded_state[scale_encoded_state < 1e-5] += 1e-5
        encoded_state_normalized = (
            encoded_state - min_encoded_state
        ) / scale_encoded_state
        return encoded_state_normalized

    def dynamics(self, hidden_state, action):
        action = torch.from_numpy(action).to(config.DEVICE).to(torch.int64)
        one_hot_action = torch.nn.functional.one_hot(
            action[:, 0], config.ACTION_DIM[0])
        one_hot_target_a = torch.nn.functional.one_hot(
            action[:, 1], config.ACTION_DIM[1])
        one_hot_target_b = torch.nn.functional.one_hot(
            action[:, 2], config.ACTION_DIM[1])

        action_one_hot = torch.cat(
            [one_hot_action, one_hot_target_a, one_hot_target_b], dim=-1).float()

        action_encodings = self.action_encodings(action_one_hot)

        lstm_state = self.flat_to_lstm_input(hidden_state)

        inputs = action_encodings
        inputs = inputs[:, None, :]

        h0, c0 = list(zip(*lstm_state))
        _, new_nested_states = self.dynamics_hidden_state_network(inputs,
                                                                  (torch.stack(h0, dim=0), torch.stack(c0, dim=0)))

        next_hidden_state = self.rnn_to_flat(
            new_nested_states)  # (8, 1024) ##DOUBLE CHECK THIS

        reward = self.dynamics_reward_network(next_hidden_state)

        # Scale encoded state between [0, 1] (See paper appendix Training)
        min_next_hidden_state = next_hidden_state.min(1, keepdim=True)[0]
        max_next_hidden_state = next_hidden_state.max(1, keepdim=True)[0]
        scale_next_hidden_state = max_next_hidden_state - min_next_hidden_state
        scale_next_hidden_state[scale_next_hidden_state < 1e-5] += 1e-5
        next_hidden_state_normalized = (
            next_hidden_state - min_next_hidden_state
        ) / scale_next_hidden_state

        return next_hidden_state_normalized, reward

    def initial_inference(self, observation):
        hidden_state = self.representation(observation)
        policy_logits, value_logits = self.prediction(hidden_state)

        value = self.value_encoder.decode_softmax(value_logits)

        reward = torch.zeros(observation.shape[0])
        reward_logits = self.reward_encoder.encode(reward)

        outputs = {
            "value": value,
            "value_logits": value_logits,
            "reward": reward,
            "reward_logits": reward_logits,
            "policy_logits": policy_logits,
            "hidden_state": hidden_state
        }
        return outputs

    @staticmethod
    def rnn_to_flat(state):
        """Maps LSTM state to flat vector."""
        states = []
        for cell_state in state:
            states.extend(cell_state)
        return torch.cat(states, dim=-1)

    @staticmethod
    def flat_to_lstm_input(state):
        """Maps flat vector to LSTM state."""
        tensors = []
        cur_idx = 0
        for size in config.RNN_SIZES:
            states = (state[Ellipsis, cur_idx:cur_idx + size],
                      state[Ellipsis, cur_idx + size:cur_idx + 2 * size])

            cur_idx += 2 * size
            tensors.append(states)
        # assert cur_idx == state.shape[-1]
        return tensors

    def recurrent_inference(self, hidden_state, action):
        next_hidden_state, reward_logits = self.dynamics(hidden_state, action)
        policy_logits, value_logits = self.prediction(next_hidden_state)

        value = self.value_encoder.decode_softmax(value_logits)
        reward = self.reward_encoder.decode_softmax(reward_logits)

        outputs = {
            "value": value,
            "value_logits": value_logits,
            "reward": reward,
            "reward_logits": reward_logits,
            "policy_logits": policy_logits,
            "hidden_state": next_hidden_state
        }
        return outputs


def mlp(
        input_size,
        layer_sizes,
        output_size,
        output_activation=torch.nn.Identity,
        activation=torch.nn.LeakyReLU,
):
    sizes = [input_size] + layer_sizes + [output_size]
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [torch.nn.Linear(sizes[i], sizes[i + 1]), act()]
    return torch.nn.Sequential(*layers).to(config.DEVICE)


# Cursed? Idk
# Linear(input, layer_size) -> RELU
#      -> Linear -> Identity -> 0
#      -> Linear -> Identity -> 1
#      ... for each size in output_size
#  -> output -> [0, 1, ... n]
class MultiMlp(nn.Module):
    def __init__(self,
                 input_size,
                 layer_sizes,
                 output_sizes,
                 output_activation=nn.Identity,
                 activation=nn.LeakyReLU):
        super().__init__()

        sizes = [input_size] + layer_sizes

        layers = []
        for i in range(len(sizes) - 1):
            layers += [torch.nn.Linear(sizes[i], sizes[i + 1]), activation()]

        # Encodes the observation to a shared hidden state between all heads
        self.encoding_layer = nn.Sequential(*layers).to(config.DEVICE)

        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(sizes[-1], size),
                output_activation()
            ) for size in output_sizes
        ]).to(config.DEVICE)

    def forward(self, x):
        # Encode the hidden state
        x = self.encoding_layer(x)

        output = [
            head(x)
            for head in self.output_heads
        ]

        return output


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
    return torch.sign(x) * \
        (torch.square(
            (torch.sqrt(4 * eps * (torch.abs(x) + 1. + eps) + 1.) - 1.) / (2. * eps)) - 1.)

# Softmax function in np because we're converting it anyway


def softmax_stable(x):
    return np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum()

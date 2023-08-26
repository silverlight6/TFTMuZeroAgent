import collections
from Models.MCTS_Util import *
from Models.abstract_model import AbstractNetwork
from Models.torch_layers import mlp, resnet, MultiMlp


NetworkOutput = collections.namedtuple(
    'NetworkOutput',
    'value reward policy_logits hidden_state')


class MuZeroNetwork(AbstractNetwork):
    def __init__(self):
        super().__init__()
        self.full_support_size = config.ENCODER_NUM_STEPS

        self.representation_network = RepNetwork(28, [256] * 16, 1, config.HIDDEN_STATE_SIZE).cuda()

        self.dynamics_network = DynNetwork(28, [256] * 16, 1, self.full_support_size).cuda()

        self.prediction_network = PredNetwork(28, [256] * 16, 1, self.full_support_size).cuda()

        min = torch.tensor(-300., dtype=torch.float32)
        max = torch.tensor(300., dtype=torch.float32)
        self.value_encoder = ValueEncoder(
            *tuple(map(inverse_contractive_mapping, (min, max))), 0)
        self.reward_encoder = ValueEncoder(
            *tuple(map(inverse_contractive_mapping, (min, max))), 0)

    def prediction(self, encoded_state):
        policy_logits, value = self.prediction_network(encoded_state)
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
        cube_action = torch.from_numpy(action_to_3d(action)).to('cuda')

        next_hidden_state, reward = self.dynamics_network(hidden_state, cube_action)

        min_next_hidden_state = next_hidden_state.min(1, keepdim=True)[0]
        max_next_hidden_state = next_hidden_state.max(1, keepdim=True)[0]
        scale_next_hidden_state = max_next_hidden_state - min_next_hidden_state
        scale_next_hidden_state[scale_next_hidden_state < 1e-5] += 1e-5

        next_hidden_state_normalized = (next_hidden_state - min_next_hidden_state) / scale_next_hidden_state

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


class RepNetwork(torch.nn.Module):
    def __init__(self, input_size, layer_sizes, output_size, encoding_size) -> torch.nn.Module:
        super().__init__()

        self.conv1 = torch.nn.Conv2d(183, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = torch.nn.BatchNorm2d(256)
        self.relu = torch.nn.ReLU(inplace=True)
        self.resnet = resnet(input_size, layer_sizes, output_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.resnet(x)

        return x

    def __call__(self, x):
        return self.forward(x)


class DynNetwork(torch.nn.Module):
    def __init__(self, input_size, layer_sizes, output_size, encoding_size) -> torch.nn.Module:
        super().__init__()

        self.conv1 = torch.nn.Conv2d(263, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = torch.nn.BatchNorm2d(256)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv_reward = torch.nn.Conv2d(256, 1, 1)
        self.bn_reward = torch.nn.BatchNorm2d(1)
        self.fc_reward = mlp(28, [config.LAYER_HIDDEN_SIZE] * config.N_HEAD_HIDDEN_LAYERS, encoding_size)
        self.resnet = resnet(input_size, layer_sizes, output_size)

    def forward(self, x, action):
        state = torch.concatenate((x, action), dim=1).type(torch.cuda.FloatTensor)
        x = self.conv1(state)
        x = self.bn(x)
        x = self.relu(x)
        x = self.resnet(x)
        new_state = x

        reward = self.conv_reward(x)
        reward = self.bn_reward(reward)
        flat = torch.flatten(reward, start_dim=1)
        reward = self.fc_reward(flat)

        return new_state, reward

    def __call__(self, x, action):
        return self.forward(x, action)

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

    @staticmethod
    def rnn_to_flat(state):
        """Maps LSTM state to flat vector."""
        states = []
        for cell_state in state:
            states.extend(cell_state)
        return torch.cat(states, dim=-1)

class PredNetwork(torch.nn.Module):
    def __init__(self, input_size, layer_sizes, output_size, encoding_size) -> torch.nn.Module:
        super().__init__()

        self.resnet = resnet(input_size, layer_sizes, output_size)
        self.conv_value = torch.nn.Conv2d(256, 3, 1)
        self.bn_value = torch.nn.BatchNorm2d(3)
        self.conv_policy = torch.nn.Conv2d(256, 3, 1)
        self.bn_policy = torch.nn.BatchNorm2d(3)
        self.relu = torch.nn.ReLU(inplace=True)
        self.fc_internal_v = torch.nn.Linear(84, 128)
        self.fc_value = mlp(128, [config.LAYER_HIDDEN_SIZE] * config.N_HEAD_HIDDEN_LAYERS, encoding_size)
        self.fc_internal_p = torch.nn.Linear(84, 128)
        self.fc_policy = MultiMlp(128, [config.LAYER_HIDDEN_SIZE] * config.N_HEAD_HIDDEN_LAYERS,
                                  config.POLICY_HEAD_SIZES, output_activation=torch.nn.Sigmoid)

    def forward(self, x):
        x = self.resnet(x)

        value = self.conv_value(x)
        value = self.bn_value(value)
        value = self.relu(value)
        value = torch.flatten(value, start_dim=1)
        value = self.fc_internal_v(value)
        value = self.relu(value)
        value = self.fc_value(value)

        policy = self.conv_policy(x)
        policy = self.bn_policy(policy)
        policy = self.relu(policy)
        policy = torch.flatten(policy, start_dim=1)
        policy = self.fc_internal_p(policy)
        policy = self.relu(policy)
        policy = self.fc_policy(policy)

        return policy, value

    def __call__(self, x):
        return self.forward(x)

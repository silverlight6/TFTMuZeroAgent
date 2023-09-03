import torch
import config
import torch.nn as nn

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

class ResLayer(torch.nn.Module):
    def __init__(self, input_channels, n_kernels) -> torch.nn.Module:
        super().__init__()

        self.conv1 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(256)
        self.conv2 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(256)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        input = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += input

        return self.relu(out)

    def __call__(self, x):
        return self.forward(x)

def resnet(input_size,
        layer_sizes,
        output_size):
    sizes = [input_size] + layer_sizes + [output_size]
    layers = []
    for i in range(1, len(sizes) - 1):
        layers += [ResLayer(sizes[i], sizes[i + 1])]
    
    return torch.nn.Sequential(*layers).cuda()


def normalize(x):
    min_encoded_state = x.min(1, keepdim=True)[0]
    max_encoded_state = x.max(1, keepdim=True)[0]
    scale_encoded_state = max_encoded_state - min_encoded_state
    scale_encoded_state[scale_encoded_state < 1e-5] += 1e-5
    encoded_state_normalized = (
        x - min_encoded_state
    ) / scale_encoded_state
    return encoded_state_normalized

# Turn normalize into a torch layer
class Normalize(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return normalize(x)

class MemoryLayer(nn.Module):
    def __init__(self, input_size, num_layers, hidden_size):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=input_size, num_layers=num_layers, hidden_size=hidden_size, batch_first=True).to(config.DEVICE)
    
    def forward(self, x):
        inputs, hidden_state = x
        lstm_state = self.flat_to_lstm_input(hidden_state)
        h0, c0 = list(zip(*lstm_state))
        _, new_nested_states = self.lstm(inputs, (torch.stack(h0, dim=0), torch.stack(c0, dim=0)))
            
        return self.rnn_to_flat(new_nested_states)
        
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
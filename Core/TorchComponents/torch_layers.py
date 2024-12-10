import config
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

def ppo_mlp(
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
        layers += [layer_init(torch.nn.Linear(sizes[i], sizes[i + 1])), act()]
    return torch.nn.Sequential(*layers).to(config.DEVICE)

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


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
                 activation=nn.LeakyReLU,
                 dropout_rate=0.1):
        super().__init__()

        # Validate activation functions
        assert callable(activation), "activation must be callable"
        assert callable(output_activation), "output_activation must be callable"

        sizes = [input_size] + layer_sizes

        layers = []
        for i in range(len(sizes) - 1):
            layers += [
                nn.Linear(sizes[i], sizes[i + 1]),
                nn.LayerNorm(sizes[i + 1]),  # Add LayerNorm
                activation(),
                nn.Dropout(dropout_rate)  # Add Dropout
            ]

        # Encodes the observation to a shared hidden state between all heads
        self.encoding_layer = nn.Sequential(*layers)

        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(sizes[-1], size),
                output_activation(),
            ) for size in output_sizes
        ])

        self.to(config.DEVICE)

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
    scale_encoded_state = torch.where(scale_encoded_state < 1e-5, scale_encoded_state + 1e-5, scale_encoded_state)
    # scale_encoded_state[scale_encoded_state < 1e-5] += 1e-5
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
    def __init__(self, input_size, num_layers, hidden_size, model_config):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=input_size, num_layers=num_layers,
                                  hidden_size=hidden_size, batch_first=True).to(config.DEVICE)
        self.model_config = model_config
    
    def forward(self, x):
        inputs, hidden_state = x
        lstm_state = self.flat_to_lstm_input(hidden_state)
        h0, c0 = list(zip(*lstm_state))
        _, new_nested_states = self.lstm(inputs, (torch.stack(h0, dim=0), torch.stack(c0, dim=0)))
            
        return self.rnn_to_flat(new_nested_states)
        
    def flat_to_lstm_input(self, state):
        """Maps flat token to LSTM state."""
        tensors = []
        cur_idx = 0
        for size in self.model_config.RNN_SIZES:
            states = (state[Ellipsis, cur_idx:cur_idx + size],
                      state[Ellipsis, cur_idx + size:cur_idx + 2 * size])

            cur_idx += 2 * size
            tensors.append(states)
        # assert cur_idx == state.shape[-1]
        return tensors

    @staticmethod
    def rnn_to_flat(state):
        """Maps LSTM state to flat token."""
        states = []
        for cell_state in state:
            states.extend(cell_state)
        return torch.cat(states, dim=-1)

class ResidualBlock(nn.Module):
    """
    A simple residual block with a skip connection.
    """
    def __init__(self, block, norm_size, local_norm=True):
        super(ResidualBlock, self).__init__()
        self.block = block
        self.normalize = local_norm
        if normalize:
            self.batch_norm = nn.BatchNorm1d(norm_size)

    def forward(self, x):
        x = x + self.block(x)
        if self.normalize:
            x = self.batch_norm(x)
        return x


class TransformerEncoder(nn.Module):
    """
    A basic Transformer encoder block.
    """
    def __init__(self, d_model, d_hidden, n_heads, n_layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()

        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, d_hidden, dropout, batch_first=True)
            for _ in range(n_layers)
        ])

        self.to(config.DEVICE)

    def forward(self, x, src_key_padding_mask=None):
        """
        Args:
            x: Input tensor of shape (sequence_length, batch_size, d_model).
               In TFT, sequence_length could be the number of champions on the board/bench
               or the number of items.
            src_key_padding_mask: Boolean tensor indicating padding positions.
                                  Shape: (batch_size, sequence_length)
                                  True values indicate padding positions.
        """

        # Iterate over the encoder layers
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, src_key_padding_mask=src_key_padding_mask)
        return x

class AlternateFeatureEncoder(torch.nn.Module):
    def __init__(self, input_size, layer_sizes, output_size, device, dropout_rate=0.1, use_layer_norm=True):
        super(AlternateFeatureEncoder, self).__init__()

        layers = []
        current_size = input_size

        # Add hidden layers
        for size in layer_sizes:
            layers.append(torch.nn.Linear(current_size, size))
            layers.append(torch.nn.ReLU())

            # Optionally add layer normalization
            if use_layer_norm:
                layers.append(torch.nn.LayerNorm(size))

            # Optionally add dropout
            layers.append(torch.nn.Dropout(dropout_rate))

            current_size = size

        # Final layer
        layers.append(torch.nn.Linear(current_size, output_size))

        self.network = torch.nn.Sequential(*layers).to(device)
        self.device = device

    def forward(self, x):
        # x = x.to(self.device)  # Ensure the input is on the correct device
        return self.network(x)


class ResidualCNNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int | str = "same",
        device: str = "cpu",
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            device=device,
        )
        self.ln1 = nn.InstanceNorm2d(out_channels, device=device)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            device=device,
        )
        self.ln2 = nn.InstanceNorm2d(out_channels, device=device)

    def forward(self, x: torch.Tensor):
        y = self.conv1(x)
        y = self.ln1(x)
        y = F.relu(y)
        y = self.conv2(y)
        y = self.ln2(x)
        y = y + x
        y = F.relu(y)
        return y

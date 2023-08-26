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


def resnet(input_size,
           layer_sizes,
           output_size):
    sizes = [input_size] + layer_sizes + [output_size]
    layers = []
    for i in range(1, len(sizes) - 1):
        layers += [ResLayer(sizes[i], sizes[i + 1])]

    return torch.nn.Sequential(*layers).cuda()

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> torch.nn.Conv2d:
    """1x1 convolution"""
    return torch.nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


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


import torch
import config
import torch.nn as nn
from Core.TorchComponents.torch_layers import MemoryLayer, Normalize, AlternateFeatureEncoder, \
    ResidualCNNBlock as ResidualBlock

class DynNetwork(torch.nn.Module):
    def __init__(self, input_size, num_layers, hidden_size, model_config) -> torch.nn.Module:
        super().__init__()

        def memory():
            return torch.nn.Sequential(
                MemoryLayer(input_size, num_layers, hidden_size, model_config),
                Normalize()
            ).to(config.DEVICE)

        self.action_embeddings = torch.nn.Embedding(model_config.POLICY_HEAD_SIZE,
                                                    model_config.ACTION_EMBEDDING_DIM).to(config.DEVICE)
        if config.GUMBEL:
            self.action_encodings = AlternateFeatureEncoder(model_config.ACTION_EMBEDDING_DIM, [
                model_config.LAYER_HIDDEN_SIZE] * 0, model_config.HIDDEN_STATE_SIZE, config.DEVICE)
        else:
            self.action_encodings = AlternateFeatureEncoder(config.ACTION_CONCAT_SIZE, [
                model_config.LAYER_HIDDEN_SIZE] * 0, model_config.HIDDEN_STATE_SIZE, config.DEVICE)

        self.dynamics_memory = memory()

        self.dynamics_reward_network = AlternateFeatureEncoder(input_size, [model_config.LAYER_HIDDEN_SIZE] * 1,
                                                               model_config.ENCODER_NUM_STEPS, config.DEVICE)
        self.model_config = model_config

    def forward(self, hidden_state, action):
        action = torch.from_numpy(action).to(config.DEVICE).to(torch.int64)
        if not config.GUMBEL:
            one_hot_action = torch.nn.functional.one_hot(action[:, 0], config.ACTION_DIM[0])
            one_hot_target_a = torch.nn.functional.one_hot(action[:, 1], config.ACTION_DIM[1])
            one_hot_target_b = torch.nn.functional.one_hot(action[:, 2], config.ACTION_DIM[1])
            action = torch.cat([one_hot_action, one_hot_target_a, one_hot_target_b], dim=-1).float()
        else:
            action = self.action_embeddings(action)

        action_encoding = self.action_encodings(action)

        inputs = action_encoding.to(torch.float32)
        inputs = inputs[:, None, :]

        new_hidden_state = self.dynamics_memory((inputs, hidden_state))

        reward = self.dynamics_reward_network(new_hidden_state)

        return new_hidden_state, reward

class PositionDynNetwork(torch.nn.Module):
    def __init__(self, input_size, num_layers, hidden_size, model_config) -> torch.nn.Module:
        super().__init__()

        def memory():
            return torch.nn.Sequential(
                MemoryLayer(input_size, num_layers, hidden_size, model_config),
                Normalize()
                # ).to(config.DEVICE)
            )

        self.action_embeddings = torch.nn.Embedding(29, model_config.ACTION_EMBEDDING_DIM).to(config.DEVICE)
        self.action_encodings = AlternateFeatureEncoder(model_config.ACTION_EMBEDDING_DIM, [
                model_config.LAYER_HIDDEN_SIZE] * 0, model_config.HIDDEN_STATE_SIZE, config.DEVICE)

        self.dynamics_memory = memory()

        # self.dynamics_reward_network = AlternateFeatureEncoder(input_size, [model_config.LAYER_HIDDEN_SIZE] * 1,
        #                                                        model_config.ENCODER_NUM_STEPS, config.DEVICE)
        self.model_config = model_config

    def forward(self, hidden_state, action):
        action = torch.from_numpy(action).to(config.DEVICE).to(torch.int64)
        action = self.action_embeddings(action)

        action_encoding = self.action_encodings(action)

        inputs = action_encoding.to(torch.float32)
        inputs = inputs[:, None, :]

        new_hidden_state = self.dynamics_memory((inputs, hidden_state))

        # reward = self.dynamics_reward_network(new_hidden_state)

        return new_hidden_state

class AtariDynamicsNetwork(nn.Module):
    def __init__(
        self,
        action_space: int,
        hidden_channels: int,
        num_blocks: int = 16,
        kernel_size: int = 3,
        device: str = "cpu",
    ):
        super().__init__()
        self.action_space = action_space
        in_channels = hidden_channels + 1

        # -- New Representation
        blocks = [
            nn.Conv2d(
                in_channels, hidden_channels, kernel_size, padding="same", device=device
            ),
        ]

        for _ in range(num_blocks - 1):
            blocks.append(
                ResidualBlock(
                    hidden_channels, hidden_channels, kernel_size, device=device
                )
            )

        self.dyn_res_blocks = nn.Sequential(*blocks)

    def forward(self, hidden_state, action):
        # action -> [...]
        device = hidden_state.device
        action = torch.tensor(action).to(device)
        B, C, H, W = hidden_state.shape
        action_one_hot = torch.ones((B, 1, H, W)).float().to(device)
        action_one_hot = (
            action[:, None, None, None] * action_one_hot / self.action_space
        )

        x = torch.cat((hidden_state, action_one_hot), dim=1)

        new_hidden_state = self.dyn_res_blocks(x)

        return new_hidden_state

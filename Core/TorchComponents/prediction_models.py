import torch
import torch.nn as nn
import config
from Core.TorchComponents.torch_layers import AlternateFeatureEncoder, MultiMlp

class PredNetwork(torch.nn.Module):
    def __init__(self, model_config) -> torch.nn.Module:
        super().__init__()

        def feature_encoder(input_size, layer_sizes, output_size):
            return torch.nn.Sequential(AlternateFeatureEncoder(input_size, layer_sizes,
                                                               output_size, config.DEVICE)).to(config.DEVICE)

        # 6 is the number of separate dynamic networks
        self.prediction_value_network = feature_encoder(model_config.HIDDEN_STATE_SIZE,
                                                        [model_config.LAYER_HIDDEN_SIZE] *
                                                        model_config.N_HEAD_HIDDEN_LAYERS,
                                                        model_config.ENCODER_NUM_STEPS)
        self.policy_network = feature_encoder(model_config.HIDDEN_STATE_SIZE,
                                              [model_config.LAYER_HIDDEN_SIZE] * model_config.N_HEAD_HIDDEN_LAYERS,
                                              model_config.POLICY_HEAD_SIZE)

        self.comp_predictor_network = MultiMlp(model_config.HIDDEN_STATE_SIZE,
                                               [model_config.LAYER_HIDDEN_SIZE] * model_config.N_HEAD_HIDDEN_LAYERS,
                                               config.TEAM_TIERS_VECTOR)

        self.final_comp_predictor_network = MultiMlp(model_config.HIDDEN_STATE_SIZE,
                                                     [model_config.LAYER_HIDDEN_SIZE] *
                                                     model_config.N_HEAD_HIDDEN_LAYERS,
                                                     config.TEAM_TIERS_VECTOR)

        self.champ_predictor_network = MultiMlp(model_config.HIDDEN_STATE_SIZE,
                                                [model_config.LAYER_HIDDEN_SIZE] * model_config.N_HEAD_HIDDEN_LAYERS,
                                                config.CHAMPION_LIST_DIM)

    def forward(self, hidden_state, training=True):

        value = self.prediction_value_network(hidden_state)
        policy = self.policy_network(hidden_state)

        if not training:
            return policy, value
        else:
            comp = self.comp_predictor_network(hidden_state)

            final_comp = self.final_comp_predictor_network(hidden_state)

            champ = self.champ_predictor_network(hidden_state)
            return policy, value, comp, final_comp, champ

class PredPositionNetwork(torch.nn.Module):
    def __init__(self, model_config) -> torch.nn.Module:
        super().__init__()

        def feature_encoder(input_size, layer_sizes, output_size):
            return torch.nn.Sequential(AlternateFeatureEncoder(input_size, layer_sizes,
                                                               output_size, config.DEVICE)).to(config.DEVICE)

        # 6 is the number of separate dynamic networks
        self.prediction_value_network = feature_encoder(model_config.HIDDEN_STATE_SIZE,
                                                        [model_config.LAYER_HIDDEN_SIZE] *
                                                        model_config.N_HEAD_HIDDEN_LAYERS,
                                                        model_config.ENCODER_NUM_STEPS)
        self.policy_network = feature_encoder(model_config.HIDDEN_STATE_SIZE,
                                              [model_config.LAYER_HIDDEN_SIZE] * model_config.N_HEAD_HIDDEN_LAYERS,
                                              29)

    def forward(self, hidden_state):

        value = self.prediction_value_network(hidden_state)
        policy = self.policy_network(hidden_state)

        return policy, value


class AtariPolicyNetwork(nn.Module):
    def __init__(self, hidden_channels: int, kernel_size: int = 3, device: str = "cpu"):
        super().__init__()

        self.pred_head = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=32,
                kernel_size=kernel_size,
                padding="same",
                device=device,
            ),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=(32 * 6 * 6), out_features=17, device=device),
        )

    def forward(self, x):
        return self.pred_head(x)


class AtariValueNetwork(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        support_size: int,
        kernel_size: int = 3,
        device: str = "cpu",
    ):
        super().__init__()

        self.value_head = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=32,
                kernel_size=kernel_size,
                padding="same",
                device=device,
            ),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(
                in_features=(32 * 6 * 6), out_features=support_size, device=device
            ),
        )

    def forward(self, x):
        return self.value_head(x)

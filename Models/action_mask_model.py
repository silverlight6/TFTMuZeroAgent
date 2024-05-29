import gymnasium as gym
import time

from Models.torch_layers import mlp, Normalize

from ray.rllib.core.models.base import ENCODER_OUT, CRITIC, ACTOR
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import ModelConfigDict
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID


torch, nn = try_import_torch()


"""
Description - Custom model so I can use a nested dict observation space. 
                Split into encoder, policy, vf for PPO purposes.
"""
class TorchActionMaskModel(TorchModelV2, nn.Module):
    """PyTorch version of above ActionMaskingModel."""

    def __init__(self,
                 obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 num_outputs: int,
                 model_config: ModelConfigDict,
                 name: str,
                 **kwargs):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.encoder = TorchActionMaskEncoderModel(obs_space, action_space, num_outputs, model_config,
                                                   "ActionMaskEncoderModel", **kwargs)

        self.pf = TorchActionMaskPolicyModel(obs_space, action_space, num_outputs, model_config,
                                             "ActionMaskPolicyModel", **kwargs)

        self.vf = TorchActionMaskValueModel(obs_space, action_space, num_outputs, model_config,
                                            "ActionMaskVFModel", **kwargs)

        self._features = None
        self.policy_id = DEFAULT_POLICY_ID

    def _forward(self, input_dict, state, seq_lens):
        hidden_state, _ = self.encoder.forward(input_dict, state, seq_lens)
        self._features = {"obs": hidden_state[ENCODER_OUT][CRITIC]}

        # Compute the unmasked logits.
        logits, _ = self.pf.forward({"obs": hidden_state[ENCODER_OUT][ACTOR]}, state, seq_lens)

        return logits, state

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        return self._forward(input_dict, state, seq_lens)

    def value_function(self):
        return self.vf.forward(self._features, [], [])[0]


class TorchActionMaskEncoderModel(TorchModelV2, nn.Module):
    """PyTorch version of above ActionMaskingModel."""

    def __init__(self,
                 obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 num_outputs: int,
                 model_config: ModelConfigDict,
                 name: str,
                 **kwargs):
        TorchModelV2.__init__(self,
                              obs_space,
                              action_space,
                              num_outputs,
                              model_config,
                              name)
        nn.Module.__init__(self)

        hidden_layer_size = model_config["custom_model_config"]["hidden_state_size"]
        num_hidden_layers = model_config["custom_model_config"]["num_hidden_layers"]
        self.device = model_config["device"] if model_config["device"] else 'cuda'

        layer_sizes = [hidden_layer_size // 2] * num_hidden_layers

        output_size = hidden_layer_size

        obs_space_local = obs_space["observations"]["player"]

        self.scalar_encoder = feature_encoder(obs_space_local["scalars"].shape[0],
                                              layer_sizes, output_size, self.device)
        self.shop_encoder = feature_encoder(obs_space_local["shop"].shape[0],
                                            layer_sizes, output_size, self.device)
        self.board_encoder = feature_encoder(obs_space_local["board"].shape[0],
                                             layer_sizes, output_size, self.device)
        self.bench_encoder = feature_encoder(obs_space_local["bench"].shape[0],
                                             layer_sizes, output_size, self.device)
        self.items_encoder = feature_encoder(obs_space_local["items"].shape[0],
                                             layer_sizes, output_size, self.device)
        self.traits_encoder = feature_encoder(obs_space_local["traits"].shape[0],
                                              layer_sizes, output_size, self.device)

        self.other_players_encoder = feature_encoder(obs_space["observations"]["opponents"].shape[0], layer_sizes,
                                                     output_size, self.device)

        self.feature_to_hidden = feature_encoder(hidden_layer_size * 7,
                                                 layer_sizes, output_size, self.device)

        self._features = None

    def _forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"]["observations"]

        scalar = self.scalar_encoder(x["player"]["scalars"])
        shop = self.shop_encoder(x["player"]["shop"])
        board = self.board_encoder(x["player"]["board"])
        bench = self.bench_encoder(x["player"]["bench"])
        items = self.items_encoder(x["player"]["items"])
        traits = self.traits_encoder(x["player"]["traits"])

        other_players = self.other_players_encoder(x["opponents"])

        full_state = torch.cat((scalar, shop, board, bench, items, traits, other_players), -1)

        hidden_state = self.feature_to_hidden(full_state)
        return {ENCODER_OUT: {CRITIC: hidden_state, ACTOR: hidden_state}}

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        return self._forward(input_dict, state, seq_lens), state


class TorchActionMaskValueModel(TorchModelV2, nn.Module):
    """PyTorch version of above ActionMaskingModel."""

    def __init__(self,
                 obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 num_outputs: int,
                 model_config: ModelConfigDict,
                 name: str,
                 **kwargs):
        TorchModelV2.__init__(self,
                              obs_space,
                              action_space,
                              num_outputs,
                              model_config,
                              name)
        nn.Module.__init__(self)

        hidden_layer_size = model_config["custom_model_config"]["hidden_state_size"]
        num_hidden_layers = model_config["custom_model_config"]["num_hidden_layers"]
        self.device = model_config["device"] if model_config["device"] else 'cuda'

        layer_sizes = [hidden_layer_size // 2] * num_hidden_layers
        self.prediction_value_network = feature_encoder(hidden_layer_size, layer_sizes, 1,
                                                        self.device, do_normalize=False)

        self._features = None

    def _forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"]
        return self.prediction_value_network(x).squeeze(1), state

    # The input_dict here is actually just a tensor but typically this calls for a dictionary so I'm calling it a dict
    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        return self._forward(input_dict, state, seq_lens)


class TorchActionMaskPolicyModel(TorchModelV2, nn.Module):
    """PyTorch version of above ActionMaskingModel."""

    def __init__(self,
                 obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 num_outputs: int,
                 model_config: ModelConfigDict,
                 name: str,
                 **kwargs):
        TorchModelV2.__init__(self,
                              obs_space,
                              action_space,
                              num_outputs,
                              model_config,
                              name)
        nn.Module.__init__(self)

        hidden_layer_size = model_config["custom_model_config"]["hidden_state_size"]
        num_hidden_layers = model_config["custom_model_config"]["num_hidden_layers"]
        self.device = model_config["device"] if model_config["device"] else 'cuda'

        layer_sizes = [hidden_layer_size // 2] * num_hidden_layers

        self.policy_network = feature_encoder(hidden_layer_size, layer_sizes,
                                              action_space.shape[0] * action_space[0].n,
                                              self.device, do_normalize=False)

    def _forward(self, input_dict, state, seq_lens):
        # Compute the unmasked logits.
        x = input_dict["obs"]
        logits = self.policy_network(x)

        # logits = torch.cat([mean, self.log_std.unsqueeze(0).repeat([len(mean), 1])], axis=1)
        return logits

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        return self._forward(input_dict, state, seq_lens), state


def feature_encoder(input_size, feature_layer_sizes, feature_output_sizes, device, do_normalize=True):
    return torch.nn.Sequential(
        mlp(input_size, feature_layer_sizes, feature_output_sizes),
        Normalize() if do_normalize else torch.nn.Identity()
    ).to(device)

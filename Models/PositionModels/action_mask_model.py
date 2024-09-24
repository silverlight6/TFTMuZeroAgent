import gymnasium as gym
import numpy as np
import config
import time

from config import ModelConfig
from Models.torch_layers import ResidualBlock, TransformerEncoder, mlp, Normalize, ppo_mlp
from torch.distributions.categorical import Categorical

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

        self.board_encoder = feature_encoder(obs_space_local["board"].shape[0],
                                             layer_sizes, output_size, self.device)
        self.traits_encoder = feature_encoder(obs_space_local["traits"].shape[0],
                                              layer_sizes, output_size, self.device)

        self.feature_to_hidden = feature_encoder(hidden_layer_size * 2,
                                                 layer_sizes, output_size, self.device)

        self._features = None

    def _forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"]["observations"]

        board = self.board_encoder(x["player"]["board"])
        traits = self.traits_encoder(x["player"]["traits"])

        full_state = torch.cat((board, traits), -1)

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


class TorchPositionModel(nn.Module):
    """PyTorch version of above ActionMaskingModel."""

    def __init__(self, model_config: ModelConfig):
        super().__init__()
        nn.Module.__init__(self)

        self.encoder = TorchPositionEncoderModel(model_config)

        self.pf = TorchPositionPolicyModel(model_config)

        self.vf = TorchPositionValueModel(model_config)

        self._features = None
        self.policy_id = DEFAULT_POLICY_ID

    def _forward(self, observation, action=None):
        hidden_state = self.encoder.forward(observation)

        # Compute the unmasked logits.
        action, log_prob, entropy = self.pf.forward({"observations": hidden_state[ENCODER_OUT][ACTOR],
                                                     "action_mask": observation["action_mask"]}, action)
        value = self.vf.forward(hidden_state[ENCODER_OUT][CRITIC])

        return action, log_prob, entropy, value

    def forward(self, input_dict, action=None):
        return self._forward(input_dict, action=None)

    def get_value(self, observation):
        return self.vf.forward(self.encoder.forward(observation)[ENCODER_OUT][CRITIC])


class TorchPositionEncoderModel(nn.Module):
    """PyTorch version of above ActionMaskingModel."""

    def __init__(self, model_config: ModelConfig):
        super().__init__()

        self.champion_embedding = torch.nn.Embedding(221, model_config.CHAMPION_EMBEDDING_DIM // 2)

        self.champion_item_embedding_1 = torch.nn.Embedding(58, model_config.CHAMPION_EMBEDDING_DIM // 8)
        self.champion_item_embedding_2 = torch.nn.Embedding(58, model_config.CHAMPION_EMBEDDING_DIM // 8)
        self.champion_item_embedding_3 = torch.nn.Embedding(58, model_config.CHAMPION_EMBEDDING_DIM // 8)

        self.champion_trait_embedding = torch.nn.Embedding(145, model_config.CHAMPION_EMBEDDING_DIM // 8)

        self.board_encoder = TransformerEncoder(
            model_config.CHAMPION_EMBEDDING_DIM,
            model_config.CHAMPION_EMBEDDING_DIM,
            model_config.N_HEADS,
            model_config.N_LAYERS
        )

        self.other_player_encoder = TransformerEncoder(
            model_config.CHAMPION_EMBEDDING_DIM,
            model_config.CHAMPION_EMBEDDING_DIM,
            model_config.N_HEADS,
            model_config.N_LAYERS
        )

        self.full_encoder = TransformerEncoder(
            model_config.CHAMPION_EMBEDDING_DIM,
            model_config.CHAMPION_EMBEDDING_DIM,
            model_config.N_HEADS,
            model_config.N_LAYERS
        )

        layer_sizes = [model_config.LAYER_HIDDEN_SIZE] * model_config.N_HEAD_HIDDEN_LAYERS
        self.trait_encoder = ppo_feature_encoder(config.TRAIT_INPUT_SIZE, layer_sizes,
                                                 model_config.CHAMPION_EMBEDDING_DIM, config.DEVICE)

        # Combine encoded features from all components
        self.feature_combiner = ppo_feature_encoder(model_config.CHAMPION_EMBEDDING_DIM * 30, layer_sizes,
                                                    model_config.HIDDEN_STATE_SIZE, config.DEVICE)

        self.feature_processor = ppo_feature_encoder(model_config.HIDDEN_STATE_SIZE, layer_sizes,
                                                     model_config.HIDDEN_STATE_SIZE, config.DEVICE)

        # Optional: Add residual connections around encoders and combiner for better gradient flow
        self.board_residual = ResidualBlock(self.board_encoder, 28)
        self.full_residual = ResidualBlock(self.full_encoder, 30)
        self.combiner_residual = ResidualBlock(self.feature_processor, model_config.HIDDEN_STATE_SIZE, local_norm=False)
        self.model_config = model_config

        self.hidden_to_actor = ppo_feature_encoder(model_config.HIDDEN_STATE_SIZE, layer_sizes,
                                                   model_config.HIDDEN_STATE_SIZE, config.DEVICE)

        self.hidden_to_critic = ppo_feature_encoder(model_config.HIDDEN_STATE_SIZE, layer_sizes,
                                                    model_config.HIDDEN_STATE_SIZE, config.DEVICE)

        self._features = None

    def _forward(self, input_dict):
        x = input_dict["observations"]

        champion_emb = self.champion_embedding(x['board'][..., 0].long())
        champion_item_emb_1 = self.champion_item_embedding_1(x['board'][..., 1].long())
        champion_item_emb_2 = self.champion_item_embedding_2(x['board'][..., 2].long())
        champion_item_emb_3 = self.champion_item_embedding_3(x['board'][..., 3].long())
        champion_trait_emb = self.champion_trait_embedding(x['board'][..., 4].long())

        champion_item_emb = torch.cat([champion_item_emb_1, champion_item_emb_2, champion_item_emb_3], dim=-1)
        cie_shape = champion_item_emb.shape
        champion_item_emb = torch.reshape(champion_item_emb, (cie_shape[0], cie_shape[1], cie_shape[2], -1))

        board_emb = torch.cat([champion_emb, champion_item_emb, champion_trait_emb], dim=-1)

        board_residual_input = torch.reshape(board_emb[:, 0], [cie_shape[0], cie_shape[2],
                                                               self.model_config.CHAMPION_EMBEDDING_DIM])

        trait_enc = self.trait_encoder(x['traits'])

        board_enc = self.board_residual(board_residual_input)

        other_player_enc = torch.cat([torch.reshape(board_emb[:, 1:],
                                                    (cie_shape[0], -1, self.model_config.CHAMPION_EMBEDDING_DIM)),
                                      trait_enc[:, 1:]], dim=1)

        other_player_enc = self.other_player_encoder(other_player_enc)
        other_player_enc = torch.sum(other_player_enc, dim=1)
        other_player_enc = other_player_enc[:, None, :]
        player_trait_enc = trait_enc[:, 0]
        player_trait_enc = player_trait_enc[:, None, :]

        cat_encode = torch.cat([board_enc, other_player_enc, player_trait_enc], dim=1)

        full_enc = self.full_encoder(cat_encode)
        full_enc = torch.reshape(full_enc, [cie_shape[0], -1])

        # Pass through final combiner network
        hidden_state = self.feature_combiner(full_enc)
        hidden_state = self.combiner_residual(hidden_state)
        hidden_critic = self.hidden_to_critic(hidden_state)
        hidden_actor = self.hidden_to_actor(hidden_state)
        return {ENCODER_OUT: {CRITIC: hidden_critic, ACTOR: hidden_actor}}

    def forward(self, input_dict):
        return self._forward(input_dict)


class TorchPositionValueModel(nn.Module):
    """PyTorch version of above ActionMaskingModel."""

    def __init__(self, model_config: ModelConfig):
        super().__init__()

        hidden_layer_size = model_config.HIDDEN_STATE_SIZE
        layer_sizes = [model_config.LAYER_HIDDEN_SIZE // 2] * model_config.N_HEAD_HIDDEN_LAYERS
        self.device = config.DEVICE

        self.prediction_value_network = ppo_feature_encoder(hidden_layer_size, layer_sizes, 1,
                                                            self.device, do_normalize=False)

        self._features = None

    def _forward(self, hidden_state):
        return self.prediction_value_network(hidden_state).squeeze(1)

    # The input_dict here is actually just a tensor but typically this calls for a dictionary so I'm calling it a dict
    def forward(self, hidden_state):
        return self._forward(hidden_state)


class TorchPositionPolicyModel(nn.Module):
    """PyTorch version of above ActionMaskingModel."""

    def __init__(self, model_config: ModelConfig):
        super().__init__()
        super().__init__()

        hidden_layer_size = model_config.HIDDEN_STATE_SIZE
        layer_sizes = [model_config.LAYER_HIDDEN_SIZE // 2] * model_config.N_HEAD_HIDDEN_LAYERS
        self.device = config.DEVICE

        self.policy_network = ppo_feature_encoder(hidden_layer_size, layer_sizes, 29 * 12,
                                                  self.device, do_normalize=False)

    def _forward(self, input_dict, action=None):
        # Compute the unmasked logits.
        x = input_dict["observations"]
        logits = self.policy_network(x)
        action_mask = input_dict["action_mask"]
        action_mask_shape = action_mask.shape
        action_mask = torch.reshape(action_mask, (action_mask_shape[0], action_mask_shape[1] * action_mask_shape[2]))

        # Mask
        inf_mask = torch.clamp(torch.log(action_mask), min=-3.4e38)
        masked_logits = logits + inf_mask
        masked_logits = torch.reshape(masked_logits, (action_mask_shape[0], action_mask_shape[1], action_mask_shape[2]))

        probs = Categorical(logits=masked_logits)
        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy()

    def forward(self, input_dict, action):
        return self._forward(input_dict, action)

def ppo_feature_encoder(input_size, feature_layer_sizes, feature_output_sizes, device, do_normalize=True):
    return torch.nn.Sequential(
        ppo_mlp(input_size, feature_layer_sizes, feature_output_sizes),
        Normalize() if do_normalize else torch.nn.Identity()
    ).to(device)

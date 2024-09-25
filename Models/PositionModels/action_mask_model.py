import numpy as np
import config
import time
import torch

from config import ModelConfig
from Models.torch_layers import ResidualBlock, TransformerEncoder, mlp, Normalize, ppo_mlp
from torch.distributions.categorical import Categorical


"""
Description - Custom model so I can use a nested dict observation space. 
                Split into encoder, policy, vf for PPO purposes.
"""
class TorchPositionModel(torch.nn.Module):
    """PyTorch version of above ActionMaskingModel."""

    def __init__(self, model_config: ModelConfig):
        super().__init__()
        torch.nn.Module.__init__(self)

        self.encoder = TorchPositionEncoderModel(model_config)

        self.pf = TorchPositionPolicyModel(model_config)

        self.vf = TorchPositionValueModel(model_config)

        self._features = None
        self.policy_id = "policy"

    def _forward(self, observation, action=None):
        hidden_critic, hidden_actor = self.encoder.forward(observation)

        # Compute the unmasked logits.
        action, log_prob, entropy = self.pf.forward({"observations": hidden_actor,
                                                     "action_mask": observation["action_mask"]}, action)
        value = self.vf.forward(hidden_critic)

        return action, log_prob, entropy, value

    def forward(self, input_dict, action=None):
        return self._forward(input_dict, action=None)

    def get_value(self, observation):
        return self.vf.forward(self.encoder.forward(observation)["Encoder_Out"]["Critic"])


class TorchPositionEncoderModel(torch.nn.Module):
    """PyTorch version of above ActionMaskingModel."""

    def __init__(self, model_config: ModelConfig):
        super().__init__()

        self.champion_embedding = torch.nn.Embedding(221, model_config.CHAMPION_EMBEDDING_DIM // 2)

        self.champion_item_embedding_1 = torch.nn.Embedding(58, model_config.CHAMPION_EMBEDDING_DIM // 8)
        self.champion_item_embedding_2 = torch.nn.Embedding(58, model_config.CHAMPION_EMBEDDING_DIM // 8)
        self.champion_item_embedding_3 = torch.nn.Embedding(58, model_config.CHAMPION_EMBEDDING_DIM // 8)

        self.champion_trait_embedding = torch.nn.Embedding(145, model_config.CHAMPION_EMBEDDING_DIM // 8)

        self.full_encoder = TransformerEncoder(
            model_config.CHAMPION_EMBEDDING_DIM,
            model_config.CHAMPION_EMBEDDING_DIM,
            model_config.N_HEADS,
            model_config.N_LAYERS
        )

        layer_sizes = [model_config.LAYER_HIDDEN_SIZE] * model_config.N_HEAD_HIDDEN_LAYERS

        self.feature_processor = ppo_feature_encoder(model_config.CHAMPION_EMBEDDING_DIM, layer_sizes,
                                                     model_config.HIDDEN_STATE_SIZE, config.DEVICE)

        # Optional: Add residual connections around encoders and combiner for better gradient flow
        self.model_config = model_config

        self.hidden_to_actor = ppo_feature_encoder(model_config.HIDDEN_STATE_SIZE, layer_sizes,
                                                   model_config.HIDDEN_STATE_SIZE, config.DEVICE)

        self.hidden_to_critic = ppo_feature_encoder(model_config.HIDDEN_STATE_SIZE, layer_sizes,
                                                    model_config.HIDDEN_STATE_SIZE, config.DEVICE)

        self.positional_embedding = torch.nn.Embedding(224, model_config.CHAMPION_EMBEDDING_DIM)

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, model_config.CHAMPION_EMBEDDING_DIM))

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

        full_embeddings = torch.cat([champion_emb, champion_item_emb, champion_trait_emb], dim=-1)
        full_embeddings = torch.reshape(full_embeddings, (cie_shape[0], cie_shape[1] * cie_shape[2], -1))

        seq_length = 224  # Number of tokens in the sequence (224 in your case)

        # Create a position index [0, 1, 2, ..., 223] repeated for each sample in the batch
        positions = torch.arange(0, seq_length).unsqueeze(0).expand(cie_shape[0], seq_length).to(config.DEVICE)

        # Get the positional encodings and add them to the full embeddings
        positional_enc = self.positional_embedding(positions)
        # print(f"positional_enc {positional_enc}")
        full_embeddings = full_embeddings + positional_enc

        # Expand the classification token to match the batch size
        cls_tokens = self.cls_token.expand(cie_shape[0], -1, -1)

        # Concatenate the cls token to the full_embeddings
        full_embeddings = torch.cat([cls_tokens, full_embeddings], dim=1)
        # print(f"hidden_state full_embeddings + position {full_embeddings}")

        # Note to future self, if I want to separate current board for some processing. Do it here but use two
        # Position encodings.

        full_enc = self.full_encoder(full_embeddings)

        cls_hidden_state = full_enc[:, 0, :]
        # print(f"cls_hidden_state {cls_hidden_state}")

        # Pass through final combiner network
        hidden_state = self.feature_processor(cls_hidden_state)
        # print(f"hidden_state {hidden_state}")
        hidden_critic = self.hidden_to_critic(hidden_state)
        hidden_actor = self.hidden_to_actor(hidden_state)

        return hidden_critic, hidden_actor

    def forward(self, input_dict):
        return self._forward(input_dict)


class TorchPositionValueModel(torch.nn.Module):
    """PyTorch version of above ActionMaskingModel."""

    def __init__(self, model_config: ModelConfig):
        super().__init__()

        hidden_layer_size = model_config.HIDDEN_STATE_SIZE
        layer_sizes = [model_config.LAYER_HIDDEN_SIZE // 2] * model_config.N_HEAD_HIDDEN_LAYERS
        self.device = config.DEVICE

        self.prediction_value_network = ppo_feature_encoder(hidden_layer_size, layer_sizes, 1,
                                                            self.device)

        self._features = None

    def _forward(self, hidden_state):
        # print(f"hidden_state critic {hidden_state}")
        pred_output = self.prediction_value_network(hidden_state).squeeze(1)
        # print(f"initial_value {pred_output}")
        return pred_output

    # The input_dict here is actually just a tensor but typically this calls for a dictionary so I'm calling it a dict
    def forward(self, hidden_state):
        return self._forward(hidden_state)


class TorchPositionPolicyModel(torch.nn.Module):
    """PyTorch version of above ActionMaskingModel."""

    def __init__(self, model_config: ModelConfig):
        super().__init__()
        super().__init__()

        hidden_layer_size = model_config.HIDDEN_STATE_SIZE
        layer_sizes = [model_config.LAYER_HIDDEN_SIZE // 2] * model_config.N_HEAD_HIDDEN_LAYERS
        self.device = config.DEVICE

        self.policy_network = ppo_feature_encoder(hidden_layer_size, layer_sizes, 29 * 12,
                                                  self.device)

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


class ppo_feature_encoder(torch.nn.Module):
    def __init__(self, input_size, layer_sizes, output_size, device, dropout_rate=0.1, use_layer_norm=True):
        super(ppo_feature_encoder, self).__init__()

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

        self.network = torch.nn.Sequential(*layers)
        self.device = device

    def forward(self, x):
        x = x.to(self.device)  # Ensure the input is on the correct device
        return self.network(x)

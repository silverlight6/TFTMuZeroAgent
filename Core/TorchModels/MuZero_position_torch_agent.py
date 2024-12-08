import time
import torch
import Models.MCTS_Util as utils
import config
from Models.abstract_model import AbstractNetwork
from Models.torch_layers import Normalize, MemoryLayer, AlternateFeatureEncoder, TransformerEncoder

from config import ModelConfig

class MuZero_Position_Network(AbstractNetwork):
    def __init__(self, model_config):
        super().__init__()
        self.full_support_size = model_config.ENCODER_NUM_STEPS

        self.representation_network = RepPositionEmbeddingNetwork(model_config)

        self.dynamics_network = PositionDynNetwork(input_size=model_config.HIDDEN_STATE_SIZE,
                                                   num_layers=model_config.NUM_RNN_CELLS,
                                                   hidden_size=model_config.LSTM_SIZE, model_config=model_config)

        self.prediction_network = PredPositionNetwork(model_config)

        map_min = torch.tensor(-300., dtype=torch.float32)
        map_max = torch.tensor(300., dtype=torch.float32)
        self.value_encoder = utils.ValueEncoder(
            *tuple(map(utils.inverse_contractive_mapping, (map_min, map_max))), 0)
        self.reward_encoder = utils.ValueEncoder(
            *tuple(map(utils.inverse_contractive_mapping, (map_min, map_max))), 0)

        self.model_config = model_config

    def prediction(self, encoded_state):
        policy_logits, value = self.prediction_network(encoded_state)
        return policy_logits, value

    def representation(self, observation):
        observation = {label: torch.from_numpy(value).to(config.DEVICE) for label, value in observation.items()}
        return self.representation_network(observation)

    def dynamics(self, x, action):
        return self.dynamics_network(x, action)

    def initial_inference(self, observation, training=False):
        hidden_state = self.representation(observation)
        policy_logits, value_logits = self.prediction(hidden_state)

        value = self.value_encoder.decode_softmax(value_logits)

        outputs = {
            "value": value,
            "value_logits": value_logits,
            "policy_logits": policy_logits,
            "hidden_state": hidden_state,
        }
        return outputs

    @staticmethod
    def rnn_to_flat(state):
        """Maps LSTM state to flat token."""
        states = []
        for cell_state in state:
            states.extend(cell_state)
        return torch.cat(states, dim=-1)

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

    def recurrent_inference(self, hidden_state, action):
        next_hidden_state = self.dynamics(hidden_state, action)

        policy_logits, value_logits = self.prediction(next_hidden_state)
        value = self.value_encoder.decode_softmax(value_logits)
        outputs = {
            "value": value,
            "value_logits": value_logits,
            "policy_logits": policy_logits,
            "hidden_state": next_hidden_state
        }
        return outputs


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

class RepPositionEmbeddingNetwork(torch.nn.Module):
    def __init__(self, model_config: ModelConfig):
        super().__init__()

        # How many layers to use in each mlp processing unit
        layer_sizes = [model_config.LAYER_HIDDEN_SIZE] * model_config.N_HEAD_HIDDEN_LAYERS

        # Embeddings for the unit are separate from item and trait. These double up for champion bench
        self.champion_embedding = torch.nn.Embedding(221, model_config.CHAMPION_EMBEDDING_DIM // 2).to(config.DEVICE)
        self.champion_item_embedding_1 = \
            torch.nn.Embedding(58, model_config.CHAMPION_EMBEDDING_DIM // 8).to(config.DEVICE)
        self.champion_item_embedding_2 = \
            torch.nn.Embedding(58, model_config.CHAMPION_EMBEDDING_DIM // 8).to(config.DEVICE)
        self.champion_item_embedding_3 = \
            torch.nn.Embedding(58, model_config.CHAMPION_EMBEDDING_DIM // 8).to(config.DEVICE)
        self.champion_trait_embedding = \
            torch.nn.Embedding(145, model_config.CHAMPION_EMBEDDING_DIM // 8).to(config.DEVICE)

        self.trait_encoder = AlternateFeatureEncoder(config.TRAIT_INPUT_SIZE, layer_sizes,
                                                     model_config.CHAMPION_EMBEDDING_DIM, config.DEVICE)

        self.round_encoder = AlternateFeatureEncoder(12, layer_sizes, model_config.CHAMPION_EMBEDDING_DIM,
                                                     config.DEVICE)

        # Main processing unit
        self.full_encoder = TransformerEncoder(
            model_config.CHAMPION_EMBEDDING_DIM,
            model_config.CHAMPION_EMBEDDING_DIM,
            model_config.N_HEADS,
            model_config.N_LAYERS
        )

        # Turn the processed representation into a hidden_state_size for the LSTM
        self.feature_processor = AlternateFeatureEncoder(model_config.CHAMPION_EMBEDDING_DIM * 4, layer_sizes,
                                                         model_config.HIDDEN_STATE_SIZE, config.DEVICE)

        # Optional: Add residual connections around encoders and combiner for better gradient flow
        self.model_config = model_config

        # Using 4 tokens instead of 1 so I can size down for the hidden state instead of size up
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 4, model_config.CHAMPION_EMBEDDING_DIM))

        self._features = None
        self.champion_embedding_dim = model_config.CHAMPION_EMBEDDING_DIM

        # Learned position embeddings instead of strait sinusoidal because tokens from different parts of the
        # observation are next to each other. It doesn't make sense to say the bench is close to the shop
        # Positional Embedding: Maximum length based on concatenated sequence length
        self.pos_embedding = torch.nn.Embedding(512, model_config.CHAMPION_EMBEDDING_DIM).to(config.DEVICE)

    def _forward(self, x):
        batch_size = x['board'].shape[0]
        champion_emb = self.champion_embedding(x['board'][..., 0].long())
        champion_item_emb_1 = self.champion_item_embedding_1(x['board'][..., 1].long())
        champion_item_emb_2 = self.champion_item_embedding_2(x['board'][..., 2].long())
        champion_item_emb_3 = self.champion_item_embedding_3(x['board'][..., 3].long())
        champion_trait_emb = self.champion_trait_embedding(x['board'][..., 4].long())

        champion_item_emb = torch.cat([champion_item_emb_1, champion_item_emb_2, champion_item_emb_3], dim=-1)
        cie_shape = champion_item_emb.shape
        champion_item_emb = torch.reshape(champion_item_emb, (batch_size, cie_shape[1], cie_shape[2], -1))
        champion_embeddings = torch.cat([champion_emb, champion_item_emb, champion_trait_emb], dim=-1)
        champion_embeddings = torch.reshape(champion_embeddings, (batch_size, -1, self.champion_embedding_dim))

        trait_encoding = self.trait_encoder(x['traits'])

        round_encoding = self.round_encoder(x['action_count'])

        full_embeddings = torch.cat([champion_embeddings, trait_encoding, round_encoding], dim=1)

        # Add positional embeddings to full_embeddings
        position_ids = torch.arange(full_embeddings.shape[1], dtype=torch.long, device=config.DEVICE).unsqueeze(0)
        pos_embeddings = self.pos_embedding(position_ids)
        full_embeddings = full_embeddings + pos_embeddings

        # Expand the classification token to match the batch size
        cls_tokens = self.cls_token.expand(cie_shape[0], -1, -1).to(config.DEVICE)

        # Concatenate the cls token to the full_embeddings
        full_embeddings = torch.cat([cls_tokens, full_embeddings], dim=1)

        # Note to future self, if I want to separate current board for some processing. Do it here but use two
        # Position encodings.
        full_enc = self.full_encoder(full_embeddings)

        cls_hidden_state = full_enc[:, 0:4, :]
        cls_hidden_state = torch.reshape(cls_hidden_state, (batch_size, -1))
        # print(f"cls_hidden_state {cls_hidden_state}")

        # Pass through final combiner network
        hidden_state = self.feature_processor(cls_hidden_state)

        return hidden_state

    def forward(self, x):
        return self._forward(x)

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

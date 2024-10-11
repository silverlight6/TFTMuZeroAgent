import time
import torch
import Models.MCTS_Util as utils
import config
from Models.abstract_model import AbstractNetwork
from Models.torch_layers import mlp, MultiMlp, Normalize, MemoryLayer, AlternateFeatureEncoder, TransformerEncoder
from config import ModelConfig


class MuZeroNetwork(AbstractNetwork):
    def __init__(self, model_config):
        super().__init__()
        self.full_support_size = model_config.ENCODER_NUM_STEPS

        self.representation_network = RepEmbeddingNetwork(model_config)

        self.dynamics_network = DynNetwork(input_size=model_config.HIDDEN_STATE_SIZE,
                                           num_layers=model_config.NUM_RNN_CELLS,
                                           hidden_size=model_config.LSTM_SIZE, model_config=model_config)

        self.prediction_network = PredNetwork(model_config)

        map_min = torch.tensor(-300., dtype=torch.float32)
        map_max = torch.tensor(300., dtype=torch.float32)
        self.value_encoder = utils.ValueEncoder(
            *tuple(map(utils.inverse_contractive_mapping, (map_min, map_max))), 0)
        self.reward_encoder = utils.ValueEncoder(
            *tuple(map(utils.inverse_contractive_mapping, (map_min, map_max))), 0)

        self.model_config = model_config

    def prediction(self, encoded_state, training=False):
        if training:
            policy_logits, value, comp, final_comp, champ = self.prediction_network(encoded_state, training)
            return policy_logits, value, comp, final_comp, champ
        else:
            policy_logits, value = self.prediction_network(encoded_state, training)
            return policy_logits, value

    def representation(self, observation):
        observation = {label: torch.from_numpy(value).to(config.DEVICE) for label, value in observation.items()}
        return self.representation_network(observation)

    def dynamics(self, x, action):
        return self.dynamics_network(x, action)

    def initial_inference(self, observation, training=False):
        hidden_state = self.representation(observation)
        if training:
            policy_logits, value_logits, comp, final_comp, champ = self.prediction(hidden_state, training)

            value = self.value_encoder.decode_softmax(value_logits)

            reward = torch.zeros(observation["shop"].shape[0])
            reward_logits = self.reward_encoder.encode(reward)

            outputs = {
                "value": value,
                "value_logits": value_logits,
                "reward": reward,
                "reward_logits": reward_logits,
                "policy_logits": policy_logits,
                "hidden_state": hidden_state,
                "comp": comp,
                "final_comp": final_comp,
                "champ": champ
            }

        else:
            policy_logits, value_logits = self.prediction(hidden_state)

            value = self.value_encoder.decode_softmax(value_logits)

            reward = torch.zeros(observation["shop"].shape[0])
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

    def recurrent_inference(self, hidden_state, action, training=False):
        next_hidden_state, reward_logits = self.dynamics(hidden_state, action)
        if training:
            policy_logits, value_logits, comp, final_comp, champ = self.prediction(next_hidden_state, training)
            value = self.value_encoder.decode_softmax(value_logits)
            reward = self.reward_encoder.decode_softmax(reward_logits)
            outputs = {
                "value": value,
                "value_logits": value_logits,
                "reward": reward,
                "reward_logits": reward_logits,
                "policy_logits": policy_logits,
                "hidden_state": next_hidden_state,
                "comp": comp,
                "final_comp": final_comp,
                "champ": champ
            }
        else:
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
                                              config.POLICY_HEAD_SIZE)

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


# Not currently using this model but leaving in case I change my mind later. Uses ObservationVector input
class RepNetwork(torch.nn.Module):
    def __init__(self, layer_sizes, output_size, model_config) -> torch.nn.Module:
        super().__init__()

        def feature_encoder(input_size, feature_layer_sizes, feature_output_sizes):
            return torch.nn.Sequential(
                AlternateFeatureEncoder(input_size, feature_layer_sizes, feature_output_sizes, config.DEVICE),
                Normalize()
            ).to(config.DEVICE)

        self.scalar_encoder = feature_encoder(config.SCALAR_INPUT_SIZE, layer_sizes, output_size)
        self.shop_encoder = feature_encoder(config.SHOP_INPUT_SIZE, layer_sizes, output_size)
        self.board_encoder = feature_encoder(config.BOARD_INPUT_SIZE, layer_sizes, output_size)
        self.bench_encoder = feature_encoder(config.BENCH_INPUT_SIZE, layer_sizes, output_size)
        self.items_encoder = feature_encoder(config.ITEMS_INPUT_SIZE, layer_sizes, output_size)
        self.traits_encoder = feature_encoder(config.TRAIT_INPUT_SIZE, layer_sizes, output_size)
        self.other_players_encoder = feature_encoder(config.OTHER_PLAYER_INPUT_SIZE, layer_sizes, output_size)

        self.feature_to_hidden = feature_encoder((model_config.HIDDEN_STATE_SIZE // 2) * 7,
                                                 [model_config.LAYER_HIDDEN_SIZE] *
                                                 model_config.N_HEAD_HIDDEN_LAYERS,
                                                 model_config.HIDDEN_STATE_SIZE)

    def forward(self, x):
        scalar = self.scalar_encoder(x["scalars"])
        shop = self.shop_encoder(x["shop"])
        board = self.board_encoder(x["board"])
        bench = self.bench_encoder(x["bench"])
        items = self.items_encoder(x["items"])
        traits = self.traits_encoder(x["traits"])
        other_players = self.other_players_encoder(x["other_players"])

        full_state = torch.cat((scalar, shop, board, bench, items, traits, other_players), -1)

        hidden_state = self.feature_to_hidden(full_state)

        return hidden_state

class RepEmbeddingNetwork(torch.nn.Module):
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

        # Technically have the other players item bench as well as their champion bench, not including for space reasons
        self.item_bench_embeddings = torch.nn.Embedding(58, model_config.CHAMPION_EMBEDDING_DIM // 2).to(config.DEVICE)

        self.shop_champion_embedding = \
            torch.nn.Embedding(221, model_config.CHAMPION_EMBEDDING_DIM * 3 // 4).to(config.DEVICE)

        # Technically have the shop_item availability but shops never have items so I am not including it.
        self.shop_trait_embedding = torch.nn.Embedding(145, model_config.CHAMPION_EMBEDDING_DIM // 4).to(config.DEVICE)

        self.scalar_encoder = AlternateFeatureEncoder(config.SCALAR_INPUT_SIZE, layer_sizes,
                                                      model_config.CHAMPION_EMBEDDING_DIM, config.DEVICE)

        self.gold_embedding = torch.nn.Embedding(61, model_config.CHAMPION_EMBEDDING_DIM).to(config.DEVICE)
        self.health_embedding = torch.nn.Embedding(101, model_config.CHAMPION_EMBEDDING_DIM).to(config.DEVICE)
        self.exp_embedding = torch.nn.Embedding(101, model_config.CHAMPION_EMBEDDING_DIM).to(config.DEVICE)
        self.game_round_embedding = torch.nn.Embedding(40, model_config.CHAMPION_EMBEDDING_DIM).to(config.DEVICE)
        self.oppo_options_embedding = torch.nn.Embedding(128, model_config.CHAMPION_EMBEDDING_DIM).to(config.DEVICE)
        self.level_embedding = torch.nn.Embedding(10, model_config.CHAMPION_EMBEDDING_DIM).to(config.DEVICE)

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

        item_bench_embeddings = torch.swapaxes(
            torch.stack([self.item_bench_embeddings(x['items'][..., i].long()) for i in range(10)]), 0, 1)
        item_bench_embeddings = torch.reshape(item_bench_embeddings, (batch_size, -1, self.champion_embedding_dim))

        bench_emb = self.champion_embedding(x['bench'][..., 0].long())
        bench_item_emb_1 = self.champion_item_embedding_1(x['bench'][..., 1].long())
        bench_item_emb_2 = self.champion_item_embedding_2(x['bench'][..., 2].long())
        bench_item_emb_3 = self.champion_item_embedding_3(x['bench'][..., 3].long())
        bench_trait_emb = self.champion_trait_embedding(x['bench'][..., 4].long())

        bench_item_emb = torch.cat([bench_item_emb_1, bench_item_emb_2, bench_item_emb_3], dim=-1)
        bi_shape = bench_item_emb.shape
        bench_item_emb = torch.reshape(bench_item_emb, (batch_size, bi_shape[1], bi_shape[2], -1))
        bench_embeddings = torch.cat([bench_emb, bench_item_emb, bench_trait_emb], dim=-1)
        bench_embeddings = torch.reshape(bench_embeddings, (batch_size, -1, self.champion_embedding_dim))

        shop_champion_emb = self.shop_champion_embedding(x['shop'][..., 0].long())
        shop_champion_trait_emb = self.shop_trait_embedding(x['shop'][..., 4].long())
        shop_embeddings = torch.cat([shop_champion_emb, shop_champion_trait_emb], dim=-1)
        shop_embeddings = torch.reshape(shop_embeddings, (batch_size, -1, self.champion_embedding_dim))
        scalar_encoding = self.scalar_encoder(x['scalars'])

        gold_embedding = self.gold_embedding(x['emb_scalars'][..., 0].long())
        health_embedding = self.health_embedding(x['emb_scalars'][..., 1].long())
        exp_embedding = self.exp_embedding(x['emb_scalars'][..., 2].long())
        game_round_embedding = self.game_round_embedding(x['emb_scalars'][..., 3].long())
        opponent_options_embedding = self.oppo_options_embedding(x['emb_scalars'][..., 4].long())
        level_embedding = self.level_embedding(x['emb_scalars'][..., 5].long())
        scalar_embeddings = torch.cat([gold_embedding, health_embedding, exp_embedding, game_round_embedding,
                                       opponent_options_embedding, level_embedding], dim=-1)
        scalar_embeddings = torch.reshape(scalar_embeddings, (batch_size, -1, self.champion_embedding_dim))

        full_embeddings = torch.cat([champion_embeddings, trait_encoding, item_bench_embeddings, bench_embeddings,
                                     shop_embeddings, scalar_encoding, scalar_embeddings], dim=1)

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

class DynNetwork(torch.nn.Module):
    def __init__(self, input_size, num_layers, hidden_size, model_config) -> torch.nn.Module:
        super().__init__()

        def memory():
            return torch.nn.Sequential(
                MemoryLayer(input_size, num_layers, hidden_size, model_config),
                Normalize()
            ).to(config.DEVICE)

        self.action_encodings = AlternateFeatureEncoder(config.ACTION_CONCAT_SIZE, [
            model_config.LAYER_HIDDEN_SIZE] * 0, model_config.HIDDEN_STATE_SIZE, config.DEVICE)

        self.dynamics_memory = memory()

        self.dynamics_reward_network = AlternateFeatureEncoder(input_size, [model_config.LAYER_HIDDEN_SIZE] * 1,
                                                               model_config.ENCODER_NUM_STEPS, config.DEVICE)
        self.model_config = model_config

    def forward(self, hidden_state, action):
        action = torch.from_numpy(action).to(config.DEVICE).to(torch.int64)
        one_hot_action = torch.nn.functional.one_hot(action[:, 0], config.ACTION_DIM[0])
        one_hot_target_a = torch.nn.functional.one_hot(action[:, 1], config.ACTION_DIM[1])
        one_hot_target_b = torch.nn.functional.one_hot(action[:, 2], config.ACTION_DIM[1])

        action_one_hot = torch.cat([one_hot_action, one_hot_target_a, one_hot_target_b], dim=-1).float()

        action_encoding = self.action_encodings(action_one_hot)

        inputs = action_encoding
        inputs = inputs[:, None, :]

        new_hidden_state = self.dynamics_memory((inputs, hidden_state))

        reward = self.dynamics_reward_network(new_hidden_state)

        return new_hidden_state, reward

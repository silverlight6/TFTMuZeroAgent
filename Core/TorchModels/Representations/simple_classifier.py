import config
import torch
import torch.nn as nn
from config import ModelConfig
from Core.TorchComponents.torch_layers import AlternateFeatureEncoder, TransformerEncoder

total_output_positions = sum(config.TEAM_TIERS_VECTOR)

class MultiTaskClassifier(nn.Module):
    def __init__(self, input_size, team_tiers_vector):
        super(MultiTaskClassifier, self).__init__()
        self.input_size = input_size
        self.team_tiers_vector = team_tiers_vector
        self.backbone = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.outputs = nn.ModuleList([
            nn.Linear(64, tier_length) for tier_length in team_tiers_vector
        ])

    def forward(self, x):
        x = self.backbone(x)
        output = [head(x) for head in self.outputs]
        return output  # List of logits for each trait

class SimpleClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x  # Raw logits

class TFTNetworkFiveClass(torch.nn.Module):
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
        self.shop_trait_embedding = torch.nn.Embedding(145, model_config.CHAMPION_EMBEDDING_DIM // 4)

        self.scalar_encoder = AlternateFeatureEncoder(config.SCALAR_INPUT_SIZE, layer_sizes,
                                                      model_config.CHAMPION_EMBEDDING_DIM, config.DEVICE)

        self.gold_embedding = torch.nn.Embedding(61, model_config.CHAMPION_EMBEDDING_DIM)
        self.health_embedding = torch.nn.Embedding(101, model_config.CHAMPION_EMBEDDING_DIM)
        self.exp_embedding = torch.nn.Embedding(101, model_config.CHAMPION_EMBEDDING_DIM)
        self.game_round_embedding = torch.nn.Embedding(40, model_config.CHAMPION_EMBEDDING_DIM)
        self.oppo_options_embedding = torch.nn.Embedding(128, model_config.CHAMPION_EMBEDDING_DIM)
        self.level_embedding = torch.nn.Embedding(10, model_config.CHAMPION_EMBEDDING_DIM)

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

        self.trait_outputs = nn.ModuleList([
            nn.Linear(model_config.HIDDEN_STATE_SIZE, tier_length) for tier_length in config.TEAM_TIERS_VECTOR
        ])

        self.champ_output = nn.ModuleList([
            nn.Linear(model_config.HIDDEN_STATE_SIZE, tier_length) for tier_length in config.CHAMPION_LIST_DIM
        ])

        self.shop_output = nn.ModuleList([
            nn.Linear(model_config.HIDDEN_STATE_SIZE, tier_length) for tier_length in [63, 63, 63, 63, 63]
        ])

        self.item_output = nn.ModuleList([
            nn.Linear(model_config.HIDDEN_STATE_SIZE, tier_length) for tier_length in [60 for _ in range(10)]
        ])

        self.scalar_output = nn.ModuleList([
            nn.Linear(model_config.HIDDEN_STATE_SIZE, tier_length) for tier_length in [100 for _ in range(3)]
        ])

        self.to(config.DEVICE)

    def _forward(self, traits, board, items, bench, shop, scalars, emb_scalars):
        # print(f"traits shape {traits.shape}")
        batch_size = traits.shape[0]
        champion_emb = self.champion_embedding(board[..., 0].long())
        champion_item_emb_1 = self.champion_item_embedding_1(board[..., 1].long())
        champion_item_emb_2 = self.champion_item_embedding_2(board[..., 2].long())
        champion_item_emb_3 = self.champion_item_embedding_3(board[..., 3].long())
        champion_trait_emb = self.champion_trait_embedding(board[..., 4].long())

        champion_item_emb = torch.cat([champion_item_emb_1, champion_item_emb_2, champion_item_emb_3], dim=-1)
        cie_shape = champion_item_emb.shape
        champion_item_emb = torch.reshape(champion_item_emb, (batch_size, cie_shape[1], cie_shape[2], -1))

        champion_embeddings = torch.cat([champion_emb, champion_item_emb, champion_trait_emb], dim=-1)
        champion_embeddings = torch.reshape(champion_embeddings, (batch_size, -1, self.champion_embedding_dim))

        trait_encoding = self.trait_encoder(traits)

        item_bench_embeddings = torch.swapaxes(
            torch.stack([self.item_bench_embeddings(items[..., i].long()) for i in range(10)]), 0, 1)
        item_bench_embeddings = torch.reshape(item_bench_embeddings, (batch_size, -1, self.champion_embedding_dim))

        bench_emb = self.champion_embedding(bench[..., 0].long())
        bench_item_emb_1 = self.champion_item_embedding_1(bench[..., 1].long())
        bench_item_emb_2 = self.champion_item_embedding_2(bench[..., 2].long())
        bench_item_emb_3 = self.champion_item_embedding_3(bench[..., 3].long())
        bench_trait_emb = self.champion_trait_embedding(bench[..., 4].long())

        bench_item_emb = torch.cat([bench_item_emb_1, bench_item_emb_2, bench_item_emb_3], dim=-1)
        bi_shape = bench_item_emb.shape
        bench_item_emb = torch.reshape(bench_item_emb, (batch_size, bi_shape[1], bi_shape[2], -1))
        bench_embeddings = torch.cat([bench_emb, bench_item_emb, bench_trait_emb], dim=-1)
        bench_embeddings = torch.reshape(bench_embeddings, (batch_size, -1, self.champion_embedding_dim))

        shop_champion_emb = self.shop_champion_embedding(shop[..., 0].long())
        shop_champion_trait_emb = self.shop_trait_embedding(shop[..., 4].long())
        shop_embeddings = torch.cat([shop_champion_emb, shop_champion_trait_emb], dim=-1)
        shop_embeddings = torch.reshape(shop_embeddings, (batch_size, -1, self.champion_embedding_dim))

        scalar_encoding = self.scalar_encoder(scalars)

        gold_embedding = self.gold_embedding(emb_scalars[..., 0].long())
        health_embedding = self.health_embedding(emb_scalars[..., 1].long())
        exp_embedding = self.exp_embedding(emb_scalars[..., 2].long())
        game_round_embedding = self.game_round_embedding(emb_scalars[..., 3].long())
        opponent_options_embedding = self.oppo_options_embedding(emb_scalars[..., 4].long())
        level_embedding = self.level_embedding(emb_scalars[..., 5].long())
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
        cls_tokens = self.cls_token.expand(batch_size, -1, -1).to(config.DEVICE)

        # Concatenate the cls token to the full_embeddings
        full_embeddings = torch.cat([cls_tokens, full_embeddings], dim=1)

        # Note to future self, if I want to separate current board for some processing. Do it here but use two
        # Position encodings.
        full_enc = self.full_encoder(full_embeddings)

        cls_hidden_state = full_enc[:, 0:4, :]
        cls_hidden_state = torch.reshape(cls_hidden_state, (batch_size, -1))

        # Pass through final combiner network
        hidden_state = self.feature_processor(cls_hidden_state)
        trait_output = [head(hidden_state) for head in self.trait_outputs]
        champ_output = [head(hidden_state) for head in self.champ_output]
        shop_output = [head(hidden_state) for head in self.shop_output]
        item_output = [head(hidden_state) for head in self.item_output]
        scalar_output = [head(hidden_state) for head in self.scalar_output]

        return trait_output, champ_output, shop_output, item_output, scalar_output

    def forward(self, traits, board, items, bench, shop, scalars, emb_scalars):
        return self._forward(traits, board, items, bench, shop, scalars, emb_scalars)


class TFTNetworkTwoClass(torch.nn.Module):
    def __init__(self, model_config: ModelConfig):
        super().__init__()

        # How many layers to use in each mlp processing unit
        layer_sizes = [model_config.LAYER_HIDDEN_SIZE] * model_config.N_HEAD_HIDDEN_LAYERS

        # Embeddings for the unit are separate from item and trait. These double up for champion bench
        self.champion_embedding = torch.nn.Embedding(221, model_config.CHAMPION_EMBEDDING_DIM // 2)
        self.champion_item_embedding_1 = \
            torch.nn.Embedding(58, model_config.CHAMPION_EMBEDDING_DIM // 8)
        self.champion_item_embedding_2 = \
            torch.nn.Embedding(58, model_config.CHAMPION_EMBEDDING_DIM // 8)
        self.champion_item_embedding_3 = \
            torch.nn.Embedding(58, model_config.CHAMPION_EMBEDDING_DIM // 8)
        self.champion_trait_embedding = \
            torch.nn.Embedding(145, model_config.CHAMPION_EMBEDDING_DIM // 8)

        self.trait_encoder = AlternateFeatureEncoder(config.TRAIT_INPUT_SIZE, layer_sizes,
                                                     model_config.CHAMPION_EMBEDDING_DIM, config.DEVICE)

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
        self.pos_embedding = torch.nn.Embedding(512, model_config.CHAMPION_EMBEDDING_DIM)

        self.trait_outputs = nn.ModuleList([
            nn.Linear(model_config.HIDDEN_STATE_SIZE, tier_length) for tier_length in config.TEAM_TIERS_VECTOR
        ])

        self.champ_output = nn.ModuleList([
            nn.Linear(model_config.HIDDEN_STATE_SIZE, tier_length) for tier_length in config.CHAMPION_LIST_DIM
        ])

        self.to(config.DEVICE)

    def _forward(self, traits, board):
        # print(f"traits shape {traits.shape}")
        batch_size = traits.shape[0]
        champion_emb = self.champion_embedding(board[..., 0].long())
        champion_item_emb_1 = self.champion_item_embedding_1(board[..., 1].long())
        champion_item_emb_2 = self.champion_item_embedding_2(board[..., 2].long())
        champion_item_emb_3 = self.champion_item_embedding_3(board[..., 3].long())
        champion_trait_emb = self.champion_trait_embedding(board[..., 4].long())

        champion_item_emb = torch.cat([champion_item_emb_1, champion_item_emb_2, champion_item_emb_3], dim=-1)
        cie_shape = champion_item_emb.shape
        champion_item_emb = torch.reshape(champion_item_emb, (batch_size, cie_shape[1], cie_shape[2], -1))

        champion_embeddings = torch.cat([champion_emb, champion_item_emb, champion_trait_emb], dim=-1)
        champion_embeddings = torch.reshape(champion_embeddings, (batch_size, -1, self.champion_embedding_dim))

        trait_encoding = self.trait_encoder(traits)

        full_embeddings = torch.cat([champion_embeddings, trait_encoding], dim=1)

        # Add positional embeddings to full_embeddings
        position_ids = torch.arange(full_embeddings.shape[1], dtype=torch.long, device=config.DEVICE).unsqueeze(0)
        pos_embeddings = self.pos_embedding(position_ids)
        full_embeddings = full_embeddings + pos_embeddings

        # Expand the classification token to match the batch size
        cls_tokens = self.cls_token.expand(batch_size, -1, -1).to(config.DEVICE)

        # Concatenate the cls token to the full_embeddings
        full_embeddings = torch.cat([cls_tokens, full_embeddings], dim=1)

        # Note to future self, if I want to separate current board for some processing. Do it here but use two
        # Position encodings.
        full_enc = self.full_encoder(full_embeddings)

        cls_hidden_state = full_enc[:, 0:4, :]
        cls_hidden_state = torch.reshape(cls_hidden_state, (batch_size, -1))

        # Pass through final combiner network
        hidden_state = self.feature_processor(cls_hidden_state)
        trait_output = [head(hidden_state) for head in self.trait_outputs]
        champ_output = [head(hidden_state) for head in self.champ_output]

        return trait_output, champ_output

    def forward(self, traits, board):
        return self._forward(traits, board)


class TFTNetworkShopClass(torch.nn.Module):
    def __init__(self, model_config: ModelConfig):
        super().__init__()

        # How many layers to use in each mlp processing unit
        layer_sizes = [model_config.LAYER_HIDDEN_SIZE] * model_config.N_HEAD_HIDDEN_LAYERS

        self.shop_champion_embedding = \
            torch.nn.Embedding(221, model_config.CHAMPION_EMBEDDING_DIM * 3 // 4).to(config.DEVICE)

        # Technically have the shop_item availability but shops never have items so I am not including it.
        self.shop_trait_embedding = torch.nn.Embedding(145, model_config.CHAMPION_EMBEDDING_DIM // 4)

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

        self.shop_output = nn.ModuleList([
            nn.Linear(model_config.HIDDEN_STATE_SIZE, tier_length) for tier_length in [63, 63, 63, 63, 63]
        ])

        self.to(config.DEVICE)

    def _forward(self, shop):
        # print(f"traits shape {traits.shape}")
        batch_size = shop.shape[0]

        shop_champion_emb = self.shop_champion_embedding(shop[..., 0].long())
        shop_champion_trait_emb = self.shop_trait_embedding(shop[..., 4].long())
        shop_embeddings = torch.cat([shop_champion_emb, shop_champion_trait_emb], dim=-1)
        shop_embeddings = torch.reshape(shop_embeddings, (batch_size, -1, self.champion_embedding_dim))

        # Add positional embeddings to full_embeddings
        position_ids = torch.arange(shop_embeddings.shape[1], dtype=torch.long, device=config.DEVICE).unsqueeze(0)
        pos_embeddings = self.pos_embedding(position_ids)
        full_embeddings = shop_embeddings + pos_embeddings

        # Expand the classification token to match the batch size
        cls_tokens = self.cls_token.expand(batch_size, -1, -1).to(config.DEVICE)

        # Concatenate the cls token to the full_embeddings
        full_embeddings = torch.cat([cls_tokens, full_embeddings], dim=1)

        # Note to future self, if I want to separate current board for some processing. Do it here but use two
        # Position encodings.
        full_enc = self.full_encoder(full_embeddings)

        cls_hidden_state = full_enc[:, 0:4, :]
        cls_hidden_state = torch.reshape(cls_hidden_state, (batch_size, -1))

        # Pass through final combiner network
        hidden_state = self.feature_processor(cls_hidden_state)
        shop_output = [head(hidden_state) for head in self.shop_output]

        return shop_output

    def forward(self, shop):
        return self._forward(shop)

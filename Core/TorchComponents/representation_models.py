import torch
import torch.nn as nn
import config
from config import ModelConfig
from Core.TorchComponents.torch_layers import AlternateFeatureEncoder, Normalize, TransformerEncoder, \
    ResidualCNNBlock as ResidualBlock


class RepMultiVectorNetwork(torch.nn.Module):
    def __init__(self, model_config):
        super().__init__()

        def feature_encoder(input_size, feature_layer_sizes, feature_output_sizes):
            return torch.nn.Sequential(
                AlternateFeatureEncoder(input_size, feature_layer_sizes, feature_output_sizes, config.DEVICE),
                Normalize()
            ).to(config.DEVICE)

        layer_sizes = [model_config.LAYER_HIDDEN_SIZE] * model_config.N_HEAD_HIDDEN_LAYERS
        output_size = model_config.HIDDEN_STATE_SIZE // 2

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

        # Pass through final combiner network
        hidden_state = self.feature_processor(cls_hidden_state)

        return hidden_state

    def forward(self, x):
        return self._forward(x)


class RepPositionEmbeddingNetwork(torch.nn.Module):
    def __init__(self, model_config: ModelConfig, use_round_count=True):
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

        self.to(config.DEVICE)

        if use_round_count:
            self.round_encoder = AlternateFeatureEncoder(12, layer_sizes, model_config.CHAMPION_EMBEDDING_DIM,
                                                         config.DEVICE)

        self.use_round_count = use_round_count

    def _forward(self, x):
        batch_size = x["traits"].shape[0]
        champion_emb = self.champion_embedding(x["board"][..., 0].long())
        champion_item_emb_1 = self.champion_item_embedding_1(x["board"][..., 1].long())
        champion_item_emb_2 = self.champion_item_embedding_2(x["board"][..., 2].long())
        champion_item_emb_3 = self.champion_item_embedding_3(x["board"][..., 3].long())
        champion_trait_emb = self.champion_trait_embedding(x["board"][..., 4].long())

        champion_item_emb = torch.cat([champion_item_emb_1, champion_item_emb_2, champion_item_emb_3], dim=-1)
        cie_shape = champion_item_emb.shape
        champion_item_emb = torch.reshape(champion_item_emb, (batch_size, cie_shape[1], cie_shape[2], -1))

        champion_embeddings = torch.cat([champion_emb, champion_item_emb, champion_trait_emb], dim=-1)
        champion_embeddings = torch.reshape(champion_embeddings, (batch_size, -1, self.champion_embedding_dim))

        trait_encoding = self.trait_encoder(x["traits"])

        if self.use_round_count:
            round_encoding = self.round_encoder(x['action_count'])

            full_embeddings = torch.cat([champion_embeddings, trait_encoding, round_encoding], dim=1)
        else:
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

        return hidden_state

    def forward(self, x):
        return self._forward(x)

class AtariObservationNetwork(nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 3,
        device: str = "cpu",
    ):
        super().__init__()

        self.down_conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=128,
            kernel_size=kernel_size,
            stride=2,
            padding=1,
            device=device,
        )

        self.res_blocks1 = nn.Sequential(
            *[
                ResidualBlock(128, 128, kernel_size, padding="same", device=device)
                for _ in range(2)
            ]
        )

        self.down_conv2 = nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=kernel_size,
            stride=2,
            padding=1,
            device=device,
        )

        self.res_blocks2 = nn.Sequential(
            *[
                ResidualBlock(256, 256, kernel_size, padding="same", device=device)
                for _ in range(3)
            ]
        )

        self.avg_pool1 = nn.AvgPool2d(kernel_size=kernel_size, stride=2, padding=1)

        self.res_blocks3 = nn.Sequential(
            *[
                ResidualBlock(256, 256, kernel_size, padding="same", device=device)
                for _ in range(3)
            ]
        )

        self.avg_pool2 = nn.AvgPool2d(kernel_size=kernel_size, stride=2, padding=1)

    def forward(self, x):
        x = self.down_conv1(x)
        x = self.res_blocks1(x)
        x = self.down_conv2(x)
        x = self.res_blocks2(x)
        x = self.avg_pool1(x)
        x = self.res_blocks3(x)
        x = self.avg_pool2(x)
        return x


class AtariRepresentationNetwork(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_blocks: int = 4,
        kernel_size: int = 3,
        device: str = "cpu",
    ):
        super().__init__()

        self.atari_downsample = AtariObservationNetwork(
            in_channels=in_channels, kernel_size=kernel_size, device=device
        )

        self.res_blocks = nn.Sequential(
            *[
                ResidualBlock(
                    hidden_channels, hidden_channels, kernel_size, device=device
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x):
        x = self.atari_downsample(x)
        x = self.res_blocks(x)
        return x
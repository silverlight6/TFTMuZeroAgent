import torch
import torch.nn

import config
from Models.torch_layers import ResidualBlock, TransformerEncoder, mlp

class BasicTokenRepModel(torch.nn.Module):
    def __init__(self, model_config):
        super().__init__()

        # TODO: Add a positional embedding tomorrow

        """
        TODO: Create a Transformer for the other players, run one encoding step
        Take a summation. Append to the board and also append the traits. 
        """

        # Input size from create_champion_vector
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

        self.trait_encoder = mlp(config.TRAIT_INPUT_SIZE,
                                 [model_config.LAYER_HIDDEN_SIZE] * model_config.N_HEAD_HIDDEN_LAYERS,
                                 model_config.CHAMPION_EMBEDDING_DIM)

        # Combine encoded features from all components
        self.feature_combiner = mlp(model_config.CHAMPION_EMBEDDING_DIM * 30,
                                    [model_config.LAYER_HIDDEN_SIZE] * model_config.N_HEAD_HIDDEN_LAYERS,
                                    model_config.HIDDEN_STATE_SIZE
                                    )

        self.feature_processor = mlp(model_config.HIDDEN_STATE_SIZE,
                                     [model_config.LAYER_HIDDEN_SIZE] * model_config.N_HEAD_HIDDEN_LAYERS,
                                     model_config.HIDDEN_STATE_SIZE)

        # Optional: Add residual connections around encoders and combiner for better gradient flow
        self.board_residual = ResidualBlock(self.board_encoder, 28)
        self.full_residual = ResidualBlock(self.full_encoder, 30)
        self.combiner_residual = ResidualBlock(self.feature_processor, model_config.HIDDEN_STATE_SIZE, local_norm=False)
        self.model_config = model_config

    def forward(self, observation):
        # --- Encode Individual Components --- #
        champion_emb = self.champion_embedding(observation['board'][..., 0].long())
        champion_item_emb_1 = self.champion_item_embedding_1(observation['board'][..., 1].long())
        champion_item_emb_2 = self.champion_item_embedding_2(observation['board'][..., 2].long())
        champion_item_emb_3 = self.champion_item_embedding_3(observation['board'][..., 3].long())
        champion_trait_emb = self.champion_trait_embedding(observation['board'][..., 4].long())

        champion_item_emb = torch.cat([champion_item_emb_1, champion_item_emb_2, champion_item_emb_3], dim=-1)
        cie_shape = champion_item_emb.shape
        champion_item_emb = torch.reshape(champion_item_emb, (cie_shape[0], cie_shape[1], cie_shape[2], -1))

        board_emb = torch.cat([champion_emb, champion_item_emb, champion_trait_emb], dim=-1)

        board_residual_input = torch.reshape(board_emb[:, 0], [cie_shape[0], cie_shape[2],
                                             self.model_config.CHAMPION_EMBEDDING_DIM])

        trait_enc = self.trait_encoder(observation['traits'])

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
        return hidden_state

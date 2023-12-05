from flax import struct

from PoroX.architectures.components.embedding import (
    EmbeddingConfig,
    PlayerSegmentFFN, OpponentSegmentFFN, SegmentConfig, expand_segments,
)
from PoroX.architectures.components.transformer import EncoderConfig
from PoroX.architectures.player_encoder import PlayerConfig

@struct.dataclass
class MuZeroConfig:
    player_config: PlayerConfig
    opponent_config: PlayerConfig
    


test_config = MuZeroConfig(
    player_config= PlayerConfig(
        embedding=EmbeddingConfig( # Hidden state of 128
            champion_embedding_size=30,
            item_embedding_size=10,
            trait_embedding_size=8,
        ),
        # Total: 75
        # 28 board, 9 bench, 5 shop, 10 items, 1 trait, 22 scalars
        segment=SegmentConfig(
            segments=expand_segments([28, 9, 5, 10, 1, 22]),
        ),
        segment_ffn=PlayerSegmentFFN,
        encoder=EncoderConfig(
            num_blocks=8,
            num_heads=4,
            qkv_features=None,
            hidden_dim=None,
        )
    ),

    opponent_config=PlayerConfig(
        # embedding=EmbeddingConfig( # Hidden state of 64
        #     champion_embedding_size=12,
        #     item_embedding_size=4,
        #     trait_embedding_size=4,
        # ),
        embedding=EmbeddingConfig( # Hidden state of 128
            champion_embedding_size=30,
            item_embedding_size=10,
            trait_embedding_size=8,
        ),
        # Total: 54
        # 28 board, 9 bench, 10 items, 1 trait, 6 scalars
        segment=SegmentConfig(
            segments=expand_segments([28, 9, 10, 1, 6]),
        ),
        segment_ffn=OpponentSegmentFFN,
        encoder=EncoderConfig(
            num_blocks=4,
            num_heads=2,
            qkv_features=None,
            hidden_dim=None,
        )
    )
)
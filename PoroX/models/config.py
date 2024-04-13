from flax import struct

from PoroX.models.components.embedding import (
    EmbeddingConfig,
    GlobalPlayerSegmentFFN, SegmentConfig, expand_segments,
)
from PoroX.models.components.transformer import EncoderConfig
from PoroX.models.player_encoder import PlayerConfig

@dataclass
class MuZeroConfig:
    player_encoder: PlayerConfig
    cross_encoder: EncoderConfig
    merge_encoder: EncoderConfig

test_config = MuZeroConfig(
    player_encoder= PlayerConfig(
        embedding=EmbeddingConfig( # Hidden state of 256
            champion_embedding_size=37,
            item_embedding_size=20,
            trait_embedding_size=20,
        ),
        
        # Total: 75
        # 28 board, 9 bench, 5 shop, 10 items, 1 trait, 12 scalars, 3 matchups, 1 player_id
        segment=SegmentConfig(
            segments=expand_segments([28, 9, 5, 10, 1, 12, 3, 1]),
        ),
        segment_ffn=GlobalPlayerSegmentFFN,
        encoder=EncoderConfig(
            num_blocks=8,
            num_heads=4,
            qkv_features=None,
            hidden_dim=None,
        ),
    ),
    cross_encoder=EncoderConfig(
        num_blocks=3,
        num_heads=2,
        qkv_features=None,
        hidden_dim=None,
    ),
    merge_encoder=EncoderConfig(
        num_blocks=4,
        num_heads=4,
        qkv_features=None,
        hidden_dim=None,
    )
)
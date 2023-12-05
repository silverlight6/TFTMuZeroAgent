from flax import linen as nn
from flax import struct

from PoroX.modules.observation import PlayerObservation

from PoroX.architectures.components.embedding import (
    PlayerEmbedding, EmbeddingConfig,
    SegmentEncoding, SegmentConfig, 
    LearnedPositionalEncoding
)
from PoroX.architectures.components.transformer import Encoder, EncoderConfig

@struct.dataclass
class PlayerConfig:
    embedding: EmbeddingConfig
    segment: SegmentConfig
    segment_ffn: nn.Module
    encoder: EncoderConfig


class PlayerEncoder(nn.Module):
    config: PlayerConfig
    
    @nn.compact
    def __call__(self, obs: PlayerObservation):
        # Embed Player
        player_embedding = PlayerEmbedding(self.config.embedding)(obs)

        # Apply MLP to each segment
        segment_x = self.config.segment_ffn(config=self.config.segment)(player_embedding)
        
        # Apply Segment Encoding
        segment_x = SegmentEncoding(
            config=self.config.segment)(segment_x)

        # Apply Positional Encoding
        segment_pe_x = LearnedPositionalEncoding()(segment_x)
        
        # Apply Transformer Encoder
        hidden_state = Encoder(self.config.encoder)(segment_pe_x)
        
        return hidden_state
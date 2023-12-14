from flax import linen as nn
from flax import struct
import jax.numpy as jnp

from PoroX.modules.observation import PlayerObservation, BatchedObservation
import PoroX.modules.batch_utils as batch_utils

from PoroX.architectures.components.embedding import (
    PlayerEmbedding, EmbeddingConfig,
    SegmentEncoding, SegmentConfig, 
    LearnedPositionalEncoding
)
from PoroX.architectures.components.transformer import (
    EncoderConfig,
    Encoder, CrossAttentionEncoder
)

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

class GlobalPlayerEncoder(nn.Module):
    config: PlayerConfig
    
    def setup(self):
        self.player_embedding = PlayerEmbedding(self.config.embedding)
    
    @nn.compact
    def __call__(self, obs: BatchedObservation):
        # 0 is the player, rest are opponents
        # First we embed all the players
        player_embeddings = self.player_embedding(obs.players)
        opponent_embeddings = self.player_embedding(obs.opponents)
        
        all_embeddings = jnp.concatenate([
            player_embeddings,
            opponent_embeddings
        ], axis=-3)
        
        # Then we segment the players
        segment_x = self.config.segment_ffn(config=self.config.segment)(all_embeddings)
        # Segment Encoding
        segment_x = SegmentEncoding(
            config=self.config.segment)(segment_x)
        # Positional Encoding
        segment_pe_x = LearnedPositionalEncoding()(segment_x)
        # Transformer Encoder
        global_states = Encoder(self.config.encoder)(segment_pe_x)
        
        return global_states
        
    
class CrossPlayerEncoder(nn.Module):
    config: EncoderConfig
    
    @nn.compact
    def __call__(self, player_state: jnp.array, opponent_state: jnp.array):
        # Player State: [...B, S1, L]
        # Opponent State: [...B, P, S2, L]
        
        # Broadcast Player State to [...B, P, S1, L]
        def expand_and_get_shape(player_state, opponent_state):
            broadcasted_shape = player_state.shape[:-2] + opponent_state.shape[-3:-2] + player_state.shape[-2:]
            expanded_player_state = jnp.expand_dims(player_state, axis=-3)
            return broadcasted_shape, expanded_player_state

        # There are different methods we can go about performing cross attention
        # Method 1: Perform Cross Attention on EACH opponent embedding
        def opponentwise_cross_attention(player_state, opponent_state):
            # [...B, P, S1, L]
            shape, expanded_player_state = expand_and_get_shape(player_state, opponent_state)
            broadcasted_player_state = jnp.broadcast_to(expanded_player_state, shape=shape)
            
            # Perform Cross Attention
            global_state = CrossAttentionEncoder(self.config)(broadcasted_player_state, context=opponent_state)
            
            # To combine the player state and the global state, we can either:
            # 1. Concatenate the two
            def concat(player_state, global_state):
                return jnp.concatenate([
                    player_state,
                    global_state
                ], axis=-3)

            # 2. Sum the global state and then concatenate
            def sum_concat(player_state, global_state):
                # Sum the global state across the P dimension
                # summed_global_state = jnp.sum(global_state, axis=-3, keepdims=True)
                # return jnp.concatenate([
                #     player_state,
                #     summed_global_state
                # ], axis=-3)
                return jnp.sum(
                    global_state,
                    axis=-3,
                )
            
            # Concatenate the two
            # global_state = concat(expanded_player_state, global_state)

            # Sum the global state and then concatenate
            global_state = sum_concat(expanded_player_state, global_state)
            
            return global_state
        
        # Method 2: Perform Cross Attention on the sum of all opponent embeddings
        def global_cross_attention(player_state, opponent_state):
            # Sum the opponent state across the P dimension [...B, P, S2, L] -> [...B, 1, S2, L]
            summed_opponent_state = jnp.sum(opponent_state, axis=-3, keepdims=True)

            # [...B, 1, S1, L]
            _, expanded_player_state = expand_and_get_shape(player_state, opponent_state)
            
            # Perform Cross Attention
            global_state = CrossAttentionEncoder(self.config)(expanded_player_state, context=summed_opponent_state)
            
            # Concatenate the two
            global_state = jnp.concatenate([
                expanded_player_state,
                global_state
            ], axis=-3)
            
            return global_state
        
        return opponentwise_cross_attention(player_state, opponent_state)
        # return global_cross_attention(player_state, opponent_state)
from typing import Any, Callable, Optional

from flax import linen as nn
from flax import struct
import jax.numpy as jnp

from PoroX.modules.observation import PlayerObservation

from PoroX.architectures.components.scalar_encoder import ScalarEncoder
from PoroX.architectures.components.fc import MLP

# -- Config -- #
@struct.dataclass
class EmbeddingConfig:
    # Champion Embedding
    champion_embedding_size: int = 30
    item_embedding_size: int = 10
    trait_embedding_size: int = 8
    stats_size: int = 12
    
    # Scalar Config
    scalar_min_value: int = 0
    scalar_max_value: int = 200

    # General Embedding Config
    num_champions: int = 60
    num_items: int = 60
    num_traits: int = 27
        
def vector_size(config: EmbeddingConfig):
    return (
        config.champion_embedding_size +
        config.item_embedding_size * 3 +
        config.trait_embedding_size * 7 +
        config.stats_size
    )
        
# -- Champion Embedding -- #

class ChampionEmbedding(nn.Module):
    """
    Embeds a champion vector into a latent space

    Champion Vector:
    0: championID
    1-3: itemID
    4-10: traitID
    11-22: stats
    """
    config: EmbeddingConfig
    
    @nn.compact
    def __call__(self, x):
        ids = jnp.int16(x[..., :11])
        stats_vector = x[..., 11:]
        championID = ids[..., 0]
        itemIDs = ids[..., 1:4]
        traitIDs = ids[..., 4:11]
        
        champion_embedding = nn.Embed(
            num_embeddings=self.config.num_champions,
            features=self.config.champion_embedding_size)(championID)
        item_embedding = nn.Embed(
            num_embeddings=self.config.num_items,
            features=self.config.item_embedding_size)(itemIDs)
        trait_embedding = nn.Embed(
            num_embeddings=self.config.num_traits,
            features=self.config.trait_embedding_size)(traitIDs)
        
        batch_shape = item_embedding.shape[:-2]

        item_embedding = jnp.reshape(
            item_embedding,
            newshape=(*batch_shape, self.config.item_embedding_size * 3)
        )

        trait_embedding = jnp.reshape(
            trait_embedding,
            newshape=(*batch_shape, self.config.trait_embedding_size * 7)
        )

        return jnp.concatenate([
            champion_embedding,
            item_embedding,
            trait_embedding,
            stats_vector
        ], axis=-1)
        
# -- Item Embedding -- #
        
class ItemBenchEmbedding(nn.Module):
    """
    Embeds item bench ids to a latent space
    """
    config: EmbeddingConfig

    @nn.compact
    def __call__(self, x):
        ids = jnp.int16(x)

        bench_item_embedding = nn.Embed(
            num_embeddings=self.config.num_items,
            features=vector_size(self.config))(ids)

        return bench_item_embedding

# -- Trait Embedding -- #
    
class TraitEmbedding(nn.Module):
    """
    Embeds trait ids to a latent space
    """
    config: EmbeddingConfig
    
    @nn.compact
    def __call__(self, x):
        return MLP(features=[
            self.config.num_traits,
            vector_size(self.config)
            ])(x)
    
# -- Player Embedding -- #
        
class PlayerEmbedding(nn.Module):
    config: EmbeddingConfig
    
    @nn.compact
    def __call__(self, x: PlayerObservation):
        champion_embeddings = ChampionEmbedding(self.config)(x.champions)
        item_bench_embeddings = ItemBenchEmbedding(self.config)(x.items)
        trait_embeddings = TraitEmbedding(self.config)(x.traits)
        trait_embeddings = jnp.expand_dims(trait_embeddings, axis=-2)

        scalar_embeddings = ScalarEncoder(
            min_value=self.config.scalar_min_value,
            max_value=self.config.scalar_max_value,
            num_steps=vector_size(self.config)
        ).encode(x.scalars)

        player_embedding = jnp.concatenate([
            champion_embeddings,
            item_bench_embeddings,
            trait_embeddings,
            scalar_embeddings,
        ], axis=-2)
        
        return player_embedding
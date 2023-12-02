from flax import linen as nn
import jax.numpy as jnp

from PoroX.modules.observation import PlayerObservation

from PoroX.architectures.components.scalar_encoder import ScalarEncoder
from PoroX.architectures.components.mlp import MLP

class ChampionEmbedding(nn.Module):
    """
    Embeds a champion vector into a latent space

    Champion Vector:
    0: championID
    1-3: itemID
    4-10: traitID
    11-22: stats
    """
    champion_embedding_size: int
    item_embedding_size: int
    trait_embedding_size: int
    stats_size: int
    
    def setup(self):
        self.champion_embedding = nn.Embed(num_embeddings=1, features=self.champion_embedding_size)
        self.item_embedding = nn.Embed(num_embeddings=3, features=self.item_embedding_size)
        self.trait_embedding = nn.Embed(num_embeddings=7, features=self.trait_embedding_size)
        
        self.vector_size = (
            self.champion_embedding_size +
            self.item_embedding_size * 3 +
            self.trait_embedding_size * 7 +
            self.stats_size
        )
        
    def __call__(self, x):
        ids = jnp.int16(x[..., :11]) # Turn float16 to int16
        stats_vector = x[..., 11:]
        championID = ids[..., 0]
        itemIDs = ids[..., 1:4]
        traitIDs = ids[..., 4:11]
        
        champion_embedding = self.champion_embedding(championID)
        item_embedding = self.item_embedding(itemIDs)
        trait_embedding = self.trait_embedding(traitIDs)
        
        batch_shape = item_embedding.shape[:-2]
        
        item_embedding = jnp.reshape(
            item_embedding,
            newshape=(*batch_shape, self.item_embedding_size*3)
        )

        trait_embedding = jnp.reshape(
            trait_embedding,
            newshape=(*batch_shape, self.item_embedding_size*7)
        )
        
        return jnp.concatenate([
            champion_embedding,
            item_embedding,
            trait_embedding,
            stats_vector
        ], axis=-1)
        
class ItemBenchEmbedding(nn.Module):
    """
    Embeds item bench ids to a latent space
    """
    bench_item_embedding_size: int

    def setup(self):
        num_items = 10

        self.bench_item_embedding = nn.Embed(num_embeddings=num_items, features=self.bench_item_embedding_size)
        
    def __call__(self, x):
        ids = jnp.int16(x) # Turn float16 to int16

        bench_item_embedding = self.bench_item_embedding(ids)

        return bench_item_embedding
    
class TraitEmbedding(nn.Module):
    """
    Embeds trait ids to a latent space
    """
    trait_embedding_size: int
    
    def setup(self):
        self.num_traits = 26
        self.mlp = MLP(features=[
            self.num_traits,
            self.trait_embedding_size
        ])
        
    def __call__(self, x):
        return self.mlp(x)
        
class PlayerEmbedding(nn.Module):
    champion_embedding_size: int = 30
    item_embedding_size: int = 10
    trait_embedding_size: int = 10
    stats_size: int = 12
    
    def setup(self):
        self.champion_embedding = ChampionEmbedding(
            champion_embedding_size=self.champion_embedding_size,
            item_embedding_size=self.item_embedding_size,
            trait_embedding_size=self.trait_embedding_size,
            stats_size=self.stats_size
        )

        self.scalar_encoder = ScalarEncoder(
            min_value=0,
            max_value=200, 
            num_steps=self.champion_embedding.vector_size)

        self.item_bench_embedding = ItemBenchEmbedding(
            bench_item_embedding_size=self.champion_embedding.vector_size
        )

        self.trait_embedding = TraitEmbedding(
            trait_embedding_size=self.champion_embedding.vector_size,
        )
        
    def __call__(self, x: PlayerObservation):
        champion_embeddings = self.champion_embedding(x.champions)

        scalar_embeddings = self.scalar_encoder.encode(x.scalars)

        item_bench_embeddings = self.item_bench_embedding(x.items)

        trait_embeddings = self.trait_embedding(x.traits)
        trait_embeddings = jnp.expand_dims(trait_embeddings, axis=-2)
        
        return jnp.concatenate([champion_embeddings, scalar_embeddings, item_bench_embeddings, trait_embeddings], axis=-2)
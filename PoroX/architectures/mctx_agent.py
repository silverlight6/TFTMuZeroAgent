from flax import linen as nn

from PoroX.modules.observation import BatchedObservation
from PoroX.architectures.components.embedding import PlayerEmbedding

class RepresentationNetwork(nn.Module):
    """
    Represnetation Network to encode observation into a latent space
    
    Observation:
    - Player: PlayerObservation
    - Opponents: [PlayerObservation...]
    """
    champion_embedding_size: int = 30
    item_embedding_size: int = 10
    trait_embedding_size: int = 10
    stats_size: int = 12
    
    def setup(self):
        self.player_embedding = PlayerEmbedding(
            champion_embedding_size=self.champion_embedding_size,
            item_embedding_size=self.item_embedding_size,
            trait_embedding_size=self.trait_embedding_size,
            stats_size=self.stats_size
        )

        self.opponent_embedding = PlayerEmbedding(
            champion_embedding_size=self.champion_embedding_size,
            item_embedding_size=self.item_embedding_size,
            trait_embedding_size=self.trait_embedding_size,
            stats_size=self.stats_size
        )
    
    def __call__(self, obs: BatchedObservation):
        player_embedding = self.player_embedding(obs.player)
        opponent_embedding = self.opponent_embedding(obs.opponents)
        
        # TODO: MHSA on player_embedding
        # TODO: MHSA on opponent embeddings
        
        # TODO: Cross attention on player and opponent embeddings
        # TODO: Concatenate player_embedding + cross attention on opponent embeddings
        
        return player_embedding, opponent_embedding



"""
Stochastic MuZero Network:

Representation Network: hidden_state = R(observation)
Prediction Network: policy_logits, value = P(hidden_state)
Afterstate Dynamics Network: afterstate = AD(hidden_state, action)
Afterstate Prediction Network: chance_outcomes, afterstate_value = AP(afterstate)
Dynamics Network: hidden_state, reward = D(afterstate, chance_outcomes)

"""

class MCTSAgent:
    def act(self, obs):
        actions = {
            player_id: 0 for player_id in obs.keys()
        }

        return actions
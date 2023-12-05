from flax import linen as nn
from flax import struct
from jax import numpy as jnp

from PoroX.modules.observation import BatchedObservation
from PoroX.architectures.player_encoder import PlayerEncoder
from PoroX.architectures.config import MuZeroConfig
    
class RepresentationNetwork(nn.Module):
    """
    Represnetation Network to encode observation into a latent space
    
    Observation:
    - Player: PlayerObservation
    - Opponents: [PlayerObservation...]
    """
    config: MuZeroConfig
    
    @nn.compact
    def __call__(self, obs: BatchedObservation):
        # MHSA on player embedding
        player_state = PlayerEncoder(self.config.player_config)(obs.player)

        # MHSA on opponent embeddings
        opponent_state = PlayerEncoder(self.config.opponent_config)(obs.opponents)
        
        
        # TODO: Cross attention on player and opponent embeddings
        # TODO: Concatenate player_embedding + cross attention on opponent embeddings
        
        return player_state, opponent_state

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
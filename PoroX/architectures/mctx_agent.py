from flax import linen as nn
from flax import struct
from jax import numpy as jnp

from PoroX.modules.observation import BatchedObservation
import PoroX.modules.batch_utils as batch_utils

from PoroX.architectures.player_encoder import PlayerEncoder, CrossPlayerEncoder
from PoroX.architectures.components.transformer import EncoderConfig
from PoroX.architectures.config import MuZeroConfig
    
class RepresentationNetwork(nn.Module):
    """
    Represnetation Network to encode observation into a latent space
    
    Observation:
    - Player: PlayerObservation
    - Opponents: [PlayerObservation...]
    """
    config: MuZeroConfig
    cross_attention_config: EncoderConfig = EncoderConfig(
        num_blocks=1,
        num_heads=2,
    )
    
    @nn.compact
    def __call__(self, obs: BatchedObservation):
        # MHSA on player embedding
        player_state = PlayerEncoder(self.config.player_config)(obs.player) # [...B, S1, L]

        # MHSA on opponent embeddings
        opponent_state = PlayerEncoder(self.config.opponent_config)(obs.opponents) # [...B, P, S2, L]
        
        # Cross Attention on player and opponent embeddings
        global_state = CrossPlayerEncoder(self.cross_attention_config)(player_state, opponent_state) # [...B, N, S1, L]
        
        return player_state, global_state

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
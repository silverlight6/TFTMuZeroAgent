from flax import linen as nn
from jax import numpy as jnp
import jax

from PoroX.modules.observation import BatchedObservation
import PoroX.modules.batch_utils as batch_utils

from PoroX.models.player_encoder import PlayerEncoder, CrossPlayerEncoder, GlobalPlayerEncoder
from PoroX.models.components.transformer import EncoderConfig, CrossAttentionEncoder
from PoroX.models.config import MuZeroConfig
    
class RepresentationNetwork(nn.Module):
    """
    Represnetation Network to encode observation into a latent space
    
    Observation:
    - Player: PlayerObservation
    - Opponents: [PlayerObservation...]
    """
    config: MuZeroConfig
    
    # Don't ask what's going on here, I have no idea
    @nn.compact
    def __call__(self, obs: BatchedObservation):
        states = GlobalPlayerEncoder(self.config.player_encoder)(obs)
        
        # Split states into player and opponents
        player_shape = obs.players.champions.shape[-3]
        
        player_states = states[..., :player_shape, :, :]
        opponent_states = states[..., player_shape:, :, :]
        
        expanded_opponent_state = jnp.expand_dims(opponent_states, axis=-4)
        broadcast_shape = opponent_states.shape[:-4] + player_states.shape[-3:-2] + opponent_states.shape[-3:]
        broadcasted_opponent = jnp.broadcast_to(expanded_opponent_state, broadcast_shape)
        
        player_ids = obs.players.scalars[..., 0].astype(jnp.int32)
        
        # Create a mask that masks out the player's own state
        mask = jnp.arange(player_states.shape[-3]) == player_ids[..., None]
        masked_opponent_states = broadcasted_opponent * mask[..., None, None]
        # I have no fucking clue what I'm doing but it jit compiles...
        
        cross_states = CrossPlayerEncoder(self.config.cross_encoder)(player_states, masked_opponent_states)
        
        merged_states = CrossAttentionEncoder(self.config.merge_encoder)(player_states, context=cross_states)
        
        return merged_states


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
import jax
import jax.numpy as jnp
from PoroX.modules.observation import BatchedObservation, PlayerObservation

# Broadcast Player State to [...B, P, S1, L]
@jax.jit
def expand_and_get_shape(player_state, opponent_state):
    broadcasted_shape = player_state.shape[:-2] + opponent_state.shape[-3:-2] + player_state.shape[-2:]
    expanded_player_state = jnp.expand_dims(player_state, axis=-3)
    return broadcasted_shape, expanded_player_state

@jax.jit
def collect(collection):
    return jax.tree_map(lambda *xs: jnp.stack(xs), *collection)

@jax.jit
def concat(collection, axis=-3):
    return jax.tree_map(lambda *xs: jnp.concatenate(xs, axis=axis), *collection)

@jax.jit
def split(collection, schema):
    """
    Splits a batched collection back into a list of collections based on a schema.
    """
    # TODO: Figure out how to do this with jax.tree_map
    ...

@jax.jit
def collect_obs(obs: dict):
    """
    Batches a dict of player observations to a BatchedObservation.

    obs: dict of observations from TFTEnv.step

    {
        "player_{id}": {
            "player": PlayerObservation,
            "action_mask": jnp.ndarray,
            "opponents": [PublicObservation...]
        }
        ...
    }
    to:
    BatchedObservation(
        player=[PlayerObservation...]
        action_mask=[jnp.ndarray...]
        opponents=[[PublicObservation...]...]
    )
    """
    
    obs = [
        # Default collect all obs for each player
        BatchedObservation(
            players=player_obs["player"],
            action_mask=player_obs["action_mask"],
            opponents=collect(player_obs["opponents"])
        )
        
        # Combine players and masked players
        # BatchedObservation(
        #     players=collect([player_obs["player"], *player_obs["opponents"]]),
        #     action_mask=player_obs["action_mask"]
        # )
        for player_obs in obs.values()
    ]
    
    return collect(obs)

@jax.jit
def collect_shared_obs(obs):
    """
    Instead of collecting 7 opponent obs for each player,
    we collect all 8 opponent obs as a single BatchedObservation.
    This will save a lot of computation for the encoder.
    """
    
    first_obs = next(iter(obs.values()))
    opponent_obs = first_obs["opponents"]
    
    player_obs = [
        player["player"]
        for player in obs.values()
    ]
    
    action_mask = [
        player["action_mask"]
        for player in obs.values()
    ]
    
    return BatchedObservation(
        players=collect(player_obs),
        action_mask=collect(action_mask),
        opponents=collect(opponent_obs)
    )
    
def collect_env_obs(obs):
    """
    Batches a list of dicts of player observations to a BatchedObservation.
    Essentially the output from multiple TFTEnv.step calls in a list.
    
    TODO: This won't work when some players are terminated in some envs.
    """
    
    batched_obs = [
        collect_obs(game_obs)
        for game_obs in obs
    ]
    
    return collect(batched_obs)
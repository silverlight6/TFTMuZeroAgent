import jax
import jax.numpy as jnp
from PoroX.modules.observation import BatchedObservation

@jax.jit
def merge_dimensions(x, dim1, dim2):
    shape = x.shape
    new_shape = shape[:dim1] + (shape[dim1] * shape[dim2],) + shape[dim2+1:]
    return jnp.reshape(x, new_shape)

@jax.jit
def collect(collection):
    return jax.tree_map(lambda *xs: jnp.stack(xs), *collection)

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
        BatchedObservation(
            player=player_obs["player"],
            action_mask=player_obs["action_mask"],
            opponents=collect(player_obs["opponents"])
        )
        for player_obs in obs.values()
    ]
    
    return collect(obs)
    
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
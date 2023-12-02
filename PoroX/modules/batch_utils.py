import jax
import jax.numpy as jnp
from PoroX.modules.observation import BatchedObservation

@jax.jit
def collect(collection):
    return jax.tree_map(lambda *xs: jnp.stack(xs), *collection)

def collect_obs(obs: dict):
    """
    obs: dict of observations

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
    
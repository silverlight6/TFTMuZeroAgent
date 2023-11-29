from flax import linen as nn

class RepresentationNetwork(nn.Module):
    """
    Represnetation Network to encode observation into a latent space
    
    Observation:
    - Champion Vectors (Board, Bench, Shop)
    - Item Vectors (Item Bench)
    - Trait Vectors
    - Scalars (Encoded as scalar)

    """
    
    ...
    

class MCTSAgent:
    def act(self, obs):
        return {
            player_id: 0 for player_id in obs.keys()
        }
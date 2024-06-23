import numpy as np
import chex
from Simulator.observation.token.observation import ObservationToken

@chex.dataclass(frozen=True)
class PlayerObservation:
    champions: chex.ArrayDevice
    scalars: chex.ArrayDevice
    items: chex.ArrayDevice
    traits: chex.ArrayDevice
    
@chex.dataclass(frozen=True)
class BatchedObservation:
    players: PlayerObservation
    action_mask: chex.ArrayDevice
    opponents: PlayerObservation


class PoroXObservation(ObservationToken):
    def __init__(self, player):
        super().__init__(player)
        
        self.board_zeros = np.zeros_like(self.board_vector, dtype=np.float32)
        self.bench_zeros = np.zeros_like(self.bench_vector, dtype=np.float32)
        self.shop_zeros = np.zeros_like(self.shop_vector, dtype=np.float32)
        self.item_bench_zeros = np.zeros_like(self.item_bench_vector, dtype=np.float32)
        self.trait_zeros = np.zeros_like(self.trait_vector, dtype=np.float32)
        self.public_zeros = np.zeros_like(self.public_scalars, dtype=np.float32)
        self.private_zeros = np.zeros_like(self.private_scalars, dtype=np.float32)
        self.game_zeros = np.zeros_like(self.game_scalars, dtype=np.float32)
        
        self.public_zeros[0] = player.player_num

    def fetch_player_observation(self):
        """Fetch Public Observation."""
        champions = np.concatenate([
            self.board_vector,
            self.bench_vector,
            self.shop_vector
        ])
        scalars = np.concatenate([
            self.public_scalars,
            self.private_scalars,
            self.game_scalars
        ])
        
        return PlayerObservation(
            champions=champions,
            scalars=scalars,
            items=self.item_bench_vector,
            traits=self.trait_vector
        )
        
    def fetch_public_observation(self):
        """Fetch Public Observation."""
        champions = np.concatenate([
            self.board_vector,
            self.bench_vector,
            self.shop_zeros  # MASK
        ])
        scalars = np.concatenate([
            self.public_scalars,
            self.private_zeros,  # MASK
            self.game_zeros  # MASK
        ])
        
        return PlayerObservation(
            champions=champions,
            scalars=scalars,
            items=self.item_bench_vector,
            traits=self.trait_vector
        )

    def fetch_dead_observation(self):
        """Fetch Dead Observation."""
        champions = np.concatenate([
            self.board_zeros,
            self.bench_zeros,
            self.shop_zeros
        ])
        scalars = np.concatenate([
            self.public_zeros,
            self.private_zeros,
            self.game_zeros
        ])
        
        return PlayerObservation(
            champions=champions,
            scalars=scalars,
            items=self.item_bench_zeros,
            traits=self.trait_zeros
        )

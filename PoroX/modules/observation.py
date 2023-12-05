import numpy as np
from dataclasses import dataclass
import chex
from Simulator.porox.observation import ObservationVector
from PoroX.architectures.components.scalar_encoder import ScalarEncoder

@chex.dataclass(frozen=True)
class PlayerObservation:
    champions: chex.ArrayDevice
    scalars: chex.ArrayDevice
    items: chex.ArrayDevice
    traits: chex.ArrayDevice
    
@chex.dataclass(frozen=True)
class BatchedObservation:
    player: PlayerObservation
    action_mask: chex.ArrayDevice
    opponents: PlayerObservation


class PoroXObservation(ObservationVector):
    def fetch_player_observation(self):
        """Fetch Public Observation."""
        champions = np.concatenate(
            (self.board_vector, self.bench_vector, self.shop_vector)
        )
        scalars = np.concatenate(
            (self.game_scalars, self.public_scalars, self.private_scalars)
        )
        
        return PlayerObservation(
            champions=champions,
            scalars=scalars,
            items=self.item_bench_vector,
            traits=self.trait_vector
        )
        
    def fetch_public_observation(self):
        """Fetch Public Observation."""
        champions = np.concatenate(
            (self.board_vector, self.bench_vector)
        )
        scalars = self.public_scalars
        
        return PlayerObservation(
            champions=champions,
            scalars=scalars,
            items=self.item_bench_vector,
            traits=self.trait_vector
        )

    def fetch_dead_observation(self):
        """Fetch Dead Observation."""
        champion_shape = (
            self.board_vector.shape[0] + self.bench_vector.shape[0],
            self.champion_vector_length
        )
        
        return PlayerObservation(
            champions=np.zeros(champion_shape),
            scalars=np.zeros_like(self.public_scalars),
            items=np.zeros_like(self.item_bench_vector),
            traits=np.zeros_like(self.trait_vector)
        )
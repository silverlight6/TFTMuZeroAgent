import numpy as np
import config

from Simulator.observation.vector.observation import ObservationVector

class SingleVectorObservation(ObservationVector):
    """Observation object that stores the observation for a player."""

    def __init__(self, player):
        super().__init__(player)

    def fetch_player_observation(self):
        """Fetch the PlayerObservation for a player.

        PlayerObservation:
            scalars: [Game Values, Public Scalars, Private Scalars]
            board: board token
            bench: bench token
            shop: shop token
            items: item token
            traits: trait token
        """

        return np.concatenate([
            self.public_scalars,
            self.private_scalars,
            self.board_vector,
            self.bench_vector,
            self.shop_vector,
            self.item_bench_vector,
            self.trait_vector
        ])

    def fetch_player_position_observation(self):
        """Fetch the PlayerObservation for a player.

        PlayerObservation:
            board: board token
            traits: trait token
        """
        return np.concatenate([
            self.board_vector,
            self.trait_vector
        ])

    def fetch_public_observation(self):
        """Fetch the PlayerPublicObservation for a player.

        PlayerPublicObservation:
            scalars: [Public Scalars]
            board: board token
            bench: bench token
            items: item token
            traits: trait token

        Commenting out the bench and items to save on storage space. Can comment back in a future iteration
        """
        return np.concatenate([
            self.public_scalars,
            self.board_vector,
            self.trait_vector
        ])

    def fetch_public_position_observation(self):
        """Fetch the PlayerPublicObservation for a player.

                PlayerPublicObservation:
                    board: board token
                    traits: trait token

                Commenting out the bench and items to save on storage space. Can comment back in a future iteration
        """
        return np.concatenate([
            self.board_vector,
            self.trait_vector
        ])

    def fetch_dead_observation(self):
        """Zero out public observations for all dead players"""
        return np.concatenate([
            np.zeros(self.public_scalars.shape, dtype=np.float32),
            np.zeros(self.board_vector.shape, dtype=np.float32),
            np.zeros(self.trait_vector.shape, dtype=np.float32)
        ])

    def fetch_dead_position_observation(self):
        """Zero out public observations for all dead players"""
        return np.concatenate([
            np.zeros(self.board_vector.shape, np.float32),
            np.zeros(self.trait_vector.shape, np.float32)
        ])

    @staticmethod
    def observation_to_input(observation):
        other_players = np.concatenate(
            [
                np.concatenate(
                    [
                        observation["opponents"][x]["board"],
                        observation["opponents"][x]["scalars"],
                        observation["opponents"][x]["traits"],
                    ],
                    axis=-1,
                )
                for x in range(config.NUM_PLAYERS)
            ],
            axis=-1,  # Concatenate opponent data along axis 0
        )
        return np.concatenate([observation["player"], other_players])

    @staticmethod
    def observation_to_dictionary(observation):
        """Converts a list of observations to a batched dictionary."""

        return {
            key: np.stack([obs[key] for obs in observation])
            for key in observation[0].keys()
        }

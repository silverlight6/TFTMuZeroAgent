import numpy as np
import config

from Simulator.observation.vector.observation import ObservationVector

class SingleVectorObservation(ObservationVector):
    """Observation object that stores the observation for a player."""

    def __init__(self, player):
        super().__init__(player)

        self.public_game_image = np.zeros([84, 84, 1])
        self.private_game_image = np.zeros([84, 84, 1])

        # So I want to build an image in the 84 x 84 --> 7056 values
        # 84 / 7 -> 12 So the scale should be 12.
        # 4 * 12 --> 48
        # Add another

    '''
    __________________________________________
    | B1  | B2  | B3  |  B4 |  B5 |  B6 |  B7 | 12 x 84
    | B1  | B2  | B3  |  B4 |  B5 |  B6 |  B7 | 12 x 84
    | B1  | B2  | B3  |  B4 |  B5 |  B6 |  B7 | 12 x 84
    | B1  | B2  | B3  |  B4 |  B5 |  B6 |  B7 | 12 x 84
    __________________________________________ 48 x 84
    | b1  | b2  | b3  |  b4 |  b5 |  b6 |  b7 | 12 x 84
    | b8  | b9  | s1  |  s2 |  s3 |  s4 |  s5 | 12 x 84
    __________________________________________ 72 x 84
    | i1 | i2 | i3 | i4 | i5 | i6 | i7 | i8 | i9 | i10 | 4 x 80|
    |v1|v2|v3|v4|v5|v6|v7|v8|v9|v10|v11|v12|v13|v14| 4 * 84
    | Traits                                       | 4 * 84
    __________________________________________
    Each one of these will have a 1pixel padding of 0s.
    This is for the main game. Positioning will have ordering in 
    the place where items and bench are.
    Itemization is the same but with borders being different values 
    depending on whose turn it is.
    '''

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
            key: np.stack([obs[key] for obs in observation]) for key in observation[0].keys()
        }




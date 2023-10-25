import numpy as np

from Simulator.porox.observation.observation_helper import ObservationHelper


class Observation(ObservationHelper):
    """Observation object that stores the observation for a player."""

    def __init__(self, player):
        super().__init__()

        self.player = player

        # --- Game Values --- #
        self.game_scalars = self.create_game_scalars(player)

        # --- Public Scalars --- #
        self.public_scalars = self.create_public_scalars(
            player)

        # --- Private Scalars --- #
        self.private_scalars = self.create_private_scalars(
            player)

        # --- Public Vectors --- #
        self.board_vector = self.create_board_vector(player)
        self.bench_vector = self.create_bench_vector(player)
        self.item_bench_vector = self.create_item_bench_vector(
            player)
        self.trait_vector = self.create_trait_vector(player)

        # --- Private Vectors --- #
        self.shop_vector = self.create_shop_vector(player)

        # --- Mask --- #
        self.exp_mask = self.create_exp_action_mask(player)
        self.buy_mask = self.create_buy_action_mask(player)
        self.move_sell_board_mask, self.move_sell_bench_mask = self.create_move_and_sell_action_mask(
            player)
        self.item_mask = self.create_item_action_mask(player)

        # --- Track Conditional Values so we don't update the mask unnecessarily --- #
        self.current_level = player.level

    def update_exp_action(self, action):
        """Update vectors and masks related to the exp action.

        Updated Values:
            - Gold, Exp, Level
            - If level up, then bench masks need to be updated

        """
        pass

    def update_refresh_action(self, action):
        """Update vectors and masks related to the refresh action.

        Updated Values:
            - Shop
        """
        pass

    def update_buy_action(self, action):
        """Update vectors and masks related to the buy action.

        Updated Values:
            - Shop, Bench, Gold
        """
        pass

    def update_move_sell_board_action(self, action):
        """Update vectors and masks related to the move sell board action.

        Updated Values:
            - Board, Bench, Gold
        """
        pass

    def update_move_sell_bench_action(self, action):
        """Update vectors and masks related to the move sell bench action.

        Updated Values:
            - Bench, Gold
        """
        pass

    def update_item_action(self, action):
        """Update vectors and masks related to the item action.

        Updated Values:
            - Bench, Item Bench, Gold
        """
        pass

    def fetch_player_observation(self, action):
        """Fetch the PlayerObservation for a player.

        PlayerObservation:
            scalars: [Game Values, Public Scalars, Private Scalars]
            board: board vector
            bench: bench vector
            shop: shop vector
            items: item vector
            traits: trait vector
        """
        pass

        return {
            "scalars": np.concatenate(
                self.game_scalars,
                self.public_scalars,
                self.private_scalars
            ),
            "board": self.board_vector,
            "bench": self.bench_vector,
            "shop": self.shop_vector,
            "items": self.item_bench_vector,
            "traits": self.trait_vector,
        }

    def fetch_public_observation(self):
        """Fetch the PlayerPublicObservation for a player.

        PlayerPublicObservation:
            scalars: [Public Scalars]
            board: board vector
            bench: bench vector
            items: item vector
            traits: trait vector
        """
        return {
            "scalars": self.public_scalars,
            "board": self.board_vector,
            "bench": self.bench_vector,
            "items": self.item_bench_vector,
            "traits": self.trait_vector,
        }

    def fetch_action_mask(self):
        """Fetch the action mask for a player.

        Mask: (5, 11, 38)  # Same as action space

        This one is a bit tricky, as we need to format each mask to match the action space.

        Action Space is an 5x11x38 Dimension MultiDiscrete Tensor
                     11
           |P|L|R|B|B|B|B|B|B|B|S|
           |b|b|b|B|B|B|B|B|B|B|S|
        5  |b|b|b|B|B|B|B|B|B|B|S| x 38
           |b|b|b|B|B|B|B|B|B|B|S|
           |I|I|I|I|I|I|I|I|I|I|S|

        P = Pass Action
        |P| : 0

        L = Level Action
        |L| : 1

        R = Refresh Action
        |R| : 2

        B = Board Slot
          3           9 10
       0 |B|B|B|B|B|B|B|
         |B|B|B|B|B|B|B|
         |B|B|B|B|B|B|B|
       3 |B|B|B|B|B|B|B|
       4

        b = Bench Slot
          0   2 3
       1 |b|b|b|
         |b|b|b|
       3 |b|b|b|
       4

        I = Item Slot
          0                 9 10
       4 |I|I|I|I|I|I|I|I|I|I|
       5

        S = Shop Slot
          10
       0 |S|
         |S|
         |S|
         |S|
       4 |S|
       5
        """

        action_mask = np.zeros((5, 11, 38))

        # --- Pass is always available --- #
        action_mask[0][0][0] = 1

        # --- Level mask --- #
        action_mask[0][1][0] = self.level_mask

        # --- Refresh mask --- #
        action_mask[0][2][0] = self.refresh_mask

        # --- Board mask --- #
        # [3:10, 0:3, :] is the board mask
        action_mask[3:10, 0:3, :] = self.move_sell_board_mask

        # --- Bench mask --- #
        # Reshape from [9, 38] to [3, 3, 38]
        bench_mask = self.move_sell_bench_mask.reshape(3, 3, 38)

        # [1:4, 0:3, :] is the bench mask
        action_mask[1:4, 0:3, :] = bench_mask

        # --- Item mask --- #
        # [4, 0:10, :] is the item mask
        action_mask[4, 0:10, :] = self.item_mask

        # --- Buy mask --- #
        # [0:5, 10, :] is the buy mask
        action_mask[0:5, 10, :] = self.buy_mask

        return action_mask

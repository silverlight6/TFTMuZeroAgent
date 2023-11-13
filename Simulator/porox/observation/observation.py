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

        # --- Action Mask --- #
        self.exp_mask = self.create_exp_action_mask(player)
        self.refresh_mask = self.create_refresh_action_mask(player)
        self.buy_mask = self.create_buy_action_mask(player)
        self.move_sell_board_mask, self.move_sell_bench_mask = self.create_move_and_sell_action_mask(
            player)
        self.item_mask = self.create_item_action_mask(player)

        # --- Track Conditional Values so we don't update the mask unnecessarily --- #
        self.current_level = player.level
        
    def update_game_round(self):
        """Update vectors and masks after game round.

        After a battle, your hp, gold, exp, and shop changes
        `refresh_all_shops` calls `update_refresh_action`, so no use calling those twice
        Round might be a carousel/minion round, so just update everything just in case
        """
        self.game_scalars = self.create_game_scalars(self.player)
        self.public_scalars = self.create_public_scalars(self.player)
        self.board_vector = self.create_board_vector(self.player)
        self.bench_vector = self.create_bench_vector(self.player)
        self.item_bench_vector = self.create_item_bench_vector(self.player)

        self.move_sell_board_mask, self.move_sell_bench_mask = self.create_move_and_sell_action_mask(self.player)
        self.item_mask = self.create_item_action_mask(self.player)

    def update_exp_action(self, action):
        """Update vectors and masks related to the exp action.

        Updated Values:
            - Gold, Exp, Level
            - If level up, then bench masks need to be updated

        """
        self.public_scalars = self.create_public_scalars(self.player)
        self.private_scalars = self.create_private_scalars(self.player)

        self.exp_mask = self.create_exp_action_mask(self.player)
        self.refresh_mask = self.create_refresh_action_mask(self.player)
        self.buy_mask = self.create_buy_action_mask(self.player)
        self.move_sell_board_mask, self.move_sell_bench_mask = self.create_move_and_sell_action_mask(self.player)

    def update_refresh_action(self, action):
        """Update vectors and masks related to the refresh action.

        Updated Values:
            - Shop
        """
        self.private_scalars = self.create_private_scalars(self.player)
        self.shop_vector = self.create_shop_vector(self.player)

        self.exp_mask = self.create_exp_action_mask(self.player)
        self.refresh_mask = self.create_refresh_action_mask(self.player)
        self.buy_mask = self.create_buy_action_mask(self.player)

    def update_buy_action(self, action):
        """Update vectors and masks related to the buy action.

        Updated Values:
            - Shop, Bench, Gold
        """
        self.private_scalars = self.create_private_scalars(self.player)
        self.bench_vector = self.create_bench_vector(self.player)
        self.shop_vector = self.create_shop_vector(self.player)
        self.trait_vector = self.create_trait_vector(self.player)

        self.exp_mask = self.create_exp_action_mask(self.player)
        self.refresh_mask = self.create_refresh_action_mask(self.player)
        self.buy_mask = self.create_buy_action_mask(self.player)
        self.move_sell_board_mask, self.move_sell_bench_mask = self.create_move_and_sell_action_mask(self.player)
        self.item_mask = self.create_item_action_mask(self.player)

    def update_move_sell_action(self, action):
        """Update vectors and masks related to the move sell board/bench action.

        Updated Values:
            - Board, Bench, Gold
        """
        self.private_scalars = self.create_private_scalars(self.player)
        self.public_scalars = self.create_public_scalars(self.player)
        self.board_vector = self.create_board_vector(self.player)
        self.bench_vector = self.create_bench_vector(self.player)
        self.trait_vector = self.create_trait_vector(self.player)
        self.item_bench_vector = self.create_item_bench_vector(self.player)

        self.exp_mask = self.create_exp_action_mask(self.player)
        self.refresh_mask = self.create_refresh_action_mask(self.player)
        self.buy_mask = self.create_buy_action_mask(self.player)
        self.move_sell_board_mask, self.move_sell_bench_mask = self.create_move_and_sell_action_mask(self.player)
        self.item_mask = self.create_item_action_mask(self.player)

    def update_item_action(self, action):
        """Update vectors and masks related to the item action.

        Updated Values:
            - Bench, Item Bench, Gold
        """
        self.board_vector = self.create_board_vector(self.player)
        self.bench_vector = self.create_bench_vector(self.player)
        self.item_bench_vector = self.create_item_bench_vector(self.player)

        self.item_mask = self.create_item_action_mask(self.player)

    def fetch_player_observation(self):
        """Fetch the PlayerObservation for a player.

        PlayerObservation:
            scalars: [Game Values, Public Scalars, Private Scalars]
            board: board vector
            bench: bench vector
            shop: shop vector
            items: item vector
            traits: trait vector
        """

        return {
            "scalars": np.concatenate([
                self.game_scalars,
                self.public_scalars,
                self.private_scalars
            ]),
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

    def fetch_dead_observation(self):
        """Zero out public observations for all dead players"""
        return {
            "scalars": np.zeros(self.public_scalars.shape),
            "board": np.zeros(self.board_vector.shape),
            "bench": np.zeros(self.bench_vector.shape),
            "shop": np.zeros(self.shop_vector.shape),
            "items": np.zeros(self.item_bench_vector.shape),
            "traits": np.zeros(self.trait_vector.shape),
        }

    def fetch_action_mask_v1(self):
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
        action_mask[0][1][0] = self.exp_mask

        # --- Refresh mask --- #
        action_mask[0][2][0] = self.refresh_mask

        # --- Board mask --- #
        # [0:4, 3:10, :] is the board mask
        action_mask[0:4, 3:10, :] = np.rot90(self.move_sell_board_mask, k=1)

        # --- Bench mask --- #
        # Reshape from [9, 38] to [3, 3, 38]
        bench_mask = self.move_sell_bench_mask.reshape(3, 3, 38)

        # [0:3, 1:4, :] is the bench mask
        action_mask[1:4, 0:3, :] = bench_mask

        # --- Item mask --- #
        # [4, 0:10, :] is the item mask
        action_mask[4, 0:10, :] = self.item_mask

        # --- Buy mask --- #
        # [0:5, 10, :] is the buy mask
        action_mask[0:5, 10, 0] = self.buy_mask

        return action_mask

    def fetch_action_mask_v2(self):
        """Fetch action mask for v2 action space.
        
        v2 action space: (55, 38)
            55 
        1 | | | | | ... | | x 38
        
        55 :
        0-27 -> Board Slots (28)
        28-36 -> Bench Slots (9)
        37-46 -> Item Bench Slots (10)
        47-51 -> Shop Slots (5)
        52 -> Pass
        53 -> Level
        54 -> Refresh
        
        38 :
        0-27 -> Board Slots
        28-36 -> Bench Slots
        37 -> Sell Slot
        
        Saves me the headache of trying to figure out how to format the mask
        """
        
        action_mask = np.zeros((55, 38))
        
        # --- Board mask --- #
        # Board mask is currently (7, 4, 38), so we need to reshape it to (28, 38)
        action_mask[0:28, :] = np.reshape(self.move_sell_board_mask, (-1, 38))
        
        # --- Bench mask --- #
        action_mask[28:37, :] = self.move_sell_bench_mask

        # --- Item bench mask --- #
        action_mask[37:47, :] = self.item_mask
        
        # --- Shop mask --- #
        action_mask[47:52, 0] = self.buy_mask

        # --- Pass is always available --- #
        action_mask[52, 0] = 1

        # --- Level mask --- #
        action_mask[53, 0] = self.exp_mask

        # --- Refresh mask --- #
        action_mask[54, 0] = self.refresh_mask
        
        return action_mask
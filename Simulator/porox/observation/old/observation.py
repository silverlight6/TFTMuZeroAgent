import numpy as np

from Simulator.porox.observation.old.observation_helper import ObservationHelper

"""
DEPRECATED FILE: This file is no longer used in the simulator.
It is kept here for reference.
"""

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
        self.board_vector = self.create_board_vector(self.player)
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

def action_space_to_action_v1(action):
    """
    Action Space is an 5x11x38 Dimension MultiDiscrete Tensor
                 11
       |P|L|R|B|B|B|B|B|B|B|S| 
       |b|b|b|B|B|B|B|B|B|B|S|
    5  |b|b|b|B|B|B|B|B|B|B|S| x 38
       |b|b|b|B|B|B|B|B|B|B|S|
       |I|I|I|I|I|I|I|I|I|I|S|

    P = Pass Action
    L = Level Action
    R = Refresh Action
    B = Board Slot
    b = Bench Slot
    I = Item Slot
    S = Shop Slot

    Pass, Level, Refresh, and Shop are single action spaces,
    meaning we only use the first dimension of the MultiDiscrete Space

    Board, Bench, and Item are multi action spaces,
    meaning we use all 3 dimensions of the MultiDiscrete Space

    0-26 -> Board Slots
    27-36 -> Bench Slots
    37 -> Sell Slot

    Board and Bench use all 38 dimensions,
    Item only uses 37 dimensions, as you cannot sell an item

    """
    row, col, index = action // 38 // 11, action // 38 % 11, action % 38
    
    action = []
    
    # Pass Action
    if row == 0 and col == 0:
        action = [0, 0, 0]
    # Level Action
    elif row == 0 and col == 1:
        action = [1, 0, 0]
    # Refresh Action
    elif row == 0 and col == 2:
        action = [2, 0, 0]
    
    # Board Slot
    elif row < 4 and (col >= 3 and col <= 9):
        pos = (col-3, 3-row)
        from_loc = pos[0] * 4 + pos[1]
        
        if index == 37:
            action = [4, from_loc, 0]
        else:
            to_loc = index
            action = [5, from_loc, to_loc]
            
    # Bench Slot
    elif (row >= 1 and row <= 3) and (col >= 0 and col <= 2):
        pos = (col, row-1)
        from_loc = pos[0] * 3 + pos[1] + 28
        
        if index == 37:
            action = [4, from_loc, 0]
        else:
            to_loc = index
            action = [5, from_loc, to_loc]
            
    # Buy Slot
    elif col == 10:
        action = [3, row, 0]
    
    # Item Slot
    elif row == 4 and col < 10:
        from_loc = col
        to_loc = index
        action = [6, from_loc, to_loc]

    return action
    
def action_space_to_action_v2(action):
    """Converts an action sampled from the action space to an action that can be performed.
    
    Action Space: (55, 38)
    55:
    |0|1|2|3|4|5|6|7|8|9|10|11|12|13|...|27| (Board Slots)
    |28|29|30|31|32|33|34|35|36| (Bench Slots)
    |37|38|39|40|41|42|43|44|45|46| (Item Bench Slots)
    |47|48|49|50|51| (Shop Slots)
    |52| (Pass)
    |53| (Level)
    |54| (Refresh)
    
    38:
    0-27 -> Board Slots
    28-36 -> Bench Slots
    37 -> Sell Slot
    
    """
    
    col, index = action // 38, action % 38
    
    action = []

    # Board and Bench slots
    if col < 37:
        from_loc = col

        if index == 37:
            action = [4, from_loc, 0]
        else:
            to_loc = index
            action = [5, from_loc, to_loc]
            
    # Item Bench Slots
    elif col < 47:
        from_loc = col - 37
        to_loc = index
        action = [6, from_loc, to_loc]
        
    # Shop Slots
    elif col < 52:
        action = [3, col - 47, 0]
        
    # Pass Action
    elif col == 52:
        action = [0, 0, 0]
    
    # Level Action
    elif col == 53:
        action = [1, 0, 0]
    
    # Refresh Action
    elif col == 54:
        action = [2, 0, 0]
        
    return action
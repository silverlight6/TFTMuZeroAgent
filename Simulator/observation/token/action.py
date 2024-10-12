import numpy as np
from gymnasium.spaces import MultiDiscrete

from Simulator.observation.interface import ActionBase, ActionVectorBase
from Simulator.observation.util import Util

class ActionToken(ActionBase, ActionVectorBase):
    def __init__(self, player):
        super().__init__()
        self.player = player
        
        self.util = Util()

        # --- Action Mask --- #
        self.exp_mask = self.create_exp_action_mask(player)
        self.refresh_mask = self.create_refresh_action_mask(player)
        self.buy_mask = self.create_buy_action_mask(player)
        self.move_sell_board_mask, self.move_sell_bench_mask = self.create_move_and_sell_action_mask(
            player)
        self.item_mask = self.create_item_action_mask(player)
        
    @staticmethod
    def action_space():
        """
        v2 Action Space is an 55x38 Dimension MultiDiscrete Tensor to keep my sanity
        
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
        
        """
        return MultiDiscrete([55, 38])
    
    @staticmethod
    def action_space_to_action(action):
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

    def fetch_action_mask(self):
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
        
        action_mask = np.zeros((55, 38), dtype=np.float32)
        
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

    def update_action_mask(self, action):
        action_type, x1, x2 = action
        
        # Pass Action
        if action_type == 0:
            ...
        
        # Level Action
        elif action_type == 1:
            self.update_exp_action(action)

        # Refresh Action
        elif action_type == 2:
            self.update_refresh_action(action)

        # Buy Action
        elif action_type == 3:
            self.update_buy_action(action)

        # Sell Action
        elif action_type == 4:
            self.update_move_sell_action(action)

        # Move/Sell Action
        elif action_type == 5:
            self.update_move_sell_action(action)

        # Item action
        elif action_type == 6:
            self.update_item_action(action)
        
        else:
            print(f"Action Type is invalid: {action}")
        
    def update_game_round(self):
        """Update vectors and masks after game round.

        After a battle, your hp, gold, exp, and shop changes
        Round might be a carousel/minion round, so just update everything just in case
        """
        self.move_sell_board_mask, self.move_sell_bench_mask = self.create_move_and_sell_action_mask(self.player)
        self.item_mask = self.create_item_action_mask(self.player)

    def update_exp_action(self, action):
        """Update vectors and masks related to the exp action.

        Updated Values:
            - Gold, Exp, Level
            - If level up, then bench masks need to be updated

        """
        self.exp_mask = self.create_exp_action_mask(self.player)
        self.refresh_mask = self.create_refresh_action_mask(self.player)
        self.buy_mask = self.create_buy_action_mask(self.player)
        self.move_sell_board_mask, self.move_sell_bench_mask = self.create_move_and_sell_action_mask(self.player)

    def update_refresh_action(self, action):
        """Update vectors and masks related to the refresh action.

        Updated Values:
            - Shop
        """
        self.exp_mask = self.create_exp_action_mask(self.player)
        self.refresh_mask = self.create_refresh_action_mask(self.player)
        self.buy_mask = self.create_buy_action_mask(self.player)

    def update_buy_action(self, action):
        """Update vectors and masks related to the buy action.

        Updated Values:
            - Shop, Bench, Gold
        """
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
        self.item_mask = self.create_item_action_mask(self.player)

    def create_exp_action_mask(self, player):
        """Create exp action mask

        Invalid:
            - Player is max level
            - Player has less than 4 gold

        Exp Action Vector: (1)
        """

        exp_action_mask = 0

        if player.gold >= player.exp_cost and player.level != player.max_level:
            exp_action_mask = 1

        # print(f"player.gold --> {player.gold} and exp_mask --> {exp_action_mask}")
        return exp_action_mask

    def create_refresh_action_mask(self, player):
        """Create refresh action mask

        Invalid:
            - Player has less than 2 gold

        Refresh Action Vector: (1)
        """

        refresh_action_mask = 1

        if player.gold < 2:
            refresh_action_mask = 0

        return refresh_action_mask

    def create_buy_action_mask(self, player):
        """Create buy action mask

        Invalid:
            - Player has no champions in the shop
            - Player has no room on their bench
            - Player doesn't have enough gold to buy the champion

        Buy Action Vector: (5)
        """

        buy_action_mask = np.zeros(5, dtype=np.float32)

        if player.bench_full() or player.shop_empty():
            return buy_action_mask

        for i, champion in enumerate(player.shop_champions):
            champion_cost = self.util.get_champion_cost(champion)

            if champion and player.gold >= champion_cost:
                buy_action_mask[i] = 1

        return buy_action_mask

    def create_move_and_sell_action_mask(self, player):
        """Create move and sell action masks

        Invalid:
            - No champion is selected for both from and to
            - Board to bench when is invalid champion (Sandguard, Dummy)
            - Bench to board when board is full
            - Bench to bench

        0-27 -> Board Slots
        28-37 -> Bench Slots
        38 -> Sell Slot

        Move Board Action Vector: (7, 4, 38)
        Move Bench Action Vector: (9, 38)
        """

        move_sell_board_action_mask = np.zeros((7, 4, 38), dtype=np.float32)
        move_sell_bench_action_mask = np.zeros((9, 38), dtype=np.float32)

        # --- Utility Masks --- #
        max_items = max(player.item_bench.count(None), 3)

        invalid_champion_mask = np.ones(28, dtype=np.float32)
        board_champion_mask = np.zeros(28, dtype=np.float32)
        bench_to_bench_mask = np.zeros(9, dtype=np.float32)

        default_board_mask = np.ones(38, dtype=np.float32)
        invalid_board_mask = np.concatenate([np.ones(28, dtype=np.float32), np.zeros(9, dtype=np.float32),
                                             np.zeros(1, dtype=np.float32)])

        # --- Board Mask --- #
        for x in range(len(player.board)):
            for y in range(len(player.board[x])):
                if champion := player.board[x][y]:
                    # Update utility masks to be used for bench mask
                    board_champion_mask[x * 4 + y] = 1
                    if champion.target_dummy or champion.overlord:
                        invalid_champion_mask[x * 4 + y] = 0

                        # Update board mask if champion is on the board
                        move_sell_board_action_mask[x][y] = invalid_board_mask
                    else:
                        move_sell_board_action_mask[x][y] = default_board_mask

                        # Update sell mask if items won't overflow item bench
                        if len(champion.items) > max_items:
                            move_sell_board_action_mask[x][y][38] = 0
                # Can't move to same location
                move_sell_board_action_mask[x][y][x * 4 + y] = 0

        # --- Bench Mask --- #
        # When board is not full, all board indices are valid
        # Except invalid champions (Sandguard, Dummy)
        if player.num_units_in_play < player.max_units:
            board_mask = np.ones(28, dtype=np.float32) * invalid_champion_mask
        else:
            board_mask = board_champion_mask * invalid_champion_mask

        bench_mask = np.concatenate([board_mask, bench_to_bench_mask, np.ones(1, dtype=np.float32)])

        for idx, champion in enumerate(player.bench):
            if champion:
                move_sell_bench_action_mask[idx] = bench_mask
                
                # Can't move to same location
                move_sell_bench_action_mask[idx][28 + idx] = 0

                # Update sell mask if items won't overflow item bench
                if len(champion.items) > max_items:
                    move_sell_bench_action_mask[idx][38] = 0

        return move_sell_board_action_mask, move_sell_bench_action_mask

    def create_item_action_mask(self, player):
        """Create item action mask

        Invalid:
            - No champion is selected
            - Champion is an invalid champion (Sandguard, Dummy)
            - Champion has 3 items
            - Item is full item and champion has 2 full items and 1 component item
            - Kayn Item on a champion that is not Kayn
            - Reforge Item on a champion with no items
            - Duplicator when board is full

        Item Action Vector: (10, 38); 10 item slots, 38 will always be 0
        """

        item_action_mask = np.zeros((10, 38), dtype=np.float32)

        valid_component_mask = np.zeros(38, dtype=np.float32)
        valid_full_item_mask = np.zeros(38, dtype=np.float32)

        valid_theives_gloves_mask = np.zeros(38, dtype=np.float32)
        valid_glove_mask = np.zeros(38, dtype=np.float32)

        valid_kayn_mask = np.zeros(38, dtype=np.float32)

        valid_reforge_mask = np.zeros(38, dtype=np.float32)
        valid_duplicator_mask = np.zeros(38, dtype=np.float32)
        
        # Oh my god what a pain...
        trait_mask = {
            trait: np.ones(38) for trait in self.util.item_traits.values()
        }

        def update_masks(champion, idx):
            if champion.target_dummy or champion.overlord:
                return

            # Valid component and full item
            if len(champion.items) == 3 and self.util.is_item_component(champion.items[2]):
                valid_component_mask[idx] = 1
            elif len(champion.items) < 3:
                valid_component_mask[idx] = 1
                valid_full_item_mask[idx] = 1
                valid_glove_mask[idx] = 1

            # Valid Glove
            if len(champion.items) > 0 and champion.items[-1] == 'sparring_gloves':
                valid_glove_mask[idx] = 0

            # Valid Theives Gloves
            if len(champion.items) == 0:
                valid_theives_gloves_mask[idx] = 1

            # Valid Kayn
            if champion.kayn_form:
                valid_kayn_mask[idx] = 1

            # Valid Reforge
            if len(champion.items) > 0:
                valid_reforge_mask[idx] = 1

            # Valid Duplicator
            valid_duplicator_mask[idx] = 1
            
            # Has trait
            for trait in champion.origin:
                if trait in trait_mask:
                    trait_mask[trait][idx] = 0

        for x in range(len(player.board)):
            for y in range(len(player.board[x])):
                if champion := player.board[x][y]:
                    idx = x * 4 + y
                    update_masks(champion, idx)

        for idx, champion in enumerate(player.bench):
            if champion:
                idx += 28
                update_masks(champion, idx)

        # If bench is full, remove valid duplicator
        if player.bench_full():
            valid_duplicator_mask = np.zeros(38, dtype=np.float32)

        for idx, item in enumerate(player.item_bench):
            if item:
                # Edge cases first
                # Is glove
                if item == 'sparring_gloves':
                    item_action_mask[idx] = valid_glove_mask

                # Is theives gloves
                elif item == 'theives_gloves':
                    item_action_mask[idx] = valid_theives_gloves_mask

                # Is kayn
                elif item.startswith('kayn'):
                    item_action_mask[idx] = valid_kayn_mask

                # Is reforge
                elif item == 'reforger':
                    item_action_mask[idx] = valid_reforge_mask

                # Is duplicator
                elif item == 'champion_duplicator':
                    item_action_mask[idx] = valid_duplicator_mask
                    
                # Trait item
                elif trait := self.util.get_item_trait(item):
                    item_action_mask[idx] = valid_full_item_mask * trait_mask[trait]

                # Regular cases
                elif self.util.is_full_item(item):
                    item_action_mask[idx] = valid_full_item_mask

                elif self.util.is_item_component(item):
                    item_action_mask[idx] = valid_component_mask

        return item_action_mask

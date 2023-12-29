import numpy as np
import config

from Simulator.observation.interface import ObservationBase, ObservationUpdateBase
from Simulator.stats import COST
from Simulator.origin_class import team_traits
from Simulator.config import MAX_BENCH_SPACE, BENCH_SIZE
from Simulator.utils import item_binary_encode, champ_binary_encode
from Simulator.item_stats import item_builds, uncraftable_items
from Simulator.origin_class_stats import tiers


class ObservationVector(ObservationBase, ObservationUpdateBase):
    """Observation object that stores the observation for a player."""

    def __init__(self, player):
        super().__init__()

        self.player = player

        # --- Public Scalars --- # 
        self.public_scalars = self.create_public_scalars(player)

        # --- Private Scalars --- #
        self.private_scalars = self.create_private_scalars(player)

        # --- Public Vectors --- #
        self.board_vector = self.create_board_vector(player)
        self.bench_vector = self.create_bench_vector(player)
        self.item_bench_vector = self.create_item_bench_vector(player)
        self.trait_vector = self.create_trait_vector(player)

        # --- Private Vectors --- #
        self.shop_vector = self.create_shop_vector(player)

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

        return {
            "scalars": np.concatenate([
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
            board: board token
            bench: bench token
            items: item token
            traits: trait token

        Commenting out the bench and items to save on storage space. Can comment back in a future iteration
        """
        return {
            "scalars": self.public_scalars,
            "board": self.board_vector,
            # "bench": self.bench_vector,
            # "items": self.item_bench_vector,
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
        
    def update_observation(self, action):
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
        `refresh_all_shops` calls `update_refresh_action`, so no use calling those twice
        Round might be a carousel/minion round, so just update everything just in case
        """
        self.public_scalars = self.create_public_scalars(self.player)
        self.private_scalars = self.create_private_scalars(self.player)

        self.board_vector = self.create_board_vector(self.player)
        self.bench_vector = self.create_bench_vector(self.player)
        self.item_bench_vector = self.create_item_bench_vector(self.player)
        self.trait_vector = self.create_trait_vector(self.player)

    def update_exp_action(self, action):
        """Update vectors and masks related to the exp action.

        Updated Values:
            - Gold, Exp, Level
            - If level up, then bench masks need to be updated

        """
        self.public_scalars = self.create_public_scalars(self.player)
        self.private_scalars = self.create_private_scalars(self.player)

    def update_refresh_action(self, action):
        """Update vectors and masks related to the refresh action.

        Updated Values:
            - Shop
        """
        self.public_scalars = self.create_public_scalars(self.player)
        self.private_scalars = self.create_private_scalars(self.player)

        self.shop_vector = self.create_shop_vector(self.player)

    def update_buy_action(self, action):
        """Update vectors and masks related to the buy action.

        Updated Values:
            - Shop, Bench, Gold
        """
        self.public_scalars = self.create_public_scalars(self.player)
        self.private_scalars = self.create_private_scalars(self.player)

        self.board_vector = self.create_board_vector(self.player)
        self.bench_vector = self.create_bench_vector(self.player)
        self.shop_vector = self.create_shop_vector(self.player)
        self.trait_vector = self.create_trait_vector(self.player)

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

    def update_item_action(self, action):
        """Update vectors and masks related to the item action.

        Updated Values:
            - Bench, Item Bench, Gold
        """
        self.private_scalars = self.create_private_scalars(self.player)
        self.public_scalars = self.create_public_scalars(self.player)

        self.board_vector = self.create_board_vector(self.player)
        self.bench_vector = self.create_bench_vector(self.player)
        self.trait_vector = self.create_trait_vector(self.player)
        self.item_bench_vector = self.create_item_bench_vector(self.player)
        
    # --- Implementation of ObservationVectorBase --- #
    def create_game_scalars(self, player):
        ...

    def create_public_scalars(self, player):
        """Create public scalars for a player

        Public Scalars:
            - health: int
            - level: int
            - win streak: int
            - loss streak: int
            - max units: int
            - available units: int
            - economy: int 
        """

        streak_lvl = 0
        if player.win_streak == 4:
            streak_lvl = 0.5
        elif player.win_streak >= 5:
            streak_lvl = 1

        return np.array([
            player.health / 100,
            player.level / 10,
            player.max_units,
            player.max_units - player.num_units_in_play,
            player.win_streak,
            player.loss_streak,
            streak_lvl,
            min(player.gold // 10, 5)
        ])

    def create_private_scalars(self, player):
        """Create private scalars for a player

        Private Scalars:
            - exp: int
            - exp to next level: int
            - gold: int
        """
        exp_to_level = 0
        if player.level < player.max_level:
            exp_to_level = (player.level_costs[player.level] - player.exp) / player.level_costs[player.level]
        match_history = np.zeros(3)
        if len(player.match_history) > 2:
            match_history = player.match_history[-3::]
        # Who we can play against in the next round. / 20 to keep numbers between 0 and 1.
        # TODO: FIX THIS
        opponent_options = np.zeros(8)
        for x in range(8):
            if ("player_" + str(x)) in player.opponent_options:
                opponent_options[x] = player.opponent_options["player_" + str(x)] / 20
            else:
                opponent_options[x] = -1

        return_array = np.array([
            player.gold / 100,
            player.exp / 100,
            player.round / 30,
            exp_to_level,
            max(player.win_streak, player.loss_streak) / 30,
            player.actions_remaining
        ])

        return np.concatenate([return_array, match_history, opponent_options], axis=-1)

    # -- Champion -- #
    def create_champion_vector(self, champion):
        """Create a champion token for a champion
        
        Champion Vector:

        championID: (shape: 6)
        stars: (shape: 1)
        cost: (shape, 1)

        items: (shape: 6) [
            itemID, itemID, itemID
        ]

        """
        champion_info_array = np.zeros(config.CHAMP_ENCODING_SIZE, dtype=np.float32)
        c_index = list(COST.keys()).index(champion.name)
        champion_info_array[0:6] = champ_binary_encode(c_index)
        champion_info_array[6] = champion.stars / 3
        champion_info_array[7] = champion.cost / 5
        for ind, item in enumerate(champion.items):
            start = (ind * 6) + 7
            finish = start + 6
            i_index = []
            if item in uncraftable_items:
                i_index = list(uncraftable_items).index(item) + 1
            elif item in item_builds.keys():
                i_index = list(item_builds.keys()).index(item) + 1 + len(uncraftable_items)
            champion_info_array[start:finish] = item_binary_encode(i_index)

        # Create vectors
        return champion_info_array

    def create_board_vector(self, player):
        """Create a board token for a player

        Board Vector: (28, champion_vector_length)

        Array board layout
                        Left
            | (0,0) (0,1) (0,2) (0,3) |
            | (1,0) (1,1) (1,2) (1,3) |
            | (2,0) (2,1) (2,2) (2,3) |
    Bottom  | (3,0) (3,1) (3,2) (3,3) |  Top
            | (4,0) (4,1) (4,2) (4,3) |
            | (5,0) (5,1) (5,2) (5,3) |
            | (6,0) (6,1) (6,2) (6,3) |
                        Right

        Rotated to match the board in game
                                Top
        | (0, 3) (1, 3) (2, 3) (3, 3) (4, 3) (5, 3) (6, 3) |
  Left  | (0, 2) (1, 2) (2, 2) (3, 2) (4, 2) (5, 2) (6, 2) |
        | (0, 1) (1, 1) (2, 1) (3, 1) (4, 1) (5, 1) (6, 1) |  Right
        | (0, 0) (1, 0) (2, 0) (3, 0) (4, 0) (5, 0) (6, 0) |
                                Bottom

        """
        board_vector = np.zeros(728, dtype=np.float32)

        player.team_champion_labels[:][0] = 1
        player.team_champion_labels[:][1] = 0
        for y in range(0, 4):
            # IMPORTANT TO HAVE THE X INSIDE -- Silver is not sure why but ok.
            for x in range(0, 7):
                # when using binary encoding (6 champ + stars + cost + 3 * 6 item) = 26
                champion_info_array = np.zeros(6 * 4 + 2, dtype=np.float32)
                if player.board[x][y]:
                    champion_info_array = self.create_champion_vector(player.board[x][y])

                    c_index = list(COST.keys()).index(player.board[x][y].name)
                    # create the label for the champion to help with training
                    if c_index < len(config.CHAMPION_ACTION_DIM):
                        player.team_champion_labels[c_index - 1][0] = 0
                        player.team_champion_labels[c_index - 1][1] = 1

                # Fit the area into the designated spot in the token
                board_vector[x * 4 + y:x * 4 + y + 26] = champion_info_array
                    
        return board_vector
    
    # -- Bench Vector -- #
    def create_bench_vector(self, player):
        """Create a bench token for a player

        Bench Vector: (9, champion_vector_length)

        Array bench layout
            | (0) (1) (2) (3) (4) (5) (6) (7) (8) |

        """
        bench = np.zeros(BENCH_SIZE * config.CHAMP_ENCODING_SIZE, dtype=np.float32)
        for x_bench in range(BENCH_SIZE):
            # when using binary encoding (6 champ  + stars + chosen + 3 * 6 item) = 26
            champion_info_array = np.zeros(config.CHAMP_ENCODING_SIZE, dtype=np.float32)
            if player.bench[x_bench]:
                champion_info_array = self.create_champion_vector(player.bench[x_bench])

            bench[x_bench * config.CHAMP_ENCODING_SIZE:
                  x_bench * config.CHAMP_ENCODING_SIZE + config.CHAMP_ENCODING_SIZE] = champion_info_array
        bench_vector = bench

        return bench_vector
    
    # -- Shop Vector -- #
    def create_shop_vector(self, player):
        """Create shop token for a player
        
        Shop Vector: (45)
        
        Array Shop layout
            | (0) (1) (2) (3) (4) |

        """

        output_array = np.zeros(45)
        shop_chosen = False
        chosen_shop_index = -1
        chosen_shop = ''
        shop = player.shop
        for x in range(0, len(shop)):
            input_array = np.zeros(8)
            if shop[x]:
                chosen = 0
                if shop[x].endswith("_c"):
                    chosen_shop_index = x
                    chosen_shop = shop[x]
                    c_shop = shop[x].split('_')
                    shop[x] = c_shop[0]
                    chosen = 1
                    shop_chosen = c_shop[1]

                i_index = list(COST.keys()).index(shop[x])
                # This should update the item name section of the token
                for z in range(6, 0, -1):
                    if i_index > 2 ** (z - 1):
                        input_array[6 - z] = 1
                        i_index -= 2 ** (z - 1)
                input_array[7] = chosen

            # Input chosen mechanics once I go back and update the chosen mechanics.
            output_array[8 * x: 8 * (x + 1)] = input_array
        if shop_chosen:
            if shop_chosen == 'the':
                shop_chosen = 'the_boss'
            i_index = list(team_traits.keys()).index(shop_chosen)
            # This should update the item name section of the token
            for z in range(5, 0, -1):
                if i_index > 2 * z:
                    output_array[45 - z] = 1
                    i_index -= 2 * z
            shop[chosen_shop_index] = chosen_shop

        return output_array
    
    # -- Item Bench Vector -- #
    def create_item_bench_vector(self, player):
        """Create an item bench token for a player
        
        Item Bench Vector: (60,)

        Array bench layout
            | (0 * 6) (1 * 6) (2 * 6) (3 * 6) (4 * 6) (5 * 6) (6 * 6) (7 * 6) (8 * 6) (9 * 6) |

        """
        item_arr = np.zeros(MAX_BENCH_SPACE * 6, dtype=np.float32)
        for ind, item in enumerate(player.item_bench):
            item_info = np.zeros(6, dtype=np.float32)
            if item == 'champion_duplicator' and player.bench_full():
                item_info = item_binary_encode(list(uncraftable_items).index(item) + 1)
            elif item in uncraftable_items:
                item_info = item_binary_encode(list(uncraftable_items).index(item) + 1)
            elif item in item_builds.keys():
                item_info = item_binary_encode(list(item_builds.keys()).index(item) + 1 + len(uncraftable_items))
            item_arr[ind * 6:ind * 6 + 6] = item_info
        return item_arr
    
    def create_trait_vector(self, player):
        current_position = 0
        tiers_vector = np.zeros(config.TIERS_FLATTEN_LENGTH, dtype=np.float32)
        base_tier_values = list(tiers.values())
        player_tier_values = list(player.team_tiers.values())
        for i in range(len(base_tier_values)):
            try:
                tiers_vector[current_position + player_tier_values[i]] = 1
                player.team_tier_labels[i] = np.zeros(config.TEAM_TIERS_VECTOR[i], dtype=np.float32)
                player.team_tier_labels[i][player_tier_values[i]] = 1
                current_position += len(base_tier_values[i]) + 1
            except IndexError:
                print("index i {} with player_tier_values[i] {}".format(i, player_tier_values[i]))
                print("team_tier_labels {}".format(player.team_tier_labels))

        chosen_vector = np.zeros(5, dtype=np.float32)
        if player.chosen:
            i_index = list(player.team_composition.keys()).index(player.chosen)
            # This should update the item name section of the token
            for z in range(5, 0, -1):
                if i_index > 2 * z:
                    chosen_vector[5 - z] = 1
                    i_index -= 2 * z
        tiers_vector = np.concatenate([tiers_vector, chosen_vector], axis=-1)
        return tiers_vector

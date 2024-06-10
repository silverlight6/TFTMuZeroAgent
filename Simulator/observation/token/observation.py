import abc
import numpy as np

from Simulator.observation.util import Util
from Simulator.observation.normalization import Normalizer

from Simulator.observation.interface import ObservationBase, ObservationUpdateBase

class ObservationToken(ObservationBase, ObservationUpdateBase):
    """Observation object that stores the observation for a player."""

    def __init__(self, player):
        super().__init__()

        self.player = player

        self.util = Util()
        self.normalizer = Normalizer()
        
        # -- Items -- #
        self.item_vector_length = 1
        
        # -- Traits -- #
        self.trait_champion_vector_length = 7
        self.trait_player_vector_length = len(self.util.trait_ids)
        
        # -- Champions -- #
        # Create ids for champions
        self.stat_vector_length = 12
        # 2 for chosen, 4 for stars, 1 for cost
        self.other_stat_vector_length = 2 + 4 + 1

        # championID, items, origins, stats
        self.champion_vector_length = 1 + self.item_vector_length * 3 \
            + self.trait_champion_vector_length + self.stat_vector_length + self.other_stat_vector_length

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
            board: board token
            bench: bench token
            items: item token
            traits: trait token
        """
        return {
            "scalars": self.public_scalars,
            "board": self.board_vector,
            "bench": self.bench_vector,
            "items": self.item_bench_vector,
            "traits": self.trait_vector,
        }

    def fetch_player_position_observation(self):
        """Fetch player position observation"""
        ...

    def fetch_dead_observation(self):
        """Zero out public observations for all dead players"""
        return {
            "scalars": np.zeros(self.public_scalars.shape, dtype=np.float32),
            "board": np.zeros(self.board_vector.shape, dtype=np.float32),
            "bench": np.zeros(self.bench_vector.shape, dtype=np.float32),
            "shop": np.zeros(self.shop_vector.shape, dtype=np.float32),
            "items": np.zeros(self.item_bench_vector.shape, dtype=np.float32),
            "traits": np.zeros(self.trait_vector.shape, dtype=np.float32),
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
        self.game_scalars = self.create_game_scalars(self.player)
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
        self.game_scalars = self.create_game_scalars(self.player)
        self.public_scalars = self.create_public_scalars(self.player)
        self.private_scalars = self.create_private_scalars(self.player)

    def update_refresh_action(self, action):
        """Update vectors and masks related to the refresh action.

        Updated Values:
            - Shop
        """
        self.game_scalars = self.create_game_scalars(self.player)
        self.public_scalars = self.create_public_scalars(self.player)
        self.private_scalars = self.create_private_scalars(self.player)

        self.shop_vector = self.create_shop_vector(self.player)

    def update_buy_action(self, action):
        """Update vectors and masks related to the buy action.

        Updated Values:
            - Shop, Bench, Gold
        """
        self.game_scalars = self.create_game_scalars(self.player)
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
        self.game_scalars = self.create_game_scalars(self.player)
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
        self.game_scalars = self.create_game_scalars(self.player)
        self.private_scalars = self.create_private_scalars(self.player)
        self.public_scalars = self.create_public_scalars(self.player)

        self.board_vector = self.create_board_vector(self.player)
        self.bench_vector = self.create_bench_vector(self.player)
        self.trait_vector = self.create_trait_vector(self.player)
        self.item_bench_vector = self.create_item_bench_vector(self.player)
        
    # --- Implementation of ObservationVectorBase --- #
    def create_game_scalars(self, player):
        """Create game scalars for a player

        Game Scalars:
            - Round: int
            - Actions remaining: int
            - opponents: list of ints
            - Action History?: [[int, int, int], [int, int, int], ...]  # TODO
        """

        opponentIDs = np.zeros(3)
        count = 0
        for playerID, will_fight in player.opponent_options.items():
            if will_fight == 1 and count < 3:
                playerID = int(playerID[-1])
                opponentIDs[count] = playerID
                count += 1

        return np.array([
            player.round,
            player.actions_remaining,
            *opponentIDs,
            # TODO: Action History
        ])

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
        return np.array([
            player.player_num,
            player.health,
            player.level,
            player.win_streak,
            player.loss_streak,
            player.max_units,
            player.max_units - player.num_units_in_play,
            min(player.gold // 10, 5)
        ])

    def create_private_scalars(self, player):
        """Create private scalars for a player

        Private Scalars:
            - exp: int
            - exp to next level: int
            - gold: int
        """
        return np.array([
            player.exp,
            player.level_costs[player.level] - player.exp,
            player.gold,
        ], dtype=np.float32)

    # -- Champion -- #
    def create_champion_vector(self, champion):
        """Create a champion token for a champion
        
        Champion Vector:

        championID: int

        items: (shape: 3) [
            itemID, itemID, itemID
        ]
        
        origins: (shape: 7) [
            traitID, traitID, traitID # 3 possible traits from champion
            traitID, traitID, traitID # 3 possible traits from items
            traitID # 1 possible trait from chosen
        ]
        
        other stats: [
            chosen, stars, cost
        ]
        
        stats: (shape: 12) [
            AD, crit_chance, crit_damage, armor, MR, dodge, health, mana, AS, SP, maxmana, range
        ]
        """
        championID = self.util.get_champion_id(champion)
        
        # Item token
        item_ids = np.zeros(self.item_vector_length * 3, dtype=np.float32)
        item_modifiers = []

        # Origin Vector
        origin_ids = np.zeros(self.trait_champion_vector_length, dtype=np.float32)
        origins = champion.origin.copy()
        
        # Stats Vector
        stats = self.get_champion_stats(champion)

        # -- Items -- #
        for i, item in enumerate(champion.items):
            itemID = self.util.get_item_id(item)
            stat_modifiers = self.get_item_modifiers(item)
            
            item_ids[i] = itemID
            item_modifiers.append(stat_modifiers)
            
            if item_trait := self.util.get_item_trait(item):
                origins.append(item_trait)
                
        # -- Origins -- #
        if champion.chosen:
            origins.append(champion.chosen)
            
        for i, origin in enumerate(origins):
            traitID = self.util.get_trait_id(origin)
            origin_ids[i] = traitID
            
        # -- Stats -- #
        # Apply item modifiers
        for item_mod in item_modifiers:
            for stat, value in item_mod.items():
                stats[stat] += value
                
        # -- Other Stats -- #
        chosen = self.util.chosen_one_hot(champion.chosen)
        stars = self.util.stars_one_hot(champion.stars)
        cost = self.util.get_champion_cost(champion)
        
        # Create vectors
        return np.concatenate([
            np.array([championID]),
            item_ids,
            origin_ids,
            chosen,
            stars,
            np.array([cost]),
            np.array(list(stats.values()))
        ], dtype=np.float32)
        
    # -- Champion Util -- #
    def get_champion_stats(self, champion) -> dict:
        return {
            "AD": champion.AD,
            "crit_chance": champion.crit_chance,
            "crit_damage": champion.crit_damage,
            "armor": champion.armor,
            "MR": champion.MR,
            "dodge": champion.dodge,
            "health": champion.health,
            "mana": champion.mana,
            "AS": champion.AS,
            "SP": champion.SP,
            "maxmana": champion.maxmana,
            "range": champion.range
        }
        
    def get_item_modifiers(self, item) -> dict:
        item_stats = self.util.get_item_stats(item)

        stat_modifiers = {
            "AD": 0,
            "crit_chance": 0,
            "crit_damage": 0,
            "armor": 0,
            "MR": 0,
            "dodge": 0,
            "health": 0,
            "mana": 0,
            "AS": 0,
            "SP": 0,
        }

        for stat, value in item_stats.items():
            stat_modifiers[stat] = value

        # Special case for AS, as it is a percentage
        if stat_modifiers["AS"] > 0:
            stat_modifiers["AS"] -= 1

        # Edge cases for bloodthirster, guardian angel, and dragon's claw
        stat_modifiers.pop("lifesteal", None)
        stat_modifiers.pop("will_revive", None)
        stat_modifiers.pop("spell_damage_reduction_percentage", None)
        
        return stat_modifiers
        
    def apply_champion_normalization(self, champion_vectors):
        """Apply normalization to a champion token
        
        0: ID
        1-3: itemIDs
        4-10: originIDs
        11-22: stats
        """
        champion_vectors[:, 11:23] = self.normalizer.apply_champion_normalization(champion_vectors[:, 11:23])
            
        return champion_vectors

    # -- Board Vector -- #
    def get_board_vector_location(self, x, y):
        """Get the location of a champion on the board token"""
        return x * 4 + y

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

        board_vector = np.zeros((28, self.champion_vector_length), dtype=np.float32)
        
        for x in range(len(player.board)):
            for y in range(len(player.board[x])):
                if champion := player.board[x][y]:
                    champion_vector = self.create_champion_vector(champion)

                    loc = self.get_board_vector_location(x, y)
                    board_vector[loc] = champion_vector
                    
        return self.apply_champion_normalization(board_vector)
    
    # -- Bench Vector -- #
    def create_bench_vector(self, player):
        """Create a bench token for a player

        Bench Vector: (9, champion_vector_length)

        Array bench layout
            | (0) (1) (2) (3) (4) (5) (6) (7) (8) |

        """
        bench_vector = np.zeros((9, self.champion_vector_length), dtype=np.float32)
        
        for idx, champion in enumerate(player.bench):
            if champion:
                champion_vector = self.create_champion_vector(champion)
                bench_vector[idx] = champion_vector
                
        return self.apply_champion_normalization(bench_vector)
    
    # -- Shop Vector -- #
    def create_shop_vector(self, player):
        """Create shop token for a player
        
        Shop Vector: (5, champion_vector_length)
        
        Array Shop layout
            | (0) (1) (2) (3) (4) |

        """
        
        shop_vector = np.zeros((5, self.champion_vector_length), dtype=np.float32)

        for idx, champion in enumerate(player.shop_champions):
            if champion:
                champion_vector = self.create_champion_vector(champion)
                shop_vector[idx] = champion_vector
                
        return self.apply_champion_normalization(shop_vector)
    
    # -- Item Bench Vector -- #
    def create_item_bench_vector(self, player):
        """Create an item bench token for a player
        
        Item Bench Vector: (10,)

        Array bench layout
            | (0) (1) (2) (3) (4) (5) (6) (7) (8) (9) |

        """
        item_bench_vector = np.zeros(10, dtype=np.float32)
        
        for i, item in enumerate(player.item_bench):
            if item:
                item_bench_vector[i] = self.util.get_item_id(item)
                
        return item_bench_vector
    
    # -- Trait Vector -- #
    def compute_traits_from_board(self, champion_vectors):
        """Compute the total trait token from either board or bench
        
        Champion Vector:
        0: ID
        1-3: itemIDs
        4-10: originIDs # need this
        11-22: stats
        
        """
        traits = champion_vectors[:, 4:11].flatten()
        trait_vector = np.bincount(traits.astype(int), minlength=self.trait_player_vector_length + 1)
        
        # Remove the first element, as it is the count of empty traits
        trait_vector = trait_vector[1:]
        
        return trait_vector
    
    def create_trait_vector(self, player):
        return self.compute_traits_from_board(self.board_vector)

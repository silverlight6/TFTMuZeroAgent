import collections
import numpy as np
import config

from Simulator.stats import COST
from Simulator.origin_class import team_traits, game_comp_tiers
from Simulator.origin_class_stats import tiers
from Simulator.item_stats import items, trait_items, basic_items, item_builds


class Observation:
    """Observation object used to generate the observation for each player.

    Format:
    {
        "player": PlayerObservation
        "mask": (5, 11, 38) # Same as action space
        "opponents": [PlayerPublicObservation, PlayerPublicObservation, ...]
    }

    Game Values:
        - Round: int
        - Actions remaining: int
        - Action History?: [[int, int, int], [int, int, int], ...]

    Public Scalars:
        - health: int
        - level: int
        - win streak: int
        - loss streak: int
        - max units: int
        - available units: int

    Private Scalars:
        - exp: int
        - exp to next level: int
        - gold: int

    Public Vectors:
        - board vector: [champion vector, champion vector, ...]
        - bench vector: [champion vector, champion vector, ...]
        - item vector: [item vector, item vector, ...]
        - trait vector: [trait vector, trait vector, ...]

    Private Vectors:
        - shop vector: [champion vector, champion vector, ...]

    PlayerObservation:
        scalars: [Game Values, Public Scalars, Private Scalars]
        board: board vector
        bench: bench vector
        shop: shop vector
        items: item vector
        traits: trait vector

    PlayerPublicObservation:
        scalars: [Public Scalars]
        board: board vector
        bench: bench vector
        items: item vector
        traits: trait vector

    Champion Vector:
        item1: item vector
        item2: item vector
        item3: item vector

        origin1: trait vector
        origin2: trait vector
        origin3: trait vector
    """

    def __init__(self):
        # -- Items -- #
        # Create ids for items
        self.item_ids = {k: idx for idx, k in enumerate(items.keys())}
        # Reverse trait_items dictionary to get the trait from the item
        self.item_traits = {v: k for k, v in trait_items.items()}
        self.item_vector_length = 10

        self.item_components = set(basic_items)
        self.full_items = set(item_builds.keys())

        # -- Tiers -- #
        # Create ids for tiers
        self.tier_ids = {k: idx for idx, k in enumerate(tiers.keys())}
        # 3 possible tiers from champion, 3 possible tiers from items, 1 possible tier from chosen
        self.tier_champion_vector_length = 7
        self.tier_player_vector_length = len(self.tier_ids) + 1

        # -- Champions -- #
        # Create ids for champions
        self.champion_ids = {k: idx for idx, k in enumerate(COST.keys())}
        self.stat_vector_length = 11
        # championID, items, origins, stats
        self.champion_vector_length = 1 + self.item_vector_length * \
            3 + self.tier_champion_vector_length + self.stat_vector_length

    # --- Observation Vectors --- #

    def create_game_scalars(self, player):
        """Create game scalars for a player

        Game Scalars:
            - Round: int
            - Actions remaining: int
            - Action History?: [[int, int, int], [int, int, int], ...] # TODO
        """

        return np.array([
            player.round,
            player.max_actions - player.actions_remaining,
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
        """

        return np.array([
            player.health,
            player.level,
            player.win_streak,
            player.loss_streak,
            player.max_units,
            self.max_units - self.num_units_in_play,
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
            self.level_costs[player.level] - player.exp,
            player.gold,
        ])

    def create_champion_vector(self, champion):
        """Create a champion vector for a champion

        Champion Vector:
        id: int
        items: [
            item1: item vector
            item2: item vector
            item3: item vector
        ]

        origins: [
            origin1: traitID # Champion
            origin2: traitID # Champion
            origin3: traitID # Champion
            origin4: traitID # Item
            origin5: traitID # Item
            origin6: traitID # Item
            origin7: traitID # Chosen
        ]

        stats: [
            AD: int
            crit_chance: int
            armor: int
            MR: int
            dodge: int
            health: int
            mana: int
            AS: int
            SP: int
            maxmana: int
            range: int
        ]
        """

        # Champion ID
        if champion.target_dummy:
            championID = len(self.champion_ids)
        else:
            championID = self.champion_ids[champion.name]

        # Items
        item_vectors = np.zeros((3, self.item_vector_length))
        item_modifiers = []

        # Origins
        origin_vector = np.zeros(self.tier_champion_vector_length)
        origins = champion.origins.copy()

        # Stats
        stats = {
            "AD": champion.AD,
            "crit_chance": champion.crit_chance,
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

        # -- Items -- #

        # Create item vectors and stat modifiers
        for idx, item in enumerate(champion.items):
            item_vector, stat_modifiers = self.create_item_vector(item)

            item_vectors[idx] = item_vector
            item_modifiers.append(stat_modifiers)

            # Add trait if item is a spatula item
            if item in self.item_traits:
                origins.append(self.item_traits[item])

        # -- Origins -- #

        if champion.chosen:
            origins.append(champion.chosen)

        for idx, origin in enumerate(origins):
            originID = self.tier_ids[origin] + 1  # 0 is reserved for no origin
            origin_vector[idx] = originID

        # -- Stats -- #

        # Add stat modifiers from items to champion stats
        for stat, modifier in item_modifiers.items():
            stats[stat] += modifier

        # Special case for mana, we want a ratio of current mana to max mana
        stats["mana"] = (stats["mana"] / stats["maxmana"]) * 100

        # Special case for AS, as it is a percentage
        stats["AS"] = stats["AS"] * 100

        stats_vector = np.array(list(stats.values()))

        champion_vector = np.concatenate(
            (item_vectors, origin_vector, stats_vector))

        champion_vector = np.append(championID, champion_vector)

        return champion_vector

    def create_item_vector(self, item):
        """Create an item vector for an item

        'items' is a dictionary of item names and their stat bonuses.
        We can use this to encode an item as its index in the dictionary
        and its stat bonuses as a vector.

        Item Vector: Shape (10)
            itemID: int
            AD: int
            crit_chance: int
            armor: int
            MR: int
            dodge: int
            health: int
            mana: int
            AS: int
            SP: int

        """
        item = items[item]  # item is a dictionary of stat bonuses
        itemID = self.item_ids[item]  # itemID is the index of the item

        stat_modifiers = {
            "AD": 0,
            "crit_chance": 0,
            "armor": 0,
            "MR": 0,
            "dodge": 0,
            "health": 0,
            "mana": 0,
            "AS": 0,
            "SP": 0,
        }

        stat_encoding = stat_modifiers.copy()

        for stat, value in item.items():
            stat_modifiers[stat] = value
            stat_encoding[stat] = value

        stat_encoding = np.array(list(stat_encoding.values()))

        # Special case for AS, as it is a percentage
        if stat_modifiers["AS"] > 0:
            stat_modifiers["AS"] -= 1

        item_vector = np.append(itemID, stat_encoding)

        return item_vector, stat_modifiers

    def create_board_vector(self, player):
        """Create a board vector for a player

        Board Vector: (7, 4, champion_vector_length)

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

        board_vector = np.zeros((7, 4, player.champion_vector_length))

        for x in player.board:
            for y in player.board[x]:
                if champion := player.board[x][y]:
                    champion_vector = player.create_champion_vector(champion)

                    board_vector[x][y] = champion_vector

        return board_vector

    def create_bench_vector(self, player):
        """Create a bench vector for a player

        Bench Vector: (9, champion_vector_length)

        | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
        """

        bench_vector = np.zeros((9, self.champion_vector_length))

        for idx, champion in enumerate(player.bench):
            if champion:
                champion_vector = self.create_champion_vector(champion)
                bench_vector[idx] = champion_vector

        return bench_vector

    def create_item_bench_vector(self, player):
        """Create a item bench vector for a player

        Item Bench Vector: (10, item_vector_length)

        | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
        """

        item_bench_vector = np.zeros((10, self.item_vector_length))

        for idx, item in enumerate(player.item_bench):
            if item:
                item_vector, _ = self.create_item_vector(item)
                item_bench_vector[idx] = item_vector

        return item_bench_vector

    def create_shop_vector(self, player):
        """Create shop vector for a player

        Shop Vector: (5, champion_vector_length)

        | 0 | 1 | 2 | 3 | 4 |
        """

        shop_vector = np.zeros((5, self.champion_vector_length))

        for idx, champion in enumerate(player.shop_champions):
            if champion:
                champion_vector = self.create_champion_vector(champion)
                shop_vector[idx] = champion_vector

        return shop_vector

    def create_trait_vector(self, player):
        """Create trait vector for a player

        Each trait is represented by traitID, and traitLevel
        """

        trait_vector = np.zeros((self.tier_player_vector_length, 2))

        for origin, id in self.tier_ids.items():
            trait_vector[id][0] = id
            trait_vector[id][1] = player.team_tiers[origin]

        return trait_vector.flatten()

    # --- Action Masking --- #

    def create_exp_action_mask(self, player):
        """Create exp action mask

        Invalid:
            - Player is max level
            - Player has less than 4 gold

        Exp Action Vector: (1)
        """

        exp_action_mask = 1

        if player.gold < player.exp_cost or player.level == player.max_level:
            exp_action_mask = 0

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

        buy_action_mask = np.zeros(5)

        if player.bench_full() or player.shop_empty():
            return buy_action_mask

        for i, champion in enumerate(player.shop_champions):
            if champion and player.gold >= champion.cost:
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

        move_sell_board_action_mask = np.zeros((7, 4, 38))
        move_sell_bench_action_mask = np.zeros((9, 38))

        # --- Utility Masks --- #
        invalid_champion_mask = np.ones(28)
        board_champion_mask = np.zeros(28)
        bench_to_bench_mask = np.zeros(9)

        default_board_mask = np.ones(38)
        invalid_board_mask = np.concatenate(
            (np.ones(28), np.zeros(9), np.zeros(1)))

        # --- Board Mask --- #
        for x in player.board:
            for y in player.board[x]:
                if champion := player.board[x][y]:
                    # Update utility masks to be used for bench mask
                    board_champion_mask[x * 4 + y] = 1
                    if champion.target_dummy or champion.overlord:
                        invalid_champion_mask[x * 4 + y] = 0

                    # Update board mask if champion is on the board
                        move_sell_board_action_mask[x][y] = invalid_board_mask
                    else:
                        move_sell_board_action_mask[x][y] = default_board_mask

        # --- Bench Mask --- #
        # When board is not full, all board indices are valid
        # Except invalid champions (Sandguard, Dummy)
        if player.num_units_in_play < player.max_units:
            board_mask = np.ones(28) * invalid_champion_mask
        else:
            board_mask = board_champion_mask * invalid_champion_mask

        bench_mask = np.append(board_mask, bench_to_bench_mask, np.ones(1))

        for idx, champion in enumerate(player.bench):
            if champion:
                move_sell_bench_action_mask[idx] = bench_mask

        return move_sell_bench_action_mask, move_sell_board_action_mask

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

        item_action_mask = np.zeros((10, 38))

        valid_component_mask = np.zeros(38)
        valid_full_item_mask = np.zeros(38)

        valid_theives_gloves_mask = np.zeros(38)
        valid_glove_mask = np.zeros(38)

        valid_kayn_mask = np.zeros(38)

        valid_reforge_mask = np.zeros(38)
        valid_duplicator_mask = np.zeros(38)

        def update_masks(champion, idx):
            if champion.target_dummy or champion.overlord:
                return

            # Valid component and full item
            if (len(champion.items) == 3 and champion.items[2] in self.item_components):
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

        for x in player.board:
            for y in player.board[x]:
                if champion := player.board[x][y]:
                    idx = x * 4 + y
                    update_masks(champion, idx)

        for idx, champion in enumerate(player.bench):
            if champion:
                idx += 28
                update_masks(champion, idx)

        # If bench is full, remove valid duplicator
        if player.bench_full():
            valid_duplicator_mask = np.zeros(38)

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

                # Regular cases
                elif item in self.full_items:
                    item_action_mask[idx] = valid_full_item_mask

                elif item in self.item_components:
                    item_action_mask[idx] = valid_component_mask

        return item_action_mask

    def minmaxnorm(self, X, min, max):
        """Helper function to normalize values between 0 and 1"""
        return (X - min) / (max - min)

    def clip(self, X, min, max):
        """Helper function to clip values between min and max"""
        return np.clip(X, min, max)

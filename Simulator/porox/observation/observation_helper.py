import numpy as np

from Simulator.stats import COST
from Simulator.origin_class_stats import tiers
from Simulator.item_stats import items, trait_items, basic_items, item_builds
from Simulator.pool_stats import cost_star_values

from Simulator.porox.observation.normalization import batch_apply_z_score_champion, batch_apply_z_score_item


class ObservationHelper:
    """Observation object used to generate the observation for each player.

    Format:
    {
        "player": PlayerObservation
        "mask": (5, 11, 38)  # Same as action space
        "opponents": [PlayerPublicObservation, PlayerPublicObservation, ...]
    }

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

    Champion Vector:
        item1: item vector
        item2: item vector
        item3: item vector

        origin1: champion trait
        origin2: champion trait
        origin3: champion trait
        origin4: item trait
        origin5: item trait
        origin6: item trait
        origin7: chosen trait

        championID: int

        AD: int
        crit_chance: int
        crit_damage: int
        armor: int
        MR: int
        dodge: int
        health: int
        mana: int
        AS: float
        SP: int
        maxmana: int
        range: int

    Item Vector:
        itemID: int

        AD: int
        crit_chance: int
        crit_damage: int
        armor: int
        MR: int
        dodge: int
        health: int
        mana: int
        AS: float
        SP: int

    """

    def __init__(self):
        # -- Items -- #
        # Create ids for items
        self.item_ids = {k: idx for idx, k in enumerate(items.keys())}
        # Reverse trait_items dictionary to get the trait from the item
        self.item_traits = {v: k for k, v in trait_items.items()}
        self.item_vector_length = 11

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
        self.stat_vector_length = 12
        # championID, items, origins, stats
        self.champion_vector_length = 1 + self.item_vector_length * \
            3 + self.tier_champion_vector_length + self.stat_vector_length

    # --- Observation Vectors --- #

    def create_game_scalars(self, player):
        """Create game scalars for a player

        Game Scalars:
            - Round: int
            - Actions remaining: int
            - Action History?: [[int, int, int], [int, int, int], ...]  # TODO
        """

        return np.array([
            player.round,
            # TODO: Add this
            # player.max_actions - player.actions_remaining,
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
            player.max_units - player.num_units_in_play,
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
        ])

    # -- Public Vectors -- #
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

        board_vector = np.zeros((7, 4, self.champion_vector_length))

        for x in range(len(player.board)):
            for y in range(len(player.board[x])):
                if champion := player.board[x][y]:
                    champion_vector = self.create_champion_vector(champion)

                    board_vector[x][y] = champion_vector

        return self.apply_champion_normalization(board_vector)

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

        return self.apply_champion_normalization(bench_vector)

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

        return self.apply_item_normalization(item_bench_vector)

    def create_trait_vector(self, player):
        """Create trait vector for a player

        Each trait is represented by traitID, and traitLevel
        """

        trait_vector = np.zeros((self.tier_player_vector_length, 2))

        for origin, id in self.tier_ids.items():
            trait_vector[id][0] = id
            trait_vector[id][1] = player.team_tiers[origin]

        return trait_vector.flatten()

    # -- Private Vectors -- #
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

        return self.apply_champion_normalization(shop_vector)

    # --- Champion and Item Vectors --- #

    def create_champion_vector(self, champion):
        """Create a champion vector for a champion

        Champion Vector:
        items: (shape: 3 * item_vector_length) [
            item1: item vector
            item2: item vector
            item3: item vector
        ]

        origins: (shape: 7) [
            origin1: traitID # Champion
            origin2: traitID # Champion
            origin3: traitID # Champion
            origin4: traitID # Item
            origin5: traitID # Item
            origin6: traitID # Item
            origin7: traitID # Chosen
        ]

        championID: int

        stats: (shape: stat_vector_length) [
            AD: int
            crit_chance: int
            crit_damage: int
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
        # TODO: make target_dummy have its own championID
        if champion.target_dummy:
            championID = len(self.champion_ids)
        else:
            championID = self.champion_ids[champion.name]

        # Items
        item_vectors = np.zeros((3, self.item_vector_length))
        item_modifiers = []

        # Origins
        origin_vector = np.zeros(self.tier_champion_vector_length)
        origins = champion.origin.copy()

        # Stats
        stats = {
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

        # -- Items -- #

        # Create item vectors and stat modifiers
        for idx, item in enumerate(champion.items):
            item_vector, stat_modifiers = self.create_item_vector(item)

            try:
                item_vectors[idx] = item_vector
            except:
                print(item)
                print(stat_modifiers)
                print(item_vector)
                raise

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
        for item_mod in item_modifiers:
            for stat, modifier in item_mod.items():
                stats[stat] += modifier

        stats_vector = np.array(list(stats.values()))

        champion_vector = np.concatenate([
            item_vectors.flatten(),
            origin_vector,
            np.array([championID]),
            stats_vector
        ])

        return champion_vector

    def create_item_vector(self, item_name):
        """Create an item vector for an item

        'items' is a dictionary of item names and their stat bonuses.
        We can use this to encode an item as its index in the dictionary
        and its stat bonuses as a vector.

        Item Vector: Shape (11)
            itemID: int

            AD: int
            crit_chance: int
            crit_damage: int
            armor: int
            MR: int
            dodge: int
            health: int
            mana: int
            AS: int
            SP: int

        """
        item_stats = items[item_name]  # item_stats is a dictionary of stat bonuses
        itemID = self.item_ids[item_name]  # itemID is the index of the item

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
            
        # Edge cases for bloodthirster, guardian angel, and dragon's claw
        stat_modifiers.pop("lifesteal", None)
        stat_modifiers.pop("will_revive", None)
        stat_modifiers.pop("spell_damage_reduction_percentage", None)

        stat_encoding = np.array(list(stat_modifiers.values()))


        # Special case for AS, as it is a percentage
        if stat_modifiers["AS"] > 0:
            stat_modifiers["AS"] -= 1

        item_vector = np.append(itemID, stat_encoding)

        return item_vector, stat_modifiers

    # --- Normalization --- #

    def apply_champion_normalization(self, champion_vectors):
        """
            0-10: item1: item vector
            11-21: item2: item vector
            22-32: item3: item vector

            33: origin1: champion trait
            34: origin2: champion trait
            35: origin3: champion trait
            36: origin4: item trait
            37: origin5: item trait
            38: origin6: item trait
            39: origin7: chosen trait

            40: championID: int

            41: AD: int
            42: crit_chance: int
            43: crit_damage: int
            44: armor: int
            45: MR: int
            46: dodge: int
            47: health: int
            48: mana: int
            49: AS: float
            50: SP: int
            51: maxmana: int
            52: range: int
        """

        original_shape = champion_vectors.shape

        champion_vectors = champion_vectors.reshape(
            -1, self.champion_vector_length)

        # --- Items --- #
        # Batch of item vectors
        item_vectors = champion_vectors[:, :33].reshape(-1, 3, 11)
        item_vectors_norm = self.apply_item_normalization(item_vectors, True)
        champion_vectors[:, :33] = item_vectors_norm.reshape(-1, 33)

        # Champion Stats --- #
        champion_vectors[:, 41] = batch_apply_z_score_champion(
            champion_vectors[:, 41], 'AD')
        champion_vectors[:, 42] = batch_apply_z_score_champion(
            champion_vectors[:, 42], 'crit_chance')
        champion_vectors[:, 43] = batch_apply_z_score_champion(
            champion_vectors[:, 43], 'crit_damage')
        champion_vectors[:, 44] = batch_apply_z_score_champion(
            champion_vectors[:, 44], 'armor')
        champion_vectors[:, 45] = batch_apply_z_score_champion(
            champion_vectors[:, 45], 'MR')
        champion_vectors[:, 46] = batch_apply_z_score_champion(
            champion_vectors[:, 46], 'dodge')
        champion_vectors[:, 47] = batch_apply_z_score_champion(
            champion_vectors[:, 47], 'health')
        champion_vectors[:, 48] = batch_apply_z_score_champion(
            champion_vectors[:, 48], 'mana')
        champion_vectors[:, 49] = batch_apply_z_score_champion(
            champion_vectors[:, 49], 'AS')
        champion_vectors[:, 50] = batch_apply_z_score_champion(
            champion_vectors[:, 50], 'SP')
        champion_vectors[:, 51] = batch_apply_z_score_champion(
            champion_vectors[:, 51], 'maxmana')
        champion_vectors[:, 52] = batch_apply_z_score_champion(
            champion_vectors[:, 52], 'range')

        return champion_vectors.reshape(original_shape)

    def apply_item_normalization(self, item_vectors, from_champ=False):
        """
            0: itemID: int
            1: AD: int
            2: crit_chance: int
            3: crit_damage: int
            4: armor: int
            5: MR: int
            6: dodge: int
            7: health: int
            8: mana: int
            9: AS: float
            10: SP: int

        """
        original_shape = item_vectors.shape
        
        item_vectors = item_vectors.reshape(
            -1, self.item_vector_length
        )

        item_vectors[:, 1] = batch_apply_z_score_item(item_vectors[:, 1], 'AD')
        item_vectors[:, 2] = batch_apply_z_score_item(
            item_vectors[:, 2], 'crit_chance')
        item_vectors[:, 3] = batch_apply_z_score_item(
            item_vectors[:, 3], 'crit_damage')
        item_vectors[:, 4] = batch_apply_z_score_item(
            item_vectors[:, 4], 'armor')
        item_vectors[:, 5] = batch_apply_z_score_item(item_vectors[:, 5], 'MR')
        item_vectors[:, 6] = batch_apply_z_score_item(
            item_vectors[:, 6], 'dodge')
        item_vectors[:, 7] = batch_apply_z_score_item(
            item_vectors[:, 7], 'health')
        item_vectors[:, 8] = batch_apply_z_score_item(
            item_vectors[:, 8], 'mana')
        item_vectors[:, 9] = batch_apply_z_score_item(item_vectors[:, 9], 'AS')
        item_vectors[:, 10] = batch_apply_z_score_item(
            item_vectors[:, 10], 'SP')

        return item_vectors.reshape(original_shape)

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
            champion_cost = cost_star_values[champion.cost -
                1][champion.stars - 1]

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

        move_sell_board_action_mask = np.zeros((7, 4, 38))
        move_sell_bench_action_mask = np.zeros((9, 38))

        # --- Utility Masks --- #
        max_items = max(player.item_bench.count(None), 3)

        invalid_champion_mask = np.ones(28)
        board_champion_mask = np.zeros(28)
        bench_to_bench_mask = np.zeros(9)

        default_board_mask = np.ones(38)
        invalid_board_mask = np.concatenate(
            [np.ones(28), np.zeros(9), np.zeros(1)])

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
            board_mask = np.ones(28) * invalid_champion_mask
        else:
            board_mask = board_champion_mask * invalid_champion_mask

        bench_mask = np.concatenate([board_mask, bench_to_bench_mask, np.ones(1)])

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

        item_action_mask = np.zeros((10, 38))

        valid_component_mask = np.zeros(38)
        valid_full_item_mask = np.zeros(38)

        valid_theives_gloves_mask = np.zeros(38)
        valid_glove_mask = np.zeros(38)

        valid_kayn_mask = np.zeros(38)

        valid_reforge_mask = np.zeros(38)
        valid_duplicator_mask = np.zeros(38)
        
        # Oh my god what a pain...
        trait_mask = {
           trait: np.ones(38) for trait in self.item_traits.values() 
        }

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
                    
                # Trait item
                elif item in self.item_traits:
                    item_action_mask[idx] = valid_full_item_mask * trait_mask[self.item_traits[item]]

                # Regular cases
                elif item in self.full_items:
                    item_action_mask[idx] = valid_full_item_mask

                elif item in self.item_components:
                    item_action_mask[idx] = valid_component_mask

        return item_action_mask

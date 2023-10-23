import collections
import numpy as np
import config

from Simulator.stats import COST
from Simulator.origin_class import team_traits, game_comp_tiers
from Simulator.origin_class_stats import tiers
from Simulator.item_stats import items, trait_items


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
        # Create ids for items
        self.item_ids = {k: idx for idx, k in enumerate(items.keys())}
        # Reverse trait_items dictionary to get the trait from the item
        self.item_traits = {v: k for k, v in trait_items.items()}
        self.item_transformations = {
            "AD": lambda x: x / 2,
            "crit_chance": lambda x: x * 255,
            "armor": lambda x: x,
            "MR": lambda x: x,
            "dodge": lambda x: x * 400,
            "health": lambda x: (x ** 0.5) * 3,
            "mana": lambda x: x * 2.5,
            "AS": lambda x: (x - 1) * 2.5,
            "SP": lambda x: x * 3,
            # TODO: "range"; rfc gives range
        }
        self.item_vector_length = 10

        # Create ids for tiers
        self.tier_ids = {k: idx for idx, k in enumerate(tiers.keys())}
        # 3 possible tiers from champion, 3 possible tiers from items, 1 possible tier from chosen
        self.tier_vector_length = 7

        # Create ids for champions
        self.champion_ids = {k: idx for idx, k in enumerate(COST.keys())}
        # Trick to get the championIDs in the same order as the champions dictionary
        self.champion_transformations = {
            "AD": lambda x: x / 2,
            "crit_chance": lambda x: x * 255,
            "armor": lambda x: x,
            "MR": lambda x: x,
            "dodge": lambda x: x * 255,
            "health": lambda x: (x ** 0.5) * 3,
            "mana": lambda x: x,  # Special case dealt with later
            "AS": lambda x: x * 100,
            "SP": lambda x: x * 100,
            "maxmana": lambda x: x * 1.45,
            "range": lambda x: x * 32
        }
        self.stat_vector_length = 11

        # Trick to get the championIDs in the same order as the champions dictionary

    def create_champion_vector(self, champion):
        """
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

        # TODO: Champion ID
        if champion.target_dummy:
            championID = len(self.champion_ids)
        else:
            championID = self.champion_ids[champion.name]

        # Items
        item_vectors = np.zeros(3, self.item_vector_length).astype("uint8")
        item_modifiers = []

        # Origins
        origin_vector = np.zeros(self.tier_vector_length).astype("uint8")
        origins = champion.origins.copy()

        # Stats
        # stats_vector = np.zeros(self.stat_vector_length).astype("uint8")
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
            encoding, stats = self.create_item_vector(item)

            item_vectors[idx] = encoding
            item_modifiers.append(stats)

            # Add trait if item is a spatula item
            if item in self.item_traits:
                origins.append(self.item_traits[item])

        # -- Origins -- #
        if champion.chosen:
            origins.append(champion.chosen)

        for idx, origin in enumerate(origins):
            originID = self.tier_ids[origin] + 1
            origin_vector[idx] = originID

        # -- Stats -- #
        # Add stat modifiers from items to champion stats
        for stat, modifier in item_modifiers.items():
            stats[stat] += modifier

        # Transform stats for encoding
        for stat, value in stats.items():
            stats[stat] = self.champion_transformations[stat](value)

        # Special case for mana, we want a ratio of current mana to max mana
        stats["mana"] = (stats["mana"] / stats["maxmana"]) * 255

        # Min-max normalization
        # stats_vector = self.minmaxnorm(np.array(list(stats.values())), 0, 255)

        # Clip values between 0 and 255
        stats_vector = self.clip(
            np.array(list(stats.values())), 0, 255).astype("uint8")

        return {
            "id": championID,
            "items": item_vectors,
            "origins": origin_vector,
            "stats": stats_vector
        }

    def create_item_vector(self, item):
        """
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

        stats = {
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

        encoding = stats.copy()

        for stat, value in item.items():
            stats[stat] = value
            encoding[stat] = self.item_transformations[stat](value)

        # Min-max normalization
        # encoding = self.minmaxnorm(np.array(list(encoding.values())), 0, 255)

        # Clip values between 0 and 255
        encoding = self.clip(
            np.array(list(encoding.values())), 0, 255).astype("uint8")

        # Special case for AS, as it is a percentage
        if stats["AS"] > 0:
            stats["AS"] -= 1

        return np.append(itemID, encoding), stats

    def minmaxnorm(self, X, min, max):
        """Helper function to normalize values between 0 and 1"""
        return (X - min) / (max - min)

    def clip(self, X, min, max):
        """Helper function to clip values between min and max"""
        return np.clip(X, min, max)

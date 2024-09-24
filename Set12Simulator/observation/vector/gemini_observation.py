import numpy as np
import config
from Simulator.config import MAX_CHAMPION_IN_SET, MAX_ITEMS_IN_SET
from Simulator.item_stats import items
from Simulator.observation.vector.observation import ObservationVector
from Simulator.origin_class_stats import tiers
from Simulator.stats import COST

class GeminiObservation(ObservationVector):
    """Observation object that stores the observation for a player."""

    def __init__(self, player):
        super().__init__(player)

    def create_public_scalars(self, player):
        return np.array([
            player.level / player.max_level,  # Normalized current level
            player.health / 100,  # Normalized current health
            player.win_streak,  # Current win streak
            player.loss_streak,  # Current loss streak
        ], dtype=np.float32)

    def create_private_scalars(self, player):
        """Create a vector of private scalar features."""
        # Estimate a reasonable maximum gold based on game stage
        max_gold_estimate = 10 + player.round * 10
        return np.array([
            player.gold / max_gold_estimate,  # Normalized current gold
            player.exp / player.level_costs[player.level],  # Normalized exp towards next level
            player.num_units_in_play / player.max_units,  # Normalized units on board
        ], dtype=np.float32)

    # -- Champion -- #
    def create_champion_vector(self, champion):
        """Create a feature vector for a single champion."""
        if champion is None:
            return np.zeros(
                MAX_CHAMPION_IN_SET + 1 + MAX_ITEMS_IN_SET * 3 + 2 + 1 + len(list(tiers.keys())), dtype=np.float32
            )
        champion_vector = np.zeros(MAX_CHAMPION_IN_SET, dtype=np.float32)
        champion_vector[list(COST.keys()).index(champion.name) - 1] = 1.0  # One-hot encode champion ID

        star_level = np.array([champion.stars / 3], dtype=np.float32)  # Normalized star level

        item_vector = np.zeros(MAX_ITEMS_IN_SET * 3, dtype=np.float32)
        for i, item_id in enumerate(champion.items):
            if item_id is not None:
                item_vector[i * MAX_ITEMS_IN_SET + list(items.keys()).index(item_id)] = 1.0  # One-hot encode item IDs

        if champion.maxmana != 0:
            # Normalized health and mana
            health_mana = np.array([champion.health / champion.max_health, champion.mana / champion.maxmana],
                                   dtype=np.float32)
        else:
            # Normalized health
            health_mana = np.array([champion.health / champion.max_health, 1], dtype=np.float32)

        is_chosen = np.array([1.0 if champion.chosen else 0.0], dtype=np.float32)

        doubled_trait_vector = np.zeros(len(list(tiers.keys())), dtype=np.float32)
        if champion.chosen:  # Assuming champion.doubled_trait stores the trait ID
            doubled_trait_vector[list(tiers.keys()).index(champion.chosen)] = 1.0

        return np.concatenate(
            [champion_vector, star_level, item_vector, health_mana, is_chosen, doubled_trait_vector]
        )

    def create_board_vector(self, player):
        """Create a tensor representation of the board."""
        board_vector = np.zeros((len(player.board[0]) * len(player.board), MAX_CHAMPION_IN_SET + 1 +
                                 MAX_ITEMS_IN_SET * 3 + 2 + 1 + len(list(tiers.keys()))), dtype=np.float32)
        player.team_champion_labels[:, 0] = 1
        player.team_champion_labels[:, 1] = 0
        for row in range(len(player.board)):
            for col in range(len(player.board[row])):
                champion = player.board[row][col]
                board_vector[row * len(player.board[0]) + col] = self.create_champion_vector(champion)
                if champion:
                    c_index = list(COST.keys()).index(player.board[row][col].name)
                    # create the label for the champion to help with training
                    if c_index < len(config.CHAMPION_ACTION_DIM):
                        player.team_champion_labels[c_index - 1, 0] = 0
                        player.team_champion_labels[c_index - 1, 1] = 1
        return board_vector

    # -- Bench Vector -- #
    def create_bench_vector(self, player):
        """Create a tensor representation of the bench."""
        bench_vector = np.zeros((len(player.bench), MAX_CHAMPION_IN_SET + 1 +
                                 MAX_ITEMS_IN_SET * 3 + 2 + 1 + len(list(tiers.keys()))), dtype=np.float32)
        for i, champion in enumerate(player.bench):
            bench_vector[i] = self.create_champion_vector(champion)
        return bench_vector

    # -- Shop Vector -- #
    def create_shop_vector(self, player):
        """Create a tensor representation of the shop."""
        shop_vector = np.zeros((len(player.shop), MAX_CHAMPION_IN_SET + 1 + 1), dtype=np.float32)
        for i, champion in enumerate(player.shop_champions):
            if champion is not None:
                champion_vec = np.zeros(MAX_CHAMPION_IN_SET, dtype=np.float32)
                champion_vec[list(COST.keys()).index(champion.name) - 1] = 1.0
                star_level = np.array([champion.stars / 3], dtype=np.float32)
                chosen = np.array([int(bool(champion.chosen))], dtype=np.float32)
                shop_vector[i] = np.concatenate([champion_vec, star_level, chosen])
        return shop_vector

    # -- Item Bench Vector -- #
    def create_item_bench_vector(self, player):
        """Create a tensor representation of the item bench."""
        item_bench_vector = np.zeros((len(player.item_bench), MAX_ITEMS_IN_SET + 1), dtype=np.float32)
        for i, item_id in enumerate(player.item_bench):
            if item_id is not None:
                c_index = list(items.keys()).index(item_id)
                item_bench_vector[i, c_index] = 1.0
        return item_bench_vector

    def create_trait_vector(self, player):
        """Create a vector representation of active traits and their tiers."""
        trait_vector = np.zeros(len(config.TEAM_TIERS_VECTOR), dtype=np.float32)
        player_tier_values = list(player.team_tiers.values())
        for i, [trait, tier] in enumerate(player.team_composition.items()):
            trait_vector[i] = tier / tiers[trait][-1]  # Normalize tier
            player.team_tier_labels[i] = np.zeros(config.TEAM_TIERS_VECTOR[i], dtype=np.float32)
            player.team_tier_labels[i][player_tier_values[i]] = 1
        return trait_vector

    @staticmethod
    def observation_to_input(observation):
        other_players = np.concatenate(
            [
                np.concatenate(
                    [
                        observation["opponents"][x]["board"].flatten(),
                        observation["opponents"][x]["scalars"],
                        observation["opponents"][x]["traits"],
                    ],
                    axis=-1, dtype=np.float32
                )
                for x in range(config.NUM_PLAYERS)
            ],
            axis=0, dtype=np.float32  # Concatenate opponent data along axis 0
        )

        return {
            "scalars": observation["player"]["scalars"],
            "shop": observation["player"]["shop"],
            "board": observation["player"]["board"],
            "bench": observation["player"]["bench"],
            "items": observation["player"]["items"],
            "traits": observation["player"]["traits"],
            "other_players": other_players
        }

    @staticmethod
    def observation_to_dictionary(observation):
        """Converts a list of observations to a batched dictionary."""

        return {
            key: np.stack([obs[key] for obs in observation])
            for key in observation[0].keys()
        }




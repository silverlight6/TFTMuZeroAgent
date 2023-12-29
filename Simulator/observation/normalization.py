import numpy as np

from Simulator.stats import AD, AS, HEALTH, ARMOR, MR, MANA, MAXMANA, RANGE, DODGE
from Simulator.item_stats import items
from Simulator.config import STARMULTIPLIER, CRIT_CHANCE, CRIT_DAMAGE, SP

"""
Utility functions for normalizing stats.
Also contains Normalizer class for applying normalization.
"""

# --- Utility functions --- #
def calculate_residuals(stats):
    """Calculate mu and sigma for each stat."""
    residuals = {}
    for stat, value in stats.items():
        mu = np.mean(value)
        sigma = np.std(value) + 1e-9 # Just to ensure no zeros
        residuals[stat] = (mu, sigma)

    return residuals


def calculate_minmax(stats):
    """Calculate min and max for each stat."""
    minmax = {}
    for stat, value in stats.items():
        minimum = np.min(value)
        maximum = np.max(value)
        minmax[stat] = (minimum, maximum)

    return minmax

# --- Item stats --- #
def calculate_item_statistics():
    """Calculate mu, sigma, min, and max for each stat using the items dictionary."""

    item_stats = {
        "AD": [],
        "crit_chance": [],
        "crit_damage": [],
        "armor": [],
        "MR": [],
        "dodge": [],
        "health": [],
        "mana": [],
        "AS": [],
        "SP": [],
    }

    for item, stats in items.items():
        for stat, value in stats.items():
            if stat in item_stats:
                item_stats[stat].append(value)

    # Calculate mu and sigma for each stat
    item_residuals = calculate_residuals(item_stats)

    # Calculte min and max for each stat
    item_minmax = calculate_minmax(item_stats)

    return item_residuals, item_minmax


# --- Champion stats --- #
def calculate_champion_statistics():
    """Calculate mu, sigma, min, and max for each stat using their respective dictionaries."""

    stats = {
        "AD": [],
        "crit_chance": [],
        "crit_damage": [],
        "armor": [],
        "MR": [],
        "dodge": [],
        "health": [],
        "mana": [],
        "AS": [],
        "SP": [],
        "maxmana": [],
        "range": [],
    }

    def calculate_star_stats(name, STAT):
        """Calculate stats that change depending on champion star level"""
        for champion, stat in STAT.items():
            # Special case for galio who's in this list for some reason
            if type(stat) is list:
                continue

            for star in range(1, 4):
                scaled_stat = stat * (STARMULTIPLIER ** (star - 1))
                stats[name].append(scaled_stat)

    def calculate_regular_stats(name, STAT):
        """Calculate stats that are the same for all star levels"""
        for champion, stat in STAT.items():
            # Special case for galio who's in this list for some reason
            if type(stat) is list:
                continue

            stats[name].append(stat)

    # Only Health and AD change depending on star level
    # All other stats are the same for all star levels
    calculate_star_stats("AD", AD)
    stats["crit_chance"] = [CRIT_CHANCE]
    stats["crit_damage"] = [CRIT_DAMAGE]
    calculate_regular_stats("armor", ARMOR)
    calculate_regular_stats("MR", MR)
    stats["dodge"] = [DODGE]
    calculate_star_stats("health", HEALTH)
    calculate_regular_stats("mana", MANA)
    calculate_regular_stats("AS", AS)
    stats["SP"] = [SP]
    calculate_regular_stats("maxmana", MAXMANA)
    calculate_regular_stats("range", RANGE)

    # Calculate mu and sigma for each stat
    champion_residuals = calculate_residuals(stats)

    # Calculte min and max for each stat
    champion_minmax = calculate_minmax(stats)

    return champion_residuals, champion_minmax


item_residuals, item_minmax = calculate_item_statistics()
champion_residuals, champion_minmax = calculate_champion_statistics()

# --- Apply normalization --- #
class Normalizer:
    def __init__(self):
        self.item_residuals, self.item_minmax = item_residuals, item_minmax
        self.champion_residuals, self.champion_minmax = champion_residuals, champion_minmax

    def batch_apply_z_score_item(self, stat_batch, stat_name):
        """Calcualte z-score for a numpy array of item stats."""
        mean, std = item_residuals[stat_name]

        return (stat_batch - mean) / std


    def batch_apply_z_score_champion(self, stat_batch, stat_name):
        """Calcualte z-score for a numpy array of champion stats."""
        mean, std = champion_residuals[stat_name]

        return (stat_batch - mean) / std


    def batch_apply_minmax_item(self, stat_batch, stat_name):
        """Calcualte min-max for a numpy array of item stats."""
        minimum, maximum = item_minmax[stat_name]

        return (stat_batch - minimum) / (maximum - minimum)


    def batch_apply_minmax_champion(self, stat_batch, stat_name):
        """Calcualte min-max for a numpy array of champion stats."""
        minimum, maximum = champion_minmax[stat_name]

        return (stat_batch - minimum) / (maximum - minimum)
    
    def apply_champion_normalization(self, champion_stats):
        """
            0: AD
            1: crit_chance
            2: crit_damage
            3: armor
            4: MR
            5: dodge
            6: health
            7: mana
            8: AS
            9: SP
            10: maxmana
            11: range
        """
        
        original_shape = champion_stats.shape
        
        champion_stats = champion_stats.reshape(-1, original_shape[-1])

        champion_stats[:, 0] = self.batch_apply_z_score_champion(champion_stats[:, 0], "AD")
        champion_stats[:, 1] = self.batch_apply_z_score_champion(champion_stats[:, 1], "crit_chance")
        champion_stats[:, 2] = self.batch_apply_z_score_champion(champion_stats[:, 2], "crit_damage")
        champion_stats[:, 3] = self.batch_apply_z_score_champion(champion_stats[:, 3], "armor")
        champion_stats[:, 4] = self.batch_apply_z_score_champion(champion_stats[:, 4], "MR")
        champion_stats[:, 5] = self.batch_apply_z_score_champion(champion_stats[:, 5], "dodge")
        champion_stats[:, 6] = self.batch_apply_z_score_champion(champion_stats[:, 6], "health")
        champion_stats[:, 7] = self.batch_apply_z_score_champion(champion_stats[:, 7], "mana")
        champion_stats[:, 8] = self.batch_apply_z_score_champion(champion_stats[:, 8], "AS")
        champion_stats[:, 9] = self.batch_apply_z_score_champion(champion_stats[:, 9], "SP")
        champion_stats[:, 10] = self.batch_apply_z_score_champion(champion_stats[:, 10], "maxmana")
        champion_stats[:, 11] = self.batch_apply_z_score_champion(champion_stats[:, 11], "range")
        
        return champion_stats.reshape(original_shape)
    
    def apply_item_normalization(self, item_stats):
        """
            0: AD
            1: crit_chance
            2: crit_damage
            3: armor
            4: MR
            5: dodge
            6: health
            7: mana
            8: AS
            9: SP
        """
        
        original_shape = item_stats.shape

        item_stats = item_stats.reshape(-1, original_shape[-1])

        item_stats[:, 0] = self.batch_apply_z_score_item(item_stats[:, 0], "AD")
        item_stats[:, 1] = self.batch_apply_z_score_item(item_stats[:, 1], "crit_chance")
        item_stats[:, 2] = self.batch_apply_z_score_item(item_stats[:, 2], "crit_damage")
        item_stats[:, 3] = self.batch_apply_z_score_item(item_stats[:, 3], "armor")
        item_stats[:, 4] = self.batch_apply_z_score_item(item_stats[:, 4], "MR")
        item_stats[:, 5] = self.batch_apply_z_score_item(item_stats[:, 5], "dodge")
        item_stats[:, 6] = self.batch_apply_z_score_item(item_stats[:, 6], "health")
        item_stats[:, 7] = self.batch_apply_z_score_item(item_stats[:, 7], "mana")
        item_stats[:, 8] = self.batch_apply_z_score_item(item_stats[:, 8], "AS")
        item_stats[:, 9] = self.batch_apply_z_score_item(item_stats[:, 9], "SP")

        return item_stats.reshape(original_shape)
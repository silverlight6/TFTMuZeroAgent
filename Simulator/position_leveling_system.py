import config
from Simulator.battle_generator import BattleGenerator, base_level_config

class PositionLevelingSystem:
    def __init__(self):

        self.levels = [
            {**base_level_config},
            {**base_level_config, "num_unique_champions": 4},
            {**base_level_config, "num_unique_champions": 6},
            {**base_level_config, "num_unique_champions": 12},
            {**base_level_config, "num_unique_champions": 12, "current_level": 4},
            {**base_level_config, "num_unique_champions": 6, "current_level": 4, "max_cost": 2},
            {**base_level_config, "num_unique_champions": 12, "current_level": 4, "max_cost": 2},
            {**base_level_config, "num_unique_champions": 12, "current_level": 5, "max_cost": 2},
            {**base_level_config, "num_unique_champions": 6, "current_level": 5, "max_cost": 3},
            {**base_level_config, "num_unique_champions": 12, "current_level": 5, "max_cost": 3},
            {**base_level_config, "num_unique_champions": 12, "current_level": 5, "max_cost": 3,
             "two_star_unit_percentage": 0.25},
            {**base_level_config, "num_unique_champions": 12, "current_level": 5, "max_cost": 3,
             "two_star_unit_percentage": 0.5},
            {**base_level_config, "current_level": 6, "sample_from_pool": True},
            {**base_level_config, "current_level": 7, "sample_from_pool": True, "chosen": True},
            {**base_level_config, "current_level": 7, "sample_from_pool": True, "chosen": True, "num_items": 1,
             "extra_randomness": True},
            {**base_level_config, "current_level": 7, "sample_from_pool": True, "chosen": True, "num_items": 1,
             "extra_randomness": True, "two_star_unit_percentage": 0.25},
            {**base_level_config, "current_level": 7, "sample_from_pool": True, "chosen": True, "num_items": 1,
             "extra_randomness": True, "two_star_unit_percentage": 0.25, "three_star_unit_percentage": 0.1},
            {**base_level_config, "current_level": 7, "sample_from_pool": True, "chosen": True, "num_items": 1,
             "extra_randomness": True, "two_star_unit_percentage": 0.25, "three_star_unit_percentage": 0.2},
            {**base_level_config, "current_level": 7, "sample_from_pool": True, "chosen": True, "num_items": 1,
             "extra_randomness": True, "two_star_unit_percentage": 0.5, "three_star_unit_percentage": 0.1},
            {**base_level_config, "current_level": 7, "sample_from_pool": True, "chosen": True, "num_items": 2,
             "extra_randomness": True, "two_star_unit_percentage": 0.5, "three_star_unit_percentage": 0.2},
            {**base_level_config, "current_level": 7, "sample_from_pool": True, "chosen": True, "num_items": 2,
             "extra_randomness": True, "two_star_unit_percentage": 0.5, "three_star_unit_percentage": 0.2},
            {**base_level_config, "current_level": 8, "sample_from_pool": True, "chosen": True, "num_items": 2,
             "extra_randomness": True, "two_star_unit_percentage": 0.5, "three_star_unit_percentage": 0.2},
            {**base_level_config, "current_level": 8, "sample_from_pool": True, "chosen": True, "num_items": 3,
             "extra_randomness": True, "two_star_unit_percentage": 0.5, "three_star_unit_percentage": 0.2},
            {**base_level_config, "current_level": 9, "sample_from_pool": True, "chosen": True, "num_items": 3,
             "extra_randomness": True, "two_star_unit_percentage": 0.5, "three_star_unit_percentage": 0.2},
            {**base_level_config, "current_level": 9, "sample_from_pool": True, "chosen": True, "num_items": 3,
             "extra_randomness": True, "two_star_unit_percentage": 0.8, "three_star_unit_percentage": 0.5},
        ]

        self.battle_levels = [
            {**base_level_config},
            {**base_level_config, "num_unique_champions": 12},
            {**base_level_config, "num_unique_champions": 12, "current_level": 4},
            {**base_level_config, "num_unique_champions": 6, "current_level": 4, "max_cost": 2},
            {**base_level_config, "num_unique_champions": 12, "current_level": 4, "max_cost": 2},
            {**base_level_config, "num_unique_champions": 6, "current_level": 5, "max_cost": 2},
            {**base_level_config, "num_unique_champions": 12, "current_level": 5, "max_cost": 2},
            {**base_level_config, "num_unique_champions": 12, "current_level": 5, "max_cost": 3},
            {**base_level_config, "num_unique_champions": 12, "current_level": 5, "max_cost": 3,
             "two_star_unit_percentage": 0.25},
            {**base_level_config, "num_unique_champions": 12, "current_level": 5, "max_cost": 3,
             "two_star_unit_percentage": 0.5},
            {**base_level_config, "current_level": 6, "sample_from_pool": True, "two_star_unit_percentage": 0.25},
            {**base_level_config, "current_level": 7, "sample_from_pool": True, "chosen": True,
             "two_star_unit_percentage": 0.25},
            {**base_level_config, "current_level": 7, "sample_from_pool": True, "chosen": True, "num_items": 1,
             "extra_randomness": True},
            {**base_level_config, "current_level": 7, "sample_from_pool": True, "chosen": True, "num_items": 1,
             "extra_randomness": True, "two_star_unit_percentage": 0.25},
            {**base_level_config, "current_level": 7, "sample_from_pool": True, "chosen": True, "num_items": 1,
             "extra_randomness": True, "two_star_unit_percentage": 0.25, "three_star_unit_percentage": 0.1},
            {**base_level_config, "current_level": 7, "sample_from_pool": True, "chosen": True, "num_items": 1,
             "extra_randomness": True, "two_star_unit_percentage": 0.25, "three_star_unit_percentage": 0.2},
            {**base_level_config, "current_level": 7, "sample_from_pool": True, "chosen": True, "num_items": 1,
             "extra_randomness": True, "two_star_unit_percentage": 0.5, "three_star_unit_percentage": 0.1},
            {**base_level_config, "current_level": 7, "sample_from_pool": True, "chosen": True, "num_items": 2,
             "extra_randomness": True, "two_star_unit_percentage": 0.5, "three_star_unit_percentage": 0.2},
            {**base_level_config, "current_level": 7, "sample_from_pool": True, "chosen": True, "num_items": 2,
             "extra_randomness": True, "two_star_unit_percentage": 0.5, "three_star_unit_percentage": 0.2},
            {**base_level_config, "current_level": 8, "sample_from_pool": True, "chosen": True, "num_items": 2,
             "extra_randomness": True, "two_star_unit_percentage": 0.5, "three_star_unit_percentage": 0.2},
            {**base_level_config, "current_level": 8, "sample_from_pool": True, "chosen": True, "num_items": 3,
             "extra_randomness": True, "two_star_unit_percentage": 0.5, "three_star_unit_percentage": 0.2},
            {**base_level_config, "current_level": 9, "sample_from_pool": True, "chosen": True, "num_items": 3,
             "extra_randomness": True, "two_star_unit_percentage": 0.5, "three_star_unit_percentage": 0.2},
            {**base_level_config, "current_level": 9, "sample_from_pool": True, "chosen": True, "num_items": 3,
             "extra_randomness": True, "two_star_unit_percentage": 0.8, "three_star_unit_percentage": 0.5},
        ]

        self.level = 0

        if config.SINGLE_PLAYER:
            # This one is a little faster progressing.
            self.battle_generator = BattleGenerator(self.battle_levels[self.level])
        else:
            # This one has a little less variance to allow for more gradual growth.
            self.battle_generator = BattleGenerator(self.levels[self.level])

    def generate_battle(self):
        return self.battle_generator.generate_battle()

    def generate_preset_battle(self):
        return self.battle_generator.generate_set_battle(self.level + 3)

    def level_up(self):
        self.level += 1
        self.battle_generator = BattleGenerator(self.levels[self.level])
        # print(f"Leveling to level {self.level}")

    def single_player_level_up(self):
        self.level += 1
        self.battle_generator = BattleGenerator(self.battle_levels[self.level])
        if self.level > 10:
            print(f"Leveling single player to level {self.level}")

from Simulator.battle_generator import BattleGenerator

class PositionLevelingSystem:
    def __init__(self):

        base_level_config = {
            "num_unique_champions": 3,
            "max_cost": 1,
            "num_items": 0,
            "current_level": 3,
            "chosen": False,
            "sample_from_pool": False,
            "two_star_unit_percentage": 0,
            "three_star_unit_percentage": 0,
            "scenario_info": True,
            "extra_randomness": False
        }

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
            {**base_level_config, "current_level": 6, "sample_from_pool": True, "chosen": True},
            {**base_level_config, "current_level": 6, "sample_from_pool": True, "chosen": True, "num_items": 1},
        ]

        self.level = 0

        self.battle_generator = BattleGenerator(self.levels[self.level])

    def generate_battle(self):
        return self.battle_generator.generate_battle()

    def level_up(self):
        self.level += 1
        self.battle_generator = BattleGenerator(self.levels[self.level])

from Simulator.generators.battle_generator import BattleGenerator, base_level_config

class PositionLevelingSystem:
    def __init__(self, single_player=False):
        # Positioning curriculum: gradual early game, then a smooth climb through
        # pool-sampled boards (levels 6–9). Length matches the 36 combat rounds
        # in the single-player campaign schedule.
        self.levels = [
            # --- Early: fixed low-cost boards ---
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
            # --- Level 6: introduce pool sampling before jumping to 7 ---
            {**base_level_config, "current_level": 6, "sample_from_pool": True},
            {**base_level_config, "current_level": 6, "sample_from_pool": True,
             "two_star_unit_percentage": 0.15},
            {**base_level_config, "current_level": 6, "sample_from_pool": True,
             "two_star_unit_percentage": 0.3},
            {**base_level_config, "current_level": 6, "sample_from_pool": True, "chosen": True,
             "two_star_unit_percentage": 0.3},
            {**base_level_config, "current_level": 6, "sample_from_pool": True, "chosen": True,
             "num_items": 1, "two_star_unit_percentage": 0.3},
            # --- Level 7: items / stars ramp ---
            {**base_level_config, "current_level": 7, "sample_from_pool": True, "chosen": True},
            {**base_level_config, "current_level": 7, "sample_from_pool": True, "chosen": True,
             "two_star_unit_percentage": 0.2},
            {**base_level_config, "current_level": 7, "sample_from_pool": True, "chosen": True,
             "num_items": 1, "extra_randomness": True},
            {**base_level_config, "current_level": 7, "sample_from_pool": True, "chosen": True,
             "num_items": 1, "extra_randomness": True, "two_star_unit_percentage": 0.15},
            {**base_level_config, "current_level": 7, "sample_from_pool": True, "chosen": True,
             "num_items": 1, "extra_randomness": True, "two_star_unit_percentage": 0.25},
            {**base_level_config, "current_level": 7, "sample_from_pool": True, "chosen": True,
             "num_items": 1, "extra_randomness": True, "two_star_unit_percentage": 0.25,
             "three_star_unit_percentage": 0.05},
            {**base_level_config, "current_level": 7, "sample_from_pool": True, "chosen": True,
             "num_items": 1, "extra_randomness": True, "two_star_unit_percentage": 0.25,
             "three_star_unit_percentage": 0.1},
            {**base_level_config, "current_level": 7, "sample_from_pool": True, "chosen": True,
             "num_items": 1, "extra_randomness": True, "two_star_unit_percentage": 0.35,
             "three_star_unit_percentage": 0.15},
            {**base_level_config, "current_level": 7, "sample_from_pool": True, "chosen": True,
             "num_items": 1, "extra_randomness": True, "two_star_unit_percentage": 0.5,
             "three_star_unit_percentage": 0.1},
            {**base_level_config, "current_level": 7, "sample_from_pool": True, "chosen": True,
             "num_items": 2, "extra_randomness": True, "two_star_unit_percentage": 0.4,
             "three_star_unit_percentage": 0.15},
            {**base_level_config, "current_level": 7, "sample_from_pool": True, "chosen": True,
             "num_items": 2, "extra_randomness": True, "two_star_unit_percentage": 0.5,
             "three_star_unit_percentage": 0.2},
            # --- Level 8 ---
            {**base_level_config, "current_level": 8, "sample_from_pool": True, "chosen": True,
             "num_items": 2, "extra_randomness": True, "two_star_unit_percentage": 0.4,
             "three_star_unit_percentage": 0.15},
            {**base_level_config, "current_level": 8, "sample_from_pool": True, "chosen": True,
             "num_items": 2, "extra_randomness": True, "two_star_unit_percentage": 0.5,
             "three_star_unit_percentage": 0.2},
            {**base_level_config, "current_level": 8, "sample_from_pool": True, "chosen": True,
             "num_items": 3, "extra_randomness": True, "two_star_unit_percentage": 0.5,
             "three_star_unit_percentage": 0.2},
            {**base_level_config, "current_level": 8, "sample_from_pool": True, "chosen": True,
             "num_items": 3, "extra_randomness": True, "two_star_unit_percentage": 0.6,
             "three_star_unit_percentage": 0.3},
            # --- Level 9 ---
            {**base_level_config, "current_level": 9, "sample_from_pool": True, "chosen": True,
             "num_items": 3, "extra_randomness": True, "two_star_unit_percentage": 0.5,
             "three_star_unit_percentage": 0.2},
            {**base_level_config, "current_level": 9, "sample_from_pool": True, "chosen": True,
             "num_items": 3, "extra_randomness": True, "two_star_unit_percentage": 0.65,
             "three_star_unit_percentage": 0.3},
            {**base_level_config, "current_level": 9, "sample_from_pool": True, "chosen": True,
             "num_items": 3, "extra_randomness": True, "two_star_unit_percentage": 0.8,
             "three_star_unit_percentage": 0.4},
            {**base_level_config, "current_level": 9, "sample_from_pool": True, "chosen": True,
             "num_items": 3, "extra_randomness": True, "two_star_unit_percentage": 0.8,
             "three_star_unit_percentage": 0.5},
        ]

        # Single-player campaign curriculum (faster early progression). Same
        # length as the 36 generated-opponent combat rounds.
        self.battle_levels = [
            # --- Early: fixed low-cost boards ---
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
            {**base_level_config, "num_unique_champions": 12, "current_level": 5, "max_cost": 3,
             "two_star_unit_percentage": 0.75},
            # --- Level 6: multi-step bridge (was a single entry) ---
            {**base_level_config, "current_level": 6, "sample_from_pool": True},
            {**base_level_config, "current_level": 6, "sample_from_pool": True,
             "two_star_unit_percentage": 0.15},
            {**base_level_config, "current_level": 6, "sample_from_pool": True,
             "two_star_unit_percentage": 0.25},
            {**base_level_config, "current_level": 6, "sample_from_pool": True,
             "two_star_unit_percentage": 0.4},
            {**base_level_config, "current_level": 6, "sample_from_pool": True, "chosen": True,
             "two_star_unit_percentage": 0.3},
            {**base_level_config, "current_level": 6, "sample_from_pool": True, "chosen": True,
             "num_items": 1, "two_star_unit_percentage": 0.3},
            # --- Level 7 ---
            {**base_level_config, "current_level": 7, "sample_from_pool": True, "chosen": True},
            {**base_level_config, "current_level": 7, "sample_from_pool": True, "chosen": True,
             "two_star_unit_percentage": 0.25},
            {**base_level_config, "current_level": 7, "sample_from_pool": True, "chosen": True,
             "num_items": 1, "extra_randomness": True},
            {**base_level_config, "current_level": 7, "sample_from_pool": True, "chosen": True,
             "num_items": 1, "extra_randomness": True, "two_star_unit_percentage": 0.15},
            {**base_level_config, "current_level": 7, "sample_from_pool": True, "chosen": True,
             "num_items": 1, "extra_randomness": True, "two_star_unit_percentage": 0.25},
            {**base_level_config, "current_level": 7, "sample_from_pool": True, "chosen": True,
             "num_items": 1, "extra_randomness": True, "two_star_unit_percentage": 0.25,
             "three_star_unit_percentage": 0.05},
            {**base_level_config, "current_level": 7, "sample_from_pool": True, "chosen": True,
             "num_items": 1, "extra_randomness": True, "two_star_unit_percentage": 0.25,
             "three_star_unit_percentage": 0.1},
            {**base_level_config, "current_level": 7, "sample_from_pool": True, "chosen": True,
             "num_items": 1, "extra_randomness": True, "two_star_unit_percentage": 0.35,
             "three_star_unit_percentage": 0.15},
            {**base_level_config, "current_level": 7, "sample_from_pool": True, "chosen": True,
             "num_items": 1, "extra_randomness": True, "two_star_unit_percentage": 0.5,
             "three_star_unit_percentage": 0.1},
            {**base_level_config, "current_level": 7, "sample_from_pool": True, "chosen": True,
             "num_items": 2, "extra_randomness": True, "two_star_unit_percentage": 0.4,
             "three_star_unit_percentage": 0.15},
            {**base_level_config, "current_level": 7, "sample_from_pool": True, "chosen": True,
             "num_items": 2, "extra_randomness": True, "two_star_unit_percentage": 0.5,
             "three_star_unit_percentage": 0.2},
            # --- Level 8 ---
            {**base_level_config, "current_level": 8, "sample_from_pool": True, "chosen": True,
             "num_items": 2, "extra_randomness": True, "two_star_unit_percentage": 0.4,
             "three_star_unit_percentage": 0.15},
            {**base_level_config, "current_level": 8, "sample_from_pool": True, "chosen": True,
             "num_items": 2, "extra_randomness": True, "two_star_unit_percentage": 0.5,
             "three_star_unit_percentage": 0.2},
            {**base_level_config, "current_level": 8, "sample_from_pool": True, "chosen": True,
             "num_items": 3, "extra_randomness": True, "two_star_unit_percentage": 0.5,
             "three_star_unit_percentage": 0.2},
            {**base_level_config, "current_level": 8, "sample_from_pool": True, "chosen": True,
             "num_items": 3, "extra_randomness": True, "two_star_unit_percentage": 0.6,
             "three_star_unit_percentage": 0.3},
            # --- Level 9 ---
            {**base_level_config, "current_level": 9, "sample_from_pool": True, "chosen": True,
             "num_items": 3, "extra_randomness": True, "two_star_unit_percentage": 0.5,
             "three_star_unit_percentage": 0.2},
            {**base_level_config, "current_level": 9, "sample_from_pool": True, "chosen": True,
             "num_items": 3, "extra_randomness": True, "two_star_unit_percentage": 0.65,
             "three_star_unit_percentage": 0.3},
            {**base_level_config, "current_level": 9, "sample_from_pool": True, "chosen": True,
             "num_items": 3, "extra_randomness": True, "two_star_unit_percentage": 0.8,
             "three_star_unit_percentage": 0.4},
            {**base_level_config, "current_level": 9, "sample_from_pool": True, "chosen": True,
             "num_items": 3, "extra_randomness": True, "two_star_unit_percentage": 0.8,
             "three_star_unit_percentage": 0.5},
        ]

        self.level = 0

        if single_player:
            # This one is a little faster progressing.
            self.battle_generator = BattleGenerator(self.battle_levels[self.level])
        else:
            # This one has a little less variance to allow for more gradual growth.
            self.battle_generator = BattleGenerator(self.levels[self.level])

    def generate_battle(self):
        return self.battle_generator.generate_battle()

    def generate_preset_battle(self):
        return self.battle_generator.generate_set_battle(self.level + 3,  True)

    def level_up(self):
        if self.level >= len(self.levels) - 1:
            return
        self.level += 1
        self.battle_generator = BattleGenerator(self.levels[self.level])

    def single_player_level_up(self):
        if self.level >= len(self.battle_levels) - 1:
            return
        self.level += 1
        self.battle_generator = BattleGenerator(self.battle_levels[self.level])
        if self.level > 10:
            print(f"Leveling single player to level {self.level}")

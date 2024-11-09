import numpy as np
import config
import random
from Simulator.pool import pool
from Simulator.player import Player
from Simulator.utils import coord_to_x_y, x_y_to_1d_coord
from Simulator.item_stats import item_builds, thieves_gloves_items
from Simulator.champion import champion
from Simulator.observation.token.action import ActionToken
from Simulator.default_agent_stats import ONE_COST_UNITS, TWO_COST_UNITS, THREE_COST_UNITS, FOUR_COST_UNITS, FIVE_COST_UNITS


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
    "extra_randomness": False,
    "stationary": True,
    "azir": False,
    "kayn": False,
    "thieves_gloves": False,
    "set_test_position": False,
    "test_position": [0, 0],
    "fill_bench": False,
    "test_mode": False,
    "allow_trait_items": False,
}

"""
Description - This is the Battle Generator class
              Generates random battles with random items and random number of players with random positioning
"""
class BattleGenerator:
    def __init__(self, generator_config):
        self.generator_config = generator_config
        unit_cost_list = [ONE_COST_UNITS, TWO_COST_UNITS, THREE_COST_UNITS, FOUR_COST_UNITS, FIVE_COST_UNITS]
        self.list_of_units = []
        self.sample_from_pool = self.generator_config["sample_from_pool"]
        if not self.sample_from_pool:
            for cost in range(self.generator_config["max_cost"]):
                if cost == self.generator_config["max_cost"] - 1:
                    self.list_of_units.append(sample_with_limit(unit_cost_list[cost],
                                                                self.generator_config["num_unique_champions"], seed=8))
                else:
                    self.list_of_units.append(sample_with_limit(unit_cost_list[cost], 12, seed=8))
        self.list_of_units = sum(self.list_of_units, [])
        self.stationary_coords = []
        if self.generator_config["stationary"]:
            random.seed(8)
            self.stationary_coords = random.sample(list(range(0, 28)), 12)
            if self.generator_config["set_test_position"]:
                if x_y_to_1d_coord(self.generator_config["test_position"][0],
                                   self.generator_config["test_position"][1]) not in self.stationary_coords:
                    self.stationary_coords[0] = x_y_to_1d_coord(self.generator_config["test_position"][0],
                                                                self.generator_config["test_position"][1])

        self.set_composition = [[champion('vi'), champion('katarina'), champion('nunu')],
                                [champion('tahmkench'), champion('katarina', itemlist=['hand_of_justice']),
                                 champion('nunu'), champion('annie')],
                                [champion('tahmkench'), champion('katarina', itemlist=['hand_of_justice']),
                                 champion('jinx'), champion('vi'), champion('vayne', chosen='duelist')]
                                ]

        self.set_oppo_composition = [[champion('sylas'), champion('maokai'), champion('diana', stars=2)],
                                     [champion('sylas'), champion('vi'), champion('vayne'),
                                      champion('janna', stars=2)],
                                     [champion('tahmkench', stars=2), champion('vi', stars=2), champion('elise'),
                                      champion('vayne', stars=2), champion('twistedfate', stars=2)]
                                     ]


    """
    Description - Generates random game state to use for battles.
    Inputs - Config that is defined in the position_leveling_system
    """
    def generate_battle(self):
        base_pool = pool()
        player_list = [Player(base_pool, player_num) for player_num in range(config.NUM_PLAYERS)]
        for player in player_list:
            level = self.generator_config["current_level"]
            item_count = self.generator_config["num_items"]
            if self.generator_config["extra_randomness"]:
                level = level + np.random.randint(-2, 3)
                if level > 9:
                    level = 9
                item_count += np.random.randint(-1, 2)
                if item_count < 0:
                    item_count = 0
                elif item_count > 3:
                    item_count = 3
            player.level = level
            player.max_units = level
            # These two are needed for masks
            allow_chosen = self.generator_config["chosen"] and random.choice([True, False])
            player.shop = base_pool.sample(player, 5, allow_chosen=allow_chosen)
            player.shop_champions = player.create_shop_champions()
            if self.generator_config["scenario_info"]:
                if self.generator_config["test_mode"]:
                    player.gold = np.random.randint(20, 60)
                    player.round = np.random.randint(3, 30)
                else:
                    player.gold = np.random.randint(0, 60)
                    player.round = np.random.randint(0, 100) % (level * 3)
                player.exp = np.random.randint(0, player.level_costs[level])
                player.health = np.random.randint(1, 101)
            action_mask = ActionToken(player)
            self.add_champions(player, action_mask, base_pool)
            # Add these back in later after I see proof of learning
            self.add_items_to_champions(player, action_mask, item_count)
            self.add_items_to_item_bench(player)
            if self.generator_config["fill_bench"]:
                self.add_champions_to_bench(player, base_pool)
        if config.NUM_PLAYERS > 1:
            player_returns = random.sample(range(0, config.NUM_PLAYERS), 2)
            return [player_list[player_returns[0]], player_list[player_returns[1]],
                    {f"player_{player.player_num}": player for player in player_list}]
        else:
            return [player_list[0], None, None]

    def add_champions(self, player, action_mask, base_pool):
        if not self.sample_from_pool:
            list_of_champs = sample_with_limit(self.list_of_units, player.max_units)
        else:
            list_of_champs = []
        for i in range(player.max_units):
            if self.sample_from_pool:
                random_champ = base_pool.sample(player, 1, allow_chosen=False)
                if i == 0 and (self.generator_config["azir"] or self.generator_config["kayn"]):
                    if self.generator_config["azir"]:
                        success = player.add_to_bench(champion('azir'))
                    else:
                        success = player.add_to_bench(champion('kayn'))
                else:
                    success = player.add_to_bench(champion(random_champ[0]))
                if not success:
                    print("I was not successful")
                    continue
                # unit was tripled
                if success and player.bench[0] is None:
                    # i is always 2 or greater since it needs 3 units to triple.
                    # Putting this here, so we can always fill up the board
                    i -= 2
                    continue
            else:
                if random.random() < self.generator_config["two_star_unit_percentage"]:
                    player.add_to_bench(champion(list_of_champs[i], stars=2))
                if random.random() < self.generator_config["three_star_unit_percentage"]:
                    player.add_to_bench(champion(list_of_champs[i], stars=3))
                else:
                    player.add_to_bench(champion(list_of_champs[i]))
            _, bench_mask = action_mask.create_move_and_sell_action_mask(player)
            if self.generator_config["stationary"]:
                coord = self.stationary_coords[i]
            else:
                coord = np.random.randint(0, 28)
            coord_x, coord_y = coord_to_x_y(coord)
            if bench_mask[0][coord]:
                player.move_bench_to_board(0, coord_x, coord_y)
            else:
                move_failure = 0
                while not bench_mask[0][coord]:
                    coord = np.random.randint(0, 28)
                    coord_x, coord_y = coord_to_x_y(coord)
                    if bench_mask[0][coord]:
                        player.move_bench_to_board(0, coord_x, coord_y)
                    else:
                        move_failure += 1
                        if move_failure > 10:
                            _, bench_mask = action_mask.create_move_and_sell_action_mask(player)
                            print(f"crisis (x, y) -> {coord_x, coord_y} with unit {player.bench[0]}")
                            print(f"full bench {player.bench}")
                            print(f"player level {player.level}")
                            break

    def add_items_to_champions(self, player, action_mask, item_count):
        move_failures = 0
        for x in range(len(player.board)):
            for y in range(len(player.board[0])):
                if player.board[x][y] and move_failures <= 9:
                    for i in range(item_count):
                        if i == 0 and self.generator_config["thieves_gloves"]:
                            player.add_to_item_bench("thieves_gloves")
                            item_mask = action_mask.create_item_action_mask(player)
                            if item_mask[move_failures][x_y_to_1d_coord(x, y)]:
                                player.move_item(move_failures, x, y)
                            else:
                                move_failures += 1
                        else:
                            if self.generator_config["allow_trait_items"]:
                                random_item = random.sample(list(item_builds.keys()), 1)
                            else:
                                random_item = random.sample(thieves_gloves_items, 1)
                            player.add_to_item_bench(random_item[0])
                            item_mask = action_mask.create_item_action_mask(player)
                            if item_mask[move_failures][x_y_to_1d_coord(x, y)] and \
                                    not (random_item[0] == "thieves_gloves" and i > 0):
                                player.move_item(move_failures, x, y)
                            else:
                                move_failures += 1
                                if move_failures > 9:
                                    break

    def add_items_to_item_bench(self, player):
        while not player.item_bench_full(1):
            if self.generator_config["allow_trait_items"]:
                random_item = random.sample(list(item_builds.keys()), 1)
            else:
                random_item = random.sample(thieves_gloves_items, 1)
            player.add_to_item_bench(random_item[0])

    def add_champions_to_bench(self, player, base_pool):
        i = 0
        while not player.bench_full():
            random_champ = base_pool.sample(player, 1, allow_chosen=False)
            player.add_to_bench(champion(random_champ[0]))
            i += 1
            if i > 10:
                print(f"ADD_CHAMPIONS_TO_BENCH has an issue {i}")

    def generate_set_battle(self, level=3, set_position=True):
        base_pool = pool()
        player_list = [Player(base_pool, player_num) for player_num in range(2)]

        player_team = self.set_composition[level - 3]
        oppo_team = self.set_oppo_composition[level - 3]

        for num, player in enumerate(player_list):
            player.level = level
            player.max_units = level
            for i in range(player.max_units):
                if num == 0:
                    player.add_to_bench(player_team[i])
                else:
                    player.add_to_bench(oppo_team[i])
                # Start with this, change it later.
                if set_position:
                    coord = self.stationary_coords[i]
                else:
                    coord = np.random.randint(0, 28)
                coord_x, coord_y = coord_to_x_y(coord)
                # TODO: Turn the next few lines into a method of it's own so I don't have to copy and paste.
                action_mask = ActionToken(player)
                _, bench_mask = action_mask.create_move_and_sell_action_mask(player)
                if bench_mask[0][coord]:
                    player.move_bench_to_board(0, coord_x, coord_y)
                else:
                    move_failure = 0
                    while not bench_mask[0][coord]:
                        coord = np.random.randint(0, 28)
                        coord_x, coord_y = coord_to_x_y(coord)
                        if bench_mask[0][coord]:
                            player.move_bench_to_board(0, coord_x, coord_y)
                        else:
                            move_failure += 1
                            if move_failure > 10:
                                _, bench_mask = action_mask.create_move_and_sell_action_mask(player)
                                print(f"crisis (x, y) -> {coord_x, coord_y} with unit {player.bench[0]}")
                                print(f"full bench {player.bench}")
                                print(f"player level {player.level}")
                                break

        return [player_list[0], player_list[1], {f"player_{player.player_num}": player for player in player_list}]



def sample_with_limit(units, x, seed=None):
    """Samples x units from the list, but returns the whole list if x is larger.

    Args:
      units: The list of units to sample from.
      x: The number of units to sample.
      seed: Which seed to use for RNG

    Returns:
      A list of x sampled units, or the entire list if x is larger than the list size.
    """
    if seed is not None:
        random.seed(seed)
    if x <= len(units):
        return random.sample(units, x)
    else:
        return units

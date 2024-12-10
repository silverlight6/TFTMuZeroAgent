import math
import time
import numpy as np
import random
from Simulator import champion, origin_class
import Simulator.utils as utils
import Simulator.config as config
from Simulator.item_stats import basic_items, item_builds, thieves_gloves_items, \
    starting_items, trait_items, items

from Simulator.stats import COST
from Simulator.pool_stats import cost_star_values
from Simulator.origin_class_stats import tiers, fortune_returns
from math import floor
from config import DEBUG, CHAMPION_ACTION_DIM, TIERS_FLATTEN_LENGTH, TEAM_TIERS_VECTOR, ALLOW_SPILL

from Simulator.observation.token.action import ActionToken  # Here for debugging purposes, will be removed later
from Simulator.default_agent import Default_Agent


"""
Description - This is the base player class
              Stores all values relevant to an individual player in the game
Inputs      - pool_pointer: Pool object pointer
                pointer to the pool object, used for updating the pool on buy and sell commands
              player_num: Int
                An identifier for the player, used in match_making for combats
"""
class Player:
    def __init__(self, pool_pointer, player_num):

        self.player_num = player_num

        # Everyone shares the pool object.
        # Required for buying champions to and from the pool
        self.pool_obj = pool_pointer

        # --- Public Scalar Values ---
        self.level = 1
        self.health = 100

        # --- Private Scalar Values ---

        self.exp = 0
        self.gold = 0

        # Streak Values
        self.win_streak = 0
        self.loss_streak = 0

        # For purposes of gold generation if fortune trait is active
        self.fortune_loss_streak = 0

        # --- Public Objects ---
        # Bench Champions
        # 9 slots for champions
        # | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
        self.bench = [None for _ in range(config.BENCH_SIZE)]

        # Board Champions
        # 7 rows, 4 columns
        """
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

        self.board = [[None for _ in range(config.BOARD_Y)] for _ in range(config.BOARD_X)]

        # List of items, there is no object for this so this is a string array
        # 10 slots for items
        # | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
        self.item_bench = [None for _ in range(config.ITEM_BENCH_SIZE)]

        # --- Private Objects ---
        # List of shop champions in string format
        # if chosen, will have _c appended to the end
        # 5 slots for champions
        # | 0 | 1 | 2 | 3 | 4 |
        self.shop = [None for _ in range(config.SHOP_SIZE)]
        self.shop_champions = [None for _ in range(config.SHOP_SIZE)]

        # List of team compositions
        self.team_composition = origin_class.game_compositions_base[self.player_num].copy()
        # List of tiers of each trait.
        self.team_tiers = origin_class.game_comp_tiers_base[self.player_num].copy()
        self.tiers_vector = np.zeros(TIERS_FLATTEN_LENGTH, dtype=np.float32).copy()
        self.team_tier_labels = [np.zeros(tier_size, dtype=np.int8) for tier_size in TEAM_TIERS_VECTOR]
        self.team_champion_labels = np.zeros([len(CHAMPION_ACTION_DIM), 2], dtype=np.int8)

        # --- Game Related Variables ---
        self.round = 0
        self.actions_remaining = 0  # Will reset to max_actions from player_manager

        # --- Board Related Variables ---
        # Triple catalog tracks the star level of each champion in the player's possession
        self.triple_catalog = []
        self.num_units_in_play = 0
        self.max_units = 1

        # --- Game Configuration Variables ---
        self.refresh_cost = 2
        self.exp_cost = 4
        # Amount of gold required to level differs based on level
        self.level_costs = [0, 2, 2, 6, 10, 20, 36, 56, 80, 100]
        self.max_level = 9

        # --- Reward Variables ---
        # Using this to track the reward gained by each player for the AI to train.
        self.reward = 0.0

        # reward levers
        self.refresh_reward = 0
        self.minion_count_reward = 0
        self.mistake_reward = 0
        self.level_reward = 0
        self.item_reward = 0
        self.won_game_reward = 0
        self.prev_rewards = 0
        self.damage_reward = 1

        # An array to record match history
        self.match_history = [0.5, 0.5, 0.5]

        self.start_time = time.time_ns()

        # --- Simulator Variables ---
        # opponent and opponent_board not currently used
        # Leaving here in case we want to add values to the observation that include previous opponent
        self.opponent = None  # Other player, player object
        # Other player's board for combat, not sure if I will use this.
        self.opponent_board = None

        self.chosen = False  # Does this player have a chosen unit already
        self.log = []

        # Boolean for fought this round or not
        self.combat = False

        # Putting this here to show the next possible opponent
        self.possible_opponents = {"player_" + str(player_id): config.MATCHMAKING_WEIGHTS
                                   for player_id in range(config.NUM_PLAYERS)}
        self.possible_opponents["player_" + str(self.player_num)] = -1
        self.opponent_options = {
            "player_" + str(player_id): 0 for player_id in range(config.NUM_PLAYERS)}

        # --- Edge Case Variables ---
        self.kayn_turn_count = 0
        self.kayn_transformed = False
        self.kayn_form = None

        self.thieves_gloves_loc = []

        # --- Loot Orb and Item Pool ---
        # Start with two copies of each item in your item pool
        self.item_pool = []
        self.refill_item_pool()
        self.refill_item_pool()

        # Context For Loot Orbs
        self.orb_history = []

        self.default_agent = Default_Agent()
        self.default_player = False

    # --- Exp Action --- #
    def buy_exp_action(self):
        """Buys exp for the player and levels up if possible.

        Conditions:
            - Must have enough gold to buy exp (4 gold)
            - Must not be max level

        Returns:
            bool: True if action was performed successfully, False otherwise.
        """

        # Can't buy exp if you are max level or don't have enough gold
        if self.gold < self.exp_cost or self.level == self.max_level:
            if DEBUG:
                print(f"Did not have gold to buy_exp, gold {self.gold} with level {self.level} and shop {self.shop}")
            return False

        self.gold -= self.exp_cost
        self.exp += self.exp_cost
        self.level_up()  # Level up if you have enough exp

        self.print(f"exp to {self.exp} on level {self.level}")
        return True

    def level_up(self):
        """Levels up the player if they have enough exp.

        Conditions:
            - Must have enough exp to level up
            - Must not be max level

        Returns:
            bool: True if action was performed successfully, False otherwise.
        """
        # Can't level up if you are max level or don't have enough exp
        if self.level >= self.max_level or self.exp < self.level_costs[self.level]:
            return False

        self.exp -= self.level_costs[self.level]
        self.level += 1
        self.max_units += 1

        self.print(f"leveled to {self.level}")

        # Only needed if it's possible to level more than once in one transaction
        self.level_up()

        if self.level == self.max_level:
            self.exp = 0

        return True

    # --- Refresh Action --- #
    def refresh_shop_action(self):
        """Refreshes the shop with new champions.

        Conditions:
            - Must have enough gold to refresh (2 gold)

        Returns:
            bool: True if action was performed successfully, False otherwise.
        """

        if self.gold < self.refresh_cost:
            if DEBUG:
                print(f"Did not have gold to refresh with gold {self.gold} and refresh_cost {self.refresh_cost}")
            return False

        self.gold -= self.refresh_cost
        self.refresh_shop()
        self.print("Refreshed shop")

        return True

    def refresh_shop(self):
        """Refreshes the shop with new champions.

        Returns:
            bool: True if action was performed successfully, False otherwise.
        """

        self.shop = self.pool_obj.sample(self, 5)
        self.shop_champions = self.create_shop_champions()

        return True

    def create_shop_champions(self):
        """Utility function that creates the champion objects from the shop."""

        shop_champions = []

        for champion_name in self.shop:
            if champion_name is None:
                a_champion = None
            elif champion_name.endswith("_c"):
                champion_name = champion_name[:-2]
                # Chosen champions are defined as `<name>_<trait>_c`
                champion_name, chosen_trait = champion_name.split("_")
                a_champion = champion.champion(champion_name, chosen=chosen_trait)
            else:
                a_champion = champion.champion(champion_name)

            shop_champions.append(a_champion)

        return shop_champions

    # --- Buy Action --- #
    def buy_shop_action(self, x1):
        """Buys a champion from the shop.

        Conditions:
            - Must have enough gold to buy the champion
            - Shop index must be between 0 and 4 and must not be empty

        Args:
            x1 (int): Index of the champion to buy. Must be between 0 and 4.

        Returns:
            bool: True if action was performed successfully, False otherwise.
        """
        if x1 < 0 or x1 > 4:
            if DEBUG:
                print("Invalid shop index")
            return False

        a_champion = self.shop_champions[x1]

        # No champion in shop slot
        if a_champion is None:
            if DEBUG:
                print("Shop slot is empty")
            return False

        bought_champion = self.buy_champion(a_champion)

        if bought_champion:
            self.shop[x1] = None
            self.shop_champions[x1] = None
            self.print(f"Bought champion {a_champion.name}")
            return True

    def buy_champion(self, a_champion):
        """Buys a champion from the pool.

        Also updates some edge cases like kayn and chosen champions.
        If possible, will also upgrade the champion.
        If the bench is full, will autosell the champion.

        Conditions:
            - Must have enough gold to buy the champion

        Args:
            a_champion: Champion object to buy and add to the bench.

        Returns:
            bool: True if action was performed successfully, False otherwise.
        """
        champion_cost = cost_star_values[a_champion.cost - 1][a_champion.stars - 1]

        # I don't know what the second condition is for but I don't intend to find out...
        if champion_cost > self.gold or a_champion.cost == 0:
            if DEBUG:
                print(f"No gold to buy champion, {champion_cost}, {self.gold}, {a_champion.name}")
                action = ActionToken(self)
                np.set_printoptions(threshold=np.inf)
                print(self.shop)
                print([(c.cost, c.stars) for c in self.shop_champions])
                print(action.buy_mask)
                print(self.gold)
                print('----------------- OBS MASK ---------------')
                print(np.reshape(action.fetch_action_mask(), (55, 38)))
            return False

        if a_champion.name == 'kayn':
            a_champion.kayn_form = self.kayn_form

        self.gold -= champion_cost

        champion_added = self.add_to_bench(a_champion)

        # Putting this outside success because when the bench is full. It auto sells the champion.
        # Which adds another to the pool and need this here to remove the fake copy from the pool
        self.pool_obj.update_pool(a_champion, -1)

        if champion_added:
            self.print(
                f"Spending gold on champion {a_champion.name} with cost = {champion_cost}, "
                f"remaining gold {self.gold} and chosen = {a_champion.chosen}")
        else:
            if DEBUG:
                print("Did not buy champion successfully")

        return champion_added

    def add_to_bench(self, a_champion, from_carousel=False):
        """Adds a champion to the bench.

        If the champion can be upgraded, will upgrade the champion.
        If the bench is full, will autosell the champion.
        If conditions are met, will add champion to the next open spot on the bench.

        The from_carousel flag is currently unused.
        Previously it was used to make sure that we didn't give a mistake reward when the bench was full.

        Conditions:
            - Must not be target dummy or azir sandguard

        Args:
            a_champion: Champion object to add to the bench.
            from_carousel: If the champion is from the carousel. Defaults to False.

        Returns:
            bool: True if action was performed successfully, False otherwise.
        """
        # Upgrade champion if possible
        golden, triple_success = self.update_triple_catalog(a_champion)
        if not triple_success:
            if DEBUG:
                print("Could not update triple catalog for champion " +
                      a_champion.name)
            return False

        if golden:
            return True

        # Get next open spot on bench
        bench_loc = self.bench_vacancy()

        # If bench is full, autosell champion
        if bench_loc < 0:
            self.sell_champion(a_champion, field=False)

            # Not successful if was not added from carousel
            if not from_carousel:
                if DEBUG:
                    print(f"add_to_bench but full, champion -> {a_champion.name}, round --> {self.round}, "
                          f"units_in_play {self.num_units_in_play}, max_units {self.max_units}")
                return False
            else:
                return True

        # Add champion to bench location
        self.bench[bench_loc] = a_champion
        a_champion.bench_loc = bench_loc

        # If champion is chosen, update chosen
        if a_champion.chosen:
            self.chosen = a_champion.chosen

        self.print("Adding champion {} with items {} and chosen {} to bench".format(
            a_champion.name, a_champion.items, a_champion.chosen))

        # If champion has thieves gloves, add to thieves gloves list
        if self.bench[bench_loc].items and self.bench[bench_loc].items[0] == 'thieves_gloves':
            self.thieves_gloves_loc.append([bench_loc, -1])
            self.thieves_gloves(bench_loc, -1)

        return True

    # --- Triple Catalog Functions --- #
    def update_triple_catalog(self, a_champion):
        """Checks if a champion can be upgraded and upgrades it if possible.

        If the champion can be upgraded, will upgrade the champion.
        If not, then it will add the champion to the triple catalog.

        Args:
            a_champion (champion): Champion object to check for upgrade.

        Returns:
            bool: If the champion was golden (3 stars) or not.
            bool: If operation was successful or not.
        """
        for entry in self.triple_catalog:
            if entry["name"] == a_champion.name and entry["level"] == a_champion.stars:
                entry["num"] += 1
                if entry["num"] == 3:
                    self.golden(a_champion)
                    return True, True
                return False, True
        if a_champion.stars > 3:
            return False, False
        self.triple_catalog.append(
            {"name": a_champion.name, "level": a_champion.stars, "num": 1})
        return False, True

    def remove_triple_catalog(self, a_champion, golden=False) -> bool:
        """Removes a champion from the triple catalog.

        Called when a champion is sold from the board or bench.

        Args:
            a_champion (champion): Champion object to remove from the triple catalog.
            golden (bool, optional): If the champion is golden. Defaults to False.

        Returns:
            bool: True if operation was successful, False otherwise.
        """
        gold = False
        if golden:
            for unit in self.triple_catalog:
                if unit["name"] == a_champion.name and unit["level"] == a_champion.stars:
                    self.triple_catalog.remove(unit)
                    return True
            return True
        for idx, i in enumerate(self.triple_catalog):
            if i["name"] == a_champion.name and i["level"] == a_champion.stars:
                i["num"] -= 1
                if i["num"] == 0:
                    self.triple_catalog.pop(idx)
                return True
            if i["name"] == a_champion.name and i["level"] - 1 == a_champion.stars:
                gold = True
        if gold:
            self.print("Trying to fix bug for {} with level {}".format(
                a_champion.name, a_champion.stars))
            return True
        if a_champion.stars >= 2:
            self.print("Failed to remove " + str(a_champion.name) +
                       " from triple catalog with stars = " + str(a_champion.stars))
        self.print("could not find champion " + a_champion.name + " with star = "
                   + str(a_champion.stars) + " in the triple catelog")
        self.print("{}".format(self.triple_catalog))
        if DEBUG:
            print("could not find champion " + a_champion.name + " with star = "
                  + str(a_champion.stars) + " in the triple catelog")
        return False

    def num_in_triple_catelog(self, a_champion):
        """Checks how many of a champion are in the triple catalog.

        Args:
            a_champion (champion): Champion object to check for.

        Returns:
            int: Number of champions in the triple catalog.
        """
        num = 0
        for entry in self.triple_catalog:
            if entry["name"] == a_champion.name and entry["level"] == a_champion.stars:
                num += 1
        return num

    def generate_tier_vector(self):
        # Create a vector where there is a 1 for the tier of the specific trait.
        current_position = 0
        self.tiers_vector = np.zeros(TIERS_FLATTEN_LENGTH, dtype=np.int8)
        base_tier_values = list(tiers.values())
        player_tier_values = list(self.team_tiers.values())
        for i in range(len(base_tier_values)):
            try:
                self.tiers_vector[current_position + player_tier_values[i]] = 1
                self.team_tier_labels[i] = np.zeros(TEAM_TIERS_VECTOR[i], dtype=np.int8)
                self.team_tier_labels[i][player_tier_values[i]] = 1
                current_position += len(base_tier_values[i]) + 1
            except IndexError:
                print("index i {} with player_tier_values[i] {}".format(i, player_tier_values[i]))
                print("team_tier_labels {}".format(self.team_tier_labels))

    def golden(self, a_champion) -> champion:
        """Upgrades a champion to the next star level of the champion.

        Despite the name, it is also called to upgrade 1-stars to a 2-star champion.
        Transfers items over to the new champion.

        TODO: Currently it requires a bench space... we need to change this so it doesn't.

        Args:
            a_champion (champion): Champion object to upgrade.

        Returns:
            champion: The upgraded champion.
        """
        x = -1
        y = -1
        chosen = False
        b_champion = champion.champion(a_champion.name, stars=a_champion.stars,
                                       itemlist=a_champion.items, chosen=a_champion.chosen)
        for i in range(0, len(self.bench)):
            if self.bench[i]:
                if self.bench[i].name == a_champion.name and self.bench[i].stars == a_champion.stars:
                    x = i
                    if self.bench[i].chosen:
                        chosen = self.bench[i].chosen
                    self.sell_from_bench(i, golden=True)
        for i in range(0, 7):
            for j in range(0, 4):
                if self.board[i][j]:
                    if self.board[i][j].name == a_champion.name and self.board[i][j].stars == a_champion.stars:
                        x = i
                        y = j
                        if self.board[i][j].chosen:
                            chosen = self.board[i][j].chosen
                        self.sell_champion(
                            self.board[i][j], golden=True, field=True)
        b_champion.chosen = chosen
        b_champion.golden()
        if chosen:
            b_champion.new_chosen()

        temp = None
        if self.bench_full():
            temp = self.bench[0]
            self.bench[0] = None
        self.add_to_bench(b_champion)
        if y != -1:
            self.move_bench_to_board(b_champion.bench_loc, x, y)
        if temp:
            self.bench[0] = temp
        self.print("champion {} was made golden".format(b_champion.name))
        return b_champion

    # --- Sell Action --- #
    def sell_action(self, x1):
        if x1 < 0 or x1 > 36:
            if DEBUG:
                print("Invalid sell index")
            return False

        if x1 < 28:
            x, y = utils.coord_to_x_y(x1)

            if self.board[x][y] is None:
                if DEBUG:
                    print("No champion to sell")
                return False

            self.sell_champion(self.board[x][y])

        else:
            bench_loc = x1 - 28
            self.sell_from_bench(bench_loc)

    # TODO: Verify that the bench / board vectors are being updated somewhere in the same operation as this method call.
    def sell_champion(self, s_champion, golden=False, field=True) -> bool:
        """Sells a champion from the board.

        This should only be called when trying to sell a champion from the field and the bench is full
        This can occur after a carousel round where you get a free champion and it can enter the field
        Even if you already have too many units in play. The default behavior will be sell that champion.

        Args:
            s_champion (Champion): Champion object to sell
            golden (bool, optional): True: Don't update the pool or grant gold. Defaults to False.
            field (bool, optional): True: unit is being sold from the field so decrement units in play. Defaults to True.

        Returns:
            bool: True: Unit successful sold. False: Was unable to sell unit due to remove from triple catalog, return item or target dummy.
        """
        # Need to add the behavior that on carousel when bench is full, add to board.
        if not (self.remove_triple_catalog(s_champion, golden=golden) and self.return_item(s_champion) and not
                s_champion.target_dummy):
            self.reward += self.mistake_reward
            self.print("Could not sell champion " + s_champion.name)
            if DEBUG:
                print("Could not sell champion " + s_champion.name)
            return False
        if not golden:
            self.gold += cost_star_values[s_champion.cost -
                                          1][s_champion.stars - 1]
            self.pool_obj.update_pool(s_champion, 1)
        if s_champion.chosen:
            self.chosen = False
        if s_champion.x != -1 and s_champion.y != -1:
            if self.board[s_champion.x][s_champion.y].name == 'azir':
                coords = self.board[s_champion.x][s_champion.y].sandguard_overlord_coordinates
                self.board[s_champion.x][s_champion.y].overlord = False
                for coord in coords:
                    self.board[coord[0]][coord[1]] = None
            self.board[s_champion.x][s_champion.y] = None
        if field:
            self.num_units_in_play -= 1
        self.print("selling champion " + s_champion.name + " with stars = " + str(s_champion.stars) + " from position ["
                   + str(s_champion.x) + ", " + str(s_champion.y) + "]")
        return True

    def sell_from_bench(self, location, golden=False) -> bool:
        """Selling unit from the bench

        Args:
            location (int): Which location on the bench to sell from
            golden (bool, optional): True: Don't update the pool or grant gold. Defaults to False.

        Returns:
            bool: True: Unit successful sold. False: Was unable to sell unit due to remove from triple catalog, return item or target dummy.
        """
        if self.bench[location]:
            if not (self.remove_triple_catalog(self.bench[location], golden=golden) and
                    self.return_item_from_bench(location)):
                self.print("Mistake in sell from bench with {} and level {}".format(self.bench[location],
                                                                                    self.bench[location].stars))
                self.reward += self.mistake_reward
                if DEBUG:
                    print("Could not remove from triple catalog or return item")
                return False
            if not golden:
                self.gold += cost_star_values[self.bench[location].cost -
                                              1][self.bench[location].stars - 1]
                self.pool_obj.update_pool(self.bench[location], 1)
            if self.bench[location].chosen:
                self.chosen = False
            return_champ = self.bench[location]
            self.print("selling champion " + self.bench[location].name + " with stars = " +
                       str(self.bench[location].stars) + " from bench_location " + str(location))
            self.bench[location] = None
            return return_champ
        if DEBUG:
            print("Nothing at bench location")
        return False

    # --- Move Action --- #
    def move_champ_action(self, x1, x2):
        """Moves champion from one location to another.

        0-27 -> Board Slots
        28-36 -> Bench Slots

        Conditions:
            - x1 must be between 0 and 36
            - x2 must be between 0 and 36
            - x1 must have a champion

        Args:
            x1 (int): Index of the champion to move.
            x2 (int): Index of the location to move to.

        Returns:
            bool: True if action was performed successfully, False otherwise.
        """
        if (x1 < 0 or x1 > 36) or (x2 < 0 or x2 > 36):
            if DEBUG:
                print(f"Invalid move index 1 -> [x1, x2] -> [{x1}, {x2}]")
            return False

        # I forgot why, but the smaller index needed to be first... maybe not?
        move_loc_from = min(x1, x2)
        move_loc_to = max(x1, x2)

        # Board to (Bench/Board)
        if move_loc_from < 28:
            # Board to Board
            if move_loc_to < 28:
                x1, y1 = utils.coord_to_x_y(move_loc_from)
                x2, y2 = utils.coord_to_x_y(move_loc_to)

                if self.board[x1][y1]:
                    self.move_board_to_board(x1, y1, x2, y2)
                elif self.board[x2][y2]:
                    self.move_board_to_board(x2, y2, x1, y1)
                else:
                    if DEBUG:
                        print(f"from {move_loc_from}, to {move_loc_to} with {self.board[x1][y1]} {self.board[x2][y2]}")
                        print(f"No champion to move for player {self.player_num} in round {self.round}")
                    return False

            # Board to Bench
            else:
                x1, y1 = utils.coord_to_x_y(move_loc_from)
                bench_loc = move_loc_to - 28

                if self.bench[bench_loc]:
                    self.move_bench_to_board(bench_loc, x1, y1)
                elif not self.bench_full():
                    self.move_board_to_bench(x1, y1, bench_loc)
                else:
                    if DEBUG:
                        print(f"Bench is full for player {self.player_num}")
                    return False
        else:
            if DEBUG:
                print("Currently, bench to bench is disabled")
            return False

        return True

    """
    Description - Moves a unit from bench to board if possible. Will switch if max units on board and board slot is used
    Inputs - dcord: Int
                    For example, 27 -> 6 for x and 3 for y
    Outputs - x: Int
                    x_coord
                  y: Int
                    y_coord
    """

    def move_bench_to_board(self, bench_x, board_x, board_y):
        if 0 <= bench_x < 9 and self.bench[bench_x] and 7 > board_x >= 0 and 4 > board_y >= 0:
            if self.num_units_in_play < self.max_units or self.board[board_x][board_y] is not None:
                # TO DO - IMPLEMENT AZIR TURRET SPAWNS
                m_champion = self.bench[bench_x]
                m_champion.x = board_x
                m_champion.y = board_y
                self.bench[bench_x] = None
                if self.board[board_x][board_y]:
                    if not self.move_board_to_bench(board_x, board_y):
                        self.bench[bench_x] = m_champion
                        m_champion.x = bench_x
                        m_champion.y = -1
                        self.print("Failed to move {} from bench {} to board [{}, {}]; {} already on board"
                                   .format(self.bench[bench_x].name, bench_x, board_x, board_y,
                                           self.board[board_x][board_y].name))
                        if DEBUG:
                            print("Failed to move {} from bench {} to board [{}, {}]; {} already on board"
                                  .format(self.bench[bench_x].name, bench_x, board_x, board_y,
                                          self.board[board_x][board_y].name))
                        return False
                self.board[board_x][board_y] = m_champion
                # tracking thiefs gloves location
                if len(m_champion.items) > 0:
                    if m_champion.items[0] == 'thieves_gloves':
                        self.thieves_gloves_loc_update(
                            bench_x, -1, board_x, board_y)
                if m_champion.name == 'azir':
                    # There should never be a situation where the board is to fill to fit the sand guards.
                    sand_coords = self.find_azir_sandguards(board_x, board_y)
                    self.board[board_x][board_y].overlord = True
                    self.board[board_x][board_y].sandguard_overlord_coordinates = sand_coords
                self.num_units_in_play += 1
                self.print("moved {} from bench {} to board [{}, {}]".format(self.board[board_x][board_y].name,
                                                                             bench_x, board_x, board_y))
                self.update_team_tiers()
                return True
        self.reward += self.mistake_reward
        if DEBUG:
            print(f"Outside board range, bench: {self.bench[bench_x]}, board: {self.board[board_x][board_y]}, \
                             bench_x: {bench_x}, board_x: {board_x}, board_y: {board_y}, \
                             with units in play {self.num_units_in_play} and max units {self.max_units}")
        return False

    """
    Description - Moves a champion to the first open bench slot available
    Inputs      - x, y: Int
                    coords on the board to move to the board
    Outputs     - True if successful
                  False if coords are outside allowable range or could not sell unit
    """

    def move_board_to_bench(self, x, y, x_bench=None) -> bool:
        if x_bench is None:
            x_bench = self.bench_vacancy()
        if 0 <= x < 7 and 0 <= y < 4 and 0 <= x_bench < 9:
            if self.bench[x_bench] and self.board[x][y] and not self.board[x][y].target_dummy:
                s_champion = self.bench[x_bench]
                if self.board[x][y]:
                    self.bench[x_bench] = self.board[x][y]
                    if self.board[x][y].name == 'azir':
                        coords = self.board[x][y].sandguard_overlord_coordinates
                        self.board[x][y].overlord = False
                        for coord in coords:
                            self.board[coord[0]][coord[1]] = None
                    self.board[x][y] = s_champion
                    self.board[x][y].x = x
                    self.board[x][y].y = y
                    self.bench[x_bench].x = x_bench
                    self.bench[x_bench].y = -1
                    if self.bench[x_bench].items and self.bench[x_bench].items[0] == 'thieves_gloves':
                        self.thieves_gloves_loc_update(x_bench, -1, x, y)
                    if self.board[x][y].items and self.board[x][y].items[0] == 'thieves_gloves':
                        self.thieves_gloves_loc_update(x, y, x_bench, -1)
                    self.update_team_tiers()
                    self.print("moved {} from board [{}, {}] to bench {}".format(self.bench[x_bench].name, x, y, x_bench) +
                               ", and moved {} from bench {} to board [{}, {}]".format(self.board[x][y].name, x_bench, x, y))
                    return True
                else:
                    self.reward += self.mistake_reward
                if DEBUG:
                    print("Unit not on board slot")
                return False
            else:
                if self.board[x][y] and not self.board[x][y].target_dummy:
                    # Dealing with edge case of azir
                    if self.board[x][y].name == 'azir':
                        coords = self.board[x][y].sandguard_overlord_coordinates
                        self.board[x][y].overlord = False
                        for coord in coords:
                            self.board[coord[0]][coord[1]] = None
                    self.bench[x_bench] = self.board[x][y]
                    if self.board[x][y]:
                        self.print("moved {} from board [{}, {}] to bench".format(
                            self.board[x][y].name, x, y))
                    self.board[x][y] = None
                    self.bench[x_bench].x = x_bench
                    self.bench[x_bench].y = -1
                    self.num_units_in_play -= 1
                    if self.bench[x_bench].items and self.bench[x_bench].items[0] == 'thieves_gloves':
                        self.thieves_gloves_loc_update(x_bench, -1, x, y)
                    self.update_team_tiers()
                    return True
        self.reward += self.mistake_reward
        if DEBUG:
            print(f"Move board to bench outside board limits: {x}, {y}, {x_bench}, {self.bench[x_bench]}, {self.board[x][y]}")
            # action = ActionToken(self)
            # np.set_printoptions(threshold=np.inf)
            print(f"player # = {self.player_num} with game_round {self.round}")
            # print(self.board)
            # print(action.move_sell_board_mask)
            # print(self.bench)
            # print(action.move_sell_bench_mask)
            # print('----------------- OBS MASK ---------------')
            # print(np.reshape(action.fetch_action_mask(), (55, 38)))
        return False

    """
    Description - Switches the contents of two squares on the board
    Inputs      - x1, y1: Int
                    coords on the board for square 1
                  x2, y2: Int
                    coords on the board for square 2
    Outputs     - True if successful
                  False if outside board limit or nothing on either square
    """

    def move_board_to_board(self, x1, y1, x2, y2) -> bool:
        if 0 <= x1 < 7 and 0 <= y1 < 4 and 0 <= x2 < 7 and 0 <= y2 < 4:
            if self.board[x1][y1] and self.board[x2][y2]:
                temp_champ = self.board[x2][y2]
                if (self.board[x1][y1].items and self.board[x1][y1].items[0] == 'thieves_gloves') or \
                   (self.board[x2][y2].items and self.board[x2][y2].items[0] == 'thieves_gloves'):
                    self.thieves_gloves_loc_update(x1, y1, x2, y2)
                self.board[x2][y2] = self.board[x1][y1]
                self.board[x1][y1] = temp_champ
                self.board[x1][y1].x = x1
                self.board[x1][y1].y = y1
                self.board[x2][y2].x = x2
                self.board[x2][y2].y = y2
                if self.board[x1][y1].name == 'sandguard' or self.board[x2][y2].name == 'sandguard':
                    for x in range(7):
                        for y in range(4):
                            if self.board[x][y] and self.board[x][y].name == 'azir':
                                if [x1, y1] in self.board[x][y].sandguard_overlord_coordinates and \
                                        [x2, y2] not in self.board[x][y].sandguard_overlord_coordinates:
                                    self.board[x][y].sandguard_overlord_coordinates.remove([x1, y1])
                                    self.board[x][y].sandguard_overlord_coordinates.append([x2, y2])
                                elif [x2, y2] in self.board[x][y].sandguard_overlord_coordinates and \
                                        [x1, y1] not in self.board[x][y].sandguard_overlord_coordinates:
                                    self.board[x][y].sandguard_overlord_coordinates.remove([x2, y2])
                                    self.board[x][y].sandguard_overlord_coordinates.append([x1, y1])
                self.print("moved {} and {} from board [{}, {}] to board [{}, {}]"
                           .format(self.board[x1][y1].name, self.board[x2][y2].name, x1, y1, x2, y2))
                return True
            elif self.board[x1][y1]:
                if self.board[x1][y1].items and self.board[x1][y1].items[0] == 'thieves_gloves':
                    self.thieves_gloves_loc_update(x1, y1, x2, y2)
                self.board[x2][y2] = self.board[x1][y1]
                self.board[x1][y1] = None
                self.board[x2][y2].x = x2
                self.board[x2][y2].y = y2
                if self.board[x2][y2].name == 'sandguard':
                    for x in range(7):
                        for y in range(4):
                            if self.board[x][y] and self.board[x][y].name == 'azir':
                                if [x1, y1] in self.board[x][y].sandguard_overlord_coordinates:
                                    self.board[x][y].sandguard_overlord_coordinates.remove([
                                                                                           x1, y1])
                                    self.board[x][y].sandguard_overlord_coordinates.append([
                                                                                           x2, y2])
                self.print("moved {} from board [{}, {}] to board [{}, {}]".format(
                    self.board[x2][y2].name, x1, y1, x2, y2))
                return True
            # if called in wrong order
            elif self.board[x2][y2]:
                return self.move_board_to_board(x2, y2, x1, y1)
        self.reward += self.mistake_reward
        if DEBUG:
            print(f"Outside board limits -> ({x1}, {y1}) to ({x2}, {y2})")
            if 0 <= x1 < 7 and 0 <= y1 < 4 and 0 <= x2 < 7 and 0 <= y2 < 4:
                if self.board[x1][y1]:
                    print(f"At board {x1}, {y1} -> {self.board[x1][y1].name}")
                if self.board[x2][y2]:
                    print(f"At board {x2}, {y2} -> {self.board[x2][y2].name}")
        return False

    # --- Item Action --- #
    def move_item_action(self, x1, x2):
        """Move item from item bench to champion.

        Conditions:
            x1 must be between 0 and 9
            x2 must be between 0 and 36

        Args:
            x1 (int): Index of the item bench.
            x2 (int): Index of the champion to move the item to.

        Returns:
            bool: True if action was performed successfully, False otherwise.
        """
        if (x1 < 0 or x1 > 9) or (x2 < 0 or x2 > 36):
            if DEBUG:
                print(f"Invalid move index 2 -> [x1, x2] -> [{x1}, {x2}]")
            return False

        item_loc = x1
        move_loc = x2

        if move_loc < 28:
            x, y = utils.coord_to_x_y(move_loc)
            self.move_item_to_board(item_loc, x, y)
        else:
            bench_loc = move_loc - 28
            self.move_item_to_bench(item_loc, bench_loc)

        return True

    def add_to_item_bench(self, item):
        """Adds an item to the item bench.

        Conditions:
            - Item bench is not full

        Args:
            item (str): Item to add to the item bench.

        Returns:
            bool: True if action was performed successfully, False otherwise.
        """
        if self.item_bench_full(1):
            if DEBUG:
                print("Failed to add item to item bench")
            return False

        bench_loc = self.item_bench_vacancy()
        self.item_bench[bench_loc] = item

    # --- Query Functions ---
    def bench_full(self):
        """Queries if the bench is full.

        Returns:
            bool: True if bench is full, False otherwise.
        """
        return all(self.bench)

    def bench_vacancy(self):
        """Returns the spot on the champion bench where there is a vacancy.

        Returns:
            int: location on bench where there is a vacancy, -1 otherwise.
        """
        for idx, champ in enumerate(self.bench):
            if not champ:
                return idx
        return -1

    def default_guide(self, champion_list):
        self.default_agent.set_default_guide(champion_list)

    def default_policy(self, game_round, shop, mask):
        return self.default_agent.policy(self, shop, game_round, mask)

    def item_bench_full(self, num_of_items=0) -> bool:
        counter = 0
        for i in self.item_bench:
            if i:
                counter += 1
        if counter + num_of_items > len(self.item_bench):
            return True
        else:
            return False

    def item_bench_vacancy(self) -> int or False:
        for free_slot, u in enumerate(self.item_bench):
            if not u:
                return free_slot
        return False

    def item_bench_open_slots(self) -> int:
        return self.item_bench.count(None)

    def shop_empty(self):
        """Queries if the shop is empty.

        Returns:
            bool: True if shop is empty, False otherwise.
        """
        return not all(self.shop)

    # --- Game Mechanics Functions --- #
    def gold_income(self, t_round):
        self.exp += 2
        self.level_up()
        if t_round <= 4:
            starting_round_gold = [0, 2, 2, 3, 4]
            self.gold += floor(self.gold / 10)
            self.gold += starting_round_gold[t_round]
            return
        interest = min(floor(self.gold / 10), 5)
        self.gold += interest
        self.gold += 5
        if self.win_streak == 2 or self.win_streak == 3 or self.loss_streak == 2 or self.loss_streak == 3:
            self.gold += 1
        elif self.win_streak == 4 or self.loss_streak == 4:
            self.gold += 2
        elif self.win_streak >= 5 or self.loss_streak >= 5:
            self.gold += 3

    def end_turn_actions(self):
        # autofill the board.
        num_units_to_move = self.max_units - self.num_units_in_play
        position_found = -1
        for _ in range(num_units_to_move):
            found_position = False
            for i in range(position_found + 1, len(self.bench)):
                if self.bench[i]:
                    for x in range(len(self.board)):
                        for y in range(len(self.board[0])):
                            if self.board[x][y] is None:
                                self.move_bench_to_board(i, x, y)
                                found_position = True
                                position_found = i
                                break
                        if found_position:
                            break
                if found_position:
                    break
        self.print(
            f"Spaces for units left to fight {self.max_units - self.num_units_in_play}")
        # update board to survive combat = False, will update after combat if they survived
        # update board to participated in combat
        for x in range(len(self.board)):
            for y in range(len(self.board[0])):
                if self.board[x][y]:
                    self.board[x][y].participated_in_combat = True
                    self.board[x][y].survive_combat = False
        # update bench to did not participate in combat
        for x in range(len(self.bench)):
            if self.bench[x]:
                self.bench[x].participated_in_combat = False
                self.bench[x].survive_combat = False

    # --- Edge Case Functions ---
    """
    Description - Finds locations to put azir's sandguards when he first gets put on the field
    Inputs - Azir's x and y coordinates
    Outputs - Two sets of coordinates
    """

    def find_azir_sandguards(self, azir_x, azir_y):
        coords_candidates = self.find_free_squares(azir_x, azir_y)
        x = 6
        y = 3
        while len(coords_candidates) < 2:
            hexes = self.find_free_squares(x, y)
            for hex in hexes:
                if hex not in coords_candidates:
                    coords_candidates.append(hex)
            x -= 1
            if x == -1:
                x = 6
                y -= 1

        for x in range(2):
            self.board[coords_candidates[x][0]][coords_candidates[x][1]] = \
                champion.champion('sandguard',kayn_form=self.kayn_form, target_dummy=True)
        coords = [coords_candidates[0], coords_candidates[1]]
        return coords

    """
    Description - Finds free squares around a coordinate
    Inputs - Coordinate
    Outputs - Free squares surrounding the coordinate
    """

    def find_free_squares(self, x, y):
        if x < 0 or x > 6 or y < 0 or y > 3:
            return []
        directions = [
            [[+1, 0], [+1, +1], [0, -1],
             [0, +1], [-1, 0], [-1, +1]],
            [[+1, -1], [+1, 0], [0, -1],
             [0, +1], [-1, -1], [-1, 0]],
        ]

        parity = y & 1
        neighbors = []
        for c in directions[parity]:
            nY = c[0] + y
            nX = c[1] + x
            if (0 <= nY < 4 and 0 <= nX < 7) and not self.board[x][y]:
                neighbors.append([nX, nY])
        return neighbors

    def get_champion_labels(self):
        self.update_comp_labels()
        return self.team_champion_labels

    def get_item_labels(self):
        item_labels = np.zeros([len(self.item_bench), config.MAX_ITEMS_IN_SET + 1], np.int8)
        for i, item_name in enumerate(self.item_bench):
            if item_name is None:
                item_labels[i][-1] = 1
            else:
                c_index = list(items.keys()).index(item_name)
                item_labels[i][c_index - 1] = 1
        return item_labels

    def get_scalar_labels(self):
        scalar_labels = np.zeros([3, 100], np.int8)
        if self.gold < 100:
            scalar_labels[0][self.gold] = 1
        else:
            scalar_labels[0][-1] = 1
        scalar_labels[1][self.exp] = 1
        scalar_labels[2][self.health - 1] = 1
        return scalar_labels

    def get_shop_labels(self):
        shop_labels = np.zeros([len(self.shop), config.MAX_CHAMPION_IN_SET + 1], np.int8)
        for i, champion_name in enumerate(self.shop):
            if champion_name is None:
                shop_labels[i][-1] = 1
            elif champion_name.endswith("_c"):
                champion_name = champion_name[:-2]
                # Chosen champions are defined as `<name>_<trait>_c`
                champion_name, _ = champion_name.split("_")
                c_index = list(COST.keys()).index(champion_name)
                shop_labels[i][c_index - 1] = 1
            else:
                c_index = list(COST.keys()).index(champion_name)
                shop_labels[i][c_index - 1] = 1
        return shop_labels

    def get_tier_labels(self):
        self.generate_tier_vector()
        return self.team_tier_labels

    """
    Description - Checks if there's a kayn on the board
    """

    def kayn_check(self) -> bool:
        for x in range(0, 7):
            for y in range(0, 4):
                if self.board[x][y]:
                    if self.board[x][y].name == "kayn":
                        return True
        return False

    """
    Description - Gives the player the kayn items that allow the player to select kayn's form
    Outputs - Sets kayn_transformed to true so it only happens once
    """

    def kayn_transform(self):
        if not self.kayn_transformed:
            if not self.item_bench_full(2):
                self.add_to_item_bench('kayn_shadowassassin')
                self.add_to_item_bench('kayn_rhast')
                self.kayn_transformed = True

    """
    Description - Moves an item or consumable from the item bench to either the player bench or player board
    Inputs      - item bench slot to move, [x, y] location to be moved to (y = -1 for the bench)
    Outputs     - boolean whether or not the move was successful
    """
    # TODO : Item combinations.
    # TODO : Documentation and setting up correct masks.

    def move_item(self, xBench, x, y) -> bool:
        board = False
        if y >= 0:
            champ = self.board[x][y]
            board = True
        if y == -1:
            champ = self.bench[x]
        if self.item_bench[xBench] and champ and not champ.target_dummy:
            # thiefs glove exception
            self.print("moving {} to {} with items {}".format(
                self.item_bench[xBench], champ.name, champ.items))
            # kayn item support
            if self.item_bench[xBench] == 'kayn_shadowassassin' or \
                    self.item_bench[xBench] == 'kayn_rhast':
                if champ.name == 'kayn':
                    self.transform_kayn(self.item_bench[xBench])
                    return True
                if DEBUG:
                    print("Applying kayn item on not kayn")
                return False
            if self.item_bench[xBench] == 'champion_duplicator':
                if COST[champ.name] != 0:
                    if not self.bench_full():
                        self.add_to_bench(champion.champion(
                            champ.name, chosen=champ.chosen, kayn_form=champ.kayn_form))
                        self.item_bench[xBench] = None
                        return True
                return False
            if self.item_bench[xBench] == 'magnetic_remover':
                if len(champ.items) > 0:
                    if not self.item_bench_full(len(champ.items)):
                        while len(champ.items) > 0:
                            self.item_bench[self.item_bench_vacancy(
                            )] = champ.items[0]
                            if champ.items[0] in trait_items.values():
                                champ.origin.pop(-1)
                                self.update_team_tiers()
                            champ.items.pop(0)
                        self.item_bench[xBench] = None
                        return True
                if DEBUG:
                    print("Applying magnetic remover to a champion with no items")
                return False
            if self.item_bench[xBench] == 'reforger':
                return self.use_reforge(xBench, x, y)
            if self.item_bench[xBench] == 'thieves_gloves':
                if len(champ.items) < 1:
                    champ.items.append(self.item_bench[xBench])
                    self.item_bench[xBench] = None
                    self.thieves_gloves_loc.append([x, y])
                    self.thieves_gloves(x, y)
                    return True
                if DEBUG:
                    print("Trying to add thieves gloves to unit with a separate item")
                return False
            # TODO: Clean up this code, we already checked for thieves_glove by this point
            if ((len(champ.items) < 3 and self.item_bench[xBench] != "thieves_gloves") or
                    (champ.items and champ.items[-1] in basic_items and self.item_bench[xBench]
                     in basic_items and len(champ.items) == 3)):
                for trait, name in enumerate(trait_items.values()):
                    if self.item_bench[xBench] == name:
                        item_trait = list(trait_items.keys())[trait]
                        if item_trait in champ.origin:
                            if DEBUG:
                                print(
                                    "Trying to add trait item to unit with that trait")
                            return False
                        else:
                            champ.origin.append(item_trait)
                            self.update_team_tiers()
                # only execute if you have items
                if len(champ.items) > 0:
                    # implement the item combinations here. Make exception with thieves gloves
                    if champ.items[-1] in basic_items and self.item_bench[xBench] in basic_items:
                        item_build_values = item_builds.values()
                        item_index = 0
                        item_names = list(item_builds.keys())
                        for index, items in enumerate(item_build_values):
                            if ((champ.items[-1] == items[0] and self.item_bench[xBench] == items[1]) or
                                    (champ.items[-1] == items[1] and self.item_bench[xBench] == items[0])):
                                item_index = index
                                break
                        for trait, names in enumerate(list(trait_items.values())):
                            if item_names[item_index] == names:
                                item_trait = list(trait_items.keys())[trait]
                                if item_trait in champ.origin:
                                    if DEBUG:
                                        print(
                                            "trying to combine trait item to unit with that trait")
                                    return False
                                else:
                                    champ.origin.append(item_trait)
                                    self.update_team_tiers()
                        if item_names[item_index] == "thieves_gloves":
                            if champ.num_items != 1:
                                if DEBUG:
                                    print(
                                        "Trying to combine thieves gloves in unit with a separate item",  x, y)
                                return False
                            else:
                                self.thieves_gloves_loc.append([x, y])
                        self.item_bench[xBench] = None
                        champ.items.pop()
                        champ.items.append(item_names[item_index])
                        if champ.items[0] == 'thieves_gloves':
                            self.thieves_gloves(x, y)
                        self.reward += .2 * self.item_reward
                        self.print("{} reward for combining two basic items into a {}"
                                   .format(.2 * self.item_reward, item_names[item_index]))
                    elif champ.items[-1] in basic_items and self.item_bench[xBench] not in basic_items:
                        basic_piece = champ.items.pop()
                        champ.items.append(self.item_bench[xBench])
                        champ.items.append(basic_piece)
                        self.item_bench[xBench] = None
                    else:
                        champ.items.append(self.item_bench[xBench])
                        self.item_bench[xBench] = None
                else:
                    champ.items.append(self.item_bench[xBench])
                    self.item_bench[xBench] = None
                self.print("After Move {} to {} with items {}".format(self.item_bench[xBench], champ.name,
                                                                      champ.items))
                return True
            elif len(champ.items) < 1 and self.item_bench[xBench] == "thieves_gloves":
                champ.items.append(self.item_bench[xBench])
                self.item_bench[xBench] = None
                self.print("After Move {} to {} with items {}".format(
                    self.item_bench[xBench], champ.name, champ.items))
                self.thieves_gloves_loc.append([x, -1])
                return True
        # last case where 3 items but the last item is a basic item and the item to input is also a basic item
        self.reward += self.mistake_reward
        if DEBUG:
            print(
                f"Failed to add item {self.item_bench[xBench]} in slot {xBench} to {champ} in {x}, {y}.")
            if champ:
                print(f'{champ} had {len(champ.items)} items')
        return False

    """
    Description - Uses the move_item method to move an item or consumable from the item bench to the player bench
    Inputs      - see move_item
    Outputs     - see move_item
    """

    def move_item_to_bench(self, xBench, x):
        self.move_item(xBench, x, -1)

    """
    Description - Uses the move_item method to move an item or consumable from the item bench to the board
    Inputs      - see move_item
    Outputs     - see move_item
    """

    def move_item_to_board(self, xBench, x, y):
        self.move_item(xBench, x, y)

    # --- Pass Action --- #
    def pass_action(self):
        """Does not update the player state.

        Returns:
            bool: True if action was performed successfully, False otherwise.
        """
        self.print("Passing Action")
        return True

    # --- Log Functions --- #
    """
    Description -
    Inputs      -
    Outputs     -
    """

    def print(self, msg):
        self.printt('{:<120}'.format('{:<8}'.format(self.player_num)
                                     + '{:<20}'.format(str((time.time_ns() - self.start_time)/1000)) + msg))

    """
    Description -
    Inputs      -
    Outputs     -
    """

    def printBench(self, log=True):
        for i in range(len(self.bench)):
            if self.bench[i]:
                if log:
                    self.print(str(i) + ": " + self.bench[i].name)
                else:
                    print(self.bench[i].name + ", ")

    """
    Description -
    Inputs      -
    Outputs     -
    """

    def printComp(self, log=True, to_console=False):
        keys = list(self.team_composition.keys())
        values = list(self.team_composition.values())
        tier_values = list(self.team_tiers.values())
        self.prev_rewards = self.reward
        for i in range(len(self.team_composition)):
            if values[i] != 0:
                if log:
                    self.print("{}: {}, tier: {}".format(keys[i], values[i], tier_values[i]))
                if to_console:
                    print("{}: {}, tier: {}".format(keys[i], values[i], tier_values[i]))
        for x in range(7):
            for y in range(4):
                if self.board[x][y]:
                    self.print("at ({}, {}), champion {}, with level = {}, items = {}, and chosen = {}".format(x, y,
                               self.board[x][y].name, self.board[x][y].stars,
                               self.board[x][y].items, self.board[x][y].chosen))
                    if to_console:
                        print("at ({}, {}), champion {}, with level = {}, items = {}, and chosen = {}".format(x, y,
                               self.board[x][y].name, self.board[x][y].stars,
                               self.board[x][y].items, self.board[x][y].chosen))
        self.print("Player level {} with gold {}, max_units = {}, ".format(self.level, self.gold, self.max_units) +
                   "num_units_in_play = {}, health = {}, ".format(self.num_units_in_play, self.health) +
                   "default {}".format(self.default_player))
        if to_console:
            print("Player level {} with gold {}, max_units = {}, ".format(self.level, self.gold, self.max_units) +
                  "num_units_in_play = {}, health = {}, ".format(self.num_units_in_play, self.health) +
                  "default {}".format(self.default_player))

    """
    Description -
    Inputs      -
    Outputs     -
    """

    def printItemBench(self, log=True):
        for i, item in enumerate(self.item_bench):
            if item:
                if log:
                    self.print(str(i) + ": " + item)
                else:
                    print(str(i) + ": " + item)

    """
    Description -
    Inputs      -
    Outputs     -
    """

    def printShop(self, shop):
        self.print("Shop with level " + str(self.level) + ": " +
                   shop[0] + ", " + shop[1] + ", " + shop[2] + ", " + shop[3] + ", " + shop[4])

    """
    Description -
    Inputs      -
    Outputs     -
    """

    def printt(self, msg):
        if config.PRINTMESSAGES:
            self.log.append(msg)

    def random_item_from_pool(self):
        """Picks and returns item to player

        Returns:
            str: item to be returned.
        """
        item = random.choice(self.item_pool)
        self.remove_from_pool(item)
        return item

    def refill_item_pool(self):
        """Refills the item pool"""
        self.item_pool.extend(starting_items)

    def remove_from_pool(self, item):
        """Removes item from item pool

        Args:
            item (str): item to be removed from item pool.
        """
        self.item_pool.remove(item)

    def reinit_numpy_arrays(self):
        self.team_champion_labels = np.zeros([len(CHAMPION_ACTION_DIM), 2], dtype=np.int8)


    # TODO: Handle case where item_bench if full
    # TODO: Thieves_gloves bug appeared again on self.thieves_gloves_loc.remove([x, -1])
    def return_item_from_bench(self, x) -> bool:
        """Returns item from a given champion.

        Args:
            x (int): Bench location of champion to sell.

        Returns:
            bool: True if item was returned to item bench. False if item was not able to be returned.
        """
        # if the unit exists
        if self.bench[x]:
            # skip if there are no items, trying to save a little processing time.
            if self.bench[x].items:

                # thieves_gloves_loc_always needs to be cleared even if there's not enough room on bench
                if self.bench[x].items[0] == 'thieves_gloves':
                    self.thieves_gloves_loc.remove([x, -1])
                    self.bench[x].items = ['thieves_gloves']

                # if I have enough space on the item bench for the number of items needed
                if not self.item_bench_full(len(self.bench[x].items)):
                    # Each item in possession
                    for i in self.bench[x].items:
                        # thieves glove exception
                        self.item_bench[self.item_bench_vacancy()] = i
                        self.print("returning " + i + " to the item bench")
                # if there is only one or two spots left on the item_bench and thieves_gloves is removed
                elif not self.item_bench_full(1) and self.bench[x].items[0] == "thieves_gloves":
                    self.item_bench[self.item_bench_cvacancy()
                                    ] = self.bench[x].items[0]
                    self.print(
                        "returning " + str(self.bench[x].items[0]) + " to the item bench")
                self.bench[x].items = []
            return True
        if DEBUG:
            print("No units at bench location {}".format(x))
        self.print("No units at bench location {}".format(x))
        return False

    def return_item(self, a_champion) -> bool:
        """Returns item from a given champion.

        Only used when bench is full and trying to add unit from carousel or minion round.

        Args:
            a_champion (Champion): Champion object to sell.

        Returns:
            bool: True: Able to return the item to the item bench.
                  False: Unable to return the item or method called with a NULL champion.
        """
        # if the unit exists
        if a_champion:
            # skip if there are no items, trying to save a little processing time.
            if a_champion.items:
                # thieves_gloves_location needs to be removed whether there's room on the bench or not
                if a_champion.items[0] == 'thieves_gloves':
                    if [a_champion.x, a_champion.y] in self.thieves_gloves_loc:
                        self.thieves_gloves_loc.remove(
                            [a_champion.x, a_champion.y])
                    a_champion.items = ['thieves_gloves']
                # if I have enough space on the item bench for the number of items needed
                if not self.item_bench_full(len(a_champion.items)):
                    # Each item in possession
                    for item in a_champion.items:
                        if item in trait_items.values():
                            a_champion.origin.pop(-1)
                            self.update_team_tiers()
                        self.item_bench[self.item_bench_vacancy()] = item
                        self.print("returning " + item + " to the item bench")

                # if there is only one or two spots left on the item_bench and thieves_gloves is removed
                elif not self.item_bench_full(1) and a_champion.items[0] == "thieves_gloves":
                    self.item_bench[self.item_bench_vacancy()
                                    ] = a_champion.items[0]
                    self.print("returning " +
                               str(a_champion.items[0]) + " to the item bench")
                else:
                    self.print("Could not remove item {} from champion {}".format(
                        a_champion.items, a_champion.name))
                    if DEBUG:
                        print("Could not remove item {} from champion {}".format(
                            a_champion.items, a_champion.name))
                    return False
                a_champion.items = []

            return True
        if DEBUG:
            print("Null champion")
        return False

    def reset_state(self):
        """Used in unit tests to allow for a cleaner state."""
        self.bench = [None for _ in range(9)]
        self.board = [[None for _ in range(4)] for _ in range(7)]
        self.item_bench = [None for _ in range(10)]
        self.gold = 0
        self.level = 1
        self.exp = 0
        self.health = 100
        self.max_units = 1
        self.num_units_in_play = 0

    def spill_reward(self, damage):
        """Gives reward for taking damage

        Args:
            damage (int): amount of damage taken
        """
        if ALLOW_SPILL:
            self.reward += self.damage_reward * damage
            self.print("Spill reward of {} received".format(
                self.damage_reward * damage))

    def start_round(self, t_round):
        """Does all operations that happen at the start of the round.

        This includes gold, reward, kayn updates and thieves gloves

        Args:
            t_round (int): current game round
        """
        self.start_time = time.time_ns()
        self.round = t_round
        self.reward += self.num_units_in_play * self.minion_count_reward
        self.gold_income(self.round)
        if self.kayn_check():
            self.kayn_turn_count += 1
        if self.kayn_turn_count >= 3:
            self.kayn_transform()
        for x in self.thieves_gloves_loc:
            if (x[1] != -1 and self.board[x[0]][x[1]]) or self.bench[x[0]]:
                self.thieves_gloves(x[0], x[1])

        self.printComp()
        self.printBench()
        self.printItemBench()

    def state_empty(self):
        """Checks if there are no possible actions in the state

        Returns:
            bool: True if there are no possible actions, False otherwise
        """
        # Need both in case of an empty shop.
        # Assume that if all that is left is a chosen unit, that is the same as no unit.
        if self.gold == 0 or all((v is None or v.endswith("_c") or COST[v] > self.gold) for v in self.shop):
            for xbench in self.bench:
                if xbench:
                    return False
            for x_board in range(len(self.board)):
                for y_board in range(len(self.board[0])):
                    if self.board[x_board][y_board]:
                        return False
            return True
        else:
            return False

    def team_origin_class(self):
        team = self.board
        for trait in self.team_composition.keys():
            self.team_composition[trait] = 0
        unique_champions = []
        for x in range(len(self.board)):
            for y in range(len(self.board[x])):
                if team[x][y]:
                    if team[x][y].name not in unique_champions:
                        unique_champions.append(team[x][y].name)
                        for trait in team[x][y].origin:
                            self.team_composition[trait] += 1
                    for item in team[x][y].items:
                        if item in trait_items.values():
                            item_index = list(trait_items.values()).index(item)
                            class_trait = list(trait_items.keys())[item_index]
                            self.team_composition[class_trait] += 1

    def thieves_gloves(self, x, y) -> bool:
        """Gives new thieves gloves items to a champion in thieves_gloves_loc

        Args:
            x (int): x coordinate of the unit to give items to
            y (int): y coordinate of the unit to give items to

        Returns:
            bool: True if successful, False otherwise
        """
        r1 = random.randint(0, len(thieves_gloves_items) - 1)
        r2 = random.randint(0, len(thieves_gloves_items) - 1)
        while r1 == r2:
            r2 = random.randint(0, len(thieves_gloves_items) - 1)
        self.print("thieves_gloves: {} and {}".format(
            thieves_gloves_items[r1], thieves_gloves_items[r2]))
        if y >= 0:
            self.board[x][y].items = ['thieves_gloves']
            self.board[x][y].items.append(thieves_gloves_items[r1])
            self.board[x][y].items.append(thieves_gloves_items[r2])
            return True
        elif y == -1:
            self.bench[x].items = ['thieves_gloves']
            self.bench[x].items.append(thieves_gloves_items[r1])
            self.bench[x].items.append(thieves_gloves_items[r2])
            return True
        else:
            if DEBUG:
                print("Could not assign thieves glove items")
            return False

    def thieves_gloves_loc_update(self, x1, y1, x2, y2):
        """Checks if either of 2 coordinates is in thieves_gloves_loc and swaps it for the one that isn't

        Args:
            x1 (int): x coordinate of the first loc to swap
            y1 (int): y coordinate of the first loc to swap
            x2 (int): x coordinate of the second loc to swap
            y2 (int): y coordinate of the second loc to swap
        """
        if [x1, y1] in self.thieves_gloves_loc and [x2, y2] in self.thieves_gloves_loc:
            return True
        elif [x1, y1] in self.thieves_gloves_loc:
            self.thieves_gloves_loc.remove([x1, y1])
            self.thieves_gloves_loc.append([x2, y2])
            self.thieves_mask_update(x1, y1, x2, y2)
        elif [x2, y2] in self.thieves_gloves_loc:
            self.thieves_gloves_loc.remove([x2, y2])
            self.thieves_gloves_loc.append([x1, y1])
            self.thieves_mask_update(x2, y2, x1, y1)

    def thieves_mask_update(self, x1, y1, x2, y2):
        """Updates the thieves glove mask

        Args:
            x1 (int): x coordinate of the unit to remove
            y1 (int): y coordinate of the unit to remove
            x2 (int): x coordinate of the unit to add
            y2 (int): y coordinate of the unit to add
        """
        coord_remove = utils.x_y_to_1d_coord(x1, y1)

        coord_add = utils.x_y_to_1d_coord(x2, y2)

    def transform_kayn(self, kayn_item):
        """Transforms Kayn into either shadowassassin or rhast based on which item the player used

        Args:
            kayn_item (str): Either 'kayn_shadowassassin' or 'kayn_rhast'

        Returns:
            None: Transforms all Kayns (board, bench and shop)
        """
        self.kayn_form = kayn_item
        for x in range(len(self.item_bench)):
            if self.item_bench[x] == 'kayn_shadowassassin' or self.item_bench[x] == 'kayn_rhast':
                self.item_bench[x] = None
        for x in range(7):
            for y in range(4):
                if self.board[x][y]:
                    if self.board[x][y].name == 'kayn':
                        self.board[x][y].kayn_form = kayn_item
        for x in range(9):
            if self.bench[x]:
                if self.bench[x].name == 'kayn':
                    self.bench[x].kaynform = kayn_item

    def update_comp_labels(self):
        self.team_champion_labels[:, 0] = 1
        self.team_champion_labels[:, 1] = 0

        for x in range(len(self.board)):
            for y in range(len(self.board[x])):
                if self.board[x][y]:
                    c_index = list(COST.keys()).index(self.board[x][y].name)
                    # create the label for the champion to help with training
                    if c_index <= len(CHAMPION_ACTION_DIM):
                        self.team_champion_labels[c_index - 1, 0] = 0
                        self.team_champion_labels[c_index - 1, 1] = 1

    def update_team_tiers(self):
        """Updates the team_tiers dictionary with the current team composition.

        Connects to the team_origin_class to update team composition
        the team_tiers dictionary with the current team composition.
        """
        self.team_origin_class()
        if self.chosen in self.team_composition.keys():
            self.team_composition[self.chosen] += 1
        for trait in self.team_composition:
            counter = 0
            while self.team_composition[trait] >= tiers[trait][counter]:
                counter += 1
                if counter >= len(tiers[trait]):
                    break
            self.team_tiers[trait] = counter
        origin_class.game_comp_tiers[self.player_num] = self.team_tiers

    def use_reforge(self, xBench, x, y) -> bool:
        """Handles all reforger functions

        Args:
            xBench (int): reforger's slot on the item bench
            x (int): champion x coordinate
            y (int): champion y coordinate

        Returns:
            bool: Reforges the items if there's room on item bench, otherwise returns false
        """
        board = False
        trait_item_list = list(trait_items.values())
        if y >= 0:
            champ = self.board[x][y]
            board = True
        elif y == -1:
            champ = self.bench[x]
        if len(champ.items) > 0 and not self.item_bench_full(len(champ.items)):
            for item in champ.items:
                if item == 'spatula':
                    self.item_bench[self.item_bench_vacancy()] = 'spatula'
                elif item in starting_items:
                    r = random.randint(0, 7)
                    while starting_items[r] == item:
                        r = random.randint(0, 7)
                    self.item_bench[self.item_bench_vacancy()
                                    ] = starting_items[r]
                elif item in trait_item_list:
                    r = random.randint(0, 7)
                    while trait_item_list[r] == item:
                        r = random.randint(0, 7)
                    self.item_bench[self.item_bench_vacancy()
                                    ] = trait_item_list[r]
                elif item in thieves_gloves_items:
                    r = random.randint(0, len(thieves_gloves_items) - 1)
                    while thieves_gloves_items[r] == item:
                        r = random.randint(0, len(thieves_gloves_items) - 1)
                    self.item_bench[self.item_bench_vacancy()
                                    ] = thieves_gloves_items[r]
                else:   # this will only ever be thieves gloves
                    r = random.randint(0, len(thieves_gloves_items) - 1)
                    self.item_bench[self.item_bench_vacancy()
                                    ] = thieves_gloves_items[r]
            champ.items = []
            self.item_bench[xBench] = None
            return True
        if DEBUG:
            print("could not use reforge")
        return False

    # --- Match Outcome Functions --- #
    def won_game(self):
        """Called at the conclusion of the game to the player who won the game"""
        self.reward += self.won_game_reward
        self.print("+0 reward for winning game")

    def won_ghost(self):
        """Same as loss_round but if the opponent was a ghost"""
        if not self.combat:
            self.win_streak += 1
            self.loss_streak = 0
            self.gold += 1
            self.print("won round against a ghost")
            self.match_history.append(1)

            if self.team_tiers['fortune'] > 0:
                if self.fortune_loss_streak >= len(fortune_returns):
                    self.gold += math.ceil(fortune_returns[len(fortune_returns) - 1] +
                                           15 * (self.fortune_loss_streak - len(fortune_returns)))
                    self.fortune_loss_streak = 0
                    return
                self.gold += math.ceil(
                    fortune_returns[self.fortune_loss_streak])
                self.fortune_loss_streak = 0

    def won_round(self, damage):
        """Keeps track of win_streaks, rewards, gold and other values related to winning a combat round.

        Args:
            damage (int): Amount of damage inflicted in the combat round
        """
        if not self.combat:
            self.win_streak += 1
            self.loss_streak = 0
            self.gold += 1
            self.reward += self.damage_reward * damage
            self.print(str(self.damage_reward * damage) + " reward for winning round against player " +
                       str(self.opponent.player_num))
            self.match_history.append(1)

            if self.team_tiers['fortune'] > 0:
                if self.fortune_loss_streak >= len(fortune_returns):
                    self.gold += math.ceil(fortune_returns[len(fortune_returns) - 1] +
                                           15 * (self.fortune_loss_streak - len(fortune_returns)))
                    self.fortune_loss_streak = 0
                    return
                self.gold += math.ceil(
                    fortune_returns[self.fortune_loss_streak])
                self.fortune_loss_streak = 0

    """
    Description - Handles all variables related to losing rounds
    Inputs - The amount of damage resulting from the loss(calculated in game_round)
    """
    # TODO: Separate losing a combat round and a minion round. They have differences related to win_streaks and classes

    def loss_round(self, damage):
        if not self.combat:
            self.loss_streak += 1
            self.win_streak = 0
            self.reward -= self.damage_reward * damage
            self.print(str(-self.damage_reward * damage) +
              " reward for losing round against player " + str(self.opponent.player_num))
            self.match_history.append(0)

            if self.team_tiers['fortune'] > 0:
                self.fortune_loss_streak += 1
                if self.team_tiers['fortune'] > 1:
                    self.fortune_loss_streak += 1

    def __eq__(self, other):
        for x in range(len(self.board)):
            for y in range(len(self.board[0])):
                if self.board[x][y] and other.board[x][y]:
                    if not self.board[x][y].is_equal(other.board[x][y]):
                        return False
                elif self.board[x][y] or other.board[x][y]:
                    return False
        for x in range(len(self.bench)):
            if self.bench[x] and other.bench[x]:
                if not self.bench[x].is_equal(other.bench[x]):
                    return False
            elif self.bench[x] or other.bench[x]:
                return False
        for x in range(len(self.item_bench)):
            if self.item_bench[x] != other.item_bench[x]:
                return False
        if self.gold != other.gold or self.level != other.level \
                or self.exp != other.exp or self.health != other.health or self.reward != other.reward:
            return False
        return True


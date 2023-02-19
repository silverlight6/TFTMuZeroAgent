import math
import config
import time
import numpy as np
import random
from Simulator import champion, origin_class
import Simulator.utils as utils

from Simulator.item_stats import basic_items, item_builds, thieves_gloves_items, \
    starting_items, trait_items, uncraftable_items

from Simulator.stats import COST
from Simulator.pool_stats import cost_star_values
from Simulator.origin_class_stats import tiers, fortune_returns
from math import floor


# This is the base player class
# Stores all values relevant to an individual player in the game
class player:

    CHAMPION_INFORMATION = 12
    BOARD_SIZE = 28
    BENCH_SIZE = 9
    MAX_CHAMPION_IN_SET = 58
    UNCRAFTABLE_ITEM = len(uncraftable_items)
    MAX_BENCH_SPACE = 10
    CHAMP_ENCODING_SIZE = 26

    def __init__(self, pool_pointer, player_num):

        self.gold = 0
        self.level = 1
        self.exp = 0
        self.health = 100
        self.player_num = player_num

        self.win_streak = 0  # For purposes of gold generation at start of turn
        self.loss_streak = 0  # For purposes of gold generation at start of turn
        self.fortune_loss_streak = 0  # For purposes of gold generation if fortune trait is active

        # array of champions, since order does not matter, can be unordered list
        self.bench = [None for _ in range(9)]
        # Champion array, this is a 7 by 4 array.
        self.board = [[None for _ in range(4)] for _ in range(7)]
        # List of items, there is no object for this so this is a string array
        self.item_bench = [None for _ in range(10)]

        # opponent and opponent_board not currently used
        # Leaving here in case we want to add values to the observation that include previous opponent
        self.opponent = None  # Other player, player object
        self.opponent_board = None  # Other player's board for combat, not sure if I will use this.
        self.chosen = False  # Does this player have a chosen unit already
        self.log = []

        # I need to comment how this works.
        self.triple_catalog = []
        self.num_units_in_play = 0
        self.max_units = 1
        self.exp_cost = 4
        self.round = 0

        # This could be in a config file, but we could implement something that alters the
        # Amount of gold required to level that differs player to player
        self.level_costs = [0, 2, 2, 6, 10, 20, 36, 56, 80, 100]
        self.max_level = 9

        # 2 spot for each item(2 component) 10 slots
        self.item_vector = np.zeros(self.MAX_BENCH_SPACE * 6)

        # This time we only need 5 bits total
        self.chosen_vector = np.zeros(5)

        # player related info split between what other players can see and what they can't see
        self.player_public_vector = np.zeros(7)
        self.player_private_vector = np.zeros(9)

        # Encoding board as an image, so we can run convolutions on it.
        self.board_vector = np.zeros(728)  # 7x4 board, with 7x4 encoding
        self.bench_vector = np.zeros(self.BENCH_SIZE * self.CHAMP_ENCODING_SIZE)

        self.decision_mask = np.ones(6, dtype=np.int8)
        self.shop_mask = np.ones(5, dtype=np.int8)
        self.board_mask = np.ones(28, dtype=np.int8)
        self.bench_mask = np.ones(9, dtype=np.int8)
        self.item_mask = np.ones(10, dtype=np.int8)
        self.shop_costs = np.ones(5)

        # Using this to track the reward gained by each player for the AI to train.
        self.reward = 0.0

        # cost to refresh
        self.refresh_cost = 2

        # reward for refreshing
        self.refresh_reward = 0
        self.minion_count_reward = 0
        self.mistake_reward = 0
        self.level_reward = 5
        self.item_reward = 1
        self.won_game_reward = 0
        self.prev_rewards = 0

        # Everyone shares the pool object.
        # Required for buying champions to and from the pool
        self.pool_obj = pool_pointer

        # Boolean for fought this round or not
        self.combat = False
        # List of team compositions
        self.team_composition = origin_class.game_compositions[self.player_num]
        # List of tiers of each trait.
        self.team_tiers = origin_class.game_comp_tiers[self.player_num]

        # An array to record match history
        self.match_history = []

        self.start_time = time.time_ns()

        # Putting this here to show the next possible opponent
        self.possible_opponents = [100 for _ in range(config.NUM_PLAYERS)]
        self.possible_opponents[self.player_num] = -1

        self.kayn_turn_count = 0
        self.kayn_transformed = False
        self.kayn_form = None

        self.thieves_gloves_loc = []

        self.action_vector = np.array([1, 0, 0, 0, 0, 0, 0, 0])
        self.current_action = 0
        self.action_complete = False
        self.action_values = []

        # Start with two copies of each item in your item pool
        self.item_pool = []
        self.refill_item_pool()
        self.refill_item_pool()

        # Context For Loot Orbs
        self.orb_history = []

    # Return value for use of pool.
    # Also I want to treat buying a unit with a full bench as the same as buying and immediately selling it
    def add_to_bench(self, a_champion):  # add champion to reduce confusion over champion from import
        # try to triple second
        golden, triple_success = self.update_triple_catalog(a_champion)
        if not triple_success:
            self.print("Could not update triple catalog for champion " + a_champion.name)
            return False
        if golden:
            return True
        if self.bench_full():
            self.sell_champion(a_champion, field=False)
            self.reward += self.mistake_reward
            return False
        bench_loc = self.bench_vacancy()
        self.bench[bench_loc] = a_champion
        a_champion.bench_loc = bench_loc
        if a_champion.chosen:
            self.print("Adding chosen champion {} of type {}".format(a_champion.name, a_champion.chosen))
            self.chosen = a_champion.chosen
        # print("items are = " + str(a_champion.items))
        self.print("Adding champion {} with items {} to bench".format(a_champion.name, a_champion.items))
        a_champion.num_items = len(a_champion.items)
        if self.bench[bench_loc].items and self.bench[bench_loc].items[0] == 'thieves_gloves':
            self.thieves_gloves_loc.append([bench_loc, -1])
            self.thieves_gloves(bench_loc, -1)
        self.generate_bench_vector()
        return True

    def add_to_item_bench(self, item):
        if self.item_bench_full(1):
            self.reward += self.mistake_reward
            return False
        bench_loc = self.item_bench_vacancy()
        self.item_bench[bench_loc] = item
        self.generate_item_vector()

    def bench_full(self):
        for u in self.bench:
            if not u:
                return False
        return True

    def bench_vacancy(self):
        for free_slot, u in enumerate(self.bench):
            if not u:
                return free_slot
        return False

    def buy_champion(self, a_champion):
        if self.gold == 0 or cost_star_values[a_champion.cost - 1][a_champion.stars - 1] > self.gold \
                or a_champion.cost == 0:
            self.reward += self.mistake_reward
            return False
        self.gold -= cost_star_values[a_champion.cost - 1][a_champion.stars - 1]
        if a_champion.name == 'kayn':
            a_champion.kayn_form = self.kayn_form
        success = self.add_to_bench(a_champion)
        # Putting this outside success because when the bench is full. It auto sells the champion.
        # Which adds another to the pool and need this here to remove the fake copy from the pool
        self.pool_obj.update_pool(a_champion, -1)
        if success:
            # Leaving this out because the agent will learn to simply buy everything and sell everything
            # I want it to just buy what it needs to win rounds.
            # self.reward += 0.005 * cost_star_values[a_champion.cost - 1][a_champion.stars - 1]
            self.print("Spending gold on champion {}".format(a_champion.name) + " with cost = " +
                       str(cost_star_values[a_champion.cost - 1][a_champion.stars - 1])
                       + ", remaining gold " + str(self.gold) + " and chosen = " + str(a_champion.chosen))
            self.generate_player_vector()
        return success

    def buy_exp(self):
        if self.gold < self.exp_cost or self.level == self.max_level:
            self.reward += self.mistake_reward
            return False
        self.gold -= 4
        # self.reward += 0.02
        self.print("exp to {} on level {}".format(self.exp, self.level))
        self.exp += 4
        self.level_up()
        self.generate_player_vector()
        return True

    def decide_vector_generation(self, x):
        if x:
            self.generate_board_vector()
        else:
            self.generate_bench_vector()

    def end_turn_actions(self):
        # auto-fill the board.
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
        # print comp and bench for log purposes at end of turn
        self.printComp()
        self.printBench()
        self.printItemBench()

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
            self.board[coords_candidates[x][0]][coords_candidates[x][1]] = champion.champion('sandguard',
                                    kayn_form=self.kayn_form, target_dummy=True)
        coords = [coords_candidates[0], coords_candidates[1]]
        return coords

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

    def findItem(self, name):
        for c, i in enumerate(self.item_bench):
            if i == name:
                return c
        return False

    def generate_board_vector(self):
        for y in range(0, 4):
            # IMPORTANT TO HAVE THE X INSIDE -- Silver is not sure why but ok.
            for x in range(0, 7):
                # when using binary encoding (6 champ  + stars + chosen + 3 * 6 item) = 26
                champion_info_array = np.zeros(6 * 4 + 2)
                if self.board[x][y]:
                    self.board_mask[4 * x + y] = 1
                    curr_champ = self.board[x][y]
                    c_index = list(COST.keys()).index(curr_champ.name)
                    champion_info_array[0:6] = utils.champ_binary_encode(c_index)
                    champion_info_array[6] = curr_champ.stars / 3
                    champion_info_array[7] = curr_champ.cost / 5
                    for ind, item in enumerate(curr_champ.items):
                        start = (ind * 6) + 7
                        finish = start + 6
                        i_index = []
                        if item in uncraftable_items:
                            i_index = list(uncraftable_items).index(item) + 1
                        elif item in item_builds.keys():
                            i_index = list(item_builds.keys()).index(item) + 1 + len(uncraftable_items)
                        champion_info_array[start:finish] = utils.item_binary_encode(i_index)
                else:
                    self.board_mask[4 * x + y] = 0

                # Fit the area into the designated spot in the vector
                self.board_vector[x * 4 + y:x * 4 + y + 26] = champion_info_array

    def generate_bench_vector(self):
        bench = np.zeros(self.BENCH_SIZE * self.CHAMP_ENCODING_SIZE)
        for x_bench in range(len(self.bench)):
            # when using binary encoding (6 champ  + stars + chosen + 3 * 6 item) = 26
            champion_info_array = np.zeros(6 * 4 + 2)
            if self.bench[x_bench]:
                curr_champ = self.bench[x_bench]
                c_index = list(COST.keys()).index(curr_champ.name)
                champion_info_array[0:6] = utils.champ_binary_encode(c_index)
                champion_info_array[6] = curr_champ.stars / 3
                champion_info_array[7] = curr_champ.cost / 5
                for ind, item in enumerate(curr_champ.items):
                    start = (ind * 6) + 8
                    finish = start + 6
                    if item in uncraftable_items:
                        i_index = list(uncraftable_items).index(item) + 1
                    elif item in item_builds.keys():
                        i_index = list(item_builds.keys()).index(item) + 1 + len(uncraftable_items)
                    champion_info_array[start:finish] = utils.item_binary_encode(i_index)
                self.bench_mask[x_bench] = 1
            else:
                self.bench_mask[x_bench] = 0
            bench[x_bench*self.CHAMP_ENCODING_SIZE:
                    x_bench*self.CHAMP_ENCODING_SIZE + self.CHAMP_ENCODING_SIZE] = champion_info_array
        self.bench_vector = bench

    def generate_chosen_vector(self):
        output_array = np.zeros(5)
        if self.chosen:
            i_index = list(self.team_composition.keys()).index(self.chosen)
            # This should update the item name section of the vector
            for z in range(5, 0, -1):
                if i_index > 2 * z:
                    output_array[5 - z] = 1
                    i_index -= 2 * z
        self.chosen_vector = output_array

    # return output_array
    def generate_item_vector(self):
        item_arr = np.zeros(self.MAX_BENCH_SPACE * 6)
        for ind, item in enumerate(self.item_bench):
            item_info = np.zeros(6)
            if item in uncraftable_items:
                item_info = utils.item_binary_encode(list(uncraftable_items).index(item) + 1)
                self.item_mask[ind] = 1
            elif item in item_builds.keys():
                item_info = utils.item_binary_encode(list(item_builds.keys()).index(item) + 1 + len(uncraftable_items))
                self.item_mask[ind] = 1
            else:
                self.item_mask[ind] = 0
            item_arr[ind*6:ind*6 + 6] = item_info
        self.item_vector = item_arr

    def generate_private_player_vector(self):
        self.player_private_vector[0] = self.gold / 100
        self.player_private_vector[1] = self.exp / 100
        self.player_private_vector[2] = self.round / 30

        exp_to_level = 0
        if self.level < self.max_level:
            exp_to_level = self.level_costs[self.level] - self.exp
        self.player_private_vector[4] = exp_to_level
        self.player_private_vector[5] = max(self.win_streak, self.loss_streak)
        if len(self.match_history) > 2:
            self.player_private_vector[6] = self.match_history[-3]
            self.player_private_vector[7] = self.match_history[-2]
            self.player_private_vector[8] = self.match_history[-1]

        # Decision mask parameters
        if self.gold < 4:
            self.decision_mask[4] = 0
            if self.gold < 2:
                self.decision_mask[5] = 0
            else:
                self.decision_mask[5] = 1
        else:
            self.decision_mask[4] = 1
            self.decision_mask[5] = 1

        for idx, cost in enumerate(self.shop_costs):
            if self.gold < cost or self.bench_full():
                self.shop_mask[idx] = 0
            elif cost != 0 and self.gold >= cost:
                self.shop_mask[idx] = 1

    def generate_public_player_vector(self):
        self.player_private_vector[0] = self.health / 100
        self.player_public_vector[1] = self.level / 10
        self.player_public_vector[2] = self.max_units / 10
        self.player_private_vector[3] = self.max_units / 10
        self.player_public_vector[4] = self.num_units_in_play / self.max_units
        self.player_public_vector[5] = min(floor(self.gold / 10), 5) / 5.0
        streak_lvl = 0
        if self.win_streak == 4:
            streak_lvl = 0.5
        elif self.win_streak >= 5:
            streak_lvl = 1
        self.player_private_vector[6] = streak_lvl

    def generate_player_vector(self):
        self.generate_public_player_vector()
        self.generate_private_player_vector()

    # This takes every occurrence of a champion at a given level and returns 1 of a higher level.
    # Transfers items over. The way I have it would mean it would require bench space.
    def golden(self, a_champion):
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
                        self.sell_champion(self.board[i][j], golden=True, field=True)
        b_champion.chosen = chosen
        b_champion.golden()
        if chosen:
            b_champion.new_chosen()

        # Leaving this code here in case I want to give rewards for triple
        # if b_champion.stars == 2:
        #     self.reward += 0.05
        #     self.print("+0.05 reward for making a level 2 champion")
        # if b_champion.stars == 3:
        #     self.reward += 1.0
        #     self.print("+1.0 reward for making a level 3 champion")

        self.add_to_bench(b_champion)
        if y != -1:
            self.move_bench_to_board(b_champion.bench_loc, x, y)
        self.print("champion {} was made golden".format(b_champion.name))
        return b_champion

    # TO DO: FORTUNE TRAIT - HUGE EDGE CASE - GOOGLE FOR MORE INFO - FORTUNE - TFT SET 4
    # Including base_exp income here

    # This gets called before any of the neural nets happen. This is the start of the round
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
        self.generate_player_vector()

    # num of items to be added to bench, set 0 if not adding.
    # I need to redo this to see how many slots within the length of the array are currently full.
    def item_bench_full(self, num_of_items=0):
        counter = 0
        for i in self.item_bench:
            if i:
                counter += 1
        if counter + num_of_items > len(self.item_bench):
            return True
        else:
            return False

    def item_bench_vacancy(self):
        for free_slot, u in enumerate(self.item_bench):
            if not u:
                return free_slot
        return False

    # checking if kayn is on the board
    def kayn_check(self):
        for x in range(0, 7):
            for y in range(0, 4):
                if self.board[x][y]:
                    if self.board[x][y].name == "kayn":
                        return True
        return False

    def kayn_transform(self):
        if not self.kayn_transformed:
            if not self.item_bench_full(2):
                self.add_to_item_bench('kayn_shadowassassin')
                self.add_to_item_bench('kayn_rhast')
                self.kayn_transformed = True

    def level_up(self):
        if self.level < self.max_level and self.exp >= self.level_costs[self.level]:
            self.exp -= self.level_costs[self.level]
            self.level += 1
            self.max_units += 1
            if self.level >= 5:
                self.reward += 0.5 * self.level_reward
                self.print("+{} reward for leveling to level {}".format(0.5 * self.level_reward, self.level))
            # Only needed if it's possible to level more than once in one transaction
            self.level_up()

        if self.level == self.max_level:
            self.exp = 0

    def loss_round(self, damage):
        if not self.combat:
            self.loss_streak += 1
            self.win_streak = 0
            self.reward -= 0.5 * damage
            self.print(str(-0.5 * damage) + " reward for losing round")
            self.match_history.append(0)

            if self.team_tiers['fortune'] > 0:
                self.fortune_loss_streak += 1
                if self.team_tiers['fortune'] > 1:
                    self.fortune_loss_streak += 1

    # location to pick which unit from bench goes to board.
    def move_bench_to_board(self, bench_x, board_x, board_y):
        # print("bench_x = " + str(bench_x) + " with len(self.bench) = " + str(len(self.bench)))
        if 0 <= bench_x < 9 and self.bench[bench_x] and 7 > board_x >= 0 and 4 > board_y >= 0:
            if self.num_units_in_play < self.max_units:
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
                        return False
                self.board[board_x][board_y] = m_champion
                # tracking thiefs gloves location
                if len(m_champion.items) > 0:
                    if m_champion.items[0] == 'thieves_gloves':
                        self.thieves_gloves_loc_update(bench_x, -1, board_x, board_y)
                if m_champion.name == 'azir':
                    # There should never be a situation where the board is too fill to fit the sandguards.
                    sand_coords = self.find_azir_sandguards(board_x, board_y)
                    self.board[board_x][board_y].overlord = True
                    self.board[board_x][board_y].sandguard_overlord_coordinates = sand_coords
                self.num_units_in_play += 1
                self.generate_bench_vector()
                self.generate_board_vector()
                self.print("moved {} from bench {} to board [{}, {}]".format(self.board[board_x][board_y].name,
                                                                             bench_x, board_x, board_y))
                self.update_team_tiers()
                return True
        self.reward += self.mistake_reward
        return False

    # automatically put the champion at the end of the open bench
    def move_board_to_bench(self, x, y):
        if 0 <= x < 7 and 0 <= y < 4:
            if self.bench_full():
                if self.board[x][y]:
                    if not self.sell_champion(self.board[x][y], field=True):
                        return False
                    self.print("sold from board [{}, {}]".format(x, y))
                    self.generate_board_vector()
                    self.update_team_tiers()
                    return True
                self.reward += self.mistake_reward
                return False
            else:
                if self.board[x][y] and not self.board[x][y].target_dummy:
                    # Dealing with edge case of azir
                    if self.board[x][y].name == 'azir':
                        coords = self.board[x][y].sandguard_overlord_coordinates
                        self.board[x][y].overlord = False
                        for coord in coords:
                            self.board[coord[0]][coord[1]] = None
                    bench_loc = self.bench_vacancy()
                    self.bench[bench_loc] = self.board[x][y]
                    if self.board[x][y]:
                        self.print("moved {} from board [{}, {}] to bench".format(self.board[x][y].name, x, y))
                    self.board[x][y] = None
                    self.bench[bench_loc].x = bench_loc
                    self.bench[bench_loc].y = -1
                    self.num_units_in_play -= 1
                    if self.bench[bench_loc].items and self.bench[bench_loc].items[0] == 'thieves_gloves':
                        self.thieves_gloves_loc_update(bench_loc, -1, x, y)
                    self.generate_bench_vector()
                    self.generate_board_vector()
                    self.update_team_tiers()
                    return True
        self.reward += self.mistake_reward
        return False

    def move_board_to_board(self, x1, y1, x2, y2):
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
                self.generate_board_vector()
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
                                    self.board[x][y].sandguard_overlord_coordinates.remove([x1, y1])
                                    self.board[x][y].sandguard_overlord_coordinates.append([x2, y2])
                self.print("moved {} from board [{}, {}] to board [{}, {}]".format(self.board[x2][y2].name, x1, y1, x2, y2))
                self.generate_board_vector()
                return True
        self.reward += self.mistake_reward
        return False

    # TO DO : Item combinations.
    # Move item from item_bench to champion_bench
    def move_item(self, xBench, x, y):
        board = False
        if y >= 0:
            champ = self.board[x][y]
            board = True
        if y == -1:
            champ = self.bench[x]
        if self.item_bench[xBench] and champ and not champ.target_dummy:
            # thiefs glove exception
            self.print("moving {} to {} with items {}".format(self.item_bench[xBench], champ.name, champ.items))
            # kayn item support
            if self.item_bench[xBench] == 'kayn_shadowassassin' or \
                    self.item_bench[xBench] == 'kayn_rhast':
                if champ.name == 'kayn':
                    self.transform_kayn(self.item_bench[xBench])
                    self.generate_item_vector()
                    self.decide_vector_generation(board)
                    return True
                return False
            if self.item_bench[xBench] == 'champion_duplicator':
                if not self.bench_full():
                    self.gold += champ.cost
                    self.buy_champion(champ)
                    self.item_bench[xBench] = None
                    self.generate_item_vector()
                    self.decide_vector_generation(board)
                    return True
                return False
            if self.item_bench[xBench] == 'magnetic_remover':
                if len(champ.items) > 0:
                    if not self.item_bench_full(len(champ.items)):
                        while len(champ.items) > 0:
                            self.item_bench[self.item_bench_vacancy()] = champ.items[0]
                            if champ.items[0] in trait_items.values():
                                champ.origin.pop(-1)
                                self.update_team_tiers()
                            champ.items.pop(0)
                        self.item_bench[xBench] = None
                        self.generate_item_vector()
                        self.decide_vector_generation(board)
                        return True
                return False
            if self.item_bench[xBench] == 'reforger':
                return self.use_reforge(xBench, x, y)
            if self.item_bench[xBench] == 'thieves_gloves':
                if len(champ.items) < 1:
                    champ.items.append(self.item_bench[xBench])
                    self.item_bench[xBench] = None
                    champ.num_items += 3
                    self.thieves_gloves_loc.append([x, y])
                    self.thieves_gloves(x, y)
                    self.generate_item_vector()
                    self.decide_vector_generation(board)
                    return True
                return False
            if ((champ.num_items < 3 and self.item_bench[xBench] != "thieves_gloves") or
                    (champ.items and champ.items[-1] in basic_items and self.item_bench[xBench]
                     in basic_items and champ.num_items == 3)):
                for trait, name in enumerate(trait_items.values()):
                    if self.item_bench[xBench] == name:
                        item_trait = list(trait_items.keys())[trait]
                        if item_trait in champ.origin:
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
                                    return False
                                else:
                                    champ.origin.append(item_trait)
                                    self.update_team_tiers()
                        if item_names[item_index] == "thieves_gloves":
                            if champ.num_items != 1:
                                return False
                            else:
                                champ.num_items += 2
                                self.thieves_gloves_loc.append([x, y])
                        self.item_bench[xBench] = None
                        champ.items.pop()
                        champ.items.append(item_names[item_index])
                        if champ.items[0] == 'thieves_gloves':
                            self.thieves_gloves(x, y)
                        self.reward += .2 * self.item_reward
                        self.print(
                            ".2 reward for combining two basic items into a {}".format(item_names[item_index]))
                    elif champ.items[-1] in basic_items and self.item_bench[xBench] not in basic_items:
                        basic_piece = champ.items.pop()
                        champ.items.append(self.item_bench[xBench])
                        champ.items.append(basic_piece)
                        self.item_bench[xBench] = None
                        champ.num_items += 1
                    else:
                        champ.items.append(self.item_bench[xBench])
                        self.item_bench[xBench] = None
                        champ.num_items += 1
                else:
                    champ.items.append(self.item_bench[xBench])
                    self.item_bench[xBench] = None
                    champ.num_items += 1
                self.print("After Move {} to {} with items {}".format(self.item_bench[xBench], champ.name,
                                                                      champ.items))
                self.generate_item_vector()
                self.decide_vector_generation(board)
                return True
            elif champ.num_items < 1 and self.item_bench[xBench] == "thieves_gloves":
                champ.items.append(self.item_bench[xBench])
                self.item_bench[xBench] = None
                champ.num_items += 3
                self.generate_item_vector()
                self.decide_vector_generation(board)
                self.print("After Move {} to {} with items {}".format(self.item_bench[xBench], champ.name,
                                                                      champ.items))
                self.thieves_gloves_loc.append([x, -1])
                return True
        # last case where 3 items but the last item is a basic item and the item to input is also a basic item
        self.reward += self.mistake_reward
        return False

    def move_item_to_bench(self, xBench, x):
        self.move_item(xBench, x, -1)

    def move_item_to_board(self, xBench, x, y):
        self.move_item(xBench, x, y)

    def num_in_triple_catelog(self, a_champion):
        num = 0
        for entry in self.triple_catalog:
            if entry["name"] == a_champion.name and entry["level"] == a_champion.stars:
                # print("champion name: " + a_champion.name + " and their level is : " + str(a_champion.stars))
                num += 1
        return num

    def print(self, msg):
        self.printt('{:<120}'.format('{:<8}'.format(self.player_num)
                                     + '{:<20}'.format(str(time.time_ns() - self.start_time)) + msg))

    def printBench(self, log=True):
        for i in range(len(self.bench)):
            if self.bench[i]:
                if log:
                    self.print(str(i) + ": " + self.bench[i].name)
                else:
                    print(self.bench[i].name + ", ")

    def printComp(self, log=True):
        keys = list(self.team_composition.keys())
        values = list(self.team_composition.values())
        tier_values = list(self.team_tiers.values())
        self.print("Reward gained last round = {}".format(self.reward - self.prev_rewards))
        self.prev_rewards = self.reward
        for i in range(len(self.team_composition)):
            if values[i] != 0:
                if log:
                    self.print("{}: {}, tier: {}".format(keys[i], values[i], tier_values[i]))
        for x in range(7):
            for y in range(4):
                if self.board[x][y]:
                    self.print("at ({}, {}), champion {}, with level = {}, items = {}, and chosen = {}".format(x, y,
                               self.board[x][y].name, self.board[x][y].stars,
                               self.board[x][y].items, self.board[x][y].chosen))
        self.print("Player level {} with gold {}, max_units = {}, ".format(self.level, self.gold, self.max_units) +
                   "num_units_in_play = {}, health = {}".format(self.num_units_in_play, self.health))

    def printItemBench(self, log=True):
        for i, item in enumerate(self.item_bench):
            if item:
                if log:
                    self.print(str(i) + ": " + item)
                else:
                    print(str(i) + ": " + item)

    def printShop(self, shop):
        self.print("Shop with level " + str(self.level) + ": " +
                   shop[0] + ", " + shop[1] + ", " + shop[2] + ", " + shop[3] + ", " + shop[4])

    def printt(self, msg):
        if config.PRINTMESSAGES:
            self.log.append(msg)

    # if(config.PRINTMESSAGES): print(msg)

    def refresh(self):
        if self.gold >= self.refresh_cost:
            self.gold -= self.refresh_cost
            self.reward += self.refresh_reward * self.refresh_cost
            self.print("Reward for refreshing shop is " + str(self.refresh_reward * self.refresh_cost))
            return True
        self.reward += self.mistake_reward
        return False

    # This is always going to be from the bench
    def return_item_from_bench(self, x):
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
                    self.item_bench[self.item_bench_cvacancy()] = self.bench[x].items[0]
                    self.print("returning " + self.bench[x].items[0] + " to the item bench")
                self.bench[x].items = []
                self.bench[x].num_items = 0
            self.generate_item_vector()
            return True
        self.print("No units at bench location {}".format(x))
        return False

    def return_item(self, a_champion):
        # if the unit exists
        if a_champion:
            # skip if there are no items, trying to save a little processing time.
            if a_champion.items:
                # thieves_gloves_location needs to be removed whether there's room on the bench or not
                if a_champion.items[0] == 'thieves_gloves':
                    if [a_champion.x, a_champion.y] in self.thieves_gloves_loc:
                        self.thieves_gloves_loc.remove([a_champion.x, a_champion.y])
                    a_champion.items = ['thieves_gloves']
                # if I have enough space on the item bench for the number of items needed
                if not self.item_bench_full(a_champion.num_items):
                    # Each item in possession
                    for item in a_champion.items:
                        if item in trait_items.values():
                            a_champion.origin.pop(-1)
                            self.update_team_tiers()
                        self.item_bench[self.item_bench_vacancy()] = item
                        self.print("returning " + item + " to the item bench")

                # if there is only one or two spots left on the item_bench and thieves_gloves is removed
                elif not self.item_bench_full(1) and a_champion.items[0] == "thieves_gloves":
                    self.item_bench[self.item_bench_vacancy()] = a_champion.items[0]
                    self.print("returning " + a_champion.items[0] + " to the item bench")
                else:
                    self.print("Could not remove item {} from champion {}".format(a_champion.items, a_champion.name))
                    return False
                a_champion.items = []
                a_champion.num_items = 0
                self.generate_item_vector()

            return True
        return False

    # called when selling a unit
    def remove_triple_catalog(self, a_champion, golden=False):
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
            self.print("Trying to fix bug for {} with level {}".format(a_champion.name, a_champion.stars))
            return True
        if a_champion.stars >= 2:
            self.print("Failed to remove " + str(a_champion.name) +
                       " from triple catalog with stars = " + str(a_champion.stars))
        self.print("could not find champion " + a_champion.name + " with star = "
                   + str(a_champion.stars) + " in the triple catelog")
        self.print("{}".format(self.triple_catalog))
        return False

    # This should only be called when trying to sell a champion from the field and the bench is full
    # This can occur after a carousel round where you get a free champion and it can enter the field
    # Even if you already have too many units in play. The default behavior will be sell that champion.
    # sell champion to reduce confusion over champion from import
    def sell_champion(self, s_champion, golden=False, field=True):
        # Need to add the behavior that on carousel when bench is full, add to board.
        if not (self.remove_triple_catalog(s_champion, golden=golden) and self.return_item(s_champion) and not \
                s_champion.target_dummy):
            self.reward += self.mistake_reward
            self.print("Could not sell champion " + s_champion.name)
            return False
        if not golden:
            self.gold += cost_star_values[s_champion.cost - 1][s_champion.stars - 1]
            self.pool_obj.update_pool(s_champion, 1)
        if s_champion.chosen:
            self.chosen = False
        if s_champion.x != -1 and s_champion.y != -1:
            self.board[s_champion.x][s_champion.y] = None
        if field:
            self.num_units_in_play -= 1
        self.print("selling champion " + s_champion.name + " with stars = " + str(s_champion.stars))
        return True

    def sell_from_bench(self, location, golden=False):
        # Check if champion has items
        # Are there any champions with special abilities on sell.
        if self.bench[location]:
            if not (self.remove_triple_catalog(self.bench[location], golden=golden) and
                    self.return_item_from_bench(location)):
                self.print("Mistake in sell from bench with {} and level {}".format(self.bench[location],
                                                                                    self.bench[location].stars))
                self.reward += self.mistake_reward
                return False
            if not golden:
                self.gold += cost_star_values[self.bench[location].cost - 1][self.bench[location].stars - 1]
                self.pool_obj.update_pool(self.bench[location], 1)
            if self.bench[location].chosen:
                self.chosen = False
            return_champ = self.bench[location]
            self.print("selling champion " + self.bench[location].name + "with stars = " +
                       str(self.bench[location].stars))
            self.bench[location] = None
            self.generate_bench_vector()
            return return_champ
        return False

    def thieves_gloves(self, x, y):
        r1 = random.randint(0, len(thieves_gloves_items) - 1)
        r2 = random.randint(0, len(thieves_gloves_items) - 1)
        while r1 == r2:
            r2 = random.randint(0, len(thieves_gloves_items) - 1)
        self.print("thieves_gloves: {} and {}".format(thieves_gloves_items[r1], thieves_gloves_items[r2]))
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
            return False

    def thieves_gloves_loc_update(self, x1, y1, x2, y2):
        if [x1, y1] in self.thieves_gloves_loc and [x2, y2] in self.thieves_gloves_loc:
            return True
        elif [x1, y1] in self.thieves_gloves_loc:
            self.thieves_gloves_loc.remove([x1, y1])
            self.thieves_gloves_loc.append([x2, y2])
        elif [x2, y2] in self.thieves_gloves_loc:
            self.thieves_gloves_loc.remove([x2, y2])
            self.thieves_gloves_loc.append([x1, y1])

    def transform_kayn(self, kayn_item):
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

    def update_shop_costs(self, shop_costs):
        self.shop_costs = shop_costs

    def update_team_tiers(self):
        self.team_composition = origin_class.team_origin_class(self)
        if self.chosen in self.team_composition.keys():
            self.team_composition[self.chosen] += 1
        for trait in self.team_composition:
            counter = 0
            # print("Trait {} with number {}".format(trait, self.team_composition[trait]))
            while self.team_composition[trait] >= tiers[trait][counter]:
                # print("Trait {} with number {}, counter -> {}".format(trait, self.team_composition[trait], counter))
                counter += 1
                if counter >= len(tiers[trait]):
                    break
            self.team_tiers[trait] = counter
        origin_class.game_comp_tiers[self.player_num] = self.team_tiers

    # Method for keeping track of which units are golden
    # It calls golden which then calls add_to_bench which calls this again with the goldened unit
    # Parameters -> champion to be added to the catalog
    # Returns -> boolean if the unit was goldened, boolean for successful operation.
    def update_triple_catalog(self, a_champion):
        for entry in self.triple_catalog:
            if entry["name"] == a_champion.name and entry["level"] == a_champion.stars:
                entry["num"] += 1
                if entry["num"] == 3:
                    self.golden(a_champion)
                    return True, True
                return False, True
        if a_champion.stars > 3:
            return False, False
        self.triple_catalog.append({"name": a_champion.name, "level": a_champion.stars, "num": 1})
        return False, True

    # print("adding " + champion.name + " to triple_catalog")

    def use_reforge(self,xBench, x, y):
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
                    self.item_bench[self.item_bench_vacancy()] = starting_items[r]
                elif item in trait_item_list:
                    r = random.randint(0, 7)
                    while trait_item_list[r] == item:
                        r = random.randint(0, 7)
                    self.item_bench[self.item_bench_vacancy()] = trait_item_list[r]
                elif item in thieves_gloves_items:
                    r = random.randint(0, len(thieves_gloves_items) - 1)
                    while thieves_gloves_items[r] == item:
                        r = random.randint(0, len(thieves_gloves_items) - 1)
                    self.item_bench[self.item_bench_vacancy()] = thieves_gloves_items[r]
                else:   # this will only ever be thiefs gloves
                    r = random.randint(0, len(thieves_gloves_items) - 1)
                    self.item_bench[self.item_bench_vacancy()] = thieves_gloves_items[r]
            champ.items = []
            champ.num_items = 0
            self.item_bench[xBench] = None
            self.generate_item_vector()
            self.decide_vector_generation(board)
            return True
        return False

    def start_round(self, t_round):
        self.start_time = time.time_ns()
        self.round = t_round
        self.reward += self.num_units_in_play * self.minion_count_reward
        # self.print(str(self.num_units_in_play * self.minion_count_reward) + " reward for minions in play")
        self.gold_income(self.round)
        self.generate_player_vector()
        if self.kayn_check():
            self.kayn_turn_count += 1
        if self.kayn_turn_count >= 3:
            self.kayn_transform()
        for x in self.thieves_gloves_loc:
            self.thieves_gloves(x[0], x[1])

    def won_game(self):
        self.reward += self.won_game_reward
        self.print("+0 reward for winning game")

    def won_round(self, damage):
        if not self.combat:
            self.win_streak += 1
            self.loss_streak = 0
            self.gold += 1
            self.reward += 0.5 * damage
            self.print(str(0.5 * damage) + " reward for winning round")
            self.match_history.append(1)

            if self.team_tiers['fortune'] > 0:
                if self.fortune_loss_streak >= len(fortune_returns):
                    self.gold += math.ceil(fortune_returns[len(fortune_returns) - 1] +
                                           15 * (self.fortune_loss_streak - len(fortune_returns)))
                    self.fortune_loss_streak = 0
                    return
                self.gold += math.ceil(fortune_returns[self.fortune_loss_streak])
                self.fortune_loss_streak = 0

    def won_ghost(self, damage):
        self.reward += 0.02 * damage
        self.print(str(0.02 * damage) + " reward for someone losing to ghost")

    # Item pool mechanic utilities
    def refill_item_pool(self):
        self.item_pool.extend(starting_items)
    
    def remove_from_pool(self, item):
        self.item_pool.remove(item)

    def random_item_from_pool(self):
        item = random.choice(self.item_pool)
        self.remove_from_pool(item)
        return item
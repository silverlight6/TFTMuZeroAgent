import champion
import config
import numpy as np
import pool
from item_stats import items as item_list, basic_items, item_builds
from stats import COST
from math import floor
from champion_functions import MILLIS
from pool_stats import cost_star_values


# Let me create a bit of a TODO list at the top here
# 
class player:
    def __init__(self, pool_pointer, player_num):

        self.gold = 0
        self.level = 0
        self.exp = 0
        self.health = 100
        self.player_num = player_num

        self.win_streak = 0  # For purposes of gold generation at start of turn
        self.loss_streak = 0  # For purposes of gold generation at start of turn

        # array of champions, since order does not matter, can be unordered list
        self.bench = [None for _ in range(9)]
        # Champion array, this is a 7 by 4 array.
        self.board = [[None for _ in range(4)] for _ in range(7)]
        # List of items, there is no object for this so this is a string array
        self.item_bench = [None for _ in range(10)]

        self.opponite = None  # Other player, player object
        self.opponite_board = None  # Other player's board for combat, not sure if I will use this.
        self.chosen = False  # Does this player have a chosen unit already
        self.log = []

        # I need to comment how this works.
        self.triple_catelog = []
        self.num_units_in_play = 0
        self.max_units = 0
        self.exp_cost = 4

        self.level_costs = [2, 2, 6, 10, 20, 36, 56, 80]

        # We have 28 board slots. Each slot has a champion info.
        # 6 spots for champion. 2 spots for the level. 1 spot for chosen.
        # 6 spots for items. 3 item slots.
        # (6 * 3 + 6 + 2 + 1) * 28 = 756
        self.board_vector = np.zeros(756)

        # We have 9 bench slots. Same rules as above
        self.bench_vector = np.zeros(243)

        # This time we only need 6 bits per slot with 10 slots
        self.item_vector = np.zeros(60)

        # Using this to track the reward gained by each player for the AI to train.
        self.reward = 0.0

        # cost to refrsh
        self.refresh_cost = 2

        # reward for refreshing
        self.refresh_reward = .01

        self.pool_obj = pool_pointer

        # Boolean for fought this round or not
        self.combat = False

    # Return value for use of pool.
    # Also I want to treat buying a unit with a full bench as the same as buying and immediately selling it
    def add_to_bench(self, a_champion):  # add champion to reduce confusion over champion from import
        # try to triple second
        self.update_triple_catelog(a_champion)
        if self.bench_full():
            self.sell_champion(a_champion)
            self.reward -= 0.01
            return False
        bench_loc = self.bench_vacency()
        self.bench[bench_loc] = a_champion
        a_champion.bench_loc = bench_loc
        # print("items are = " + str(a_champion.items))
        return True

    def add_to_item_bench(self, item):
        if self.item_bench_full(1):
            self.reward -= 0.01
            return False
        bench_loc = self.item_bench_vacency()
        self.item_bench[bench_loc] = item

    def bench_full(self):
        for u in self.bench:
            if (not u):
                return False
        return True

    def bench_vacency(self):
        for free_slot, u in enumerate(self.bench):
            if (not u):
                return free_slot
        return False

    def buy_champion(self, a_champion):
        if a_champion.cost > self.gold or a_champion.cost == 0:
            self.reward -= 0.01
            return False
        self.gold -= cost_star_values[a_champion.cost - 1][a_champion.stars - 1]
        self.reward += 0.01 * cost_star_values[a_champion.cost - 1][a_champion.stars - 1]
        # Feel free to uncomment these if you need to see what is happening with every reward.
        # These are going to proc so often that I am leaving them commented for now.
        # self.print("+{} reward for spending gold".format(0.01 * cost_star_values[a_champion.cost][a_champion.stars]))
        # print("Buying " + a_champion.name + " to the bench")
        self.add_to_bench(a_champion)
        return True

    def buy_exp(self):
        if self.gold < self.exp_cost:
            self.reward -= 0.01
            return False
        self.gold -= 4
        self.reward += 0.04
        # Feel free to uncomment these if you need to see what is happening with every reward.
        # These are going to proc so often that I am leaving them commented for now.
        # self.print("+{} reward for spending gold".format(0.04))
        self.exp += 4
        self.level_up()
        return True

    def findItem(self, name):
        for c, i in enumerate(self.item_bench):
            if i == name:
                return c
        return False

    def generate_board_vector(self):
        # 27 - length of each component, 7 - x axis, 4 - y axis
        output_array = np.zeros(27 * 7 * 4)
        for x in range(0, 7):
            for y in range(0, 4):
                input_array = np.zeros(27)
                if self.board[x][y]:
                    # start with champion name
                    c_index = list(COST.keys()).index(self.board[x][y].name)
                    # This should update the champion name section of the vector
                    for z in range(0, 6, -1):
                        if c_index > 2 * z:
                            input_array[z] = 1
                            c_index -= 2 * z
                    if self.board[x][y].level == 1:
                        input_array[6:8] = [0, 1]
                    if self.board[x][y].level == 2:
                        input_array[6:8] = [1, 0]
                    if self.board[x][y].level == 3:
                        input_array[6:8] = [1, 1]
                    if self.board[x][y].chosen == True:
                        input_array[8] = 1
                    else:
                        input_array[8] = 0

                    if champion.items:
                        for i in range(0, 3):
                            if self.board[x][y].items[i]:
                                i_index = list(item_list.keys()).index(self.board[x][y].items[i])
                                # This should update the item name section of the vector
                                for z in range(0, 6, -1):
                                    if i_index > 2 * z:
                                        input_array[9 + 6 * (i + 1) - z] = 1
                                        i_index -= 2 * z
                lower_bound = 27 * (x + 7 * y)
                output_array[lower_bound: lower_bound + 27] = input_array
        self.board_vector = output_array

    # return output_array

    def generate_bench_vector(self):
        output_array = np.zeros(27 * 9)
        for x in range(0, 9):
            input_array = np.zeros(27)
            if self.bench[x]:
                # start with champion name
                c_index = list(COST.keys()).index(self.bench[x].name)
                # This should update the champion name section of the vector
                for z in range(0, 6, -1):
                    if c_index > 2 * z:
                        input_array[z] = 1
                        c_index -= 2 * z
                if self.bench[x].level == 1:
                    input_array[6:8] = [0, 1]
                if self.bench[x].level == 2:
                    input_array[6:8] = [1, 0]
                if self.bench[x].level == 3:
                    input_array[6:8] = [1, 1]
                if self.bench[x].chosen == True:
                    input_array[8] = 1
                else:
                    input_array[8] = 0

                if champion.items:
                    for i in range(0, 3):
                        if self.bench[x].items[i]:
                            i_index = list(item_list.keys()).index(self.bench[x].items[i])
                            # This should update the item name section of the vector
                            for z in range(0, 6, -1):
                                if i_index > 2 * z:
                                    input_array[9 + 6 * (i + 1) - z] = 1
                                    i_index -= 2 * z
            lower_bound = 27 * x
            output_array[lower_bound: lower_bound + 27] = input_array
        self.bench_vector = output_array

    # return output_array

    def generate_item_vector(self):
        for x in range(0, len(self.item_bench)):
            input_array = np.zeros(6)
            if self.item_bench[x]:
                i_index = list(item_list.keys()).index(self.item_bench[x])
                # This should update the item name section of the vector
                for z in range(0, 6, -1):
                    if i_index > 2 * z:
                        input_array[6 - z] = 1
                        i_index -= 2 * z
            self.item_array[6 * x: 6 * (x + 1)] = input_array

    # return self.item_array

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
                    if self.bench[i].chosen: chosen = self.bench[i].chosen
                    self.sell_from_bench(i, True)
        for i in range(0, 7):
            for j in range(0, 4):
                if self.board[i][j]:
                    if self.board[i][j].name == a_champion.name and self.board[i][j].stars == a_champion.stars:
                        x = i
                        y = j
                        if self.board[i][j].chosen: chosen = self.board[i][j].chosen
                        self.sell_champion(self.board[i][j], True)
        b_champion.chosen = chosen
        b_champion.golden()
        if chosen:
            b_champion.new_chosen()
        # print(str(b_champion.stars))
        if b_champion.stars == 2:
            self.reward += 1.0
            self.print("+1 reward for making a level 2 champion")
            # print("champion " + b_champion.name + " has turned golden")
        if b_champion.stars == 3:
            self.reward += 5.0
            self.print("+5 reward for making a level 3 champion")
            print("champion " + b_champion.name + " has turned mega golden")

        self.add_to_bench(b_champion)
        if y != -1:
            self.move_bench_to_board(b_champion.bench_loc, x, y)
        self.printt("champion {} was made golden".format(b_champion.name))
        return b_champion

    # TO DO: FORTUNE TRAIT - HUGE EDGE CASE - GOOGLE FOR MORE INFO - FORTUNE - TFT SET 4
    # Including base_exp income here

    def gold_income(self, t_round):  # time of round since round is a keyword
        self.exp += 2
        self.level_up()
        if (t_round <= 4):
            starting_round_gold = [2, 2, 3, 4]
            self.gold += starting_round_gold[t_round]
            self.gold += floor(self.gold / 10)
            return
        self.gold += 5
        self.gold += floor(self.gold / 10)
        if (self.win_streak == 2 or self.win_streak == 3 or self.loss_streak == 2 or self.loss_streak == 3):
            self.gold += 1
        elif (self.win_streak == 3 or self.loss_streak == 3):
            self.gold += 2
        elif (self.win_streak >= 4 or self.loss_streak >= 4):
            self.gold += 3

    # num of items to be added to bench, set 0 if not adding.
    # This is going to crash because the item_bench is set to initialize all to NULL but len to 10.
    # I need to redo this to see how many slots within the length of the array are currently full.
    def item_bench_full(self, num_of_items):
        for i in self.item_bench:
            if not i:
                return False
        return True

    def item_bench_vacency(self):
        for free_slot, u in enumerate(self.item_bench):
            if not u:
                return free_slot
        return False

    def level_up(self):
        if self.exp >= self.level_costs[self.level]:
            self.exp -= self.level_costs[self.level]
            self.level += 1
            self.max_units += 1
            if self.level >= 5:
                self.reward += 2.0
                self.print("+2 reward for leveling to level {}".format(self.level))
            self.level_up()

    # location to pick which unit from bench goes to board.
    def move_bench_to_board(self, bench_x, board_x, board_y):
        # print("bench_x = " + str(bench_x) + " with len(self.bench) = " + str(len(self.bench)))
        if bench_x >= 0 and bench_x < 9 and self.bench[bench_x] \
                and board_x < 7 and board_x >= 0 \
                and board_y < 4 and board_y >= 0:

            if self.num_units_in_play < self.max_units:
                m_champion = self.bench[bench_x]
                self.bench[bench_x] = None
                m_champion.x = board_x
                m_champion.y = board_y
                if (self.board[board_x][board_y]):
                    self.move_board_to_bench(board_x, board_y)
                self.board[board_x][board_y] = m_champion
                return True
        self.reward -= 0.01
        return False

    # automatically put the champion at the end of the open bench
    # Will likely have to deal with azir and other edge cases here.
    # Kinda of the attitude that I will let those issues sting me first and deal with them
    # When they come up and appear.
    def move_board_to_bench(self, x, y):
        if self.bench_full():
            if self.board[x][y]:
                self.sell_champion(self.board[x][y])
            self.reward -= 0.01
            return False
        else:
            if self.board[x][y]:
                bench_loc = self.bench_vacency()
                self.bench[bench_loc] = self.board[x][y]
                self.board[x][y] = None
                return True
            else:
                self.reward -= 0.01
                return False

    # TO DO : Item combinations.
    # Move item from item_bench to champion_bench
    def move_item_to_bench(self, xBench, x):
        if self.item_bench[xBench]:
            if self.bench[x]:
                # theives glove exception
                self.print("moving {} to unit {}".format(self.item_bench[xBench], self.bench[x].name))
                if ((self.bench[x].num_of_items < 3 and self.item_bench[
                    xBench] != "thiefs_gloves") or  # item is not thiefs golves
                        (self.bench[x].items[-1] in basic_items and self.item_bench[x] in basic_items and self.bench[
                            x].num_of_items == 3)):  # adding item& have 3 or more items already
                    # implement the item combinations here. Make exception with theives gloves
                    if self.bench[x].items[-1] in basic_items and self.item_bench[x] in basic_items:
                        item_build_values = item_builds.values()
                        item_index = 0
                        for index, items in enumerate(item_build_values):
                            if ((self.bench[x].items[-1] == items[0] and self.item_bench[x] == items[1]) or
                                    (self.bench[x].items[-1] == items[1] and self.item_bench[x] == items[0])):
                                item_index = index
                                break
                        if item_builds.keys()[item_index] == "theifs_gloves":
                            if self.bench[x].num_of_items != 1:
                                return False
                            else:
                                self.bench[x].num_of_items += 2
                        self.item_bench[xBench] = None
                        self.bench[x].items.append(item_builds.keys()[item_index])
                        self.reward += .5
                        self.print(
                            ".5 reward for combining two basic items into a {}".format(item_builds.keys()[item_index]))
                    else:
                        self.bench[x].items.append(self.item_bench[xBench])
                        self.item_bench[xBench] = next_observation
                        self.bench[x].num_of_items += 1
                    return True
                elif self.bench[x].num_of_items < 1 and self.item_bench[xBench] == "thiefs_gloves":
                    self.bench[x].items.append(self.item_bench[xBench])
                    self.item_bench[xBench] = None
                    self.bench[x].num_of_items += 3
                    return True
            # last case where 3 items but the last item is a basic item and the item to input is also a basic item
        self.reward -= 0.01
        return False

    def move_item_to_board(self, xBench, x, y):
        if self.item_bench[xBench]:
            if self.board[x][y]:
                # theives glove exception
                self.print("moving {} to unit {}".format(self.item_bench[xBench], self.board[x][y].name))
                if ((len(self.board[x][y].items) < 3 and self.item_bench[xBench] != "thiefs_gloves") or
                        (self.board[x][y].items[-1] in basic_items and self.item_bench[x] in basic_items and len(
                            self.board[x][y].items) == 3)):
                    # implement the item combinations here. Make exception with theives gloves
                    if self.board[x][y].items and self.board[x][y].items[-1] in basic_items and self.item_bench[
                        x] in basic_items:
                        item_build_values = item_builds.values()
                        item_index = 0
                        for index, items in enumerate(item_build_values):
                            if ((self.board[x][y].items[-1] == items[0] and self.item_bench[x] == items[1]) or \
                                    (self.board[x][y].items[-1] == items[1] and self.item_bench[x] == items[0])):
                                item_index = index
                                break
                        if list(item_builds.keys())[item_index] == "theifs_gloves":
                            if self.board[x][y].num_of_items != 1:
                                return False
                            else:
                                self.board[x][y].num_of_items += 2
                        self.item_bench[xBench] = None
                        self.board[x][y].items.append(list(item_builds.keys())[item_index])
                        self.reward += .5
                        self.print(".5 reward for combining two basic items into a {}".format(
                            list(item_builds.keys())[item_index]))
                    else:
                        self.board[x][y].items.append(self.item_bench[xBench])
                        self.item_bench[xBench] = None
                        self.board[x][y].num_items += 1
                    return True
                elif self.board[x][y].num_items < 1 and self.item_bench[xBench] == "thiefs_gloves":
                    self.board[x][y].items.append(self.item_bench[xBench])
                    self.item_bench[xBench] = None
                    self.board[x][y].num_items += 3
                    return True
                self.print("moving {} to unit {} failed".format(self.item_bench[xBench], self.board[x][y].name))
        self.reward -= 0.01
        return False

    def print(self, msg):
        self.printt('{:<120}'.format('{:<8}'.format(self.player_num) + msg) + str(MILLIS()))

    def printBench(self, log=True):
        for i in self.bench:
            if i:
                if log:
                    i.print()
                else:
                    print(i.name + ", ")

    def printItemBench(self, log=True):
        for i in self.item_bench:
            if i:
                if log:
                    self.printt('{:<120}'.format('{:<8}', format(self.player_num) + self.item_bench))
                else:
                    print('{:<120}'.format('{:<8}', format(self.player_num) + self.item_bench))

    def printt(self, msg):
        if (config.PRINTMESSAGES): self.log.append(msg)

    # if(config.PRINTMESSAGES): print(msg)

    def refresh(self):
        if self.gold >= self.refresh_cost:
            self.gold -= self.refresh_cost
            self.reward += self.refresh_reward * self.refresh_cost
            return True
        self.reward -= 0.01
        return False

    # This is always going to be from the bench
    def return_item_from_bench(self, x):
        # if the unit exists
        if self.bench[x]:
            # skip if there are no items, trying to save a little processing time.
            if self.bench[x].items:
                # if I have enough space on the item bench for the number of items needed
                if (not self.item_bench_full(len(self.bench[x].items))):
                    # Each item in posesstion
                    for i in self.bench[x].items:
                        # theives glove exception
                        self.item_bench[self.item_bench_vacency()] = i
                # if there is only one or two spots left on the item_bench and thiefs_gloves is removed
                elif (not self.item_bench_full(1) and self.bench[x].items[0] == "thiefs_gloves"):
                    self.item_bench[self.item_bench_vacency()] = self.bench[x].items[0]
                self.bench[x].items = []
                self.bench[x].num_of_items = 0
            return True
        return False

    def return_item_from_board(self, x, y):
        # if the unit exists
        if self.board[x][y]:
            # skip if there are no items, trying to save a little processing time.
            if self.board[x][y].items:
                # if I have enough space on the item bench for the number of items needed
                if (not self.item_bench_full(self.board[x][y].num_items)):
                    # Each item in posesstion
                    for i in self.board[x][y].items:
                        # theives glove exception
                        self.item_bench[self.item_bench_vacency()] = i
                # if there is only one or two spots left on the item_bench and thiefs_gloves is removed
                elif (not self.item_bench_full(1) and self.board[x][y].items[0] == "thiefs_gloves"):
                    self.item_bench[self.item_bench_vacency()] = self.board[x][y].items[0]
                self.board[x][y].items = []
                self.board[x][y].num_items = 0
            return True
        return False

    # called when selling a unit
    def remove_triple_catelog(self, a_champion):
        for c, i in enumerate(self.triple_catelog):
            if (i["name"] == a_champion.name and i["level"] == a_champion.stars):
                i["num"] -= 1
                if i["num"] == 0:
                    self.triple_catelog.pop(c)
                    return True
        return False

    # This should only be called when trying to sell a champion from the field and the bench is full
    # This can occur after a carosell round where you get a free champion and it can enter the field
    # Even if you already have too many units in play. The default behavior will be sell that champion.
    def sell_champion(self, s_champion, golden=False):  # sell champion to reduce confusion over champion from import
        # Need to add the behavior that on carosell when bench is full, add to board.
        if (not self.remove_triple_catelog(s_champion) or not self.return_item_from_board(s_champion.x, s_champion.y)):
            self.reward -= 0.01
            return False
        if not golden:
            print(
                "selling champion " + s_champion.name + " with cost = " + str(s_champion.cost) + " and stars = " + str(
                    s_champion.stars))
            self.gold += cost_star_values[s_champion.cost - 1][s_champion.stars - 1]
        if s_champion.chosen: self.chosen = False
        self.pool_obj.update(s_champion, 1)
        self.board[s_champion.x][s_champion.y] = None
        self.num_units_in_play -= 1
        return True

    def sell_from_bench(self, location, golden=False):
        # Check if champion has items
        # Are there any champions with special abilities on sell.
        # print("location on bench = " + str(location))
        if self.bench[location]:
            if (not self.remove_triple_catelog(self.bench[location]) or not self.return_item_from_bench(location)):
                return False
            if not golden: self.gold += cost_star_values[self.bench[location].cost - 1][self.bench[location].stars - 1]
            if self.bench[location].chosen: self.chosen = False
            self.pool_obj.update(self.bench[location], 1)
            return_champ = self.bench[location]
            self.bench[location] = None
            return return_champ
        return False

    def update_triple_catelog(self, a_champion):
        for entry in self.triple_catelog:
            if (entry["name"] == a_champion.name and entry["level"] == a_champion.stars):
                # print("champion name: " + a_champion.name + " and their level is : " + str(a_champion.stars))
                entry["num"] += 1
                if entry["num"] == 3:
                    b_champion = self.golden(a_champion)
                    self.update_triple_catelog(b_champion)
                return
        self.triple_catelog.append({"name": a_champion.name, "level": a_champion.stars, "num": 1})

    # print("adding " + champion.name + " to triple_catelog")

    def won_game(self):
        self.reward += 5.0
        self.print("+5 reward for winning game")

    def won_round(self):
        self.reward += 1.0
        self.print("+1 reward for winning round")

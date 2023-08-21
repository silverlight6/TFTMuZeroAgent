import numpy as np
import Simulator.champion as champion
from Simulator.default_agent_stats import *
from Simulator.pool_stats import cost_star_values
from Simulator.origin_class_stats import tiers
from Simulator.origin_class import team_traits
from Simulator.utils import x_y_to_1d_coord
from Simulator.stats import COST
from copy import deepcopy


class Default_Agent:
    def __init__(self):
        self.current_round = 0
        self.next_round = 3
        self.round_3_10_checks = [True for _ in range(5)]
        self.round_11_end_checks = [True for _ in range(5)]
        self.pairs = []
        self.require_pair_update = False
        self.round_11_clean_up = True
        self.comp_number = -1

    def policy(self, player, shop, game_round):
        self.current_round = game_round
        if game_round == 1 or game_round == 2:
            return self.round_1_2(player, shop)
        elif 2 < game_round < 11:
            if game_round == 3 and self.current_round == self.next_round:
                self.update_pairs_list(player)
            return self.round_3_10(player, shop)
        elif game_round == 11 and self.round_11_clean_up:
            return self.decide_comp(player)
        elif game_round >= 11:
            # put a check here to see if current round == next round and round == 11 to pick the comp
            return self.round_11_end(player, shop)
        print("Game_round = {}".format(game_round))
        return "0"

    def move_bench_to_empty_board(self, player, bench_location, unit):
        if unit in FRONT_LINE_UNITS:
            for displacement in range(4):
                if not player.board[3 + displacement][3]:
                    return "2_" + str(bench_location) + "_" + str(24 + displacement)
                if not player.board[3 - displacement][3]:
                    return "2_" + str(bench_location) + "_" + str(24 - displacement)
            print("Empty Front line with board {}".format(player.board))
        if unit in MIDDLE_LINE_UNITS:
            for displacement in range(4):
                if not player.board[3 + displacement][2]:
                    return "2_" + str(bench_location) + "_" + str(17 + displacement)
                if not player.board[3 - displacement][2]:
                    return "2_" + str(bench_location) + "_" + str(17 - displacement)
            print("Empty Mid line with board {}".format(player.board))
        if unit in BACK_LINE_UNITS:
            for displacement in range(4):
                if not player.board[3 + displacement][0]:
                    return "2_" + str(bench_location) + "_" + str(3 + displacement)
                if not player.board[3 - displacement][0]:
                    return "2_" + str(bench_location) + "_" + str(3 - displacement)
            print("Empty Back line with board {}".format(player.board))
        return "0"

    def check_unit_location(self, player, x, y, unit):
        if unit in FRONT_LINE_UNITS:
            if y != 3:
                for displacement in range(4):
                    if not player.board[3 + displacement][3]:
                        return "2_" + str(x_y_to_1d_coord(x, y)) + "_" + str(24 + displacement)
                    if not player.board[3 - displacement][3]:
                        return "2_" + str(x_y_to_1d_coord(x, y)) + "_" + str(24 - displacement)
                print("Check Front line with board {}".format(player.board))

        if unit in MIDDLE_LINE_UNITS:
            if y != 2:
                for displacement in range(4):
                    if not player.board[3 + displacement][2]:
                        return "2_" + str(x_y_to_1d_coord(x, y)) + "_" + str(17 + displacement)
                    if not player.board[3 - displacement][2]:
                        return "2_" + str(x_y_to_1d_coord(x, y)) + "_" + str(17 - displacement)
                print("Check Mid line with board {}".format(player.board))
        if unit in BACK_LINE_UNITS:
            if y != 0:
                for displacement in range(4):
                    if not player.board[3 + displacement][0]:
                        return "2_" + str(x_y_to_1d_coord(x, y)) + "_" + str(3 + displacement)
                    if not player.board[3 - displacement][0]:
                        return "2_" + str(x_y_to_1d_coord(x, y)) + "_" + str(3 - displacement)
                print("Check Back line with board {}".format(player.board))
        return False

    def max_unit_check(self, player):
        # place units in front or back.
        if player.num_units_in_play < player.max_units:
            for i, bench_slot in enumerate(player.bench):
                if bench_slot:
                    return self.move_bench_to_empty_board(player, 28 + i, bench_slot.name)
        return " "

    def update_pairs_list(self, player):
        list_of_units = []
        for bench_unit in player.bench:
            if bench_unit:
                list_of_units.append(bench_unit.name + "_" + str(bench_unit.stars))
        for x in range(len(player.board)):
            for y in range(len(player.board[x])):
                if player.board[x][y]:
                    list_of_units.append(player.board[x][y].name + "_" + str(player.board[x][y].stars))
        for unit in list_of_units:
            unit_count = list_of_units.count(unit)
            if unit_count > 1 and unit not in self.pairs:
                self.pairs.append(unit)

    def sell_bench_full(self, player):
        if self.comp_number != -1:
            for i, bench_unit in enumerate(player.bench):
                if bench_unit.name not in TEAM_COMPS[self.comp_number]:
                    return "4_" + str(i)
        for i, bench_unit in enumerate(player.bench):
            if (bench_unit.name + "_" + str(bench_unit.stars)) not in self.pairs and bench_unit.stars == 1:
                return "4_" + str(i)
        low_cost = 100
        position = 0
        for i, bench_unit in enumerate(player.bench):
            if bench_unit.cost < low_cost:
                position = i
        return "4_" + str(position)

    def compare_shop_unit(self, shop_unit, board, x, y):
        if shop_unit.endswith("_c"):
            c_shop = shop_unit.split('_')
            a_champion = champion.champion(c_shop[0], chosen=c_shop[1], itemlist=[])
        else:
            a_champion = champion.champion(shop_unit)
        board[x][y] = a_champion
        return self.rank_comp(board)

    def rank_comp(self, board):
        score = 0
        chosen = ''

        for x in range(7):
            for y in range(4):
                if board[x][y]:
                    score += cost_star_values[board[x][y].cost - 1][board[x][y].stars - 1]
                    if board[x][y].chosen:
                        chosen = board[x][y].chosen

        team_comp, team_tiers = self.update_team_tiers(board, chosen)
        values = list(team_comp.values())
        tier_values = list(team_tiers.values())

        for i in range(len(team_comp)):
            if values[i] != 0:
                score += values[i] * tier_values[i]

        return score

    def update_team_tiers(self, board, chosen):
        team_comp = deepcopy(team_traits)
        team_tiers = deepcopy(team_traits)
        unique_champions = []
        for x in range(0, 7):
            for y in range(0, 4):
                if board[x][y]:
                    if board[x][y].name not in unique_champions:
                        unique_champions.append(board[x][y].name)
                        for trait in board[x][y].origin:
                            team_comp[trait] += 1
        if chosen in team_comp.keys():
            team_comp[chosen] += 1
        for trait in team_comp:
            counter = 0
            while team_comp[trait] >= tiers[trait][counter]:
                counter += 1
                if counter >= len(tiers[trait]):
                    break
            team_tiers[trait] = counter
        return team_comp, team_tiers

    def round_1_2(self, player, shop):
        # buy every unit in the shop until no gold
        self.next_round = self.current_round + 1
        if player.gold > 0 and not player.bench_full():
            shop_position = 0
            for s in shop:
                if s.endswith("_c"):
                    c_shop = s.split('_')[0]
                    if COST[c_shop] * 2 - 1 > player.gold or COST[c_shop] == 1:
                        shop_position += 1
                    else:
                        break
                elif s == " " or COST[s] > player.gold:
                    shop_position += 1
                else:
                    break
            # if no gold remains and we just bought a unit
            if shop_position != 5:
                # print("buying {} with gold {} and cost {} for player {}".format(
                #     shop[shop_position], player.gold, COST[shop[shop_position]], player.player_num))
                return "1_" + str(shop_position)
        max_unit_check = self.max_unit_check(player)
        if max_unit_check != " ":
            return max_unit_check
        return "0"

    def round_3_10(self, player, shop):
        # Reset checks
        if self.current_round == self.next_round:
            self.round_3_10_checks = [True for _ in range(5)]

        if self.require_pair_update:
            self.update_pairs_list(player)
            self.require_pair_update = False

        self.next_round = self.current_round + 1

        # Verify that we have a full board.
        max_unit_check = self.max_unit_check(player)
        if max_unit_check != " ":
            return max_unit_check

        # Check if we are 4 exp from the next level. If so level.
        if player.exp == player.level_costs[player.level] - 4:
            return "5"

        # Check if bench is full. Default sell non 2 star non pair unit. If none, sell the lowest cost.
        if player.bench_full():
            return self.sell_bench_full(player)

        # Next check each shop for triples. First check if we have any pairs with the third available
        # Create check mark booleans that reset only if new round begins, so I can optimize a bit.
        if self.round_3_10_checks[0]:
            for i, shop_unit in enumerate(shop):
                if shop_unit != " ":
                    if shop_unit + "_1" in self.pairs and COST[shop_unit] <= player.gold:
                        self.require_pair_update = True
                        return "1_" + str(i)
            self.round_3_10_checks[0] = False

        # After that, check if any units on my bench will improve my comp.
        # This is rather inefficient, there are some ways to speed it up a little. I could save the positions.
        # Start with the shop. Also buy all pairs
        if self.round_3_10_checks[1]:
            base_score = self.rank_comp(player.board)
            for i, shop_unit in enumerate(shop):
                if shop_unit != " " and (not shop_unit.endswith("_c")) and COST[shop_unit] <= player.gold:
                    for x in range(len(player.board)):
                        for y in range(len(player.board[x])):
                            if player.board[x][y]:
                                # If what I am buying is a pair
                                if player.board[x][y].name == shop_unit:
                                    self.require_pair_update = True
                                    return "1_" + str(i)
                                # If it improves my comp
                                shop_score = self.compare_shop_unit(shop_unit, deepcopy(player.board), x, y)
                                if shop_score > base_score:
                                    self.require_pair_update = True
                                    return "1_" + str(i)
                elif shop_unit.endswith("_c"):
                    c_shop = shop_unit.split('_')[0]
                    if COST[c_shop] != 1 and player.gold >= COST[c_shop] * 2 - 1:
                        return "1_" + str(i)
            self.round_3_10_checks[1] = False

        # Do the same for the bench
        if self.round_3_10_checks[2]:
            base_score = self.rank_comp(player.board)
            for i, bench_unit in enumerate(player.bench):
                if bench_unit:
                    for x in range(len(player.board)):
                        for y in range(len(player.board[x])):
                            if player.board[x][y]:
                                board_copy = deepcopy(player.board)
                                board_copy[x][y] = bench_unit
                                bench_score = self.rank_comp(board_copy)
                                if bench_score > base_score:
                                    # Reset shop checks in case new trait synergies happened due to the change.
                                    self.round_3_10_checks[1] = True
                                    return "2_" + str(x_y_to_1d_coord(x, y)) + "_" + str(28 + i)
            self.round_3_10_checks[2] = False

        # Double check that the units in the front should be in the front and vise versa
        if self.round_3_10_checks[3]:
            for x in range(len(player.board)):
                for y in range(len(player.board[x])):
                    if player.board[x][y]:
                        movement = self.check_unit_location(player, x, y, player.board[x][y].name)
                        if movement:
                            return movement
            self.round_3_10_checks[3] = False

        # Lastly check the cost of the units on the bench that are not a pair with a unit on the board.
        # If selling allows us to hit 10 gold, sell until 10 gold.
        if self.round_3_10_checks[4]:
            cost = 0
            position = 0
            for i, bench_unit in enumerate(player.bench):
                if bench_unit:
                    if (bench_unit.name + "_" + str(bench_unit.stars)) not in self.pairs:
                        cost += bench_unit.cost
                        position = i

            if player.gold // 10 != (player.gold + cost) // 10 and player.gold < 50:
                return "4_" + str(position)
            self.round_3_10_checks[3] = False
        return "0"

    def decide_comp(self, player):
        # First check if current comp has any active traits with any of the comps
        # Use comp with the greatest number of synergies
        if self.comp_number == -1:
            current_comp_traits = []
            for key, tier in player.team_tiers.items():
                if key in TEAM_COMP_TRAITS:
                    if tier > 0:
                        current_comp_traits.append(key)
            if current_comp_traits:
                position = int(np.random.rand() * len(current_comp_traits))
                self.comp_number = TEAM_COMP_TRAITS.index(current_comp_traits[int(position)])
            # If not, pick a random comp
            else:
                self.comp_number = int(np.random.rand() * len(TEAM_COMP_TRAITS))

        # Now sell all units on bench that are not in the desired comp
        # Keeping pairs for now but they will be the first to sell if bench is full
        for i, bench_unit in enumerate(player.bench):
            if bench_unit and bench_unit.name not in TEAM_COMPS[self.comp_number] and \
                    (bench_unit.name + "_" + str(bench_unit.stars)) not in self.pairs:
                return "4_" + str(i)
            # Sell the chosen unit, so we can get one with our desired trait
            elif bench_unit and bench_unit.chosen and bench_unit.chosen != TEAM_COMPS[self.comp_number]:
                return "4_" + str(i)
        # Sell our current chosen unit, so we can pick a new one with our current type.
        for x in range(len(player.board)):
            for y in range(len(player.board[x])):
                if player.board[x][y] and player.board[x][y].chosen and \
                        player.board[x][y].chosen != TEAM_COMPS[self.comp_number]:
                    return "2_" + str(x_y_to_1d_coord(x, y)) + "_28"
        self.round_11_clean_up = False
        return "0"

    def round_11_end(self, player, shop):
        if self.current_round == self.next_round:
            self.round_11_end_checks = [True for _ in range(5)]

        if self.require_pair_update:
            self.update_pairs_list(player)
            self.require_pair_update = False

        self.next_round = self.current_round + 1

        # Verify that we have a full board.
        max_unit_check = self.max_unit_check(player)
        if max_unit_check != " ":
            return max_unit_check

        # Check if bench is full. Default sell non 2 star non pair unit. If none, sell the lowest cost.
        if player.bench_full():
            return self.sell_bench_full(player)

        # Look for pairs and comp units
        if self.round_11_end_checks[0]:
            for i, shop_unit in enumerate(shop):
                if shop_unit != " ":
                    if (shop_unit + "_1" in self.pairs or shop_unit in TEAM_COMPS[self.comp_number]) \
                            and COST[shop_unit] <= player.gold:
                        self.require_pair_update = True
                        return "1_" + str(i)
            self.round_11_end_checks[0] = False

        # Check for any updates on the comp from the shop
        if self.round_11_end_checks[1]:
            base_score = self.rank_comp(player.board)
            for i, shop_unit in enumerate(shop):
                if shop_unit != " " and not shop_unit.endswith("_c") and COST[shop_unit] <= player.gold:
                    for x in range(len(player.board)):
                        for y in range(len(player.board[x])):
                            if player.board[x][y]:
                                # If what I am buying is a pair
                                if player.board[x][y].name == shop_unit:
                                    self.require_pair_update = True
                                    return "1_" + str(i)
                                # If it improves my comp and is not part of the desired comp.
                                # I buy all units that are part of the desired comp above
                                shop_score = self.compare_shop_unit(shop_unit, deepcopy(player.board), x, y)
                                if shop_score > base_score and \
                                        player.board[x][y].name not in TEAM_COMPS[self.comp_number]:
                                    self.require_pair_update = True
                                    return "1_" + str(i)
                # Buy chosen unit for the given comp
                elif shop_unit.endswith("_c"):
                    c_shop = shop_unit.split('_')[0]
                    chosen_type = shop_unit.split('_')[1]
                    if COST[c_shop] != 1 and player.gold >= COST[c_shop] * 2 - 1 and \
                            chosen_type == TEAM_COMP_TRAITS[self.comp_number]:
                        return "1_" + str(i)
            self.round_11_end_checks[1] = False

        # Do the same for the bench
        if self.round_11_end_checks[2]:
            base_score = self.rank_comp(player.board)
            for i, bench_unit in enumerate(player.bench):
                if bench_unit:
                    for x in range(len(player.board)):
                        for y in range(len(player.board[x])):
                            if player.board[x][y]:
                                board_copy = deepcopy(player.board)
                                board_copy[x][y] = bench_unit
                                bench_score = self.rank_comp(board_copy)
                                # First option, both not in comp
                                if (bench_score > base_score and bench_unit.name in TEAM_COMPS[self.comp_number]) or \
                                        (player.board[x][y].name not in TEAM_COMPS[self.comp_number]
                                         and bench_unit.name in TEAM_COMPS[self.comp_number]) or \
                                        (bench_score > base_score and bench_unit.name not in TEAM_COMPS[self.comp_number]
                                         and player.board[x][y].name not in TEAM_COMPS[self.comp_number]):
                                    # Reset shop checks in case new trait synergies happened due to the change.
                                    self.round_11_end_checks[1] = True
                                    return "2_" + str(x_y_to_1d_coord(x, y)) + "_" + str(28 + i)
            self.round_11_end_checks[2] = False

        # Double check that the units in the front should be in the front and vise versa
        if self.round_11_end_checks[3]:
            for x in range(len(player.board)):
                for y in range(len(player.board[x])):
                    if player.board[x][y]:
                        movement = self.check_unit_location(player, x, y, player.board[x][y].name)
                        if movement:
                            return movement
            self.round_11_end_checks[3] = False

        # Check the cost of the units on the bench that are not a pair with a unit on the board.
        # If selling allows us to hit 10 gold, sell until 10 gold.
        if self.round_11_end_checks[4]:
            cost = 0
            position = 0
            for i, bench_unit in enumerate(player.bench):
                if bench_unit:
                    if (bench_unit.name + "_" + str(bench_unit.stars)) not in self.pairs \
                            and bench_unit.name not in TEAM_COMPS[self.comp_number]:
                        cost += bench_unit.cost
                        position = i

            if player.gold // 10 != (player.gold + cost) // 10 and player.gold < 50:
                return "4_" + str(position)
            self.round_11_end_checks[3] = False

        # TODO: Implement usage for champion duplicator
        # TODO: Implement spat usage

        # If above 50 gold and not yet level or when low health, buy exp
        if player.level < 8 and (player.gold >= 54 or (player.health < 30 and player.gold > 4)):
            return "5"

        # Refresh at level 8 or when you get a little desperate
        if (player.level == 8 and player.gold >= 54) or (player.health < 30 and player.gold > 4):
            self.round_11_end_checks[0] = True
            self.round_11_end_checks[1] = True
            self.round_11_end_checks[2] = True
            return "6"

        return "0"

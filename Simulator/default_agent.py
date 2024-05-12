import config
import numpy as np
import Simulator.champion as champion
from Simulator.default_agent_stats import *
from Simulator.pool_stats import cost_star_values
from Simulator.origin_class_stats import tiers, origin_class
from Simulator.origin_class import team_traits
from Simulator.utils import x_y_to_1d_coord
from Simulator.stats import COST, BASE_CHAMPION_LIST
from Simulator.item_stats import starting_items
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
        self.champion_buy_list = [np.array([1, 0, 0, 0, 0]) for _ in range(len(BASE_CHAMPION_LIST))]
        self.sell_chosen = 0
        self.item_guide = np.zeros(config.ITEM_CHOICE_DIM)

    def policy(self, player, shop, game_round, mask):
        self.current_round = game_round
        if game_round == 1 or game_round == 2:
            return self.round_1_2(player, shop, mask)
        elif config.CHAMP_DECIDER:
            return self.ai_round_default(player, shop, mask)
        elif 2 < game_round < 11:
            if game_round == 3 and self.current_round == self.next_round:
                self.update_pairs_list(player)
            return self.round_3_10(player, shop, mask)
        elif game_round == 11 and self.round_11_clean_up:
            return self.decide_comp(player)
        elif game_round >= 11:
            # put a check here to see if current round == next round and round == 11 to pick the comp
            return self.round_11_end(player, shop, mask)
        print("Game_round = {}".format(game_round))
        return "0"

    def move_bench_to_empty_board(self, player, bench_location, unit):
        if unit in FRONT_LINE_UNITS:
            is_full = self.is_row_full(player, 3)
            row_idx = 3 if not is_full else 1
            for displacement in range(4):
                if not player.board[3 + displacement][row_idx]:
                    return "5_" + str(bench_location) + "_" + str(x_y_to_1d_coord(3 + displacement, row_idx))
                if not player.board[3 - displacement][row_idx]:
                    return "5_" + str(bench_location) + "_" + str(x_y_to_1d_coord(3 - displacement, row_idx))
            print("Empty Front line with board {}".format(player.board))
        if unit in MIDDLE_LINE_UNITS:
            for displacement in range(4):
                if not player.board[3 + displacement][2]:
                    return "5_" + str(bench_location) + "_" + str(x_y_to_1d_coord(3 + displacement, 2))
                if not player.board[3 - displacement][2]:
                    return "5_" + str(bench_location) + "_" + str(x_y_to_1d_coord(3 - displacement, 2))
            print("Empty Mid line with board {}".format(player.board))
        if unit in BACK_LINE_UNITS:
            is_full = self.is_row_full(player, 0)
            row_idx = 0 if not is_full else 1
            for displacement in range(4):
                if not player.board[3 + displacement][row_idx]:
                    return "5_" + str(bench_location) + "_" + str(x_y_to_1d_coord(3 + displacement, row_idx))
                if not player.board[3 - displacement][row_idx]:
                    return "5_" + str(bench_location) + "_" + str(x_y_to_1d_coord(3 - displacement, row_idx))
            print("Empty Back line with board {}".format(player.board))
        return "0"

    def check_unit_location(self, player, x, y, unit):
        if unit in FRONT_LINE_UNITS:
            if y != 3:
                is_full = self.is_row_full(player, 3)
                player.print(f"row is full = {is_full}")
                row_idx = 3 if not is_full else 1
                for displacement in range(4):
                    if not player.board[3 + displacement][row_idx]:
                        return "5_" + str(x_y_to_1d_coord(x, y)) + "_" + str(x_y_to_1d_coord(3 + displacement, row_idx))
                    if not player.board[3 - displacement][row_idx]:
                        return "5_" + str(x_y_to_1d_coord(x, y)) + "_" + str(x_y_to_1d_coord(3 - displacement, row_idx))
                print("Check Front line with board {}".format(player.board))

        if unit in MIDDLE_LINE_UNITS:
            if y != 2:
                for displacement in range(4):
                    if not player.board[3 + displacement][2]:
                        return "5_" + str(x_y_to_1d_coord(x, y)) + "_" + str(x_y_to_1d_coord(3 + displacement, 2))
                    if not player.board[3 - displacement][2]:
                        return "5_" + str(x_y_to_1d_coord(x, y)) + "_" + str(x_y_to_1d_coord(3 - displacement, 2))
                print("Check Mid line with board {}".format(player.board))
        if unit in BACK_LINE_UNITS:
            if y != 0:
                is_full = self.is_row_full(player, 0)
                row_idx = 0 if not is_full else 1
                for displacement in range(4):
                    if not player.board[3 + displacement][row_idx]:
                        return "5_" + str(x_y_to_1d_coord(x, y)) + "_" + str(x_y_to_1d_coord(3 + displacement, row_idx))
                    if not player.board[3 - displacement][row_idx]:
                        return "5_" + str(x_y_to_1d_coord(x, y)) + "_" + str(x_y_to_1d_coord(3 - displacement, row_idx))
                print("Check Back line with board {}".format(player.board))
        return False

    def buy_every_shop_unit(self, player, shop, mask):
        shop_position = 0
        for s in shop:
            if not mask[47 + shop_position][0]:
                shop_position += 1
            elif s.endswith("_c"):
                c_shop = s.split('_')[0]
                if COST[c_shop] * 2 - 1 > player.gold or COST[c_shop] == 1:
                    shop_position += 1
                else:
                    break
            else:
                break
        return shop_position

    def max_unit_check(self, player, shop, mask):
        # place units in front or back.
        if player.num_units_in_play < player.max_units:
            for i, bench_slot in enumerate(player.bench):
                if bench_slot:
                    return self.move_bench_to_empty_board(player, 28 + i, bench_slot.name)
            if not any(player.bench) and player.gold > 0:
                # if no gold remains and we just bought a unit
                shop_position = self.buy_every_shop_unit(player, shop, mask)
                if shop_position != 5:
                    return "3_" + str(shop_position)
        return " "
    
    def is_row_full(self, player, row):
        for col in player.board:
            if col[row] is None:
                return False
        return True

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
                    return "4_" + str(i + 28)
        for i, bench_unit in enumerate(player.bench):
            if (bench_unit.name + "_" + str(bench_unit.stars)) not in self.pairs and bench_unit.stars == 1:
                return "4_" + str(i + 28)
        low_cost = 100
        position = 0
        for i, bench_unit in enumerate(player.bench):
            if bench_unit.cost < low_cost:
                position = i + 28
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
                if board[x][y] and board[x][y].name in BASE_CHAMPION_LIST:
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
                    if board[x][y].name not in unique_champions and board[x][y].name in BASE_CHAMPION_LIST:
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

    def set_default_guide(self, default_guide):
        self.champion_buy_list = np.eye(config.CHAMPION_ACTION_DIM[0])[default_guide[:58]]
        self.sell_chosen = default_guide[58]
        self.item_guide = np.eye(config.ITEM_CHOICE_DIM[0])[default_guide[59:]]

    def round_1_2(self, player, shop, mask):
        # buy every unit in the shop until no gold
        self.next_round = self.current_round + 1
        if player.gold > 0 and not player.bench_full():
            shop_position = self.buy_every_shop_unit(player, shop, mask)
            # if no gold remains and we just bought a unit
            if shop_position != 5:
                return "3_" + str(shop_position)
        max_unit_check = self.max_unit_check(player, shop, mask)
        if max_unit_check != " ":
            return max_unit_check
        return "0"

    def check_chosen_trait(self, string_name, trait):
        if self.champion_buy_list[BASE_CHAMPION_LIST.index(string_name)][4]:
            return True
        elif self.champion_buy_list[BASE_CHAMPION_LIST.index(string_name)][2] and trait == origin_class[string_name][0]:
            return True
        elif self.champion_buy_list[BASE_CHAMPION_LIST.index(string_name)][3] and trait == origin_class[string_name][1]:
            return True
        return False

    def round_3_10(self, player, shop, mask):
        # Reset checks
        if self.current_round == self.next_round:
            self.round_3_10_checks = [True for _ in range(6)]

        if self.require_pair_update:
            self.update_pairs_list(player)
            self.require_pair_update = False

        self.next_round = self.current_round + 1

        # Verify that we have a full board.
        max_unit_check = self.max_unit_check(player, shop, mask)
        if max_unit_check != " ":
            return max_unit_check

        # Check if we are 4 exp from the next level. If so level.
        if player.exp == player.level_costs[player.level] - 4:
            return "1"

        # Check if bench is full. Default sell non 2 star non pair unit. If none, sell the lowest cost.
        if player.bench_full():
            return self.sell_bench_full(player)

        # Next check each shop for triples. First check if we have any pairs with the third available
        # Create check mark booleans that reset only if new round begins, so I can optimize a bit.
        if self.round_3_10_checks[0]:
            for i, shop_unit in enumerate(shop):
                if mask[47 + i][0]:
                    if shop_unit + "_1" in self.pairs and not shop_unit.endswith("_c"):
                        self.require_pair_update = True
                        return "3_" + str(i)
            self.round_3_10_checks[0] = False

        # After that, check if any units on my bench will improve my comp.
        # This is rather inefficient, there are some ways to speed it up a little. I could save the positions.
        # Start with the shop. Also buy all pairs
        if self.round_3_10_checks[1]:
            base_score = self.rank_comp(player.board)
            for i, shop_unit in enumerate(shop):
                if mask[47 + i][0]:
                    if not shop_unit.endswith("_c"):
                        for x in range(len(player.board)):
                            for y in range(len(player.board[x])):
                                if player.board[x][y]:
                                    # If what I am buying is a pair
                                    if player.board[x][y].name == shop_unit:
                                        self.require_pair_update = True
                                        return "3_" + str(i)
                                    # If it improves my comp
                                    shop_score = self.compare_shop_unit(shop_unit, deepcopy(player.board), x, y)
                                    if shop_score > base_score:
                                        self.require_pair_update = True
                                        return "3_" + str(i)
                    elif shop_unit.endswith("_c"):
                        c_shop = shop_unit.split('_')[0]
                        if COST[c_shop] != 1 and player.gold >= COST[c_shop] * 3 - 1:
                            return "3_" + str(i)
            self.round_3_10_checks[1] = False

        # Do the same for the bench
        if self.round_3_10_checks[2]:
            base_score = self.rank_comp(player.board)
            for i, bench_unit in enumerate(player.bench):
                if bench_unit:
                    for x in range(len(player.board)):
                        for y in range(len(player.board[x])):
                            if player.board[x][y] and player.board[x][y] in BASE_CHAMPION_LIST:
                                board_copy = deepcopy(player.board)
                                board_copy[x][y] = bench_unit
                                bench_score = self.rank_comp(board_copy)
                                if bench_score > base_score:
                                    # Reset shop checks in case new trait synergies happened due to the change.
                                    self.round_3_10_checks[1] = True
                                    return "3_" + str(x_y_to_1d_coord(x, y)) + "_" + str(28 + i)
            self.round_3_10_checks[2] = False
            
        if self.round_3_10_checks[5]:
            # Add items to highest level units
            item_idx = []
            for i, item in enumerate(player.item_bench):
                if item is not None and not (item == "champion_duplicator" or item == "spatula"):
                    item_idx.append((item, i))
            # check how many items we can make
            champions = []
            for x in range(len(player.board)):
                for y in range(len(player.board[x])):
                    if player.board[x][y]:
                        champions.append((player.board[x][y], x_y_to_1d_coord(x, y)))
            # sort champions by level
            champions.sort(key=lambda x: x[0].cost * x[0].stars, reverse=True)
            
            # add items to champions
            for champion, coord in champions:
                if len(item_idx) == 0:
                    break
                while len(item_idx) > 0:
                    item, idx = item_idx.pop()
                    if mask[37 + idx][coord]:
                        return "6_" + str(idx) + "_" + str(coord)
        
            self.round_3_10_checks[5] = False

        # Double check that the units in the front should be in the front and vise versa
        if self.round_3_10_checks[3]:
            for x in range(len(player.board)):
                for y in range(len(player.board[x])):
                    if player.board[x][y]:
                        movement = self.check_unit_location(player, x, y, player.board[x][y].name)
                        if movement:
                            player.print(f"movement command with {player.board[x][y]} at {x} {y} to {movement}")
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
                        position = i + 28

            if player.gold // 10 != (player.gold + cost) // 10 and player.gold < 50:
                player.print(f"selling units at position for gold {position} ")
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
                return "4_" + str(i + 28)
            # Sell the chosen unit, so we can get one with our desired trait
            elif bench_unit and bench_unit.chosen and bench_unit.chosen != TEAM_COMPS[self.comp_number]:
                return "4_" + str(i + 28)
        # Sell our current chosen unit, so we can pick a new one with our current type.
        for x in range(len(player.board)):
            for y in range(len(player.board[x])):
                if player.board[x][y] and player.board[x][y].chosen and \
                        player.board[x][y].chosen != TEAM_COMPS[self.comp_number]:
                    return "4_" + str(x_y_to_1d_coord(x, y))
        self.round_11_clean_up = False
        return "0"

    def round_11_end(self, player, shop, mask):
        if self.current_round == self.next_round:
            self.round_11_end_checks = [True for _ in range(5)]

        if self.require_pair_update:
            self.update_pairs_list(player)
            self.require_pair_update = False

        self.next_round = self.current_round + 1

        # Verify that we have a full board.
        max_unit_check = self.max_unit_check(player, shop, mask)
        if max_unit_check != " ":
            return max_unit_check

        # Check if bench is full. Default sell non 2 star non pair unit. If none, sell the lowest cost.
        if player.bench_full():
            return self.sell_bench_full(player)

        # Look for pairs and comp units
        if self.round_11_end_checks[0]:
            for i, shop_unit in enumerate(shop):
                if mask[47 + i][0]:
                    if (shop_unit + "_1" in self.pairs or shop_unit in TEAM_COMPS[self.comp_number]) \
                            and COST[shop_unit] <= player.gold:
                        self.require_pair_update = True
                        # print("buying triple or comp unit {}".format(shop_unit))
                        return "3_" + str(i)
            self.round_11_end_checks[0] = False

        # Check for any updates on the comp from the shop
        if self.round_11_end_checks[1]:
            base_score = self.rank_comp(player.board)
            for i, shop_unit in enumerate(shop):
                if mask[47 + i][0]:
                    if not shop_unit.endswith("_c"):
                        for x in range(len(player.board)):
                            for y in range(len(player.board[x])):
                                if player.board[x][y]:
                                    # If what I am buying is a pair
                                    if player.board[x][y].name == shop_unit:
                                        self.require_pair_update = True
                                        # print("buying pair 11_end {}".format(shop_unit))
                                        return "3_" + str(i)
                                    # If it improves my comp and is not part of the desired comp.
                                    # I buy all units that are part of the desired comp above
                                    shop_score = self.compare_shop_unit(shop_unit, deepcopy(player.board), x, y)
                                    if shop_score > base_score and \
                                            player.board[x][y].name not in TEAM_COMPS[self.comp_number]:
                                        self.require_pair_update = True
                                        # print("buying shop_score 11_end {}".format(shop_unit))
                                        return "3_" + str(i)
                    # Buy chosen unit for the given comp
                    else:
                        c_shop = shop_unit.split('_')[0]
                        chosen_type = shop_unit.split('_')[1]
                        if COST[c_shop] != 1 and player.gold >= COST[c_shop] * 3 - 1 and \
                                chosen_type == TEAM_COMP_TRAITS[self.comp_number]:
                            # print("buying chosen 11_end {}".format(shop_unit))
                            return "3_" + str(i)
            self.round_11_end_checks[1] = False

        # Do the same for the bench
        if self.round_11_end_checks[2]:
            base_score = self.rank_comp(player.board)
            for i, bench_unit in enumerate(player.bench):
                if bench_unit:
                    for x in range(len(player.board)):
                        for y in range(len(player.board[x])):
                            if player.board[x][y] and player.board[x][y] in BASE_CHAMPION_LIST:
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
                                    return "5_" + str(x_y_to_1d_coord(x, y)) + "_" + str(28 + i)
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
                        position = i + 28

            if player.gold // 10 != (player.gold + cost) // 10 and player.gold < 50:
                return "4_" + str(position)
            self.round_11_end_checks[3] = False

        # TODO: Implement usage for champion duplicator
        # TODO: Implement spat usage

        # If above 50 gold and not yet level or when low health, buy exp
        if player.level < 8 and (player.gold >= 54 or (player.health < 30 and player.gold > 4)):
            return "1"

        # Refresh at level 8 or when you get a little desperate
        if (player.level == 8 and player.gold >= 54) or (player.health < 30 and player.gold > 4):
            self.round_11_end_checks[0] = True
            self.round_11_end_checks[1] = True
            self.round_11_end_checks[2] = True
            return "2"

        return "0"

    def ai_round_default(self, player, shop, mask):
        if self.current_round == self.next_round:
            self.round_11_end_checks = [True for _ in range(5)]

        if self.require_pair_update:
            self.update_pairs_list(player)
            self.require_pair_update = False

        self.next_round = self.current_round + 1

        # Check if we are 4 exp from the next level. If so level.
        if (player.exp == player.level_costs[player.level] - 4) and mask[53][0]:
            return "1"

        # Verify that we have a full board.
        max_unit_check = self.max_unit_check(player, shop, mask)
        if max_unit_check != " ":
            return max_unit_check

        # Check if bench is full. Default sell non 2 star non pair unit. If none, sell the lowest cost.
        if player.bench_full():
            return self.sell_bench_full(player)

        # Look for pairs and comp units
        if self.round_11_end_checks[0]:
            for i, shop_unit in enumerate(shop):
                if mask[47 + i][0]:
                    if shop_unit.endswith("_c"):
                        c_shop = shop_unit.split('_')[0]
                        chosen_type = shop_unit.split('_')[1]
                        check_chosen_trait = self.check_chosen_trait(c_shop, chosen_type)
                        # If it is more than a 1 cost champion, we have to buy, a type we are running, and want to buy.
                        if player.gold >= COST[c_shop] * 3 - 1 and check_chosen_trait:
                            return "3_" + str(i)
                    else:
                        if not self.champion_buy_list[BASE_CHAMPION_LIST.index(shop_unit)][0] and \
                                COST[shop_unit] <= player.gold:
                            self.require_pair_update = True
                            return "3_" + str(i)
            self.round_11_end_checks[0] = False

        # Do the same for the bench
        # TODO: Do every unique champion combination available on bench and board.
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
                                        (bench_score > base_score and bench_unit.name not in TEAM_COMPS[
                                            self.comp_number]
                                         and player.board[x][y].name not in TEAM_COMPS[self.comp_number]):
                                    # Reset shop checks in case new trait synergies happened due to the change.
                                    self.round_11_end_checks[1] = True
                                    return "5_" + str(x_y_to_1d_coord(x, y)) + "_" + str(28 + i)
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
                        position = i + 28

            if player.gold // 10 != (player.gold + cost) // 10 and player.gold < 50:
                return "4_" + str(position)
            self.round_11_end_checks[3] = False

        # TODO: Implement usage for champion duplicator
        # TODO: Implement spat usage

        # If above 50 gold and not yet level or when low health, buy exp
        if player.level < 8 and (player.gold >= 54 or (player.health < 30 and mask[53][0])):
            return "1"

        # Refresh at level 8 or when you get a little desperate
        if (player.level == 8 and player.gold >= 54) or (player.health < 30 and mask[54][0]):
            self.round_11_end_checks[0] = True
            self.round_11_end_checks[1] = True
            self.round_11_end_checks[2] = True
            return "2"

        return "0"

import Simulator.champion as champion
from Simulator.default_agent_stats import *
from Simulator.pool_stats import cost_star_values
from Simulator.origin_class_stats import tiers
from Simulator.origin_class import team_traits
from Simulator.utils import x_y_to_1d_coord
from Simulator.stats import COST

class Default_Agent:
    def __init__(self):
        self.current_round = 0
        self.next_round = 3
        self.round_3_10_checks = [True for _ in range(5)]
        self.pairs = []
        self.require_pair_update = False

    def policy(self, player, shop, game_round):
        self.current_round = game_round
        if game_round == 1 or game_round == 2:
            return self.round_1_2(player, shop)
        elif 2 < game_round < 11:
            if game_round == 3 and self.current_round == self.next_round:
                self.update_pairs_list(player)
            return self.round_3_10(player, shop)
        elif game_round >= 11:
            # put a check here to see if current round == next round and round == 11 to pick the comp
            return self.round_11_end(player, shop)
        print("Game_round = {}".format(game_round))
        return "0"

    def move_bench_to_empty_board(self, player, bench_location, unit):
        if unit in FRONT_LINE_UNITS:
            for displacement in range(3):
                if not player.board[3 + displacement][3]:
                    return "2_" + str(bench_location) + "_" + str(24 + displacement)
                if not player.board[3 - displacement][3]:
                    return "2_" + str(bench_location) + "_" + str(24 - displacement)
            print("I should never be here front line")
        if unit in MIDDLE_LINE_UNITS:
            for displacement in range(3):
                if not player.board[3 + displacement][2]:
                    return "2_" + str(bench_location) + "_" + str(17 + displacement)
                if not player.board[3 - displacement][2]:
                    return "2_" + str(bench_location) + "_" + str(17 - displacement)
            print("I should never be here mid line")
        if unit in BACK_LINE_UNITS:
            for displacement in range(3):
                if not player.board[3 + displacement][0]:
                    return "2_" + str(bench_location) + "_" + str(3 + displacement)
                if not player.board[3 - displacement][0]:
                    return "2_" + str(bench_location) + "_" + str(3 - displacement)
            print("I should never be here back line")
        return "0"

    def check_unit_location(self, player, x, y, unit):
        if unit in FRONT_LINE_UNITS:
            if y != 3:
                for displacement in range(3):
                    if not player.board[3 + displacement][3]:
                        return "2_" + str(x_y_to_1d_coord(x, y)) + "_" + str(24 + displacement)
                    if not player.board[3 - displacement][3]:
                        return "2_" + str(x_y_to_1d_coord(x, y)) + "_" + str(24 - displacement)
                print("I should never be here front line")

        if unit in MIDDLE_LINE_UNITS:
            if y != 2:
                for displacement in range(3):
                    if not player.board[3 + displacement][2]:
                        return "2_" + str(x_y_to_1d_coord(x, y)) + "_" + str(17 + displacement)
                    if not player.board[3 - displacement][2]:
                        return "2_" + str(x_y_to_1d_coord(x, y)) + "_" + str(17 - displacement)
                print("I should never be here mid line")
        if unit in BACK_LINE_UNITS:
            if y != 0:
                for displacement in range(3):
                    if not player.board[3 + displacement][0]:
                        return "2_" + str(x_y_to_1d_coord(x, y)) + "_" + str(3 + displacement)
                    if not player.board[3 - displacement][0]:
                        return "2_" + str(x_y_to_1d_coord(x, y)) + "_" + str(3 - displacement)
                print("I should never be here back line")
        return False

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
        team_comp = team_traits.copy()
        team_tiers = team_traits.copy()
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
        if player.gold > 0:
            shop_position = 0
            for s in shop:
                if s.endswith("_c"):
                    c_shop = s.split('_')[0]
                    if COST[c_shop] * 2 > player.gold:
                        shop_position += 1
                elif s == " " or COST[s] > player.gold:
                    shop_position += 1
                else:
                    break
            # if no gold remains and we just bought a unit
            if shop_position != 5:
                return "1_" + str(shop_position)
        # place units in front or back.
        if player.num_units_in_play < player.max_units:
            for i, bench_slot in enumerate(player.bench):
                if bench_slot:
                    dummy_return = self.move_bench_to_empty_board(player, i, bench_slot.name)
                    return dummy_return
        return "0"

    def round_3_10(self, player, shop):
        # Reset checks
        if self.current_round == self.next_round:
            print("Resetting checks")
            self.round_3_10_checks = [True for _ in range(5)]

        if self.require_pair_update:
            self.update_pairs_list(player)
            self.require_pair_update = False

        self.next_round = self.current_round + 1
        # First check if we are 4 exp from the next level. If so level.
        if player.exp == player.level_costs[player.level] - 4:
            return "5"

        # Check if bench is full. Default sell non 2 star non pair unit. If none, sell the lowest cost.
        if player.bench_full():
            for i, bench_unit in enumerate(player.bench):
                if bench_unit.name not in self.pairs and bench_unit.stars == 1:
                    return "4_" + str(i)
            low_cost = 100
            position = 0
            for i, bench_unit in enumerate(player.bench):
                if bench_unit.cost < low_cost:
                    position = i
            return "4_" + str(position)

        # Next check each shop for triples. First check if we have any pairs with the third available
        # Create check mark booleans that reset only if new round begins, so I can optimize a bit.
        if self.round_3_10_checks[0]:
            for i, shop_unit in enumerate(shop):
                if shop_unit != " ":
                    if shop_unit in self.pairs:
                        self.require_pair_update = True
                        return "1_" + str(i)
            self.round_3_10_checks[0] = False

        # After that, check if any units on my bench will improve my comp.
        # This is rather inefficient, there are some ways to speed it up a little. I could save the positions.
        # Start with the shop. Also buy all pairs
        if self.round_3_10_checks[1]:
            base_score = self.rank_comp(player.board.copy())
            for i, shop_unit in enumerate(shop):
                if shop_unit != " ":
                    for x in range(len(player.board)):
                        for y in range(len(player.board[x])):
                            if player.board[x][y]:
                                # If what I am buying is a pair
                                if player.board[x][y].name == shop_unit:
                                    self.require_pair_update = True
                                    return "1_" + str(i)
                                # If it improves my comp
                                shop_score = self.compare_shop_unit(shop_unit, player.board.copy(), x, y)
                                if shop_score > base_score:
                                    self.require_pair_update = True
                                    return "1_" + str(i)
            self.round_3_10_checks[1] = False

        # Do the same for the bench
        if self.round_3_10_checks[2]:
            base_score = self.rank_comp(player.board.copy())
            for i, bench_unit in enumerate(player.bench):
                if bench_unit != " ":
                    for x in range(len(player.board)):
                        for y in range(len(player.board[x])):
                            if player.board[x][y]:
                                board_copy = player.board.copy()
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
                        movement = self.check_unit_location(player, x, y, player.board.name)
                        if movement:
                            print("Moving board to board {}".format(movement))
                            return movement
            self.round_3_10_checks[3] = False

        # Lastly check the cost of the units on the bench that are not a pair with a unit on the board.
        # If selling allows us to hit 10 gold, sell until 10 gold.
        if self.round_3_10_checks[4]:
            cost = 0
            position = 0
            # TODO: Update this part of the method to respect 2-star units and 1-star units after a 2-star.
            for i, bench_unit in enumerate(player.bench):
                if bench_unit:
                    if bench_unit.name not in self.pairs:
                        cost += bench_unit.cost
                        position = i

            if player.gold // 10 != (player.gold + cost) // 10 and player.gold < 50:
                print("selling unit {} != {}, {}".format(player.gold, (player.gold + cost) // 10, cost))
                return "4_" + str(position)
            self.round_3_10_checks[3] = False
        return "0"

    def round_11_end(self, player, shop):
        return "0"

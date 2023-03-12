# Description: This script will verify the drop rate of a shop 
# by running the simulator 100000 times and counting the number of times
# each champion is chosen.

from Simulator.player import player
from Simulator.pool import pool
from Simulator.champion import champion
from Simulator.pool import COST_1, COST_2, COST_3, COST_4, COST_5

# Verify shop drop rates for each cost are correct

# Correct drop rates of each cost by level for set 4
CORRECT_DROP_RATES = [
    [1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [0.75, 0.25, 0, 0, 0],
    [0.55, 0.30, 0.15, 0, 0],
    [0.45, 0.33, 0.20, 0.02, 0],
    [0.25, 0.40, 0.30, 0.05, 0],
    [0.20, 0.30, 0.35, 0.14, 0.01],
    [0.15, 0.20, 0.35, 0.25, 0.05],
    [0.10, 0.15, 0.30, 0.30, 0.15],
]


def verifyShopDropRate():
    # Create a player and pool
    pool1 = pool()
    player1 = player(pool_pointer=pool1, player_num=1)

    # loop through 9 player levels
    for level in range(1, 10):
        player1.level = level
        costs = verify(pool1, player1)

        # what is the percentage of each cost chosen?
        print("Player level: " + str(level))

        total_1_cost = 0
        total_2_cost = 0
        total_3_cost = 0
        total_4_cost = 0
        total_5_cost = 0

        # loop though each cost 1 champion
        for i in range(len(costs[0])):
            print(str(list(costs[0])[i]) + ": " + str(costs[0][list(costs[0])[i]] / 100000))
            total_1_cost += costs[0][list(costs[0])[i]]

        # loop though each cost 2 champion
        for i in range(len(costs[1])):
            print(str(list(costs[1])[i]) + ": " + str(costs[1][list(costs[1])[i]] / 100000))
            total_2_cost += costs[1][list(costs[1])[i]]

        # loop though each cost 3 champion
        for i in range(len(costs[2])):
            print(str(list(costs[2])[i]) + ": " + str(costs[2][list(costs[2])[i]] / 100000))
            total_3_cost += costs[2][list(costs[2])[i]]

        # loop though each cost 4 champion
        for i in range(len(costs[3])):
            print(str(list(costs[3])[i]) + ": " + str(costs[3][list(costs[3])[i]] / 100000))
            total_4_cost += costs[3][list(costs[3])[i]]

        # loop though each cost 5 champion
        for i in range(len(costs[4])):
            print(str(list(costs[4])[i]) + ": " + str(costs[4][list(costs[4])[i]] / 100000))
            total_5_cost += costs[4][list(costs[4])[i]]

        # need to add proper statistical one way ANOVA to be extra sure, right now error is arbitrary (variance of 1%)
        assert (CORRECT_DROP_RATES[level - 1][0] - 0.01) <= total_1_cost / 100000 <= (
                    CORRECT_DROP_RATES[level - 1][0] + 0.01), \
            f"1 Cost Champion drop rate at level {level} is significantly incorrect! Control: " \
            f"{CORRECT_DROP_RATES[level - 1][0]} Sample: {total_1_cost / 100000}"

        assert (CORRECT_DROP_RATES[level - 1][1] - 0.01) <= total_2_cost / 100000 <= (
                    CORRECT_DROP_RATES[level - 1][1] + 0.01), \
            f"2 Cost Champion drop rate at level {level} is significantly incorrect! Control: " \
            f"{CORRECT_DROP_RATES[level - 1][1]} Sample: {total_2_cost / 100000}"

        assert (CORRECT_DROP_RATES[level - 1][2] - 0.01) <= total_3_cost / 100000 <= (
                    CORRECT_DROP_RATES[level - 1][2] + 0.01), \
            f"3 Cost Champion drop rate at level {level} is significantly incorrect! Control: " \
            f"{CORRECT_DROP_RATES[level - 1][2]} Sample: {total_3_cost / 100000}"

        assert (CORRECT_DROP_RATES[level - 1][3] - 0.01) <= total_4_cost / 100000 <= (
                    CORRECT_DROP_RATES[level - 1][3] + 0.01), \
            f"4 Cost Champion drop rate at level {level} is significantly incorrect! Control: " \
            f"{CORRECT_DROP_RATES[level - 1][3]} Sample: {total_4_cost / 100000}"

        assert (CORRECT_DROP_RATES[level - 1][4] - 0.01) <= total_5_cost / 100000 <= (
                    CORRECT_DROP_RATES[level - 1][4] + 0.01), \
            f"5 Cost Champion drop rate at level {level} is significantly incorrect! Control: " \
            f"{CORRECT_DROP_RATES[level - 1][4]} Sample: {total_5_cost / 100000}"

        print("Total percent (should be 1.0 or very close)  : " + str(
            (total_1_cost + total_2_cost + total_3_cost + total_4_cost + total_5_cost) / 100000))


def verify(_pool, _player):
    shop = pool.sample(self=_pool, player=_player, num=100000, idx=-1)
    # a dictionary to store the number of times each cost is chosen

    cost_1 = {}
    cost_2 = {}
    cost_3 = {}
    cost_4 = {}
    cost_5 = {}

    # the number of times each cost is chosen
    for i in range(len(shop)):
        if shop[i] in COST_1:
            if shop[i] in cost_1:
                cost_1[shop[i]] += 1
            else:
                cost_1[shop[i]] = 1
        elif shop[i] in COST_2:
            if shop[i] in cost_2:
                cost_2[shop[i]] += 1
            else:
                cost_2[shop[i]] = 1
        elif shop[i] in COST_3:
            if shop[i] in cost_3:
                cost_3[shop[i]] += 1
            else:
                cost_3[shop[i]] = 1
        elif shop[i] in COST_4:
            if shop[i] in cost_4:
                cost_4[shop[i]] += 1
            else:
                cost_4[shop[i]] = 1
        elif shop[i] in COST_5:
            if shop[i] in cost_5:
                cost_5[shop[i]] += 1
            else:
                cost_5[shop[i]] = 1
        else:
            pass

    return cost_1, cost_2, cost_3, cost_4, cost_5


def list_of_tests():
    verifyShopDropRate()

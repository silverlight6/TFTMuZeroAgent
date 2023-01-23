from enum import Enum
import random
import numpy as np
from Simulator import champion,  pool_stats
from Simulator.item_stats import item_builds as full_items, starting_items

item_list = list(full_items.keys())

# Loot Orbs
# There are 3 types of orbs:
# Common (Gray): 3 gold worth of champions, gold, or champion duplicator
# Uncommon (Blue): 6 gold worth of champions, gold, or items
# Rare (Gold): 10 gold worth of champions, gold, or spatula
# TODO add reforgers, magnetic removers, emblem tomes, and item choosers

class LootOrb(Enum):
    COMMON = {
        'three_gold': .4,
        'three_cost': .2,
        'two_cost_one_gold': .2,
        'three_one_costs': .15,
        'champion_duplicator_one_gold': .05
    }
    UNCOMMON = {
        'six_gold': .1,
        'two_three_costs': .1,
        'three_two_costs': .1,
        'one_item': .7
    }
    RARE = {
        'ten_gold': .1,
        'two_five_costs': .1,
        'spatula': .80,
    }

# Implementation of the loot that can be given to the player
# The rewards are pretty self explanatory; 'three_one_costs' gives three random one cost champions, 'one_item' gives one random item, etc.
def give_loot(player, reward):
    # I would use a match here but we're on python 3.8
    if reward == 'three_gold':
        player.gold += 3
    if reward == 'six_gold':
        player.gold += 6
    if reward == 'ten_gold':
        player.gold += 10
    if reward == 'three_one_costs':
        give_champions(player, pool_stats.COST_1, count=3)
    if reward == 'two_cost_one_gold':
        give_champions(player, pool_stats.COST_2)
        player.gold += 1
    if reward == 'three_cost':
        give_champions(player, pool_stats.COST_3)
    if reward == 'three_two_costs':
        give_champions(player, pool_stats.COST_2, count=3)
    if reward == 'two_three_costs':
        give_champions(player, pool_stats.COST_3, count=2)
    if reward == 'two_five_costs':
        give_champions(player, pool_stats.COST_5, count=2)
    if reward == 'one_item':
        give_random_item(player)
    if reward == 'full_item':
        give_random_full_item(player)
    if reward == 'champion_duplicator_one_gold':
        player.gold += 1
        player.add_to_item_bench('champion_duplicator')
    if reward == 'spatula':
        player.add_to_item_bench('spatula')


# Utility Functions

## Random Chapmions
def give_champions(player, cost, count=1):
    for _ in range(count):
        give_champion(player, cost)

def give_champion(player, cost):
    # Give gold if bench is full
    if player.bench_full():
        if cost is pool_stats.COST_1:
            player.gold += 1
        if cost is pool_stats.COST_2:
            player.gold += 2
        if cost is pool_stats.COST_3:
            player.gold += 3
        if cost is pool_stats.COST_5:
            player.gold += 5
    else:
        name = list(cost.items())[random.randint(0, len(cost) - 1)][0]
        random_champion = champion.champion(name)
        player.pool_obj.update_pool(random_champion, -1)
        player.add_to_bench(random_champion)

## Random Items
def give_random_item(player):
    player.add_to_item_bench(starting_items[random.randint(0, len(starting_items) - 1)])

def give_random_full_item(player):
    player.add_to_item_bench(item_list[random.randint(0, len(item_list) - 1)])

# Helper functions for getting orbs after minion rounds
def gen_orbs(choices, probabilities, count):
    orbs = []
    choices = np.array(choices, dtype=object)

    for _ in range(count):
        orb = np.random.choice(choices, p=probabilities)
        if type(orb) is list:
            orbs.extend(orb)
        else:
            orbs.append(orb)

    return orbs

# Gets a random reward based on the loot table (dictionary) defined in the LootOrb enum
def gen_orb_reward(loot_orb: LootOrb):
    choices = list(loot_orb.value.keys())
    probabilities = list(loot_orb.value.values())

    reward = np.random.choice(choices, p=probabilities)
    print(reward)

    return reward

# Combines `gen_orbs` and `gen_orb_reward` to directly give you the rewards from loot orbs
# e.g after raptors you might get ['one_item', 'one_item', 'one_item', 'three_gold', 'two_three_costs']
def gen_loot(choices, probabilities, count):
    loot = []

    orbs = gen_orbs(choices, probabilities, count)

    for orb in orbs:
        loot.append(gen_orb_reward(orb))

    return loot

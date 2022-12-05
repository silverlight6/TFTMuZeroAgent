from Simulator.item_stats import item_builds as full_items, basic_items, starting_items
from Simulator import champion, pool, pool_stats
import random

def minion_round(player, round):
 # simulate minion round here
    if round == 1:
        # do minion round 1 stuff...
        player.add_to_item_bench(starting_items[random.randint(0, len(starting_items) - 1)])
        ran_cost_3 = list(pool_stats.COST_3.items())[random.randint(0, len(pool_stats.COST_3) - 1)][0]
        ran_cost_3 = champion.champion(ran_cost_3)
        player.add_to_bench(ran_cost_3)
    elif round == 2:
        player.add_to_item_bench(starting_items[random.randint(0, len(starting_items) - 1)])
        player.gold += 3
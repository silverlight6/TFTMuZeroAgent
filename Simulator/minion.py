from Simulator.item_stats import item_builds as full_items, basic_items, starting_items
from Simulator import champion, pool, pool_stats
import random

item_list = list(full_items.keys())

# TODO
# add randomness to drops
# add representations of minions (Melees, Ranges, Krugs, Wolves, Raptors, Nexus, and Herald)
# add combat between player and minions

def minion_round(player, round, pool_obj):
 # simulate minion round here
    # 2 melee minions - give 1 item component
    if round == 0:
        player.add_to_item_bench(starting_items[random.randint(0, len(starting_items) - 1)])

    # 2 melee and 1 ranged minion - give 1 item component and 1 3 cost champion
    elif round == 1:
        player.add_to_item_bench(starting_items[random.randint(0, len(starting_items) - 1)])
        ran_cost_3 = list(pool_stats.COST_3.items())[random.randint(0, len(pool_stats.COST_3) - 1)][0]
        ran_cost_3 = champion.champion(ran_cost_3)
        pool_obj.update(ran_cost_3, -1)
        player.add_to_bench(ran_cost_3)

    # 2 melee minions and 2 ranged minions - give 3 gold and 1 item component
    elif round == 2:
        player.add_to_item_bench(starting_items[random.randint(0, len(starting_items) - 1)])
        player.gold += 3
        
    # 3 Krugs - give 3 gold and 3 item components
    elif round == 9:
        player.gold += 3
        for _ in range(0, 3):
            player.add_to_item_bench(starting_items[random.randint(0, len(starting_items) - 1)])

    # 1 Greater Murk Wolf and 4 Murk Wolves - give 3 gold and 3 item components
    elif round == 15:
        player.gold += 3
        for _ in range(0, 3):
            player.add_to_item_bench(starting_items[random.randint(0, len(starting_items) - 1)])

    # 1 Crimson Raptor and 4 Raptors - give 6 gold and 4 item components
    elif round == 21:
        player.gold += 6
        for _ in range(0, 4):
            player.add_to_item_bench(starting_items[random.randint(0, len(starting_items) - 1)])

    # 1 Nexus Minion - give 6 gold and a full item
    elif round == 27:
        player.gold += 6
        player.add_to_item_bench(item_list[random.randint(0, len(item_list) - 1)])

    # Rift Herald - give 6 gold and a full item
    elif round >= 33:
        player.gold += 6
        player.add_to_item_bench(item_list[random.randint(0, len(item_list) - 1)])

    # invalid round! Do nothing
    else:
        return


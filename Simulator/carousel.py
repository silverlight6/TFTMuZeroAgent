import config
from Simulator.item_stats import item_builds as full_items, basic_items, starting_items
from Simulator import champion, pool, pool_stats
import random

def carousel(players, round):
    # probability of certain arrangements during certain carousels
    # https://leagueoflegends.fandom.com/wiki/Carousel_(Teamfight_Tactics)
    alive = []
    alive.append(players[0])
    # sort the list of alive players with the lowest HP player at the beginning
    for player in players:
        if player:
            if player.health <= alive[0].health and player != alive[0]:
                alive.insert(0, player)

    champions = generateChampions(round)
    items = generateHeldItems(round)

    # give all champions on the carousel an item
    for champ in champions:
        for i in items:
            if items[i]:
                champ.add_item(items[i])
                items[i] = None
                break
    
    for player in alive:
        for champ in champions:
            player.add_to_bench(champ)
            player.generate_bench_vector()
    pass

# this will handle champion generation based on the current round
def generateChampions(round):
    pass

# handles the item generation based on the current round
# also chooses what kind of item set to generate (e.g. offensive components only, defensive, utility, etc.)
def generateHeldItems(round):
    pass

def generateAllComponents():
    # generate a list of 1 of every component + 1 random duplicate component
    for i in range(0,9):
        player.add_to_item_bench(starting_items[random.randint(0, len(starting_items) - 1)])
        ran_cost_3 = list(pool_stats.COST_3.items())[random.randint(0, len(pool_stats.COST_3) - 1)][0]
        ran_cost_3 = champion.champion(ran_cost_3)
        pool_obj.update(ran_cost_3, -1)
        player.add_to_bench(ran_cost_3)

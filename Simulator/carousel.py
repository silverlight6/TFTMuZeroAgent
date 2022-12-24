import config
from Simulator.item_stats import item_builds as full_items, basic_items, starting_items, offensive_items, defensive_items
from Simulator import champion, pool, pool_stats
import random

def carousel(players, round, pool_obj):
    # probability of certain arrangements during certain carousels
    # https://leagueoflegends.fandom.com/wiki/Carousel_(Teamfight_Tactics)
    alive = []
    alive.append(players[0])
    # sort the list of alive players with the lowest HP player at the beginning
    for player in players:
        if player:
            if player.health <= alive[0].health and player != alive[0]:
                alive.insert(0, player)

    champions = generateChampions(round, pool_obj)
    items = generateHeldItems(round, pool_obj)

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
def generateChampions(round, pool_obj):
    pass

# handles the item generation based on the current round
# also chooses what kind of item set to generate (e.g. offensive components only, defensive, utility, etc.)
def generateHeldItems(round, pool_obj):
    pass

# random helper methods for generation of item sets for carousel below

def generateAllComponents():
    # generate a list of 1 of every component + 1 random duplicate component
    items = []
    for i in range(8):
        items.append(starting_items[i])
    # only 8 components but 9 champs on the carousel, so choose a random component
    items.append(starting_items[random.randint(0, len(starting_items) - 1)])
    return items

def generateOffenseComponents():
    # generate an item list of only offense components (BF, Rod, Bow)
    items = []
    for i in range(9):
        items.append(offensive_items[i % len(offensive_items)])
    return items

def generateDefenseComponents():
    # generate an item list of only defense components (Chain Vest, Belt, Cloak)
    items = []
    for i in range(9):
        items.append(defensive_items[i % len(offensive_items)])
    return items

def generateUtilComponents():
    # generate an item list of utility components and random components (sparring glove, tear, 7 random components)
    # NEED TO DOUBLE CHECK IF THIS IS ACTUALLY HOW UTILITY CAROUSEL IS GENERATED
    items = ['sparring_gloves', 'tear_of_the_goddess']
    for _ in range(7):
        items.append(starting_items[random.randint(0, len(starting_items) - 1)])
    return items

def generateAllSpats():
    # generate an item list of spatulas
    return ['spatula' for _ in range(9)]

def generateFONs():
    # generate an item list of FoN items
    return ['force_of_nature' for _ in range(9)]

def generateAllComponentsSpat():
    items = []
    for i in range(9):
        items.append(basic_items[i])
    return items

def generateThreeSpatsRandComponents():
    items = ['spatula', 'spatula', 'spatula']
    for _ in range(6):
        items.append(starting_items[random.randint(0, len(starting_items) - 1)])
    return items

def generateAllRandomComponents():
    items = []
    for _ in range(9):
        items.append(starting_items[random.randint(0, len(starting_items) - 1)])
    return items

def generateFullItems():
    pass

def generateBFItems():
    pass

def generateVestItems():
    pass

def generateBeltItems():
    pass

def generateBowItems():
    pass

def generateCloakItems():
    pass

def generateTearItems():
    pass

def generateGlovesItems():
    pass

def generateRodItems():
    pass

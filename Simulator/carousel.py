import config
from Simulator.item_stats import item_builds as item_builds, basic_items, starting_items, offensive_items, defensive_items
from Simulator.champion import champion
from Simulator.pool_stats import COST_1, COST_2, COST_3, COST_4, COST_5
import random

# TODO:
# Choose the best champion + item combo for each player

def carousel(players, r, pool_obj):
    # probability of certain arrangements during certain carousels
    # https://leagueoflegends.fandom.com/wiki/Carousel_(Teamfight_Tactics)
    alive = []
    # sort the list of alive players with the lowest HP player at the beginning
    for player in players:
        if player:
            if len(alive) == 0:
                alive.append(player)
            elif player.health <= alive[0].health and player != alive[0]:
                alive.insert(0, player)

    champions = generateChampions(r, pool_obj)
    items = generateHeldItems(r)

    # give all champions on the carousel an item
    for i, champ in enumerate(champions):
        champ.add_item(items[i])
    # player will choose the highest cost available regardless of item
    # needs to be changed to choose the "best" choice for each player

    # alive is already ordered from lowest hp to highest
    for player in alive:
        current = champions[0]
        for champ in champions:
            if champ.cost > current.cost:
                current = champ
        player.add_to_bench(current)
        player.generate_bench_vector()
        champions.remove(current)
    # pool updating should be handled upon a player choosing a champion
    # much easier this way
        pool_obj.update_pool(current, -1)

# this will handle champion generation based on the current round
def generateChampions(r , pool_obj):
    oneCosts = list(COST_1.items())
    twoCosts = list(COST_2.items())
    threeCosts = list(COST_3.items())
    fourCosts = list(COST_4.items())
    fiveCosts = list(COST_5.items())
    # remove champions from the list that have been exhausted from the pool (checking COST_1, COST_2, etc)
    for champ, count in oneCosts:
        if count <= 0:
            oneCosts.pop(champ)
    for champ, count in twoCosts:
        if count <= 0:
            twoCosts.pop(champ)
    for champ, count in threeCosts:
        if count <= 0:
            threeCosts.pop(champ)
    for champ, count in fourCosts:
        if count <= 0:
            fourCosts.pop(champ)
    for champ, count in fiveCosts:
        if count <= 0:
            fiveCosts.pop(champ)
    carouselChamps = []
    # first carousel - all 1 costs
    if r == 0:
        for _ in range(9):
            carouselChamps.append(champion(oneCosts.pop(random.randint(0, len(oneCosts) - 1))[0]))
    # second carousel - 1 one cost, 4 two costs, 4 three costs
    elif r == 6:
        carouselChamps.append(champion(oneCosts.pop(random.randint(0, len(oneCosts) - 1))[0]))
        for _ in range(4):
            carouselChamps.append(champion(twoCosts.pop(random.randint(0, len(twoCosts) - 1))[0]))
        for _ in range(4):
            carouselChamps.append(champion(threeCosts.pop(random.randint(0, len(threeCosts) - 1))[0]))
    # third carousel - 1 one cost, 2 two costs, 3 three costs, 3 four costs
    elif r == 12:
        carouselChamps.append(champion(oneCosts.pop(random.randint(0, len(oneCosts) - 1))[0]))
        for _ in range(2):
            carouselChamps.append(champion(twoCosts.pop(random.randint(0, len(twoCosts) - 1))[0]))
        for _ in range(3):
            carouselChamps.append(champion(threeCosts.pop(random.randint(0, len(threeCosts) - 1))[0]))
        for _ in range(3):
            carouselChamps.append(champion(fourCosts.pop(random.randint(0, len(fourCosts) - 1))[0]))
    # fourth carousel and beyond - 1 one cost, 2 two costs, 2 three costs, 2 four costs, 2 five costs
    elif r >= 18:
        carouselChamps.append(champion(oneCosts.pop(random.randint(0, len(oneCosts) - 1))[0]))
        for _ in range(2):
            carouselChamps.append(champion(twoCosts.pop(random.randint(0, len(twoCosts) - 1))[0]))
        for _ in range(2):
            carouselChamps.append(champion(threeCosts.pop(random.randint(0, len(threeCosts) - 1))[0]))
        for _ in range(2):
            carouselChamps.append(champion(fourCosts.pop(random.randint(0, len(fourCosts) - 1))[0]))
        for _ in range(2):
            carouselChamps.append(champion(fiveCosts.pop(random.randint(0, len(fiveCosts) - 1))[0]))
    return carouselChamps

# handles the item generation based on the current round
# also chooses what kind of item set to generate (e.g. offensive components only, defensive, utility, etc.)
def generateHeldItems(r):
    roll = random.random()
    if r == 0:
        if roll < 0.65:
            return generateAllComponents()
        elif roll < 0.76:
            return generateOffenseComponents()
        elif roll < 0.87:
            return generateDefenseComponents()
        elif roll < 0.98:
            return generateUtilComponents()
        elif roll < 0.995:
            return generateAllSpats()
        else:
            return generateFONs()
    elif r == 6:
        if roll < 0.80:
            return generateAllComponents()
        elif roll < 0.95:
            return generateAllComponentsSpat()
        else:
            return generateThreeSpatsRandComponents()
    elif r == 12:
        if roll < 0.50:
            return generateAllRandomComponents()
        elif roll < 0.80:
            return generateAllComponents()
        elif roll < 0.95:
            return generateAllComponentsSpat()
        else:
            return generateThreeSpatsRandComponents()
    elif r == 18:
        if roll < 0.80:
            return generateAllComponents()
        elif roll < 0.95:
            return generateAllComponentsSpat()
        else:
            return generateThreeSpatsRandComponents()
    elif r == 24:
        if roll < 0.50:
            return generateComponentItems(starting_items[random.randint(0, len(starting_items) - 1)])
        elif roll < 0.754:
            return generateFullItems()
        # 3% chance for a full set of items made with each component = 24% chance for all components
        elif roll < 0.994:
            # since they are equally weighted, can just do a random selection of non-spat components
            return generateComponentItems(starting_items[random.randint(0, len(starting_items) - 1)])
        else:
            return generateFONs()
    elif r >= 30:
        return generateHalfItems()

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
    # generate an item list of all components, including a spatula
    items = []
    for i in range(9):
        items.append(basic_items[i])
    return items

def generateThreeSpatsRandComponents():
    # generate an item list of 3 spatulas and 6 random components
    items = ['spatula', 'spatula', 'spatula']
    for _ in range(6):
        items.append(starting_items[random.randint(0, len(starting_items) - 1)])
    return items

def generateAllRandomComponents():
    # generate an item list of completely random components
    items = []
    for _ in range(9):
        items.append(starting_items[random.randint(0, len(starting_items) - 1)])
    return items

def generateFullItems():
    # generate an item list of completely random items
    items = []
    # list of all possible completed items
    fullitems = list(item_builds.keys())
    for _ in range(9):
        # randomly choose an item from the full items list and simultaneously remove it (prevents duplicates)
        items.append(fullitems.pop(random.randint(0, len(fullitems) - 1)))
    return items

def generateComponentItems(component):
    # generate an item list of only items with the specified component
    # Since only 9 unique components, there should be exactly as many unique items as champions
    items = []
    for item in item_builds:
        if component in item_builds[item]:
            items.append(item)
    return items

def generateHalfItems():
    # generate an item list of half full items, half random components
    items = []
    fullitems = list(item_builds.keys())
    for _ in range(5):
        items.append(fullitems.pop(random.randint(0, len(fullitems) - 1)))
    
    for _ in range(4):
        items.append(basic_items[random.randint(0, len(basic_items) - 1)])
        
    return items

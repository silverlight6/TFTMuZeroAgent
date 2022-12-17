from Simulator.player import player
from Simulator.pool import pool
from Simulator.champion import champion
from Simulator import minion

def setup() -> player:
    """Creates fresh player and pool"""
    base_pool = pool()
    player1 = player(base_pool, 0)
    return player1
# Check health calculation from minion combat
def combatTest():
    p1 = setup()
    p1.gold = 10000
    p1.max_units = 100
    # making 3* Zilean for combat
    minion.minion_combat(p1, minion.FirstMinion(), 0)

    assert p1.health < 100, "I didn't lose any health from losing a PVE round!"

# test round 0 rewards
def rewardsTest():
    p1 = setup()
    p1.gold = 10000
    p1.max_units = 100
    # making 3* Zilean for combat
    for x in range(9):
        p1.buy_champion(champion("zilean"))

    minion.minion_round(p1, 0, p1.pool_obj)
    assert not emptyBench(p1.item_bench), "I didn't get any items from the PVE round!"

    # need to add the rest of the rounds to test

def round1Test():
    pass

def emptyBench(item_bench):
    for i in item_bench:
        if i is not None:
            return False
    return True

def list_of_tests():
    combatTest()
    rewardsTest()
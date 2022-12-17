from Simulator.player import player
from Simulator.pool import pool
from Simulator.champion import champion
from Simulator import minion

def setup() -> player:
    """Creates fresh player and pool"""
    base_pool = pool()
    player1 = player(base_pool, 0)
    return player1

def combatTest():
    p1 = setup()
    p1.gold = 10000
    p1.max_units = 100
    # making 3* Zilean for combat
    minion.minion_round(p1, 0, p1.pool_obj)
    minion.minion_round(p1, 1, p1.pool_obj)
    minion.minion_round(p1, 2, p1.pool_obj)
    minion.minion_round(p1, 8, p1.pool_obj)
    minion.minion_round(p1, 14, p1.pool_obj)
    minion.minion_round(p1, 20, p1.pool_obj)
    minion.minion_round(p1, 26, p1.pool_obj)
    minion.minion_round(p1, 33, p1.pool_obj)

    assert p1.health == 100, "I lost a PVE round despite having a 3* Zilean!"
def round0Test():
    """Testing minion method usage at round 0"""
    p1 = setup()
    p1.gold = 10000
    p1.max_units = 100
    # making 3* Zilean for combat
    for x in range(9):
        p1.buy_champion(champion("zilean"))

    minion.minion_round(p1, 0, p1.pool_obj)
    assert not emptyBench(p1.item_bench), "I didn't get any items from the PVE round!"

def round1Test():
    pass

def emptyBench(item_bench):
    for i in item_bench:
        if i is not None:
            return False
    return True

def list_of_tests():
    round0Test()
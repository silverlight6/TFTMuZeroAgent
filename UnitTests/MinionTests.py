from Simulator.player import player
from Simulator.pool import pool
from Simulator.champion import champion
from Simulator import minion

def setup() -> player:
    """Creates fresh player and pool"""
    base_pool = pool()
    player1 = player(base_pool, 0)
    return player1

def round0Test():
    """Testing minion method usage at round 0"""
    p1 = setup()
    p1.gold = 10000
    p1.max_units = 100
    # making 3* Zilean for combat
    for x in range(9):
        p1.buy_champion(champion("zilean"))
    
    minion.minion_round(p1, 0, p1.pool_obj)
    assert len(p1.item_bench) != 0, "I didn't get any items from the PVE round!"

def list_of_tests():
    round0Test()
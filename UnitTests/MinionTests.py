from Simulator.player import player
from Simulator.pool import pool
from Simulator.champion import champion
from Simulator import minion

# contains the list of round numbers where unique PVE rounds occur
rounds = [0,1,2,8,14,20,26,33]

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
    minion.minion_combat(p1, minion.FirstMinion(), 0, [None for _ in range(2)])

    assert p1.health < 100, "I didn't lose any health from losing a PVE round!"

# test if each round is dropping rewards for the player
def rewardsTest():
    p1 = setup()
    # add 3* zilean and yone to board for combat
    p1.board[0][0] = champion("zilean", None, 0, 0, 3, None, None, None, False)
    p1.board[0][1] = champion("yone", None, 0, 0, 3, None, None, None, False)

    # PVE rounds can drop champions, gold, or items, but not nothing
    for r in rounds:
        p1.gold = 0
        p1.bench = [None for _ in range(9)]
        p1.item_bench = [None for _ in range(10)]
        
        minion.minion_round(p1, r, [None for _ in range(2)])

        assert not emptyList(p1.item_bench) \
            or not emptyList(p1.bench) \
            or not p1.gold == 0, "I didn't get anything from the PVE round!"

def emptyList(listArr):
    for i in listArr:
        if i is not None:
            return False
    return True

def list_of_tests():
    combatTest()
    rewardsTest()
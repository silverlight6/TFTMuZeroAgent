from Simulator.player import player
from Simulator.pool import pool
from Simulator.champion import champion

#Creates Zilean 11 times, there are 3 1* Zileans and a 3* Zilean, correlated to update_triple_catalogue ...
def levelingChampions(p1: player):
    #create a champ
    p1.gold = 100000
    p1.max_units = 1000
    print(p1.bench)
    for x in range(11):
        loopChampion = champion("zilean")
        assert p1.buy_champion(loopChampion)
    print(p1.bench)
    for x in range(4):
        print(p1.bench[x].stars)

def setup():
    base_pool = pool()
    player1 = player(base_pool, 0)
    return player1


def list_of_tests():
    p1 = setup()
    levelingChampions(p1)

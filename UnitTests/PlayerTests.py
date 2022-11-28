from Simulator.player import player
from Simulator.pool import pool
from Simulator.champion import champion


def add_to_bench_empty_bench(p1):
    testChampion = champion("evelynn")
    assert p1.add_to_bench(testChampion)


def setup():
    base_pool = pool()
    player1 = player(base_pool, 0)
    return player1


def list_of_tests():
    p1 = setup()
    add_to_bench_empty_bench(p1)

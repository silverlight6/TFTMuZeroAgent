from Simulator.player import player
from Simulator.pool import pool
from Simulator.champion import champion


def setup(player_num=0) -> player:
    """Creates fresh player and pool"""
    base_pool = pool()
    player1 = player(base_pool, player_num)
    return player1

def thiefsGlovesTest():
    p1 = setup()
    p1.gold = 1000
    p1.max_units = 1
    p1.buy_champion(champion('nami'))
    p1.add_to_item_bench('thiefs_gloves')
    p1.move_bench_to_board(0, 0, 0)
    p1.move_item_to_board(0, 0, 0)
    assert p1.board[0][0].items[0] == 'thiefs_gloves'
    for x in range(3):
        p1.start_round(x)
    p1.move_board_to_board(0, 0, 6, 3)
    p1.start_round(3)
    p1.move_board_to_bench(6, 3)
    p1.start_round(4)
    p1.sell_from_bench(0)
    p1.buy_champion(champion('nami'))
    p1.move_item_to_bench(0, 0)
    p1.start_round(5)

def kaynTests():
    p1 = setup()
    p2 = setup(1)
    p1.gold = 500
    p2.gold = 500
    p1.max_units = 10
    p2.max_units = 10
    p1.buy_champion(champion('kayn'))
    p1.move_bench_to_board(0, 0, 0)
    for x in range(3):
        p1.start_round(x)
        p2.start_round(x)
        p2.buy_champion(champion('kayn'))
        p2.move_bench_to_board(0, x, 0)
    assert p1.kayn_transformed,  'Kayn should transform after his third round in combat'
    assert not p2.kayn_transformed
    assert p1.item_bench[0] == 'kayn_shadowassassin'
    assert p1.item_bench[1] == 'kayn_rhast'
    p2.start_round(3)
    assert p2.kayn_transformed
    p1.move_item_to_board(0, 0, 0)
    assert p2.item_bench[0] == 'kayn_shadowassassin'
    assert p2.item_bench[1] == 'kayn_rhast'
    for x in range(7):
        for y in range(4):
            if p2.board[x][y]:
                p2.move_item_to_board(1, x, y)
                break
    assert p1.kayn_form == 'kayn_shadowassassin'
    assert p2.kayn_form == 'kayn_rhast'
    p1.buy_champion(champion('kayn'))
    assert p1.bench[0].kayn_form == 'kayn_shadowassassin'
    for x in range(10):
        assert not p1.item_bench[x]

def level2Champion():
    """Creates 3 Zileans, there should be 1 2* Zilean on bench"""
    p1 = setup()
    p1.gold = 100000
    p1.max_units = 10
    for x in range(3):
        p1.buy_champion(champion("zilean"))
    assert p1.bench[0].stars == 2, "champion should be 2*"
    for x in range(1, 9):
        assert p1.bench[x] is None, "these slot should be empty"
    for x in p1.board:
        for y in x:
            assert y is None, "the board should be empty"


def level3Champion():
    """Creates 9 Zileans, there should be 1 3* Zilean on bench"""
    p1 = setup()
    p1.gold = 100000
    p1.max_units = 1000
    for x in range(3):
        p1.buy_champion(champion("zilean"))
    assert p1.bench[0].stars == 2
    for x in range(3):
        p1.buy_champion(champion("zilean"))
    assert p1.bench[1].stars == 2
    for x in range(3):
        p1.buy_champion(champion("zilean"))
    assert p1.bench[0].stars == 3, "champion should be 3*"
    for x in range(1, 9):
        assert p1.bench[x] is None, "this slot should be empty"
    for x in p1.board:
        for y in x:
            assert y is None, "the board should be empty"


def levelChampFromField():
    """buy third copy while 1 copy on field"""
    p1 = setup()
    p1.gold = 100000
    p1.max_units = 1000
    p1.buy_champion(champion("zilean"))
    p1.buy_champion(champion("zilean"))
    p1.move_bench_to_board(1, 0, 0)
    p1.buy_champion(champion("zilean"))
    for x in p1.bench:
        assert x is None, "bench should be empty"
    assert p1.board[0][0].stars == 2, "the unit placed on the field should be 2*"


# Please expand on this test or add additional tests here.
# I am sure there are some bugs with the level cutoffs for example
# Like I do not think I am hitting level 3 on the correct round without buying any exp
def buyExp():
    p1 = setup()
    p1.level_up()
    lvl = p1.level
    while p1.level < p1.max_level:
        p1.exp = p1.level_costs[p1.level + 1]
        p1.level_up()
        lvl += 1
        assert lvl == p1.level


def spamExp():
    """buys tons of experience"""
    p1 = setup()
    p1.gold = 100000
    for _ in range(1000):
        p1.buy_exp()
    assert p1.level == p1.max_level, "I should be max level"
    assert p1.exp == 0, "I should not have been able to buy experience after hitting max lvl"


def incomeTest1():
    """first test for gold income"""
    p1 = setup()
    p1.gold = 15
    p1.gold_income(5)
    assert p1.gold == 21, f"Interest calculation is messy, gold should be 21, it is {p1.gold}"


def incomeTest2():
    """Check for income cap"""
    p1 = setup()
    p1.gold = 1000
    p1.gold_income(5)
    assert p1.gold == 1010, f"Interest calculation is messy, gold should be 1010, it is {p1.gold}"


def incomeTest3():
    """Checks win streak gold"""
    p1 = setup()
    p1.gold = 0
    p1.win_streak = 0
    p1.gold_income(5)
    assert p1.gold == 5, f"Interest calculation is messy, gold should be 5, it is {p1.gold}"
    p1.gold = 0
    p1.win_streak = 1
    p1.gold_income(5)
    assert p1.gold == 5, f"Interest calculation is messy, gold should be 5, it is {p1.gold}"
    p1.gold = 0
    p1.win_streak = 2
    p1.gold_income(5)
    assert p1.gold == 6, f"Interest calculation is messy, gold should be 6, it is {p1.gold}"
    p1.gold = 0
    p1.win_streak = 3
    p1.gold_income(5)
    assert p1.gold == 6, f"Interest calculation is messy, gold should be 6, it is {p1.gold}"
    p1.gold = 0
    p1.win_streak = 4
    p1.gold_income(5)
    assert p1.gold == 7, f"Interest calculation is messy, gold should be 7, it is {p1.gold}"
    p1.gold = 0
    p1.win_streak = 5
    p1.gold_income(5)
    assert p1.gold == 8, f"Interest calculation is messy, gold should be 8, it is {p1.gold}"
    p1.gold = 0
    p1.win_streak = 500
    p1.gold_income(5)
    assert p1.gold == 8, f"Interest calculation is messy, gold should be 8, it is {p1.gold}"


def incomeTest4():
    """Checks loss streak gold"""
    p1 = setup()
    p1.gold = 0
    p1.loss_streak = 0
    p1.gold_income(5)
    assert p1.gold == 5, f"Interest calculation is messy, gold should be 5, it is {p1.gold}"
    p1.gold = 0
    p1.loss_streak = 1
    p1.gold_income(5)
    assert p1.gold == 5, f"Interest calculation is messy, gold should be 5, it is {p1.gold}"
    p1.gold = 0
    p1.loss_streak = 2
    p1.gold_income(5)
    assert p1.gold == 6, f"Interest calculation is messy, gold should be 6, it is {p1.gold}"
    p1.gold = 0
    p1.loss_streak = 3
    p1.gold_income(5)
    assert p1.gold == 6, f"Interest calculation is messy, gold should be 6, it is {p1.gold}"
    p1.gold = 0
    p1.loss_streak = 4
    p1.gold_income(5)
    assert p1.gold == 7, f"Interest calculation is messy, gold should be 7, it is {p1.gold}"
    p1.gold = 0
    p1.loss_streak = 5
    p1.gold_income(5)
    assert p1.gold == 8, f"Interest calculation is messy, gold should be 8, it is {p1.gold}"
    p1.gold = 0
    p1.loss_streak = 500
    p1.gold_income(5)
    assert p1.gold == 8, f"Interest calculation is messy, gold should be 8, it is {p1.gold}"


def list_of_tests():
    """tests all test cases"""
    thiefsGlovesTest()

    kaynTests()

    level2Champion()
    level3Champion()
    levelChampFromField()

    buyExp()
    spamExp()

    # Problem: Interest gets calculated after base income is added
    incomeTest1()
    # Problem: Interest rate not capped
    incomeTest2()
    incomeTest3()
    incomeTest4()

    # I would like to go over move commands again before writing test code for that
    pass

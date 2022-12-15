import config
from Simulator.item_stats import item_builds as full_items, basic_items, starting_items
from Simulator import champion, pool, pool_stats
import random

item_list = list(full_items.keys())

# TODO
# add randomness to drops
# add representations of minions (Melees, Ranges, Krugs, Wolves, Raptors, Nexus, and Herald)
# add combat between player and minions
# add check for champion existing in the pool before being removed

# Objects representing board configs for each minion round
# These should function similar to player objects except simplified for minion combat
class Minion:
    def __init__(self):
        # array of champions, since order does not matter, can be unordered list
        self.bench = [None for _ in range(9)]
        # Champion array, this is a 7 by 4 array.
        self.board = [[None for _ in range(4)] for _ in range(7)]
        # List of items, there is no object for this so this is a string array
class FirstMinion(Minion):
    def __init__(self):
        super().__init__(self)
        self.board[1][1] = champion('meleeminion', 1, 1, 1, None, None, None, False)
        self.board[5][1] = champion('meleeminion', 5, 1, 1, None, None, None, False)
class SecondMinion(Minion):
    def __init__(self):
        super().__init__(self)
class ThirdMinion(Minion):
    def __init__(self):
        super().__init__(self)
class Krug(Minion):
    def __init__(self):
        super().__init__(self)
class Wolf(Minion):
    def __init__(self):
        super().__init__(self)
class Raptor(Minion):
    def __init__(self):
        super().__init__(self)
class Nexus(Minion):
    def __init__(self):
        super().__init__(self)
class Herald(Minion):
    def __init__(self):
        super().__init__(self)

def minion_round(player, round, pool_obj):
 # simulate minion round here
    # 2 melee minions - give 1 item component
    if round == 0:
        player.add_to_item_bench(starting_items[random.randint(0, len(starting_items) - 1)])

    # 2 melee and 1 ranged minion - give 1 item component and 1 3 cost champion
    elif round == 1:
        player.add_to_item_bench(starting_items[random.randint(0, len(starting_items) - 1)])
        ran_cost_3 = list(pool_stats.COST_3.items())[random.randint(0, len(pool_stats.COST_3) - 1)][0]
        ran_cost_3 = champion.champion(ran_cost_3)
        pool_obj.update(ran_cost_3, -1)
        player.add_to_bench(ran_cost_3)

    # 2 melee minions and 2 ranged minions - give 3 gold and 1 item component
    elif round == 2:
        player.add_to_item_bench(starting_items[random.randint(0, len(starting_items) - 1)])
        player.gold += 3

    # 3 Krugs - give 3 gold and 3 item components
    elif round == 8:
        player.gold += 3
        for _ in range(0, 3):
            player.add_to_item_bench(starting_items[random.randint(0, len(starting_items) - 1)])

    # 1 Greater Murk Wolf and 4 Murk Wolves - give 3 gold and 3 item components
    elif round == 14:
        player.gold += 3
        for _ in range(0, 3):
            player.add_to_item_bench(starting_items[random.randint(0, len(starting_items) - 1)])

    # 1 Crimson Raptor and 4 Raptors - give 6 gold and 4 item components
    elif round == 20:
        player.gold += 6
        for _ in range(0, 4):
            player.add_to_item_bench(starting_items[random.randint(0, len(starting_items) - 1)])

    # 1 Nexus Minion - give 6 gold and a full item
    elif round == 26:
        player.gold += 6
        player.add_to_item_bench(item_list[random.randint(0, len(item_list) - 1)])

    # Rift Herald - give 6 gold and a full item
    elif round >= 33:
        player.gold += 6
        player.add_to_item_bench(item_list[random.randint(0, len(item_list) - 1)])

    # invalid round! Do nothing
    else:
        return

# modeled after combat_phase from game_round.py, except with a minion "player" versus the player
def minion_combat(player, enemy, round):
    ROUND_DAMAGE = [
            [3, 0],
            [9, 2],
            [15, 3],
            [21, 5],
            [27, 8],
            [10000, 15]
        ]
    config.WARLORD_WINS['red'] = player.win_streak
    player.end_turn_actions()
    player.start_round()

    round_index = 0
    while round > ROUND_DAMAGE[round_index][0]:
            round_index += 1

    index_won, damage = champion.run(champion.champion, player, enemy, ROUND_DAMAGE[round_index][1])
    # tie!
    if index_won == 0:
        player.health -= damage
    # player wins!
    if index_won == 1:
        pass
    # minions win! (yikes)
    if index_won == 2:
        player.health -= damage
    


import config
from Simulator import champion
from Simulator.loot_orb import LootOrb, gen_loot, gen_orb_reward, gen_orbs, give_loot


# TODO
# add better randomness to drops
# add check for champion existing in the pool before being removed
# add "abilities" for each class of minions (krugs gain 1200 hp when another krug dies, etc)
# this would probably be handled with an origin for each type of minion

# Objects representing board configs for each minion round
# These should function similar to player objects except simplified for minion combat
class Minion:
    def __init__(self):
        # Used in print statements
        self.player_num = -1
        # array of champions, since order does not matter, can be unordered list
        self.bench = [None for _ in range(9)]
        # Champion array, this is a 7 by 4 array.
        self.board = [[None for _ in range(4)] for _ in range(7)]
        self.opponent = None


# Round 1 Minions
# Gain 18 gold worth of orbs
# Either 3 blue or 2 blue and 2 gray orbs
# NOTE the excessive if statements for round 1 account for all possible scenario's I have found
class FirstMinion(Minion):
    def __init__(self):
        super().__init__()
        self.player_num = -1
        self.board[2][1] = champion.champion('meleeminion')
        self.board[5][1] = champion.champion('meleeminion')

    def drop_loot(self, history):
        choices = [LootOrb.UNCOMMON, LootOrb.COMMON, (LootOrb.COMMON, LootOrb.COMMON), (LootOrb.COMMON, LootOrb.UNCOMMON)]
        probabilities = [.45, .25, .15, .15]
        count = 1
        return gen_loot(choices, probabilities, count, history)


# Drops depend on the drops beforehand
class SecondMinion(Minion):
    def __init__(self):
        super().__init__()
        self.board[4][2] = champion.champion('meleeminion')
        self.board[1][2] = champion.champion('meleeminion')
        self.board[5][1] = champion.champion('rangedminion')

    def drop_loot(self, history):
        choices = []
        probabilities = []

        common = history.count(LootOrb.COMMON)
        uncommon = history.count(LootOrb.UNCOMMON)

        # TODO change this with a more elegant solution
        # Ensures that round 1 gives at least 2 blue orbs
        if common == 1 and uncommon == 1:
            choices = [LootOrb.COMMON, LootOrb.UNCOMMON]
            probabilities = [.5, .5]
        elif common == 2:
            choices = [LootOrb.UNCOMMON]
            probabilities = [1]
        elif common == 1:
            choices = [LootOrb.UNCOMMON, LootOrb.COMMON, (LootOrb.COMMON, LootOrb.UNCOMMON)]
            probabilities = [.45, .35, .2]
        else:
            # uncommon == 1
            choices = [LootOrb.UNCOMMON, LootOrb.COMMON, (LootOrb.COMMON, LootOrb.COMMON)]
            probabilities = [.45, .35, .2]

        count = 1
        return gen_loot(choices, probabilities, count, history)


# Drops depend on the drops beforehand
class ThirdMinion(Minion):
    def __init__(self):
        super().__init__()
        self.board[4][2] = champion.champion('meleeminion')
        self.board[1][2] = champion.champion('meleeminion')
        self.board[5][1] = champion.champion('rangedminion')
        self.board[1][1] = champion.champion('rangedminion')

    def drop_loot(self, history):
        orbs = []

        common = history.count(LootOrb.COMMON)
        uncommon = history.count(LootOrb.UNCOMMON)

        # TODO change this with a more elegant solution
        # Ensures that round 1 gives at least 2 blue orbs
        if uncommon == 1 and common == 1:
            orbs += [LootOrb.UNCOMMON, LootOrb.COMMON]
        elif uncommon == 2:
            # Can either be 2 common orbs or 1 uncommon orb
            orbs += gen_orbs([(LootOrb.COMMON, LootOrb.COMMON), LootOrb.UNCOMMON], p=[.5,.5], count=1)
        elif common == 2 and uncommon == 1:
            orbs += [LootOrb.UNCOMMON]
        else:
            # common == 2
            orbs += [LootOrb.COMMON, LootOrb.UNCOMMON]

        history += orbs

        loot = []
        for orb in orbs:
            loot.append(gen_orb_reward(orb))

        return loot


# Drops up to 3 orbs, on rare occasions krugs don't drop anything
class Krug(Minion):
    def __init__(self):
        super().__init__()
        self.board[1][3] = champion.champion('krug')
        self.board[6][3] = champion.champion('krug')
        self.board[5][1] = champion.champion('krug')

    def drop_loot(self, history):
        choices = [LootOrb.UNCOMMON, LootOrb.COMMON, LootOrb.RARE, None]
        probabilities = [.7, .2, .05, .05]
        count = 3
        return gen_loot(choices, probabilities, count, history)


# Drops up to 5 orbs, sometimes wolves don't drop anything
class Wolf(Minion):
    def __init__(self):
        super().__init__()
        self.board[1][0] = champion.champion('lesserwolf')
        self.board[2][0] = champion.champion('lesserwolf')
        self.board[4][0] = champion.champion('lesserwolf')
        self.board[5][0] = champion.champion('lesserwolf')
        self.board[3][2] = champion.champion('wolf')

    def drop_loot(self, history):
        choices = [LootOrb.UNCOMMON, LootOrb.COMMON, LootOrb.RARE, None]
        probabilities = [.6, .25, .05, .1]
        count = 5
        return gen_loot(choices, probabilities, count, history)


# Drops up to 5 orbs, sometimes raptors don't drop anything
class Raptor(Minion):
    def __init__(self):
        super().__init__()
        self.board[3][0] = champion.champion('crimsonraptor')
        self.board[2][1] = champion.champion('raptor')
        self.board[1][2] = champion.champion('raptor')
        self.board[5][1] = champion.champion('raptor')
        self.board[5][2] = champion.champion('raptor')

    def drop_loot(self, history):
        choices = [LootOrb.UNCOMMON, LootOrb.COMMON, LootOrb.RARE, None]
        probabilities = [.55, .3, .05, .1]
        count = 5
        return gen_loot(choices, probabilities, count, history)


# Drops 1 random full item and a random orb
class Nexus(Minion):
    def __init__(self):
        super().__init__()
        self.board[3][1] = champion.champion('nexusminion')

    def drop_loot(self, history):
        loot = ['full_item']

        choices = [LootOrb.UNCOMMON, LootOrb.COMMON, LootOrb.RARE]
        probabilities = [.3, .6, .1]
        count = 1

        loot += gen_loot(choices, probabilities, count, history)

        return loot


# Drops 1 random full item and 2 random orbs
class Herald(Minion):
    def __init__(self):
        super().__init__()
        self.board[3][1] = champion.champion('riftherald')
    
    def drop_loot(self, history):
        loot = ['full_item']

        choices = [LootOrb.UNCOMMON, LootOrb.COMMON, LootOrb.RARE]
        probabilities = [.3, .6, .1]
        count = 2

        loot += gen_loot(choices, probabilities, count, history)

        return loot


def minion_round(player, round, others):
    # simulate minion round here
    # 2 melee minions - give 1 item component
    if round == 0:
        minion_combat(player, FirstMinion(), round, others)

    # 2 melee and 1 ranged minion - give 1 item component and 1 3 cost champion
    elif round == 1:
        minion_combat(player, SecondMinion(), round, others)

    # 2 melee minions and 2 ranged minions - give 3 gold and 1 item component
    elif round == 2:
        minion_combat(player, ThirdMinion(), round, others)

    # 3 Krugs - give 3 gold and 3 item components
    elif round == 8:
        minion_combat(player, Krug(), round, others)

    # 1 Greater Murk Wolf and 4 Murk Wolves - give 3 gold and 3 item components
    elif round == 14:
        minion_combat(player, Wolf(), round, others)

    # 1 Crimson Raptor and 4 Raptors - give 6 gold and 4 item components
    elif round == 20:
        minion_combat(player, Raptor(), round, others)

    # 1 Nexus Minion - give 6 gold and a full item
    elif round == 26:
        minion_combat(player, Nexus(), round, others)

    # Rift Herald - give 6 gold and a full item
    elif round >= 33:
        minion_combat(player, Herald(), round, others)

    # invalid round! Do nothing
    else:
        return


# modeled after combat_phase from game_round.py, except with a minion "player" versus the player
def minion_combat(player, enemy, round, others):
    ROUND_DAMAGE = [
            [3, 0],
            [9, 2],
            [15, 3],
            [21, 5],
            [27, 8],
            [10000, 15]
        ]
    config.WARLORD_WINS['blue'] = player.win_streak
    player.end_turn_actions()

    round_index = 0
    while round > ROUND_DAMAGE[round_index][0]:
        round_index += 1

    player.opponent = enemy
    enemy.opponent = player

    index_won, damage = champion.run(champion.champion, player, enemy, ROUND_DAMAGE[round_index][1])
    # list of currently alive players at the conclusion of combat
    alive = []
    for o in others:
        if o:
            if o.health > 0 and o is not player:
                alive.append(o)
    # tie!
    if index_won == 0:
        player.loss_round(damage)
        for p in alive:
            p.ghost_won(damage / len(alive))
        player.health -= damage
    # player wins!
    if index_won == 1:
        loot = enemy.drop_loot(player.orb_history)
        for reward in loot:
            give_loot(player, reward)
    # minions win! (yikes)
    if index_won == 2:
        player.loss_round(damage)
        if len(alive) > 0:
            for p in alive:
                p.ghost_won(damage / len(alive))
        player.health -= damage

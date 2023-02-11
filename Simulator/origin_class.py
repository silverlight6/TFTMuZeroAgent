import Simulator.origin_class_stats as origin_class_stats
from Simulator import field, item_stats, items, champion_functions
import Simulator.stats as stats
import random
import config
import time

# ORIGINS AND CLASSES
# loads of similar functions

starting_time = time.time_ns()

cultist_stars = {'blue': 0, 'red': 0}  # chosen's stars counts as double
total_health_teams = {'blue': 0, 'red': 0}
galio_spawn_time = {'blue': 0, 'red': 0}

amounts = {
    'cultist': {'blue': 0, 'red': 0},           # 0  in champion.py: champion object, champion.champion_functions.py
    'divine': {'blue': 0, 'red': 0},            # 1  in champion.py: spell(), champion_functions.py: attack()
    'dusk': {'blue': 0, 'red': 0},              # 2  in origin_class.py: total_origin_class()
    'elderwood': {'blue': 0, 'red': 0},         # 3  in champion.py: main()
    'enlightened': {'blue': 0, 'red': 0},       # 4  in origin_class.py: total_origin_class()
    'exile': {'blue': 0, 'red': 0},             # 5  in origin_class.py: total_origin_class()
    'ninja': {'blue': 0, 'red': 0},             # 6  in origin_class.py: total_origin_class()
    'spirit': {'blue': 0, 'red': 0},            # 7  in ability.py: default_ability_calls()
    'the_boss': {'blue': 0, 'red': 0},          # 8  in champion.py: spell(), champion_functions.py: attack()
    'warlord': {'blue': 0, 'red': 0},           # 9  in origin_class.py: total_origin_class()
    'adept': {'blue': 0, 'red': 0},             # 10 in origin_class.py: total_origin_class()
    'assassin': {'blue': 0, 'red': 0},          # 11 in origin_class.py: total_origin_class()
    'brawler': {'blue': 0, 'red': 0},           # 12 in origin_class.py: total_origin_class()
    'dazzler': {'blue': 0, 'red': 0},           # 13 in champion.py: clear_que_dazzler(), spell()
    'duelist': {'blue': 0, 'red': 0},           # 14 in origin_class.py: total_origin_class(), champion.py: attack()
    'emperor': {'blue': 0, 'red': 0},           # 15 in origin_class.py: total_origin_class()
    'hunter': {'blue': 0, 'red': 0},            # 16 in champion.py: main()
    'keeper': {'blue': 0, 'red': 0},            # 17 in origin_class.py: total_origin_class()
    'mage': {'blue': 0, 'red': 0},              # 18 in origin_class.py: total_origin_class(), champion.py: ability()
    'mystic': {'blue': 0, 'red': 0},            # 19 in origin_class.py: total_origin_class()
    'shade': {'blue': 0, 'red': 0},             # 20 in origin_class.py: total_origin_class()
    'sharpshooter': {'blue': 0, 'red': 0},      # 21 in champion_functions.py: attack(), champion.py: spell()
    'vanguard': {'blue': 0, 'red': 0},          # 22 in origin_class.py: total_origin_class()
    'fortune': {'blue': 0, 'red': 0},           # 23 in player
    'moonlight': {'blue': 0, 'red': 0},         # 24 in origin_class.py: My own implementation
    'tormented': {'blue': 0, 'red': 0}
}

team_traits = {
    'cultist': 0,
    'divine': 0,
    'dusk': 0,
    'elderwood': 0,
    'enlightened': 0,
    'exile': 0,
    'ninja': 0,
    'spirit': 0,
    'the_boss': 0,
    'warlord': 0,
    'adept': 0,
    'assassin': 0,
    'brawler': 0,
    'dazzler': 0,
    'duelist': 0,
    'emperor': 0,
    'hunter': 0,
    'keeper': 0,
    'mage': 0,
    'mystic': 0,
    'shade': 0,
    'sharpshooter': 0,
    'vanguard': 0,
    'fortune': 0,
    'moonlight': 0,
    'tormented': 0
}

# Number of each class
game_compositions = [team_traits.copy() for _ in range(8)]

# Tier rank
game_comp_tiers = [team_traits.copy() for _ in range(8)]


def chosen(champion, value):
    if value:
        if champion.team:
            amounts[value][champion.team] += 1
            stat_change = list(filter(lambda x: x['champion'] == champion.name, origin_class_stats.chosen))[0]
            if stat_change['stat'] == 'maxmana':
                items.change_stat(champion, stat_change['stat'],
                                  getattr(champion, stat_change['stat']) * stat_change['value'], 'chosen')
            else:
                items.change_stat(champion, stat_change['stat'],
                                  getattr(champion, stat_change['stat']) + stat_change['value'], 'chosen')
        return value
    return False


def total_health(blue, red):
    global total_health_teams
    for b in blue:
        total_health_teams['blue'] += b.health
    for r in red:
        total_health_teams['red'] += r.health


def get_origin_class_tier(team, trait):
    try:
        amount = amounts[trait][team]
        amount_limits = origin_class_stats.tiers[trait]

        if trait != 'ninja':
            # Not useless because oftentimes, the first is 0 and the second is the tier number.
            if amount < amount_limits[0]:
                return 0
            if amount >= amount_limits[-1]:
                return len(amount_limits)

            for i, a in enumerate(amount_limits):
                if amount < a:
                    return i
        else:
            for i, a in enumerate(amount_limits):
                if amount == a:
                    return i + 1
            return 0
    except KeyError:
        return 0


# calculate the amount of traits per team and mark them to the 'amounts' -dict
def total_origin_class(blue_champion, red_champion):
    traits = list(amounts.keys())

    blue_team = blue_champion.own_team()
    red_team = red_champion.own_team()
    champion_data = origin_class_stats.origin_class

    counted = []

    for team in [blue_team, red_team]:  # team layer
        for c in team:                 # champions in teams
            for t in traits:           # traits
                if t in champion_data[c.name] and not [team, t, c.name] in counted:
                    amounts[t][c.team] += 1
                    counted.append([team, t, c.name])

    for t in traits:
        if t in origin_class_stats.initiate_traits:
            eval(t)(blue_team, red_team)  # origin_class_stats.py: initiate_traits

    calculate_cultist_stars(blue_team, red_team)


def team_origin_class(player):
    team = player.board
    for trait in game_compositions[player.player_num]:
        game_compositions[player.player_num][trait] = 0
    unique_champions = []
    for x in range(0, 7):
        for y in range(0, 4):
            if team[x][y]:
                if team[x][y].name not in unique_champions:
                    unique_champions.append(team[x][y].name)
                    for trait in team[x][y].origin:
                        game_compositions[player.player_num][trait] += 1
    return game_compositions[player.player_num]


def is_trait(champion, trait):
    champion_data = origin_class_stats.origin_class
    if trait in champion_data[champion.name]:
        return True
    elif trait in item_stats.trait_items and item_stats.trait_items[trait] in champion.items:
        return True

    return False


def calculate_cultist_stars(blue, red):
    teams = ['blue', 'red']
    for t in teams:
        for c in eval(t):
            if is_trait(c, 'cultist'):
                cultist_stars[t] += c.stars
                if c.chosen == 'cultist':
                    cultist_stars[t] += 1


def cultist(champion, team):
    galio_stars = get_origin_class_tier(team, 'cultist')
    galio_spawn_time[team] = champion_functions.MILLIS()
    # find the spawn point
    # which free hex has the lowest total distance to all enemy units
    enemies = champion.enemy_team()

    all_hexes = field.hexes_in_distance(0, 0, 20)
    current_spawn_hex = [[], 9999]
    coords = field.coordinates

    for h in all_hexes:
        total_distance = 0
        if not coords[h[0]][h[1]]:
            for e in enemies:
                d = field.distance({'y': e.y, 'x': e.x}, {'y': h[0], 'x': h[1]}, False)
                total_distance += d
            if total_distance < current_spawn_hex[1]:
                current_spawn_hex = [h, total_distance]

    current_spawn_hex = current_spawn_hex[0]

    champion.spawn('galio', galio_stars, current_spawn_hex[0], current_spawn_hex[1])
    galio = list(filter(lambda x: x.name == 'galio', champion.own_team()))[0]
    galio.print(' spawns')
    entry_range = stats.ABILITY_SECONDARY_RADIUS['galio'][galio.stars]

    # entry ability
    enemies_in_range = field.enemies_in_distance(galio, galio.y, galio.x, entry_range)
    for e in enemies_in_range:
        galio.spell(e, stats.ABILITY_SECONDARY_DMG['galio'][galio.stars])

        e.add_que('change_stat', -1, None, 'stunned', True)
        e.clear_que_stunned_removal()
        e.add_que('change_stat', stats.ABILITY_STUN_DURATION['galio'][galio.stars], None, 'stunned', False)


# galio crits (directed from the attack function)
def cultist_helper(champion, damage, target):

    # reverting the armor effect
    if target.armor >= 0:
        damage = damage / (100 / (100 + target.armor))
    else:
        damage = damage / (2 - 100 / (100 - target.armor))

    targets = field.enemies_in_distance(champion, target.y, target.x, 1)
    for t in targets:
        champion.spell(t, damage)


divine_attack_list = []  # [champion, attack_amount]
divine_list = []  # champion, champion, champion


def divine(champion, target, attack):

    # counting the x attacks for ascending
    if is_trait(champion, 'divine') and get_origin_class_tier(champion.team, 'divine') > 0 \
            and not champion in divine_list:
        if attack:
            divine_tier = get_origin_class_tier(champion.team, 'divine')

            if len(list(filter(lambda x: x[0] == champion, divine_attack_list))) == 0:
                divine_attack_list.append([champion, 1])
            else:
                for i, d in enumerate(divine_attack_list):
                    if d[0] == champion:
                        divine_attack_list[i][1] += 1

                        if divine_attack_list[i][1] == origin_class_stats.threshold['divine']:
                            divine_helper(champion, divine_tier)

    # following target hp (directed here from spells and attacks)
    if is_trait(target, 'divine') and get_origin_class_tier(target.team, 'divine') > 0 and not target in divine_list:
        divine_tier = get_origin_class_tier(target.team, 'divine')
        if(target.health / target.max_health < 0.5):
            divine_helper(target, divine_tier)


# ascend champion
def divine_helper(champion, divine_tier):
    divine_list.append(champion)
    champion.print(' ascends for {} seconds [divine]'.format(origin_class_stats.length['divine'][divine_tier] / 1000))

    items.change_stat(champion, 'stunned', False, 'divine')
    items.change_stat(champion, 'disarmed', False, 'divine')
    items.change_stat(champion, 'blinded', False, 'divine')

    new_damage_receiving = champion.receive_decreased_damage * origin_class_stats.receive_decreased_damage['divine']
    old_damage_receiving = champion.receive_decreased_damage # can be safely done since the only other functhion which uses this is galio's ult
    items.change_stat(champion, 'receive_decreased_damage', new_damage_receiving, 'divine')
    champion.add_que('change_stat', origin_class_stats.length['divine'][divine_tier], None, 'receive_decreased_damage', old_damage_receiving)

    items.change_stat(champion, 'deal_bonus_true_damage', origin_class_stats.deal_bonus_true_damage['divine'], 'divine')
    champion.add_que('change_stat', origin_class_stats.length['divine'][divine_tier], None, 'deal_bonus_true_damage', 0)


def dusk(blue_team, red_team):

    teams = {'blue': blue_team, 'red': red_team}

    for t in ['blue', 'red']:
        tier = get_origin_class_tier(t, 'dusk')
        if tier > 0:
            for c in teams[t]:
                items.change_stat(c, 'SP', c.SP + origin_class_stats.SP_secondary['dusk'][tier], 'dusk')
                if is_trait(c, 'dusk'):
                    items.change_stat(c, 'SP', c.SP + origin_class_stats.SP['dusk'][tier], 'dusk')


elderwood_list = {'blue': 0, 'red': 0}


def elderwood(blue_team, red_team):

    teams = {'blue': blue_team, 'red': red_team}
    for t in ['blue', 'red']:
        tier = get_origin_class_tier(t, 'elderwood')
        if tier > 0 and elderwood_list[t] < 5:
            for c in teams[t]:
                if is_trait(c, 'elderwood'):
                    items.change_stat(c, 'AD', c.AD + origin_class_stats.AD['elderwood'][tier], 'elderwood')
                    items.change_stat(c, 'SP', c.SP + origin_class_stats.SP['elderwood'][tier], 'elderwood')
                    items.change_stat(c, 'MR', c.MR + origin_class_stats.MR['elderwood'][tier], 'elderwood')
                    items.change_stat(c, 'armor', c.armor + origin_class_stats.armor['elderwood'][tier], 'elderwood')
            elderwood_list[t] += 1


def enlightened(blue_team, red_team):
    teams = {'blue': blue_team, 'red': red_team}

    for t in ['blue', 'red']:
        tier = get_origin_class_tier(t, 'enlightened')
        if(tier > 0):
            for c in teams[t]:
                if(is_trait(c, 'enlightened')):
                    items.change_stat(c, 'mana_generation', origin_class_stats.mana_generation['enlightened'][tier], 'enlightened')


def exile(blue_team, red_team):
    teams = {'blue': blue_team, 'red': red_team}

    coords = field.coordinates
    for t in ['blue', 'red']:
        tier = get_origin_class_tier(t, 'exile')
        if tier > 0:
            for c in teams[t]:
                if is_trait(c, 'exile'):

                    # the surrounding hexes must be empty or hold an enemy unit
                    applicable = True
                    neighbor_hexes = field.hexes_distance_away(c.y, c.x, 1)
                    for n in neighbor_hexes:
                        n_hex = coords[n[0]][n[1]]
                        if n_hex and n_hex.team == c.team:
                            applicable = False

                    if applicable:
                        shield_size = origin_class_stats.shield['exile'][tier] * c.max_health
                        shield_identifier = c.millis() * shield_size
                        c.add_que('shield', -1, None, None, {'amount': shield_size, 'identifier': shield_identifier,
                                                             'applier': c, 'original_amount': shield_size},
                                  {'increase': True})

                        lifesteal = origin_class_stats.lifesteal['exile'][tier]
                        if lifesteal:
                            items.change_stat(c, 'lifesteal', lifesteal, 'exile')


def ninja(blue_team, red_team):
    teams = {'blue': blue_team, 'red': red_team}

    for t in ['blue', 'red']:
        tier = get_origin_class_tier(t, 'ninja')
        if(tier > 0):
            for c in teams[t]:
                if(is_trait(c, 'ninja')):

                    items.change_stat(c, 'AD', c.AD + origin_class_stats.AD['ninja'][tier], 'ninja')
                    items.change_stat(c, 'SP', c.SP + origin_class_stats.SP['ninja'][tier], 'ninja')


spirit_list = [] # champion, champion, champion (the ones who have casted)
def spirit(champion):
    tier = get_origin_class_tier(champion.team, 'spirit')
    if(tier > 0 and champion not in spirit_list):
        multiplier = origin_class_stats.AS['spirit'][tier]
        own_team = champion.own_team()
        for o in own_team:
            items.change_stat(o, 'AS', o.AS * (champion.maxmana * multiplier / 100 + 1))
        spirit_list.append(champion)


def the_boss(champion):
    champion.done_situps = True

    own_team = champion.own_team()
    coords = field.coordinates
    if len(own_team) > 1:
        # set champion.champion = False
        # free the hex he's at
        # call helper function every x.y seconds
        # at the start of the helper function check if the rest of the team has died
        #    if has, return to the combat
        #    if not, do the sit up
        #    if health == 100%
        #        return and set pumped up = True
        items.change_stat(champion, 'champion', False)
        items.change_stat(champion, 'stunned', True)
        for c in champion.enemy_team():
            if c.target == champion:
                c.target = None
        coords[champion.y][champion.x] = None

        champion.add_que('execute_function', 0, [the_boss_helper])


# do sit-ups and return when needed
def the_boss_helper(champion):
    coords = field.coordinates

    if len(champion.own_team()) > 1:
        champion.print(' sit-up')

        # healing
        heal_amount = champion.max_health * origin_class_stats.heal['the_boss']
        if champion.health + heal_amount > champion.max_health:
            heal_amount = champion.max_health - champion.health
        items.change_stat(champion, 'health', champion.health + heal_amount)

        # pumped up -status
        if champion.health == champion.max_health:
            items.change_stat(champion, 'pumped_up', True)

        items.change_stat(champion, 'AS', champion.AS * origin_class_stats.AS['the_boss'])
        items.change_stat(champion, 'movement_delay',
                          champion.movement_delay * origin_class_stats.movement_delay['the_boss'])

    if len(champion.own_team()) == 1 or champion.pumped_up:
        items.change_stat(champion, 'champion', True)
        items.change_stat(champion, 'stunned', False)

        # find the closest free hex to the correct corner (blue team = (0,6) and red team = (7, 0))
        corner = {'blue': [0, 6], 'red': [7, 0]}
        hexes = field.hexes_in_distance(corner[champion.team][0], corner[champion.team][1], 3)
        for i, h in enumerate(hexes):
            d = field.distance({'y': corner[champion.team][0],
                                'x': corner[champion.team][1]}, {'y': h[0], 'x': h[1]}, False)
            hexes[i].append(d)

        hexes = list(filter(lambda x: (not coords[x[0]][x[1]]), hexes))
        hexes = sorted(hexes, key=lambda x: x[2])

        champion.move(hexes[0][0], hexes[0][1], True, True)
        # items.change_stat(champion, 'y', hexes[0][0])
        # items.change_stat(champion, 'x', hexes[0][1])

    else:
        champion.add_que('execute_function', origin_class_stats.length['the_boss'], [the_boss_helper, {}])


def warlord(blue_team, red_team):
    teams = {'blue': blue_team, 'red': red_team}

    for t in ['blue', 'red']:
        tier = get_origin_class_tier(t, 'warlord')
        if tier > 0:
            for c in teams[t]:
                if is_trait(c, 'warlord'):
                    wins = config.WARLORD_WINS[t]
                    if wins > 5:
                        wins = 5

                    hp_add = origin_class_stats.health['warlord'][tier]
                    hp_add = hp_add * (1 + wins * origin_class_stats.increasement['warlord'])

                    SP_add = origin_class_stats.SP['warlord'][tier] * (1 + wins *
                                                                       origin_class_stats.increasement['warlord'])

                    items.change_stat(c, 'health', c.health + hp_add, 'warlord')
                    items.change_stat(c, 'SP', c.SP + SP_add, 'warlord')


def adept(blue_team, red_team):
    teams = {'blue': blue_team, 'red': red_team}

    for t in ['blue', 'red']:
        tier = get_origin_class_tier(t, 'adept')
        if tier > 0:
            enemies = teams[t][0].enemy_team()
            for e in enemies:
                items.change_stat(e, 'AS', e.AS * origin_class_stats.AS['adept'], 'adept')
                e.add_que('change_stat', origin_class_stats.length['adept'][tier], None, 'AS', None,
                          {'ezreal': origin_class_stats.AS['adept']})


def assassin(blue_team, red_team):
    teams = {'blue': blue_team, 'red': red_team}

    for t in ['blue', 'red']:
        tier = get_origin_class_tier(t, 'assassin')
        for c in teams[t]:
            if is_trait(c, 'assassin'):

                if tier > 0:
                    items.change_stat(c, 'crit_chance',
                                      c.crit_chance + origin_class_stats.crit_chance['assassin'][tier], 'assassin')
                    items.change_stat(c, 'crit_damage',
                                      c.crit_damage + origin_class_stats.crit_damage['assassin'][tier], 'assassin')

                items.change_stat(c, 'champion', False, '  assassin')
                items.change_stat(c, 'stunned', True, '  assassin')
                field.coordinates[c.y][c.x] = None
                # allows the units to "swap" places instead of treating the hex as taken

                c.add_que('execute_function', config.LEAP_DELAY, [field.leap_to_back_line, {'trait': '  assassin'}])


def brawler(blue_team, red_team):
    teams = {'blue': blue_team, 'red': red_team}

    for t in ['blue', 'red']:
        tier = get_origin_class_tier(t, 'brawler')
        if tier > 0:
            for c in teams[t]:
                if is_trait(c, 'brawler'):

                    items.change_stat(c, 'max_health',
                                      c.max_health + origin_class_stats.health['brawler'][tier], 'brawler')
                    items.change_stat(c, 'health', c.health + origin_class_stats.health['brawler'][tier], 'brawler')


def dazzler(champion, target):
    tier = get_origin_class_tier(champion.team, 'dazzler')

    if tier > 0:
        # self.AD_reduction_cc = False #ludens counts dazzler ad reduction as crowd control so adding a flag for dat
        length = origin_class_stats.length['dazzler']
        AD_reduction = origin_class_stats.AD['dazzler'][tier]

        # if not targeted, target
        if not target.AD_reduction_cc:
            items.change_stat(target, 'AD', target.AD * AD_reduction, 'dazzler')
            items.change_stat(target, 'AD_reduction_cc', True, '  dazzler')
        # otherwise just clear the 8 second timer
        else:
            target.clear_que_dazzler()

        target.add_que('change_stat', length, None, 'AD', None, {'ashe': AD_reduction, 'dazzler': True})
        target.add_que('change_stat', length, None, 'AD_reduction_cc', False, {'dazzler': True})


# change the movement speed of the units
def duelist(blue_team, red_team):
    teams = {'blue': blue_team, 'red': red_team}

    for t in ['blue', 'red']:
        tier = get_origin_class_tier(t, 'duelist')
        if tier > 0:
            for c in teams[t]:
                if is_trait(c, 'duelist'):

                    items.change_stat(c, 'movement_delay',
                                      c.movement_delay * origin_class_stats.movement_delay['duelist'], 'duelist')


# AS changes
duelist_helper_list = []  # [champion, stacks]


def duelist_helper(champion):

    tier = get_origin_class_tier(champion.team, 'duelist')
    stacks = -1
    if tier > 0:

        if len(list(filter(lambda x: x[0] == champion, duelist_helper_list))) == 0:
            duelist_helper_list.append([champion, 1])
            stacks = 1
        else:
            for i, d in enumerate(duelist_helper_list):
                if d[0] == champion:
                    duelist_helper_list[i][1] += 1
                    stacks = duelist_helper_list[i][1]

        if stacks <= origin_class_stats.threshold['duelist']:
            end_AS = champion.AS * origin_class_stats.AS['duelist'][tier]
            if end_AS > 5.00:
                end_AS = 5
            items.change_stat(champion, 'AS', end_AS, 'duelist')


# set the emperor to be the overlord of the statue thingy
def emperor(blue_team, red_team):
    ...
    # teams = {'blue': blue_team, 'red': red_team}
    #
    # for t in ['blue', 'red']:
    #     for c in teams[t]:
    #         if c.name == 'sandguard':
    #             items.change_stat(c, 'stunned', True)
    #             daddy = field.coordinates[c.overlord_coordinates[0]][c.overlord_coordinates[1]]
    #             c.overlord = daddy
    #             daddy.underlords.append(c)


def fortune():
    return


# is called from main() every x milliseconds
def hunter(team):
    if team:
        ally_unit = team[0]
        tier = get_origin_class_tier(ally_unit.team, 'hunter')
        if tier > 0:
            for c in team:
                enemy_team = ally_unit.enemy_team()
                enemy_team = sorted(enemy_team, key=lambda x: x.health)

                if len(enemy_team) > 0 and not c.stunned and not c.blinded and not c.disarmed:
                    target = enemy_team[0]

                    bonus_damage = (origin_class_stats.AD['hunter'][tier] - 1) * c.AD
                    c.attack(bonus_damage, target, False, 'hunter')


def keeper(blue_team, red_team):
    teams = {'blue': blue_team, 'red': red_team}
    coords = field.coordinates

    for t in ['blue', 'red']:
        tier = get_origin_class_tier(t, 'keeper')
        if tier > 0:
            for c in teams[t]:
                if is_trait(c, 'keeper'):

                    length = origin_class_stats.length['keeper'][tier]
                    hexes = field.hexes_in_distance(c.y, c.x, 1)

                    for h in hexes:
                        u = coords[h[0]][h[1]]
                        if u and u.team == c.team and u.champion:
                            shield = origin_class_stats.shield['keeper'][tier]
                            if is_trait(u, 'keeper'):
                                shield *= (1 + origin_class_stats.increasement['keeper'])

                            if c.chosen:
                                shield *= 2

                            shield_identifier = round(c.max_health * shield * u.AD * (u.AS * 5))

                            u.add_que('shield', -1, None, None, {'amount': shield, 'identifier': shield_identifier,
                                                                 'applier': c, 'original_amount': shield},
                                      {'increase': True, 'expires': length})


def mage(blue_team, red_team):
    teams = {'blue': blue_team, 'red': red_team}

    for t in ['blue', 'red']:
        tier = get_origin_class_tier(t, 'mage')
        if tier > 0:
            for c in teams[t]:
                if is_trait(c, 'mage'):
                    items.change_stat(c, 'SP', c.SP + (origin_class_stats.SP['mage'][tier] - 1), 'mage')


def moonlight(blue_team, red_team):
    teams = {'blue': blue_team, 'red': red_team}

    for t in ['blue', 'red']:
        tier = get_origin_class_tier(t, 'moonlight')
        if tier > 0:
            c_level = []
            for i, c in enumerate(teams[t]):
                if is_trait(c, 'moonlight'):
                    c_level.append([c.stars, len(c.items), i])
            c_level.sort()
            for i in range(0, tier):
                teams[t][c_level[i][2]].golden()


def mystic(blue_team, red_team):
    teams = {'blue': blue_team, 'red': red_team}

    for t in ['blue', 'red']:
        tier = get_origin_class_tier(t, 'mystic')
        if tier > 0:
            for c in teams[t]:
                if is_trait(c, 'mystic'):
                    items.change_stat(c, 'MR', c.MR + origin_class_stats.MR['mystic'][tier], 'mystic')


def shade(blue_team, red_team):
    teams = {'blue': blue_team, 'red': red_team}

    for t in ['blue', 'red']:
        tier = get_origin_class_tier(t, 'shade')
        for c in teams[t]:
            if is_trait(c, 'shade'):

                items.change_stat(c, 'champion', False, '  shade')
                items.change_stat(c, 'stunned', True, '  shade')
                field.coordinates[c.y][c.x] = None
                # allows the units to "swap" places instead of treating the hex as taken

                c.add_que('execute_function', config.LEAP_DELAY, [field.leap_to_back_line, {'trait': '  shade'}])


shade_helper_list = []  # [champion, attacks]


def shade_helper(champion):
    if not champion.target: field.find_target(champion)

    tier = get_origin_class_tier(champion.team, 'shade')
    if tier > 0:
        attacks = 1
        if len(list(filter(lambda x: x[0] == champion, shade_helper_list))) == 0:
            shade_helper_list.append([champion, 1])
        else:
            for i, s in enumerate(shade_helper_list):
                if s[0] == champion:
                    shade_helper_list[i][1] += 1
                    attacks = shade_helper_list[i][1]

        if attacks % 3 == 0:
            for c in champion.enemy_team():
                if c.target == champion:
                    c.target = None

        if attacks % 4 == 0 and attacks > 1:
            if champion.target:
                if champion.target.health > 0:
                    champion.spell(champion.target, origin_class_stats.damage['shade'][tier] / champion.SP)


def sharpshooter(champion, target, damage, true_damage, spell):
    tier = get_origin_class_tier(champion.team, 'sharpshooter')
    if tier > 0 and is_trait(champion, 'sharpshooter'):

        bounces = origin_class_stats.targets['sharpshooter'][tier]
        last_target = target
        damage_multiplier = origin_class_stats.damage['sharpshooter'][tier]
        if not spell:
            damage = champion.AD

        # start ricochet loop
        for i in range(0, bounces):
            enemy_team = champion.enemy_team()
            possible_targets = list(filter(lambda x: x != last_target, enemy_team))
            random.shuffle(possible_targets)

            # continue if total more than one enemy left
            if len(possible_targets) > 0:
                next_target = possible_targets[0]
                last_target = next_target
                damage *= damage_multiplier
                true_damage *= damage_multiplier

                if spell:
                    champion.spell(next_target, damage, true_damage, False, False, 'sharpshooter')
                else:                                                               # forward AD as a parameter
                    champion.attack(true_damage, next_target, False, 'sharpshooter', champion.AD * damage_multiplier ** (i+1))


def tormented():
    return


def vanguard(blue_team, red_team):
    teams = {'blue': blue_team, 'red': red_team}

    for t in ['blue', 'red']:
        tier = get_origin_class_tier(t, 'vanguard')
        if tier > 0:
            for c in teams[t]:
                if is_trait(c, 'vanguard'):
                    items.change_stat(c, 'armor', c.armor + origin_class_stats.armor['vanguard'][tier], 'vanguard')

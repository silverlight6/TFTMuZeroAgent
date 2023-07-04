import Simulator.config as config
import Simulator.stats as stats
import Simulator.field as field
import Simulator.champion_functions as champion_functions
import Simulator.item_stats as item_stats
import Simulator.origin_class as origin_class
import Simulator.items as items
import random
from math import ceil, floor


# ALL ULT FUNCTIONS BASE HERE. NAMED:
# THE FIRST IS JUST 'champion.name'
# IF A SECOND FUNCTION IS NEEDED IT'S ALWAYS 'champion.name_ability'

# There's some repetition in the ults. Lots of logic stuff written again in the next one.
# The reason is that if something small gets changed in the ult's logic, it's easy to make the changes.
# Or maybe I was lazy and didn't exactly know how many times
# I needed to do the same in the future so never replaced them with functions.
# Of course some (quite many) helper functions are used, but not as many as I could have.
# The ones with cones and overall every ult with loads of shenanigans
# with the coordinates are rather ugly since the coordinates are hexagonal.

# For some ults there's an image named 'champion.name_ult.png' which gives some idea about what's going on
# The pics are pretty shit and were made just for my own good but decided to include them anyway

def default_ability_calls(champion):
    if not champion.target:
        field.find_target(champion)
    if not (champion.name == 'galio' and champion.stars == 1):
        champion.print(' ability triggered ')

    if champion.mana_cost_increased:
        champion.print(' {} {} --> {}'.format('mana_cost_increased', True, False))

    for i in range(0, champion.ionic_sparked):
        champion.spell(champion, champion.maxmana * item_stats.damage['ionic_spark'], 0, True)

    # spirit -trait
    if origin_class.is_trait(champion, 'spirit'):
        origin_class.spirit(champion)

    champion.spell_has_used_ludens = False  # ludens_echo helper

    champion.mana_cost_increased = False
    champion.mana = 0
    champion.castMS = champion_functions.MILLIS()


# treat the ult as an attack --> apply cooldown
def apply_attack_cooldown(champion, halved=True):
    halver = 2
    if not halved:
        halver = 1
    champion.idle = False
    champion.clear_que_idle()
    champion.add_que('clear_idle', (1 / champion.AS * 1000) / halver)


# Aatrox pulls some of the farthest enemies toward himself, then slams the ground in front of himself,
# dealing magic damage to all enemies hit.
def aatrox(champion):
    champion.add_que('change_stat', -1, None, 'ability_active', True)
    default_ability_calls(champion)

    champion.add_que('change_stat', stats.ABILITY_LENGTH[champion.name], None, 'ability_active', False)

    enemies = field.find_enemies(champion)

    enemy_amount = len(enemies)
    enemies_pulled = stats.ABILITY_TARGETS[champion.name][champion.stars]

    # pull max amount of targets. for example if 3 star aatrox wants to pull 5 but there's 3 alive, pull 2 instead
    if enemy_amount < enemies_pulled + 1:
        enemies_pulled = enemy_amount - 1
    for i in range(1, enemies_pulled + 1):

        free_hexes = []
        target_neighbors = field.find_neighbors(champion.target.y, champion.target.x)

        for n in target_neighbors:
            c = field.coordinates[n[0]][n[1]]
            if c is None:
                free_hexes.append(n)

        # break if there's no more space around the target unit. this should be implemented better
        if len(free_hexes) == 0:
            break

        for h in free_hexes:
            d = field.distance({'y': enemies[-i][0].y, 'x': enemies[-i][0].x}, {'y': h[0], 'x': h[1]}, False)
            h.append(d)
        free_hexes.sort(key=lambda x: x[2])

        enemies[-i][0].move(free_hexes[0][0], free_hexes[0][1], True)

    champion.add_que('execute_function', stats.ABILITY_LENGTH[champion.name],
                     [aatrox_ability, {'y': champion.target.y, 'x': champion.target.x}])


def aatrox_ability(champion, data):
    neighbors = field.find_neighbors(data['y'], data['x'])
    neighbors.append([data['y'], data['x']])

    c = field.coordinates
    for n in neighbors:
        if c[n[0]][n[1]] and c[n[0]][n[1]].team != champion.team and c[n[0]][n[1]].champion:
            champion.spell(c[n[0]][n[1]], stats.ABILITY_DMG[champion.name][champion.stars])

    apply_attack_cooldown(champion)


def ahri(champion):
    champion.idle = False
    champion.clear_que_idle()
    champion.add_que('clear_idle', stats.ABILITY_LENGTH[champion.name])
    default_ability_calls(champion)

    champion.add_que('execute_function', stats.ABILITY_LENGTH[champion.name],
                     [ahri_ability, {'y': champion.target.y, 'x': champion.target.x}])


def ahri_ability(champion, data):
    radius = stats.ABILITY_RADIUS[champion.name]
    # interruption
    if champion.stunned or champion.health <= 0:
        radius = stats.ABILITY_RADIUS[champion.name] - 1

    enemies = field.enemies_in_distance(champion, data['y'], data['x'], radius)
    for e in enemies:
        champion.spell(e, stats.ABILITY_DMG[champion.name][champion.stars])

    apply_attack_cooldown(champion)


def akali(champion):
    default_ability_calls(champion)
    champion.spell(champion.target, stats.ABILITY_DMG[champion.name][champion.stars])

    apply_attack_cooldown(champion)


# making a cone is not that fun
def annie(champion):
    default_ability_calls(champion)

    # 1. find hex that is our neighbor and closest to the target
    neighbors = field.find_neighbors(champion.y, champion.x)
    if neighbors:
        for n in neighbors:
            try:
                d = field.distance({'y': champion.target.y, 'x': champion.target.x}, {'y': n[0], 'x': n[1]}, False)
                n.append(d)
            except AttributeError:
                print('passing')
                pass
        # making sure that distance was added
        if len(neighbors[0]) == 3:
            neighbors = sorted(neighbors, key=lambda x: x[2])

        # 2. find hex that is 1's neigbor and furthest away from us
        # 3 leave the neighbor out of the spell which is in a line's way that's drawn from champion to 'cone_center'

        spell_target = neighbors[0]

        direction_y = spell_target[0] - champion.y + 1
        if champion.y % 2 == 0:
            direction_x = spell_target[1] - champion.x + 1
            cone_center_table = [
                [[], [-2, -1], [-2, 1]],
                [[0, -2], [], [0, 2]],
                [[], [2, -1], [2, 1]]
            ]

            leave_out_table = [
                [[], [-3, -1], [-3, 2]],
                [[0, -3], [], [0, 2]],
                [[], [3, -1], [3, 2]]
            ]

            cone_center = [champion.y + cone_center_table[direction_y][direction_x][0],
                           champion.x + cone_center_table[direction_y][direction_x][1]]
            leave_out = [champion.y + leave_out_table[direction_y][direction_x][0],
                         champion.x + leave_out_table[direction_y][direction_x][1]]
        else:
            direction_x = spell_target[1] - champion.x + 1
            cone_center_table = [
                [[-2, -1], [-2, 1], []],
                [[0, -2], [], [0, 2]],
                [[2, -1], [2, 1], []]
            ]

            leave_out_table = [
                [[-3, -2], [-3, 1], []],
                [[0, -3], [], [0, 3]],
                [[3, -2], [3, 1], []]
            ]
            cone_center = [champion.y + cone_center_table[direction_y][direction_x][0],
                           champion.x + cone_center_table[direction_y][direction_x][1]]
            leave_out = [champion.y + leave_out_table[direction_y][direction_x][0],
                         champion.x + leave_out_table[direction_y][direction_x][1]]
    else:
        cone_center = [champion.target.y, champion.target.x]
        leave_out = []

    neighbors = field.find_neighbors(cone_center[0], cone_center[1])
    neighbors.append(cone_center)
    for n in neighbors:
        if n != leave_out and n[0] >= 0 and n[1] >= 0 and n[0] < 8 and n[1] < 7 and field.coordinates[n[0]][n[1]] \
                and field.coordinates[n[0]][n[1]].team != champion.team and field.coordinates[n[0]][n[1]].champion:
            champion.spell(field.coordinates[n[0]][n[1]], stats.ABILITY_DMG[champion.name][champion.stars])

    apply_attack_cooldown(champion)
    shield_amount = stats.SHIELD_AMOUNT[champion.name][champion.stars] * champion.SP
    champion.add_que('shield', 0, None, None,
                     {'amount': shield_amount, 'identifier': champion_functions.MILLIS() * shield_amount,
                      'applier': champion, 'original_amount': shield_amount},
                     {'increase': True, 'expires': stats.SHIELD_LENGTH[champion.name]})


def aphelios(champion):
    default_ability_calls(champion)

    c = None
    while c is None:
        y = random.randint(0, 7)
        x = random.randint(0, 6)
        if y == 0 or y == 7 and x == 0 or x == 6 and field.coordinates[y][x] is None:
            c = [y, x]

    turret = champion.spawn('aphelios_turret', champion.stars, c[0], c[1], champion.team, False)
    champion.underlords.append(turret)
    champion.add_que('kill', stats.ABILITY_LENGTH[champion.name][champion.stars] * champion.SP, None, None, turret)


# also in champion_functions.py: attack()
def ashe(champion):
    default_ability_calls(champion)

    # to ensure the changes to go through before the initial attack,
    # we'll just set the change time to be CURRENT TIME - 1 MS
    # using ezreal's AS change logic
    as_gain = (stats.ABILITY_AS_GAIN[champion.name][champion.stars] - 1) * champion.SP + 1
    champion.add_que('change_stat', -1, None, 'AS', champion.AS * as_gain)
    champion.add_que('change_stat', stats.ABILITY_LENGTH[champion.name], None, 'AS', None, {'ezreal': as_gain})

    champion.add_que('change_stat', -1, None, 'ability_active', True)
    champion.add_que('change_stat', stats.ABILITY_LENGTH[champion.name], None, 'ability_active', False)


def ashe_helper(champion, data):
    target = data['target']
    items.guinsoos_rageblade(champion)  # guinsoos_rageblade
    items.statikk_shiv(champion, target)  # statikk_shiv
    items.runaans_hurricane(champion, target)  # runaans_hurricane

    if target.health < 0:
        field.find_target(champion)
        target = champion.target
    for i in range(0, stats.ABILITY_SLICES[champion.name]):
        if target:
            champion.attack(data['bonus_dmg'], target, False, ' ability',
                            champion.AD * stats.ABILITY_DAMAGE_MULTIPLIER[champion.name])


def azir(champion):
    default_ability_calls(champion)
    ability_rectangle_width = 3
    affected_hexes = field.rectangle_from_champion_to_wall_behind_target(champion, ability_rectangle_width,
                                                                         champion.target.y, champion.target.x)

    longest_line = max([len(affected_hexes[0]), len(affected_hexes[1]), len(affected_hexes[2])])

    # base_delay = stats.ABILITY_LENGTH[champion.name] / longest_line
    already_targeted = []

    # the whole azir ult executive process
    # the line of dudes is three wide
    for i in range(0, longest_line):
        for j in range(0, ability_rectangle_width):

            # if some line is longer, skip the rest of the iterations regarding this line
            if len(affected_hexes[j]) > i:
                # current coordinate
                c = field.coordinates[affected_hexes[j][i][0]][affected_hexes[j][i][1]]
                if c and c.team != champion.team and c.champion and c not in already_targeted:

                    # if this coordinate is within the pushing range, find a new coordinate for the minion
                    if affected_hexes[j][i][2] <= 3:
                        push_coordinates = None
                        push_counter = 1
                        while not push_coordinates:
                            if i + push_counter < len(affected_hexes[j]):
                                if field.coordinates[affected_hexes[j][i + push_counter][0]][
                                                     affected_hexes[j][i + push_counter][1]] is None:
                                    push_coordinates = [affected_hexes[j][i + push_counter][0],
                                                        affected_hexes[j][i + push_counter][1]]
                                else:
                                    push_counter += 1
                            else:
                                break
                        if push_coordinates:
                            c.move(push_coordinates[0], push_coordinates[1], True)
                            already_targeted.append(c)

                    # if not, then just stun for 2 seconds
                    else:
                        c.add_que('change_stat', -1, None, 'stunned', True)
                        c.clear_que_stunned_removal()
                        c.add_que('change_stat', stats.ABILITY_STUN_DURATION[champion.name][champion.stars], None,
                                  'stunned', False)
                        already_targeted.append(c)

                    champion.spell(c, stats.ABILITY_DMG[champion.name][champion.stars])
                    c.add_que('change_stat', 0, None, 'movement_delay',
                              c.movement_delay * stats.ABILITY_SLOW_AMOUNT[champion.name])
                    c.add_que('change_stat', stats.ABILITY_SLOW_DURATION[champion.name], None, 'movement_delay',
                              champion_functions.reset_stat(c, 'movement_delay'))


# forming cones with hexagonal coordinates is absolute aids
# the code is shit but does the job
def cassiopeia(champion):
    default_ability_calls(champion)
    target = champion.target

    # all coords that are 4 tiles away
    four_tiles_away = []
    for i in range(-5, 12):
        for j in range(-5, 11):
            if field.distance({'y': champion.y, 'x': champion.x}, {'y': i, 'x': j}, False) == 4:
                four_tiles_away.append([i, j])

    # get a line from the target to the tiles that are 4 distance away
    # then do the same but from the champion itself
    # if the target --> tile -line hexes are all in the champion -->
    # tile -line, choose that end tile to be the cone end point
    line_end_point = []
    for f in four_tiles_away:

        champion_line = (field.line({'y': champion.y, 'x': champion.x}, {'y': f[0], 'x': f[1]}))
        target_line = (field.line({'y': target.y, 'x': target.x}, {'y': f[0], 'x': f[1]}))

        includes_all = True
        for t in target_line:
            if t not in champion_line:
                includes_all = False
                break

        if includes_all:
            line_end_point.append([f[0], f[1]])

    if not line_end_point:
        print("Someone figure out why cassiopeia is dying")
        return
    line_end_point = line_end_point[0]
    # line_end_point = line_end_point[random.randint(0,len(line_end_point) - 1)]

    # find the cone corners. rules: 2 hexes from cone end, 4 hexes from champion
    cone_corners = []
    for f in four_tiles_away:
        d_from_champion = field.distance({'y': champion.y, 'x': champion.x}, {'y': f[0], 'x': f[1]}, False)
        d_from_end_point = field.distance({'y': line_end_point[0], 'x': line_end_point[1]}, {'y': f[0], 'x': f[1]},
                                          False)
        if d_from_champion == 4 and d_from_end_point == 2:
            cone_corners.append([f[0], f[1]])

    # the hexes between the cone end point and the corners
    side_points = []
    for f in four_tiles_away:
        d_from_first_corner = field.distance({'y': cone_corners[0][0], 'x': cone_corners[0][1]}, {'y': f[0], 'x': f[1]},
                                             False)
        d_from_second_corner = field.distance({'y': cone_corners[1][0], 'x': cone_corners[1][1]},
                                              {'y': f[0], 'x': f[1]}, False)
        d_from_end_point = field.distance({'y': line_end_point[0], 'x': line_end_point[1]}, {'y': f[0], 'x': f[1]},
                                          False)
        if d_from_end_point == 1 and (d_from_first_corner == 1 or d_from_second_corner == 1):
            side_points.append([f[0], f[1]])

    # the middle of the cone: find a certain coord (pic) and choose that and all its neighbors
    # the coord should be 2 from end point, 2 from champion and 2-3 from the corners (2 preferred).
    mid_hex = None
    mid_hex_secondary = None
    for i in range(-5, 12):
        for j in range(-5, 11):
            f = [i, j]
            d_from_first_corner = field.distance({'y': cone_corners[0][0], 'x': cone_corners[0][1]},
                                                 {'y': f[0], 'x': f[1]}, False)
            d_from_second_corner = field.distance({'y': cone_corners[1][0], 'x': cone_corners[1][1]},
                                                  {'y': f[0], 'x': f[1]}, False)
            d_from_end_point = field.distance({'y': line_end_point[0], 'x': line_end_point[1]}, {'y': f[0], 'x': f[1]},
                                              False)
            d_from_champion = field.distance({'y': champion.y, 'x': champion.x}, {'y': f[0], 'x': f[1]}, False)

            if d_from_end_point == 2 and d_from_first_corner == 2 and \
                    d_from_second_corner == 2 and d_from_champion == 2:
                mid_hex = f
            if (d_from_end_point == 2 and d_from_first_corner == 3 and
                    d_from_second_corner == 3 and d_from_champion == 2):
                mid_hex_secondary = f

    # if the hexes align the right way, there's two extra coords that need to be added. one next to each corner
    additional_coords = []
    if not mid_hex_secondary and not mid_hex:
        mid_hex = [target.y, target.x]
    elif not mid_hex:
        mid_hex = mid_hex_secondary
        for i in range(-5, 12):
            for j in range(-5, 11):
                f = [i, j]
                d_from_first_corner = field.distance({'y': cone_corners[0][0], 'x': cone_corners[0][1]},
                                                     {'y': f[0], 'x': f[1]}, False)
                d_from_second_corner = field.distance({'y': cone_corners[1][0], 'x': cone_corners[1][1]},
                                                      {'y': f[0], 'x': f[1]}, False)
                d_from_champion = field.distance({'y': champion.y, 'x': champion.x}, {'y': f[0], 'x': f[1]}, False)

                if (d_from_first_corner == 1 or d_from_second_corner == 1) and d_from_champion == 3:
                    additional_coords.append(f)

    mid_hex_neighbors = []
    if mid_hex:
        mid_hex_neighbors = field.find_neighbors(mid_hex[0], mid_hex[1])

    cone = mid_hex_neighbors
    cone.append(mid_hex)
    cone.append(side_points[0])
    cone.append(side_points[1])
    cone.append(cone_corners[0])
    cone.append(cone_corners[1])
    cone.append(line_end_point)
    if len(additional_coords) == 2:
        cone.append(additional_coords[0])
        cone.append(additional_coords[1])

    # deal the damage and stun the targets etc
    already_targeted = []
    for c in cone:
        coords = field.coordinates
        if 7 >= c[0] >= 0 <= c[1] <= 6:
            h = coords[c[0]][c[1]]
            if h and h.team != champion.team and h.champion and h not in already_targeted:
                # h.add_que('change_stat', -1, None, 'stunned', True)
                # adding this when creating 'quicksilver' but seems like we had a reason to stun them here locally
                if not ('quicksilver' in h.items and champion_functions.MILLIS() <=
                        item_stats.item_change_length['quicksilver']):
                    h.print(' {} {} --> {}'.format('stunned', h.stunned, True))
                    h.stunned = True
                    h.clear_que_stunned_removal()

                    h.add_que('change_stat', stats.ABILITY_STUN_DURATION[champion.name][champion.stars],
                              None, 'stunned', False)
                else:
                    h.print(' not stunned because wears quicksilver')

                dmg_increase = stats.ABILITY_TARGET_INCREASE_DAMAGE_RECEIVING[champion.name]
                h.print(' {} {} --> {}'.format('receive_increased_damage', h.receive_increased_damage,
                                               round(dmg_increase * champion.SP, 3)))
                h.receive_increased_damage = dmg_increase * champion.SP

                h.add_que('change_stat', stats.ABILITY_STUN_DURATION[champion.name][champion.stars],
                          None, 'receive_increased_damage', 1)
                champion.spell(h, stats.ABILITY_DMG[champion.name][champion.stars])

                already_targeted.append(h)

    apply_attack_cooldown(champion)


def diana(champion):
    default_ability_calls(champion)
    champion.add_que('change_stat', -1, None, 'ability_active', True)

    shield_identifier = champion_functions.MILLIS() * stats.SHIELD_AMOUNT[champion.name][champion.stars] * champion.SP
    shield_amount = stats.SHIELD_AMOUNT[champion.name][champion.stars] * champion.SP
    champion.add_que('shield', 0, None, None, {'amount': shield_amount, 'identifier': shield_identifier,
                                               'applier': champion, 'original_amount': shield_amount},
                     {'increase': True, 'expires': stats.SHIELD_LENGTH[champion.name]})

    neighbors = field.find_neighbors(champion.y, champion.x)
    orbs = []

    neighbor_amount = len(field.find_neighbors(champion.y, champion.x))
    n_a = neighbor_amount
    for i in range(0, stats.ABILITY_TARGETS[champion.name][champion.stars]):
        orbs.append(
            {'y': neighbors[i % n_a][0], 'x': neighbors[i % n_a][1], 'index': i % n_a, 'orbs': orbs, 'identifier': i,
             'shield_identifier': shield_identifier})

    for o in orbs:
        champion.add_que('execute_function', 0, [diana_ability, o])


# spin the orbs
def diana_ability(champion, data):
    turn_speed_per_hex = 1500 / 6

    # hit the enemy if there's someone in the orb's coordinates
    c = field.coordinates[data['y']][data['x']]
    if c and c.team != champion.team and c.champion:

        if data in data['orbs']:
            data['orbs'].remove(data)

        if len(data['orbs']) == 0:

            for s in champion.shields:
                if s['identifier'] == data['shield_identifier']:
                    shield_before = champion.shield_amount()
                    champion.shields.remove(s)
                    champion.print(
                        ' {} {} --> {}'.format('shield', ceil(shield_before), ceil(champion.shield_amount())))
                    break

            shield_identifier = champion_functions.MILLIS() * stats.SHIELD_AMOUNT[champion.name][
                champion.stars] * champion.SP
            shield_amount = stats.SHIELD_AMOUNT[champion.name][champion.stars] * champion.SP

            champion.add_que('shield', 0, None, None,
                             {'amount': shield_amount, 'identifier': shield_identifier, 'applier': champion,
                              'original_amount': shield_amount},
                             {'increase': True, 'expires': stats.SHIELD_LENGTH[champion.name]})
            champion.add_que('change_stat', 0, None, 'ability_active', False)

    # if not, spin the orbs.
    else:
        neighbors = field.find_neighbors(champion.y, champion.x)
        data['index'] += 1
        if data['index'] >= len(neighbors):
            data['index'] = 0

        data['y'] = neighbors[(data['index']) % len(neighbors)][0]
        data['x'] = neighbors[(data['index']) % len(neighbors)][1]
        for i, d in enumerate(data['orbs']):
            if d['identifier'] == data['identifier']:
                data['orbs'][i] = data

        champion.add_que('execute_function', turn_speed_per_hex, [diana_ability, data])


def elise(champion):
    default_ability_calls(champion)
    champion.add_que('change_stat', -1, None, 'ability_active', True)

    # to ensure both, the health change and health per attack -changes to happen BEFORE the auto attack goes through, 
    # we have to mark these changes as high priority
    champion.add_que('change_stat', -2, None, 'health',
                     champion.health + (stats.HEALTH[champion.name] * config.STARMULTIPLIER ** (champion.stars - 1))
                     * (stats.ABILITY_HEALTH_GAIN_PERCENTAGES[champion.name][champion.stars] - 1))

    champion.add_que('change_stat', -1, None, 'heal_per_attack',
                     champion.heal_per_attack + stats.ABILITY_HEALTH_PER_ATTACK[champion.name][champion.stars])
    champion.add_que('change_stat', -1, None, 'max_health',
                     champion.max_health * stats.ABILITY_HEALTH_GAIN_PERCENTAGES[champion.name][champion.stars])


def evelynn(champion):
    default_ability_calls(champion)

    r = random.randint(1, 100) / 100
    targets = 1
    if r > stats.ABILITY_TARGET_PROBABILITIES[champion.name][3]:
        targets = 3
    elif r > stats.ABILITY_TARGET_PROBABILITIES[champion.name][2]:
        targets = 2

    target_y = champion.target.y
    target_x = champion.target.x

    # find a bunch of close by targets and deal the dmg
    enemies_around_target = field.enemies_in_distance(champion, champion.target.y, champion.target.x, 1)
    if champion.target in enemies_around_target:
        enemies_around_target.remove(champion.target)
    enemies_around_target.insert(0, champion.target)

    for i in range(0, targets):
        if i > len(enemies_around_target) - 1:
            break

        target_dmg = stats.ABILITY_DMG[champion.name][champion.stars]
        if (enemies_around_target[i].health / enemies_around_target[i].max_health) < 0.5:
            target_dmg *= stats.ABILITY_DAMAGE_MULTIPLIER[champion.name][champion.stars]

        champion.spell(enemies_around_target[i], target_dmg)

    # find a hex that's 3 away from champion and 4 away from the target
    teleport_hexes = []
    for i in range(0, 7):
        for j in range(0, 6):
            d_from_champion = field.distance({'y': champion.y, 'x': champion.x}, {'y': i, 'x': j}, False)
            d_from_target = field.distance({'y': target_y, 'x': target_x}, {'y': i, 'x': j}, False)
            if d_from_champion == 3 and d_from_target == 4:
                teleport_hexes.append([i, j])
    # if there's none, just pick the neighbors
    if len(teleport_hexes) == 0:
        teleport_hexes = field.find_neighbors(champion.y, champion.x)

    # make sure that we teleport backwards
    if champion.y > target_y:
        teleport_hexes = list(filter(lambda x: x[0] > target_y, teleport_hexes))
    if champion.y < target_y:
        teleport_hexes = list(filter(lambda x: x[0] < target_y, teleport_hexes))

    if len(teleport_hexes) > 0:
        teleport_target = teleport_hexes[random.randint(0, len(teleport_hexes) - 1)]
        champion.clear_que_idle()
        champion.move(teleport_target[0], teleport_target[1], True)


def ezreal(champion):
    default_ability_calls(champion)
    champion.idle = False
    champion.clear_que_idle()
    champion.add_que('clear_idle', stats.ABILITY_LENGTH[champion.name])

    champion.add_que('execute_function', stats.ABILITY_LENGTH[champion.name],
                     [ezreal_ability, {'y': champion.target.y, 'x': champion.target.x}])


def ezreal_ability(champion, data):
    ability_rectangle_width = 5

    target_y = data['y']
    target_x = data['x']

    # get the hexes of which the ult passes through
    affected_hexes = field.rectangle_from_champion_to_wall_behind_target(champion, ability_rectangle_width, target_y,
                                                                         target_x)

    longest_line = max([len(affected_hexes[0]), len(affected_hexes[1]), len(affected_hexes[2]), len(affected_hexes[3]),
                        len(affected_hexes[4])])

    already_targeted = []

    for i in range(0, longest_line):
        for j in range(0, ability_rectangle_width):

            # if some line is longer, skip the rest of the iterations regarding this line
            if len(affected_hexes[j]) > i:
                # current coordinate
                c = field.coordinates[affected_hexes[j][i][0]][affected_hexes[j][i][1]]
                if c and c.team != champion.team and c.champion and c not in already_targeted:
                    champion.spell(c, stats.ABILITY_DMG[champion.name][champion.stars])

                    c.add_que('change_stat', -1, None, 'AS',
                              c.AS * stats.ABILITY_AS_DECREASE[champion.name][champion.stars])
                    c.add_que('change_stat', stats.ABILITY_AS_CHANGE_LENGTH[champion.name], None, 'AS', None,
                              {'ezreal': stats.ABILITY_AS_DECREASE[champion.name][champion.stars]})

                    already_targeted.append(c)

                if c and c.team == champion.team and c.champion and c not in already_targeted and c is not champion:
                    c.add_que('heal', -1, None, None, stats.ABILITY_HEAL[champion.name][champion.stars] * champion.SP)

                    c.add_que('change_stat', -1, None, 'AS',
                              c.AS * stats.ABILITY_AS_GAIN[champion.name][champion.stars])
                    c.add_que('change_stat', stats.ABILITY_AS_CHANGE_LENGTH[champion.name], None, 'AS', None,
                              {'ezreal': stats.ABILITY_AS_GAIN[champion.name][champion.stars]})
                    already_targeted.append(c)

    apply_attack_cooldown(champion)


def fiora(champion):
    default_ability_calls(champion)
    champion.idle = False
    champion.clear_que_idle()
    champion.add_que('clear_idle', stats.ABILITY_LENGTH[champion.name])

    champion.add_que('change_stat', -1, None, 'immune', True)
    champion.add_que('change_stat', stats.ABILITY_LENGTH[champion.name], None, 'immune', False)

    champion.add_que('execute_function', stats.ABILITY_LENGTH[champion.name], [fiora_ability, {}])


def fiora_ability(champion, data):
    if not champion.target:
        field.find_target(champion)

    if champion.target:
        champion.target.add_que('change_stat', -1, None, 'stunned', True)
        champion.target.clear_que_stunned_removal()
        champion.target.add_que('change_stat', stats.ABILITY_STUN_DURATION[champion.name][champion.stars], None,
                                'stunned', False)

        champion.spell(champion.target, stats.ABILITY_DMG[champion.name][champion.stars])

    apply_attack_cooldown(champion, False)


def garen(champion):
    default_ability_calls(champion)
    champion.add_que('change_stat', -1, None, 'ability_active', True)
    champion.add_que('change_stat', stats.ABILITY_LENGTH[champion.name], None, 'ability_active', False)

    champion.add_que('change_stat', -1, None, 'spell_damage_reduction_percentage',
                     champion.spell_damage_reduction_percentage * stats.ABILITY_SPELL_DAMAGE_REDUCTION_PERCENTAGE[
                         champion.name])
    champion.add_que('change_stat', stats.ABILITY_LENGTH[champion.name], None, 'spell_damage_reduction_percentage',
                     None, {'garen': stats.ABILITY_SPELL_DAMAGE_REDUCTION_PERCENTAGE[champion.name]})

    ms = 50
    for i in range(0, stats.ABILITY_SLICES[champion.name]):
        champion.add_que('execute_function',
                         ms + i * (stats.ABILITY_LENGTH[champion.name] / stats.ABILITY_SLICES[champion.name]),
                         [garen_ability, {}])


def garen_ability(champion, data):
    neighbors = field.find_neighbors(champion.y, champion.x)
    team = champion.team
    enemies_around = []
    for n in neighbors:
        c = field.coordinates[n[0]][n[1]]
        if c and c.team != team and c.champion:
            enemies_around.append(c)

    # if(not champion.stunned):
    for e in enemies_around:
        champion.spell(e, stats.ABILITY_DMG[champion.name][champion.stars] / stats.ABILITY_SLICES[champion.name])


def hecarim(champion):
    champion.add_que('change_stat', -1, None, 'ability_active', True)
    default_ability_calls(champion)
    champion.add_que('change_stat', stats.ABILITY_LENGTH[champion.name], None, 'ability_active', False)

    for i in range(0, stats.ABILITY_SLICES[champion.name]):
        champion.add_que('execute_function',
                         i * (stats.ABILITY_LENGTH[champion.name] / stats.ABILITY_SLICES[champion.name]),
                         [hecarim_ability, {}])


def hecarim_ability(champion, data):
    neighbors = field.find_neighbors(champion.y, champion.x)

    coords = field.coordinates
    for n in neighbors:
        c = coords[n[0]][n[1]]
        if c and c.team != champion.team and c.champion:
            champion.spell(c, stats.ABILITY_DMG[champion.name][champion.stars] / stats.ABILITY_SLICES[champion.name])
            champion.add_que('heal', -1, None, None, (
                    stats.ABILITY_HEAL[champion.name][champion.stars] /
                    stats.ABILITY_SLICES[champion.name]) * champion.SP)


def irelia(champion):
    default_ability_calls(champion)
    ability_rectangle_width = 3

    # get the hexes of which the ult passes through
    affected_hexes = field.rectangle_from_champion_to_wall_behind_target(champion, ability_rectangle_width,
                                                                         champion.target.y, champion.target.x)

    for i, line in enumerate(affected_hexes):
        if len(line) > 3:
            affected_hexes[i] = line[:3]

    longest_line = max([len(affected_hexes[0]), len(affected_hexes[1]), len(affected_hexes[2])])
    already_targeted = []

    for i in range(0, longest_line):
        for j in range(0, ability_rectangle_width):

            # if some line is longer, skip the rest of the iterations regarding this line
            if len(affected_hexes[j]) > i:
                c = field.coordinates[affected_hexes[j][i][0]][affected_hexes[j][i][1]]
                if c and c.team != champion.team and c.champion and c not in already_targeted:
                    champion.spell(c, stats.ABILITY_DMG[champion.name][champion.stars])

                    c.add_que('change_stat', -1, None, 'disarmed', True)
                    c.add_que('change_stat', stats.ABILITY_DISARM_DURATION[champion.name][champion.stars], None,
                              'disarmed', False)

                    already_targeted.append(c)

    apply_attack_cooldown(champion)


def janna(champion):
    default_ability_calls(champion)
    ally_units = []
    if champion.team == 'blue':
        ally_units = champion.blue_return()
    if champion.team == 'red':
        ally_units = champion.red_return()

    ally_units = sorted(ally_units, key=lambda x: x.health / x.max_health)

    shielded_units = stats.ABILITY_TARGETS[champion.name][champion.stars]
    if len(ally_units) > shielded_units:
        ally_units = ally_units[:shielded_units]

    for a in ally_units:
        for i in range(0, 3):
            for s in a.shields:
                if s['applier'] == champion:
                    shield_before = a.shield_amount()
                    a.shields.remove(s)
                    a.print(' {} {} --> {}'.format('shield', ceil(shield_before), ceil(a.shield_amount())))
                    break

        identifier = champion_functions.MILLIS() * stats.SHIELD_AMOUNT[champion.name][champion.stars]
        shield_amount = stats.SHIELD_AMOUNT[champion.name][champion.stars] * champion.SP
        a.add_que('shield', -1, None, None, {'amount': shield_amount, 'identifier': identifier, 'applier': champion,
                                             'original_amount': shield_amount},
                  {'increase': True, 'expires': stats.SHIELD_LENGTH[champion.name]})

        ad_gain_abs = stats.ABILITY_DMG_GAIN[champion.name][champion.stars]
        if a.AD == 0:
            percentual_ad_change = 1
        else:
            percentual_ad_change = (ad_gain_abs / a.AD) + 1
        a.add_que('change_stat', -1, None, 'AD', a.AD * percentual_ad_change)
        a.add_que('change_stat', stats.SHIELD_LENGTH[champion.name], None, 'AD', None, {'ashe': percentual_ad_change})


def jarvaniv(champion):
    default_ability_calls(champion)
    enemies_plain = champion.enemy_team()
    enemies = []

    for i, e in enumerate(enemies_plain):
        d = field.distance(champion, e, True)
        enemies.append([e, d])

    enemies = sorted(enemies, key=lambda x: x[1], reverse=True)

    target = enemies[0][0]

    # find the hex where jarvan moves to (closest free hex to the target)
    hexes = []
    coords = field.coordinates
    for i in range(0, 8):
        for j in range(0, 7):
            distance = field.distance({'y': i, 'x': j}, {'y': target.y, 'x': target.x}, False)
            hexes.append([coords[i][j], distance, i, j])

    hexes = sorted(hexes, key=lambda x: x[1])
    target_hex_y = None
    target_hex_x = None
    while target_hex_y is None:
        if not hexes[0][0]:
            target_hex_y = hexes[0][2]
            target_hex_x = hexes[0][3]
        else:
            hexes = hexes[1:]

    # now find the path to the target hex and log its surroundings
    path = field.line({'y': champion.y, 'x': champion.x}, {'y': target_hex_y, 'x': target_hex_x})

    target_neighbors = field.find_neighbors(target_hex_y, target_hex_x)
    path += target_neighbors

    already_targeted = []
    for p in path:
        # print("IN JARVIN ABILITY")
        # print(p)
        if 0 <= p[0] < 8 and 0 <= p[1] < 7:
            c = coords[p[0]][p[1]]
            if c and c.team != champion.team and c.champion and c not in already_targeted:
                c.add_que('change_stat', -1, None, 'stunned', True)
                c.clear_que_stunned_removal()
                c.add_que('change_stat', stats.ABILITY_STUN_DURATION[champion.name][champion.stars], None, 'stunned',
                          False)

                champion.spell(c, stats.ABILITY_DMG[champion.name][champion.stars])

            already_targeted.append(c)

    champion.clear_que_idle()
    champion.move(target_hex_y, target_hex_x, True)


def jax(champion):
    default_ability_calls(champion)

    champion.add_que('change_stat', -1, None, 'ability_active', True)
    champion.add_que('change_stat', -1, None, 'autoimmune', True)
    champion.add_que('change_stat', stats.ABILITY_LENGTH[champion.name], None, 'ability_active', False)
    champion.add_que('change_stat', stats.ABILITY_LENGTH[champion.name], None, 'autoimmune', False)

    champion.add_que('execute_function', stats.ABILITY_LENGTH[champion.name], [jax_ability, {}])


def jax_ability(champion, data):
    enemy_neighbors = field.enemies_in_distance(champion, champion.y, champion.x, 1)

    for e in enemy_neighbors:
        e.add_que('change_stat', -1, None, 'stunned', True)
        e.clear_que_stunned_removal()
        e.add_que('change_stat', stats.ABILITY_STUN_DURATION[champion.name][champion.stars], None, 'stunned', False)
        champion.spell(e, stats.ABILITY_DMG[champion.name][champion.stars])


def jinx(champion):
    default_ability_calls(champion)

    targets = field.enemies_in_distance(champion, champion.target.y, champion.target.x, 1)
    for t in targets:
        champion.spell(t, stats.ABILITY_DMG[champion.name][champion.stars])

        t.add_que('change_stat', -1, None, 'stunned', True)
        t.clear_que_stunned_removal()
        t.add_que('change_stat', stats.ABILITY_STUN_DURATION[champion.name][champion.stars], None, 'stunned', False)

    apply_attack_cooldown(champion)


def katarina(champion):
    default_ability_calls(champion)
    champion.add_que('change_stat', -1, None, 'ability_active', True)
    champion.idle = False
    champion.clear_que_idle()
    champion.add_que('clear_idle', stats.ABILITY_LENGTH[champion.name])

    champion.add_que('change_stat', stats.ABILITY_LENGTH[champion.name], None, 'ability_active', False)

    for i in range(0, stats.ABILITY_SLICES[champion.name]):
        champion.add_que('execute_function',
                         i * (stats.ABILITY_LENGTH[champion.name] / stats.ABILITY_SLICES[champion.name]),
                         [katarina_ability, {}])


def katarina_ability(champion, data):
    enemies_in_range = field.enemies_in_distance(champion, champion.y, champion.x, 3)
    enemies_in_range = enemies_in_range[:3]

    if not champion.stunned:
        for e in enemies_in_range:
            champion.spell(e, stats.ABILITY_DMG[champion.name][champion.stars] / stats.ABILITY_SLICES[champion.name])
            e.add_que('change_stat', 0, None, 'healing_strength', stats.ABILITY_HEALING_REDUCE[champion.name])
            e.clear_que_healing_reduction()
            e.add_que('change_stat', stats.ABILITY_HEALING_REDUCE_LENGTH[champion.name], None, 'healing_strength', 1)


def kayn(champion, data={'redash': False}):
    if not champion.stunned:
        if not champion.target:
            field.find_target(champion)
            if champion.target:
                champion.print(
                    ' has a new target: ' + '{:<8}'.format(champion.target.team) + '{:<8}'.format(champion.target.name))
            else:
                return

        default_ability_calls(champion)
        distance = field.distance(champion, champion.target, True)

        r = random.randint(0, 100)
        if ((distance > 1 or r > 50) and not data['redash']) or (r > 50 and data['redash']):
            target_neighbors = field.find_neighbors(champion.target.y, champion.target.x)
            empty_neighbors = []
            coords = field.coordinates
            for n in target_neighbors:
                if not coords[n[0]][n[1]]:
                    empty_neighbors.append(n)
                    if len(empty_neighbors) > 1:
                        dash_coords = empty_neighbors[random.randint(0, len(empty_neighbors) - 1)]
                    else:
                        dash_coords = empty_neighbors[0]
                    champion.move(dash_coords[0], dash_coords[1], True)

        enemies_in_range = field.enemies_in_distance(champion, champion.y, champion.x, 1)

        for e in enemies_in_range:
            ability_dmg = stats.ABILITY_DMG[champion.name][champion.stars]

            if champion.kayn_form == 'shadow_assassin' \
                    and champion_functions.MILLIS() < stats.ABILITY_EXTRA_DAMAGE_LENGTH[champion.name]:
                ability_dmg *= stats.ABILITY_EXTRA_DAMAGE[champion.name][champion.stars]

            champion.spell(e, ability_dmg)

            # healing if rhaast equipped
            if champion.kayn_form == 'rhast':
                damage = 0
                if e.MR >= 0:
                    damage = ability_dmg * (100 / (100 + e.MR)) * champion.SP
                else:
                    damage = ability_dmg * (2 - 100 / (100 - e.MR)) * champion.SP
                champion.add_que('heal', -1, None, None,
                                 damage * stats.ABILITY_HEALTH_PER_CAST_DAMAGE_PERCENTAGES[champion.name][
                                     champion.stars])

            # increase next spell mana cost by 33% = reduce mana by 33% of maxmana
            if not e.mana_cost_increased and e.maxmana > 0:
                mana_reduce_amount = e.maxmana * stats.ABILITY_MANA_REQUIREMENT_INCREASEMENT[champion.name]
                start_value = e.mana
                e.mana -= mana_reduce_amount
                e.print(' {} {} --> {}'.format('mana', round(start_value, 1), round(e.mana, 1)))
                e.add_que('change_stat', -1, None, 'mana_cost_increased', True)

        # add a 250ms delay after each spin
        champion.idle = False
        que = champion.que_return()
        for q in que:
            if q[1] is champion and q[0] == 'clear_idle':
                que.remove(q)
        champion.que_replace(que)
        champion.add_que('clear_idle', 350)

        if len(enemies_in_range) == 1 and len(champion.enemy_team()) > 0:
            champion.add_que('execute_function', 350, [kayn, {'redash': True}])


kennen_hits = []


def kennen(champion):
    global kennen_hits
    kennen_hits = list(filter(lambda x: x[0] != champion, kennen_hits))

    # for kenny not to ult when there's no targets in range
    # brings some extra cpu load
    if champion.target and field.distance(champion, champion.target, True) <= stats.ABILITY_RADIUS[champion.name]:
        default_ability_calls(champion)
        champion.add_que('change_stat', -1, None, 'ability_active', True)
        champion.add_que('change_stat', stats.ABILITY_LENGTH[champion.name], None, 'ability_active', False)

        for i in range(0, stats.ABILITY_SLICES[champion.name]):
            champion.add_que('execute_function',
                             i * (stats.ABILITY_LENGTH[champion.name] / stats.ABILITY_SLICES[champion.name]),
                             [kennen_ability, {}])


def kennen_ability(champion, data):
    global kennen_hits
    targets = field.enemies_in_distance(champion, champion.y, champion.x, stats.ABILITY_RADIUS[champion.name])

    if not champion.stunned:
        for e in targets:
            champion.spell(e, stats.ABILITY_DMG[champion.name][champion.stars] / stats.ABILITY_SLICES[champion.name])

            found = False
            index = -1
            target = e
            if len(kennen_hits) > 0:
                for i, v in enumerate(kennen_hits):
                    if v[0] == champion and v[1] == target:
                        found = True
                        index = i
                        break

            if found:
                kennen_hits[index][2] += 1
            else:
                kennen_hits.append([champion, target, 1])
                index = len(kennen_hits) - 1

            if kennen_hits[index][2] >= 3:
                e.add_que('change_stat', 0, None, 'stunned', True)
                e.clear_que_stunned_removal()
                e.add_que('change_stat', stats.ABILITY_STUN_DURATION[champion.name][champion.stars], None, 'stunned',
                          False)
                kennen_hits[index][2] = 0


def kindred(champion):
    default_ability_calls(champion)

    if not champion.target:
        field.find_target(champion)
        champion.print(
            ' has a new target: ' + '{:<8}'.format(champion.target.team) + '{:<8}'.format(champion.target.name))

    target = champion.target
    target_y = target.y
    target_x = target.x

    champion.spell(target, stats.ABILITY_DMG[champion.name][champion.stars])

    target.add_que('change_stat', 0, None, 'healing_strength', stats.ABILITY_HEALING_REDUCE[champion.name])
    target.clear_que_healing_reduction()
    target.add_que('change_stat', stats.ABILITY_HEALING_REDUCE_LENGTH[champion.name], None, 'healing_strength', 1)

    # find all hexes that are within 3 distance of kindred and log the distance from target to all those hexes
    potential_hexes = []
    coords = field.coordinates
    for i in range(0, 7):
        for j in range(0, 6):
            distance_to_champion = field.distance({'y': champion.y, 'x': champion.x}, {'y': i, 'x': j}, False)
            if distance_to_champion <= 3:
                distance_to_target = field.distance({'y': target_y, 'x': target_x}, {'y': i, 'x': j}, False)
                potential_hexes.append([i, j, distance_to_target])

                # find all hexes that are furthest away of the target (still under 4) and choose one random of those
    potential_hexes = sorted(potential_hexes, key=lambda x: x[2], reverse=True)
    if len(potential_hexes) > 0:
        while len(potential_hexes[0]) < 2 and potential_hexes[0][2] > 3:
            potential_hexes = potential_hexes[1:]
        potential_hexes = list(filter(lambda x: (x[2] == potential_hexes[0][2]), potential_hexes))
        leap_hex = potential_hexes[random.randint(0, len(potential_hexes) - 1)]

    else:
        leap_hex = [champion.y, champion.x]

    champion.move(leap_hex[0], leap_hex[1], True)
    apply_attack_cooldown(champion, halved=False)


def leesin(champion):
    default_ability_calls(champion)
    # draw a line from lee to target and continue it until it hits an edge
    if not champion.target:
        field.find_target(champion)
    if len(champion.enemy_team()) > 0:
        line_to_wall_behind_target = field.rectangle_from_champion_to_wall_behind_target(champion, 1, champion.target.y,
                                                                                         champion.target.x)
        if line_to_wall_behind_target[0]:
            end_point = line_to_wall_behind_target[0][-1]
        else:
            end_point = [champion.target.y, champion.target.x]

        # find the closest corner to the line's end point
        # not perfect
        e_y = end_point[0]
        e_x = end_point[1]
        end_point_original = [e_y, e_x]
        if e_x == 6 or e_x == 0:
            if e_y > champion.y:
                end_point[0] = 7
            if e_y < champion.y:
                end_point[0] = 0
            if e_y == champion.y:
                if champion.y <= 3:
                    end_point[0] = 0
                else:
                    end_point[0] = 7

        elif e_y == 7 or e_y == 0:
            if e_x > champion.x:
                end_point[1] = 6
            if e_x < champion.x:
                end_point[1] = 0
            if e_x == champion.x:
                if champion.y % 2 == 0:
                    end_point[1] = 0
                else:
                    end_point[1] = 6

        coords = field.coordinates
        kick_coords = None
        kick_out = False
        deal = False

        t = champion.target

        # if 3 star, just kill the target and other enemies strictly around it
        # Its a fun way of doing things but sure.
        if champion.stars == 3:
            ttt = field.enemies_in_distance(champion, t.y, t.x, 1)
            for tt in ttt:
                tt.die()
        else:

            # if target is on a side
            if t.y == 7 or t.y == 0 or t.x == 6 or t.x == 0:

                corners = [[0, 0], [0, 6], [7, 0], [7, 6]]

                # if they both are on a side lines, remove some corners from the list
                if champion.y == 7 or champion.y == 0 or champion.x == 6 or champion.x == 0:
                    if champion.x > t.x:
                        corners.remove([0, 6])
                        corners.remove([7, 6])
                    if champion.x < t.x:
                        corners.remove([0, 0])
                        corners.remove([7, 0])
                    if champion.y > t.y:
                        if champion.x <= t.x:
                            corners.remove([7, 6])
                        if champion.x >= t.x:
                            corners.remove([7, 0])
                    if champion.y < t.y:
                        if champion.x >= t.x:
                            corners.remove([0, 0])
                        if champion.x <= t.x:
                            corners.remove([0, 6])

                # find the closest corner
                for i, c in enumerate(corners):
                    d = field.distance({'y': t.y, 'x': t.x}, {'y': c[0], 'x': c[1]}, False)
                    corners[i].append(d)
                corners = sorted(corners, key=lambda x: x[2])
                closest_corner = corners[0]

                # if we're at the corner (distance == 0)
                if closest_corner[2] == 0:
                    kick_out = True

                # otherwise draw a line from target to the closest corner
                # if all hexes on the line are occupied, kick the target out
                else:
                    line = field.line({'y': t.y, 'x': t.x}, {'y': closest_corner[0], 'x': closest_corner[1]})
                    all_occupied = True
                    for l in line:
                        if coords:
                            if not 0 <= l[0] <= 7 or 0 <= l[1] <= 6:
                                all_occupied = False
                                break
                    if all_occupied:
                        kick_out = True

            if not kick_out:
                # draw a line from the corner to the (lee to target) -line end point and find the first free spot

                if end_point[:2] != end_point_original:
                    line = (field.line({'y': end_point[0], 'x': end_point[1]},
                                       {'y': end_point_original[0], 'x': end_point_original[1]}))
                    while not kick_coords:
                        if len(line) == 0:

                            # very much just a 'tape on a tap head' -type solution for a rare bug
                            # if the corner is very full,
                            # the unit would get kicked out rather than added to the queue of units on the side lane
                            if not (champion.target.y == 0 or champion.target.y == 7
                                    or champion.target.x == 0 or champion.target.x == 6):
                                deal = True
                                kick_coords = [champion.target.y, champion.target.x]

                            else:
                                kick_out = True
                            break
                        if not (line[0][0] == 0 or line[0][0] == 7 or line[0][1] == 0 or line[0][1] == 6)\
                                or coords[line[0][0]][line[0][1]]:
                            line = line[1:]
                        else:
                            kick_coords = line[0]

                            break

                # if the target is kicked straight to the corner, but it's occupied
                # find the closest hex to the corner that's free
                else:
                    # some close hexes
                    index = 1
                    hexes_on_sides = []
                    while len(hexes_on_sides) < 1:
                        hexes_in_distance = field.hexes_in_distance(end_point[0], end_point[1], index)
                        for n in hexes_in_distance:
                            # add only if they are on side lanes
                            if (n[0] == 0 or n[0] == 7 or n[1] == 0 or n[1] == 6) and not coords[n[0]][n[1]]:
                                hexes_on_sides.append(n)
                                d = field.distance({'y': n[0], 'x': n[1]},
                                                   {'y': end_point_original[0], 'x': end_point_original[1]}, False)
                                hexes_on_sides[-1].append(d)
                        index += 1

                    # sort by distance to the end point and choose the closest one
                    hexes_on_sides = sorted(hexes_on_sides, key=lambda x: x[2])
                    kick_coords = [hexes_on_sides[0][0], hexes_on_sides[0][1]]

                # push, stun, deal damage and dash
                if deal or not kick_out:
                    # enemies which collide with the pushed target
                    if champion.target.y != kick_coords[0] and champion.target.x != kick_coords[1]:
                        line = field.line({'y': champion.target.y, 'x': champion.target.x},
                                          {'y': kick_coords[0], 'x': kick_coords[1]})
                        for l in line:
                            c = coords[l[0]][l[1]]
                            if c and c.team != champion.team and c.champion and c != champion.target:
                                c.add_que('change_stat', 0, None, 'stunned', True)
                                c.clear_que_stunned_removal()
                                c.add_que('change_stat',
                                          stats.ABILITY_SECONDARY_STUN_DURATION[champion.name][champion.stars], None,
                                          'stunned', False)
                                champion.spell(c, stats.ABILITY_DMG[champion.name][champion.stars] / 2)
                    if champion.target is None:
                        return
                    # push to corner
                    champion.target.move(kick_coords[0], kick_coords[1], True)
                    target_y = champion.target.y
                    target_x = champion.target.x

                    champion.target.add_que('change_stat', 0, None, 'stunned', True)
                    champion.target.clear_que_stunned_removal()
                    champion.target.add_que('change_stat', stats.ABILITY_STUN_DURATION[champion.name][champion.stars],
                                            None, 'stunned', False)

                    champion.spell(champion.target, stats.ABILITY_DMG[champion.name][champion.stars])

                    # dash (only if the pushed unit is alive and outside of lee's range)
                    # find a hex that's a neighbor of the pushed unit and as close as possible to lee
                    if (champion.target and field.distance({'y': target_y, 'x': target_x},
                                                           {'y': champion.y, 'x': champion.x}, False) > champion.range):
                        neighbors = field.find_neighbors(target_y, target_x)
                        for i, n in enumerate(neighbors):
                            d = field.distance({'y': n[0], 'x': n[1]}, {'y': champion.y, 'x': champion.x}, False)
                            neighbors[i].append(d)

                        neighbors = sorted(neighbors, key=lambda x: x[2])
                        for n in neighbors:
                            if not coords[n[0]][n[1]]:
                                champion.move(n[0], n[1], True)
                                apply_attack_cooldown(champion, halved=False)
                                break
                else:
                    champion.target.die()

            else:
                champion.target.die()


# this was aids


def lillia(champion):
    default_ability_calls(champion)
    target_amount = stats.ABILITY_TARGETS[champion.name][champion.stars]
    enemy_list = champion.enemy_team()
    enemies = []

    for i, e in enumerate(enemy_list):
        d = field.distance(champion, e, True)
        enemies.append([e, d, e.health])

    enemies = sorted(enemies, key=lambda x: (x[2]), reverse=True)
    if len(enemies) > target_amount:
        enemies = enemies[:target_amount]

    for e in enemies:
        e[0].add_que('change_stat', 0, None, 'stunned', True)
        e[0].clear_que_stunned_removal()
        e[0].add_que('change_stat', stats.ABILITY_STUN_DURATION[champion.name][champion.stars], None, 'stunned', False)

    # if two lillia ults by the same unit were casted while the last one's stuns are still active,
    # remove the hp checks from the que
    """que = champion.que_return()
     for q in que:
        if(q[1] is champion and q[0] == 'execute_function' and q[3][0] == lillia_ability): 
            que.remove(q)
    champion.que_replace(que) """

    champion.add_que('execute_function', -1, [lillia_ability, {'enemies': enemies, 'i': 0}])


def lillia_ability(champion, data):
    enemies = data['enemies']
    threshold = stats.ABILITY_STUN_STOP_DMG_THRESHOLD[champion.name][champion.stars]

    for e in enemies:
        difference = e[2] - e[0].health
        if difference > threshold:
            e[0].clear_que_stunned_removal()
            e[0].add_que('change_stat', 0, None, 'stunned', False)
            champion.spell(e[0], stats.ABILITY_DMG[champion.name][champion.stars])
            data['enemies'].remove(e)

    # call the same function every 50 milliseconds until the stuns are over
    # these function calls are overwritten in case lillia ults again
    if data['i'] * 50 < stats.ABILITY_STUN_DURATION[champion.name][champion.stars]:
        # 49ms because the que adds one more ms
        champion.add_que('execute_function', 49, [lillia_ability, {'enemies': enemies, 'i': data['i'] + 1}])


# this is a bit shit as well
def lissandra(champion):
    default_ability_calls(champion)
    enemy_list = champion.enemy_team()

    target = None
    for e in enemy_list:
        if not target or e.AD > target.AD:
            target = e

    dagger_path = field.line({'y': champion.y, 'x': champion.x}, {'y': target.y, 'x': target.x})

    dagger_target = None
    coords = field.coordinates
    for d in dagger_path:
        if 0 <= d[0] < 8 and 0 <= d[1] < 7:
            # print("IN LISSANDRA ABILITY")
            # print(d)
            c = coords[d[0]][d[1]]
            if c and c.team != champion.team and c.champion:
                dagger_target = c
                break
    if not dagger_target:
        dagger_target = champion.target

    # find the first three points of the cone (blue in 'lissandra_ult.png')
    primary_neighbors = field.find_neighbors(dagger_target.y, dagger_target.x, True)
    for i, p in enumerate(primary_neighbors):
        distance = field.distance({'y': p[0], 'x': p[1]}, {'y': champion.y, 'x': champion.x}, False)
        primary_neighbors[i].append(distance)
    primary_neighbors = sorted(primary_neighbors, key=lambda x: (x[2]))
    primary_neighbors = primary_neighbors[3:]
    for i, p in enumerate(primary_neighbors):
        primary_neighbors[i] = [p[0], p[1]]

    # finding the orange circle in 'lissandra_ult.png'
    # which of the just found hexes have two of the hexes as neighbors
    primary_neighbors_count = []
    for p in primary_neighbors:
        primary_neighbors_neighbors = field.find_neighbors(p[0], p[1], True)
        primary_neighbors_count.append([p[0], p[1], 0])
        for n in primary_neighbors_neighbors:
            if n in primary_neighbors:
                primary_neighbors_count[len(primary_neighbors_count) - 1][2] += 1

    primary_neighbors_count = sorted(primary_neighbors_count, key=lambda x: (x[2]), reverse=True)
    middle_hex = [primary_neighbors_count[0][0], primary_neighbors_count[0][1]]
    side_primary_neighbors = [[primary_neighbors_count[1][0], primary_neighbors_count[1][1]],
                              [primary_neighbors_count[2][0], primary_neighbors_count[2][1]]]

    # the dark red circle in 'lissandra_ult.png'
    # neighbor of the yellow circle and two distance away from the side primary neighbors
    middle_hex_neighbors = field.find_neighbors(middle_hex[0], middle_hex[1], True)
    for i, m in enumerate(middle_hex_neighbors):
        distance0 = field.distance({'y': m[0], 'x': m[1]},
                                   {'y': side_primary_neighbors[0][0], 'x': side_primary_neighbors[0][1]}, False)
        distance1 = field.distance({'y': m[0], 'x': m[1]},
                                   {'y': side_primary_neighbors[1][0], 'x': side_primary_neighbors[1][1]}, False)
        middle_hex_neighbors[i] = [m[0], m[1], distance0, distance1]

    secondary_middle_hex = list(filter(lambda x: (x[2] == 2 and x[3] == 2), middle_hex_neighbors))
    secondary_middle_hex = [secondary_middle_hex[0][0], secondary_middle_hex[0][1]]

    # now find all the hexes within two distance from the orange circle
    # that are also within two distance of the red circle

    middle_hex_two_distance = field.hexes_in_distance(middle_hex[0], middle_hex[1], 2)
    secondary_middle_hex_two_distance = field.hexes_in_distance(secondary_middle_hex[0], secondary_middle_hex[1], 2)

    cone = list(
        set(map(tuple, middle_hex_two_distance)).intersection(set(map(tuple, secondary_middle_hex_two_distance))))

    # now add the corners (black) of the cone by finding hexes
    # that are of 4 distance from the champion and three distance from the red circle
    four_from_champion = field.hexes_distance_away(champion.y, champion.x, 4)
    three_from_red = field.hexes_distance_away(secondary_middle_hex[0], secondary_middle_hex[1], 3)

    corners = list(set(map(tuple, four_from_champion)).intersection(set(map(tuple, three_from_red))))

    for c in corners:
        cone.append(c)

    # remove the dagger target since everyone in the cone will be taking secondary damage
    cone.remove(tuple((dagger_target.y, dagger_target.x)))

    champion.spell(dagger_target, stats.ABILITY_DMG[champion.name][champion.stars])

    coords = field.coordinates
    for c in cone:
        hex_data = coords[c[0]][c[1]]
        if hex_data and champion.team != hex_data.team and hex_data.champion:
            champion.spell(hex_data, stats.ABILITY_SECONDARY_DMG[champion.name][champion.stars])


lulu_targeted = []


def lulu(champion):
    default_ability_calls(champion)
    own_team = champion.own_team()
    own_team_hp = []

    for i, o in enumerate(own_team):
        own_team_hp.append([o, o.health / o.max_health])

    own_team_hp = sorted(own_team_hp, key=lambda x: x[1])

    for o in own_team_hp:
        targeted = len(list(filter(lambda x: (x[0] == champion and x[1] == o[0]), lulu_targeted)))
        hp_amount = stats.ABILITY_HEALTH_GAIN_TOTAL[champion.name][champion.stars] * champion.SP
        if targeted != 0:
            o[0].add_que('change_stat', -1, None, 'max_health', o[0].max_health + hp_amount)
        o[0].add_que('heal', -1, None, None, hp_amount)

        neighbors = field.enemies_in_distance(o[0], o[0].y, o[0].x, 1)
        for n in neighbors:
            n.add_que('change_stat', -1, None, 'stunned', True)
            n.clear_que_stunned_removal()
            n.add_que('change_stat', stats.ABILITY_STUN_DURATION[champion.name][champion.stars], None, 'stunned', False)
        lulu_targeted.append([champion, o[0]])

        break


def lux(champion):
    default_ability_calls(champion)
    if len(champion.enemy_team()) > 0:
        enemies = field.enemies_in_distance(champion, champion.y, champion.x, 20)
        distances = []
        for e in enemies:
            d = field.distance(champion, e, True)
            distances.append([e, d])
        distances = sorted(distances, key=lambda x: x[1], reverse=True)

        if len(distances) == 0:
            target = [champion.target.y, champion.target.x]
        else:
            target = distances[0][0]

        if target:
            try:
                path = field.line({'y': champion.y, 'x': champion.x}, {'y': target.y, 'x': target.x})
            except AttributeError or ValueError:
                path = field.line({'y': champion.y, 'x': champion.x}, {'y': 0, 'x': 0})
            coords = field.coordinates
            for p in path:
                if 0 <= p[0] < 8 and 0 <= p[1] < 7:
                    # print("IN LUX ABILITY")
                    # print(p)
                    c = coords[p[0]][p[1]]
                    if c and c.team != champion.team and c.champion:
                        c.add_que('change_stat', -1, None, 'stunned', True)
                        c.clear_que_stunned_removal()
                        c.add_que('change_stat', stats.ABILITY_STUN_DURATION[champion.name][champion.stars],
                                  None, 'stunned', False)
                        champion.spell(c, stats.ABILITY_DMG[champion.name][champion.stars])


def maokai(champion):
    default_ability_calls(champion)

    targeted_hexes = [[champion.target.y, champion.target.x]]

    # target all hexes adjacent to target that are 1 away from champion (red dots in 'maokai_ult.png')
    target_neighbors = field.find_neighbors(champion.target.y, champion.target.x, True)
    for i, t in enumerate(target_neighbors):
        d = field.distance({'y': champion.y, 'x': champion.x}, {'y': t[0], 'x': t[1]}, False)
        target_neighbors[i] = [t, d]
    target_neighbors = sorted(target_neighbors, key=lambda x: x[1])
    target_neighbors = list(filter(lambda x: (x[1] == 1), target_neighbors))

    # print(target_neighbors)
    # Bug happens if there are no neighbors or if it is outside the range of the map. Have yet to test.
    if target_neighbors:
        targeted_hexes.append(target_neighbors[0][0])
        if len(target_neighbors) > 1:
            targeted_hexes.append(target_neighbors[1][0])

        # also target two hexes behind the unit (red dashes in 'maokai_ult.png')
        # the last dashed hex is 3 away from the side neighbors and 3 away from the champion
        # NOT THE MOST ELEGANT WAY TO DO THIS:

        three_from_champion = field.hexes_distance_away(champion.y, champion.x, 3)
        three_from_n0 = field.hexes_distance_away(target_neighbors[0][0][0], target_neighbors[0][0][1], 3)
        three_away = list(set(map(tuple, three_from_champion)).intersection(set(map(tuple, three_from_n0))))
        if len(target_neighbors) > 1:
            three_from_n1 = field.hexes_distance_away(target_neighbors[1][0][0], target_neighbors[1][0][1], 3)
            three_away = list(set(map(tuple, three_away)).intersection(set(map(tuple, three_from_n1))))

        # in case the dash hexes are outside the map, let's just find them independently without drawing a line
        two_from_champion = field.hexes_distance_away(champion.y, champion.x, 2)
        two_from_n0 = field.hexes_distance_away(target_neighbors[0][0][0], target_neighbors[0][0][1], 2)
        two_away = list(set(map(tuple, two_from_champion)).intersection(set(map(tuple, two_from_n0))))
        if len(target_neighbors) > 1:
            two_from_n1 = field.hexes_distance_away(target_neighbors[1][0][0], target_neighbors[1][0][1], 2)
            two_away = list(set(map(tuple, two_away)).intersection(set(map(tuple, two_from_n1))))

        if len(three_away) > 0:
            targeted_hexes.append([three_away[0][0], three_away[0][1]])
        if len(two_away) > 0:
            targeted_hexes.append([two_away[0][0], two_away[0][1]])

    coords = field.coordinates
    for t in targeted_hexes:
        if 0 <= t[0] <= 7 and 0 <= t[1] <= 6:
            c = coords[t[0]][t[1]]
            if c and c.team != champion.team and c.champion:
                champion.spell(c, stats.ABILITY_DMG[champion.name][champion.stars])

                c.add_que('change_stat', -1, None, 'AS',
                          c.AS * stats.ABILITY_AS_DECREASE[champion.name][champion.stars])
                c.add_que('change_stat', stats.ABILITY_AS_CHANGE_LENGTH[champion.name], None, 'AS', None,
                          {'ezreal': stats.ABILITY_AS_DECREASE[champion.name][champion.stars]})


def morgana(champion):
    default_ability_calls(champion)

    champion.add_que('change_stat', -1, None, 'ability_active', True)
    champion.add_que('change_stat', stats.ABILITY_LENGTH[champion.name], None, 'ability_active', False)

    # between two random targets cast where it hits the most units
    enemies = champion.enemy_team()
    random.shuffle(enemies)

    circle0 = field.enemies_in_distance(champion, enemies[0].y, enemies[0].x, stats.ABILITY_RADIUS[champion.name])
    circle1 = []
    if len(enemies) > 1:
        circle1 = field.enemies_in_distance(champion, enemies[1].y, enemies[1].x, stats.ABILITY_RADIUS[champion.name])
    target = [enemies[0].y, enemies[0].x]

    if len(circle1) > len(circle0):
        target = [enemies[1].y, enemies[1].x]

    for i in range(0, stats.ABILITY_SLICES[champion.name]):
        current_ms = i * (stats.ABILITY_LENGTH[champion.name] / stats.ABILITY_SLICES[champion.name])
        champion.add_que('execute_function', current_ms, [morgana_ability, {'coordinates': target, 'ms': current_ms}])


morgana_MR_list = []


def morgana_ability(champion, data):
    global morgana_MR_list
    targets = field.enemies_in_distance(champion, data['coordinates'][0], data['coordinates'][1],
                                        stats.ABILITY_RADIUS[champion.name])

    for t in targets:
        # if that unit hasn't been targeted by this morgana yet
        if len(list(filter(lambda x: (x[0] == champion and x[1] == t), morgana_MR_list))) == 0:
            ms_left = stats.ABILITY_LENGTH[champion.name] - data['ms']

            t.print(' {} {} --> {}'.format('MR', t.MR, t.MR * stats.ABILITY_MR_DECREASE[champion.name]))
            t.MR *= stats.ABILITY_MR_DECREASE[champion.name]
            t.add_que('change_stat', ms_left, None, 'MR', None, {'morgana': stats.ABILITY_MR_DECREASE[champion.name]})

            morgana_MR_list.append([champion, t])

        champion.spell(t, stats.ABILITY_DMG[champion.name][champion.stars] / stats.ABILITY_SLICES[champion.name])

        ability_dmg = stats.ABILITY_DMG[champion.name][champion.stars] / stats.ABILITY_SLICES[champion.name]
        if t.MR >= 0:
            damage = ability_dmg * (100 / (100 + t.MR)) * champion.SP
        else:
            damage = ability_dmg * (2 - 100 / (100 - t.MR)) * champion.SP
        champion.add_que('heal', -1, None, None, damage * stats.ABILITY_HEAL_PER_DAMAGE[champion.name][champion.stars])

    # clear the list at the end of the last slice
    if (data['ms'] == stats.ABILITY_LENGTH[champion.name] - (
            stats.ABILITY_LENGTH[champion.name] / stats.ABILITY_SLICES[champion.name])):
        morgana_MR_list = list(filter(lambda x: (x[0] != champion), morgana_MR_list))


def nami(champion):
    default_ability_calls(champion)

    enemy_team = champion.enemy_team()
    enemy_team = sorted(enemy_team, key=lambda x: field.distance(champion, x, True))
    target = enemy_team[0]

    target.add_que('change_stat', -1, None, 'stunned', True)
    target.clear_que_stunned_removal()
    target.add_que('change_stat', stats.ABILITY_STUN_DURATION[champion.name][champion.stars], None, 'stunned', False)

    champion.spell(target, stats.ABILITY_DMG[champion.name][champion.stars])

    apply_attack_cooldown(champion)


def nidalee(champion):
    default_ability_calls(champion)

    enemies = champion.enemy_team()
    enemies_distance = []
    for i, e in enumerate(enemies):
        d = field.distance(champion, e, True)
        enemies_distance.append([e, d])

    enemies_distance = (sorted(enemies_distance, key=lambda x: x[1], reverse=True))[0]

    # don't take the champion's hex into calculations
    path = (field.line({'y': champion.y, 'x': champion.x}, {'y': enemies_distance[0].y, 'x': enemies_distance[0].x}))[
           1:]
    dmg = stats.ABILITY_DMG[champion.name][champion.stars]

    coords = field.coordinates
    for p in path:
        dmg *= (1 + stats.ABILITY_DAMAGE_ADDITION_PERCENTAGE[champion.name])
        if 0 < p[0] < 8 and 0 < p[1] < 7:
            c = coords[p[0]][p[1]]
            if c and c.team != champion.team and c.champion:
                champion.spell(c, dmg)
                break

    apply_attack_cooldown(champion)


def nunu(champion):
    default_ability_calls(champion)

    if champion.health > champion.target.health:
        true_dmg = stats.ABILITY_DMG[champion.name][champion.stars] * (
                1 + stats.ABILITY_DAMAGE_ADDITION_PERCENTAGE[champion.name])
        champion.spell(champion.target, 0, true_dmg)
    else:
        champion.spell(champion.target, stats.ABILITY_DMG[champion.name][champion.stars])

    apply_attack_cooldown(champion)


# add to the path any hexes that have three neighbors in the pyke's dashing path (x-marks in 'pyke_ult.png')
def pyke(champion):
    default_ability_calls(champion)
    champion.add_que('change_stat', -1, None, 'ability_active', True)
    champion.add_que('change_stat', stats.ABILITY_LENGTH[champion.name], None, 'ability_active', False)

    # find the enemy that's furthest away
    enemies = champion.enemy_team()
    enemies_distance = []
    for i, e in enumerate(enemies):
        d = field.distance(champion, e, True)
        enemies_distance.append([e, d])

    target = (sorted(enemies_distance, key=lambda x: x[1], reverse=True))[0][0]

    # get that enemy's neighbors to find the dash target
    target_neighbors = field.find_neighbors(target.y, target.x)
    target_neighbor_distances = []
    for n in target_neighbors:
        d = field.distance({'y': champion.y, 'x': champion.x}, {'y': n[0], 'x': n[1]}, False)
        target_neighbor_distances.append([n, d])
    target_neighbor_distances = sorted(target_neighbor_distances, key=lambda x: x[1], reverse=True)

    dash_target = None
    coords = field.coordinates
    for t in target_neighbor_distances:
        if not coords[t[0][0]][t[0][1]]:
            dash_target = t[0]
            break

    # problem: what if the target unit doesn't have free neighbor slots
    # kind of a shit solution but find a free hex within two hexes of the original target
    # then replace the original target with some enemy next to the new dash target
    if not dash_target:
        second_degree_neighbors = field.hexes_in_distance(target.y, target.x, 2)
        random.shuffle(second_degree_neighbors)
        for t in second_degree_neighbors:
            if not coords[t[0]][t[1]]:
                dash_target = t
                break

        if dash_target:
            dash_target_neighbors = field.find_neighbors(dash_target[0], dash_target[1])
            random.shuffle(dash_target_neighbors)
            for n in dash_target_neighbors:
                c = coords[n[0]][n[1]]
                if c and c.team != champion.team and c.champion:
                    target = c
                    break

    # if we STILL don't have a dash target, give back 90% of pyke's mana and move on
    if not dash_target:
        champion.add_que('change_stat', -1, None, 'mana', champion.maxmana * 0.9)

    else:
        # if the dash_target is further away than the targeted champion, set that hex to be the path's second end
        path_target = [target.y, target.x]
        distance_to_dash_target = field.distance({'y': champion.y, 'x': champion.x},
                                                 {'y': dash_target[0], 'x': dash_target[1]}, False)
        distance_to_target = field.distance(champion, target, True)
        if distance_to_dash_target > distance_to_target:
            path_target = dash_target

        path = field.line({'y': champion.y, 'x': champion.x}, {'y': path_target[0], 'x': path_target[1]})
        extra_path = []

        # find all the hexes that have three neighbors which are in the path
        all_hexes = field.hexes_in_distance(0, 0, 20)
        for a in all_hexes:
            a_neighbors = field.find_neighbors(a[0], a[1])
            count = 0
            for n in a_neighbors:
                if n in path:
                    count += 1

            if count == 3:
                extra_path.append(a)

        for e in extra_path:
            path.append(e)

        champion.move(dash_target[0], dash_target[1], True)
        champion.add_que('execute_function', stats.ABILITY_LENGTH[champion.name], [pyke_ability, {'path': path}])


def pyke_ability(champion, data):
    coords = field.coordinates
    for p in data['path']:
        if 0 <= p[0] <= 7 and 0 <= p[1] <= 6:
            c = coords[p[0]][p[1]]
            if c and c.team != champion.team and c.champion:
                c.add_que('change_stat', -1, None, 'stunned', True)
                c.clear_que_stunned_removal()
                c.add_que('change_stat', stats.ABILITY_STUN_DURATION[champion.name][champion.stars], None, 'stunned', False)
                champion.spell(c, stats.ABILITY_DMG[champion.name][champion.stars])


riven_counter = []
riven_identifier_list = []


def riven(champion):
    if riven_helper(champion, {}):

        global riven_counter
        default_ability_calls(champion)
        # riven_counter needs to sustain multiple rivens on the field,
        # so there's gonna be every riven's data on the same array
        found = False
        index = -1
        if len(riven_counter) > 0:
            for i, v in enumerate(riven_counter):
                if v[0] == champion:
                    found = True
                    index = i
                    break

        if found:
            riven_counter[index][1] += 1
        else:
            riven_counter.append([champion, 1])
            index = len(riven_counter) - 1

        if len(champion.enemy_team()) > 0:
            if not champion.target:
                field.find_target(champion)
            # target neighbors and their distances to champion
            target_neighbors = field.find_neighbors(champion.target.y, champion.target.x, True)
            for i, t in enumerate(target_neighbors):
                d = field.distance({'y': champion.y, 'x': champion.x}, {'y': t[0], 'x': t[1]}, False)
                target_neighbors[i] = [t, d]

            coords = field.coordinates

            # the wave of damage
            if riven_counter[index][1] == 3:

                # target_hex = [[champion.target.y, champion.target.x]]
                corner_neighbors = []

                # target all hexes adjacent to target that are 1 away from champion (orange circles in 'riven_ult.png')
                target_neighbors = sorted(target_neighbors, key=lambda x: x[1])
                target_neighbors = list(filter(lambda x: (x[1] == 1), target_neighbors))

                # bugged out once and couldn't repeat.
                if len(target_neighbors) > 1:
                    corner_neighbors.append(target_neighbors[0][0])
                    corner_neighbors.append(target_neighbors[1][0])

                    # find the red circle in the pic
                    two_from_champion = field.hexes_distance_away(champion.y, champion.x, 2, True)
                    two_from_n0 = field.hexes_distance_away(corner_neighbors[0][0], corner_neighbors[0][1], 2, True)
                    two_from_n1 = field.hexes_distance_away(corner_neighbors[1][0], corner_neighbors[1][1], 2, True)
                    two_away = list(set(map(tuple, two_from_champion)).intersection(set(map(tuple, two_from_n0))))
                    slash_hexes = []
                    if len(two_away) > 0:
                        two_away = list(set(map(tuple, two_away)).intersection(set(map(tuple, two_from_n1))))[0]
                        slash_hexes = field.hexes_in_distance(two_away[0], two_away[1], 1)

                    slash_hexes.append(corner_neighbors[0])
                    slash_hexes.append(corner_neighbors[1])

                    for s in slash_hexes:
                        if s[0] < 8 and s[1] < 7:
                            c = coords[s[0]][s[1]]
                            if c and c.team != champion.team and c.champion:
                                champion.spell(c, stats.ABILITY_SECONDARY_DMG[champion.name][champion.stars])

                riven_counter[index][1] = 0

            # first dash to a furthest free neighboring hex of the target
            # then get a shield / reset the shield
            # damage ALL neighboring enemies
            else:

                random.shuffle(target_neighbors)
                target_neighbors = sorted(target_neighbors, key=lambda x: x[1], reverse=True)

                for t in target_neighbors:
                    t = t[0]
                    if t[0] >= 0 and t[0] <= 7 and t[1] >= 0 and t[1] <= 6:
                        c = coords[t[0]][t[1]]
                        if not c:
                            champion.move(t[0], t[1], True)
                            break

                shield_identifier = champion_functions.MILLIS() * stats.SHIELD_AMOUNT[champion.name][
                    champion.stars] * champion.SP
                shield_amount = stats.SHIELD_AMOUNT[champion.name][champion.stars] * champion.SP
                if shield_identifier in riven_identifier_list:
                    champion.add_que('shield', 0, None, None,
                                     {'amount': shield_amount + 0.001, 'identifier': shield_identifier,
                                      'applier': champion, 'original_amount': shield_amount + 0.001},
                                     {'increase': True})

                riven_identifier_list.append(shield_identifier)

                champion.add_que('shield', 0, None, None,
                                 {'amount': shield_amount, 'identifier': shield_identifier, 'applier': champion,
                                  'original_amount': shield_amount}, {'increase': True})

                for t in field.find_neighbors(champion.y, champion.x):
                    c = coords[t[0]][t[1]]
                    if c and c.team != champion.team and c.champion:
                        champion.spell(c, stats.ABILITY_DMG[champion.name][champion.stars])

            apply_attack_cooldown(champion)

    else:
        champion.attack()


# check if she has old shields left
def riven_helper(champion, data):
    if len(champion.shields) == 0: return True
    own_shields = list(filter(lambda x: (
            x['applier'] == champion and x['original_amount'] ==
            stats.SHIELD_AMOUNT[champion.name][champion.stars] * champion.SP), champion.shields))
    if len(own_shields) == 0:
        return True
    else:
        return False


def sejuani(champion):
    default_ability_calls(champion)
    champion.add_que('change_stat', -1, None, 'ability_active', True)
    champion.add_que('change_stat', stats.ABILITY_LENGTH[champion.name], None, 'ability_active', False)

    enemies = champion.enemy_team()
    enemies_distance = []

    for i, e in enumerate(enemies):
        d = field.distance(champion, e, True)
        enemies_distance.append([e, d])
    enemies_distance = sorted(enemies_distance, key=lambda x: x[1])

    target = enemies_distance[0][0]

    champion.add_que('execute_function', stats.ABILITY_LENGTH[champion.name],
                     [sejuani_ability, {'target': [target.y, target.x]}])


def sejuani_ability(champion, data):
    target_hexes = field.hexes_in_distance(data['target'][0], data['target'][1], stats.ABILITY_RADIUS[champion.name])
    coords = field.coordinates
    for t in target_hexes:
        c = coords[t[0]][t[1]]
        if (c and c.team != champion.team and c.champion):
            c.add_que('change_stat', -1, None, 'stunned', True)
            c.clear_que_stunned_removal()
            c.add_que('change_stat', stats.ABILITY_STUN_DURATION[champion.name][champion.stars], None, 'stunned', False)

            champion.spell(c, stats.ABILITY_DMG[champion.name][champion.stars])


def sett(champion):
    default_ability_calls(champion)
    if len(champion.enemy_team()) > 0:
        if not champion.target:
            field.find_target(champion)

        # add the slamming time (assumed 1000ms)
        champion.add_que('change_stat', -1, None, 'ability_active', True)
        champion.add_que('change_stat', stats.ABILITY_LENGTH[champion.name], None, 'ability_active', False)

        # slams the target forward ('sett_ult.png') and takes the target's old place.
        # if all of the possible hexes are taken, don't move but still slam.

        smash_targets = []

        target_neighbors = field.find_neighbors(champion.target.y, champion.target.x, False)
        for i, t in enumerate(target_neighbors):
            d = field.distance({'y': champion.y, 'x': champion.x}, {'y': t[0], 'x': t[1]}, False)
            target_neighbors[i] = [t, d]

        # print("IN SETT ABILITY")
        # print(target_neighbors)
        one_distance_neighbors = list(filter(lambda x: (x[1] == 1), target_neighbors))
        two_distance_neighbors = list(filter(lambda x: (x[1] == 2), target_neighbors))

        if one_distance_neighbors and two_distance_neighbors:
            # finding the preferred smash target hex (two away from the side neighbors [one distance from champion])
            # and two away from the champion itself.
            two_from_champion = field.hexes_distance_away(champion.y, champion.x, 2, True)
            two_from_n0 = field.hexes_distance_away(one_distance_neighbors[0][0][0],
                                                    one_distance_neighbors[0][0][1], 2, True)
            if len(one_distance_neighbors) > 1:
                two_from_n1 = field.hexes_distance_away(one_distance_neighbors[1][0][0],
                                                        one_distance_neighbors[1][0][1], 2, True)
                two_away = list(set(map(tuple, two_from_n0)).intersection(set(map(tuple, two_from_n1))))
            else:
                two_away = two_from_n0
            two_away = list(set(map(tuple, two_from_champion)).intersection(set(map(tuple, two_away))))
            # add into 'smash_targets' in the next order: first we have the preferred hex (solid red line)
            # then the next two will be the secondary smash targets
            if two_away:
                smash_targets.append([two_away[0][0], two_away[0][1]])
                two_distance_neighbors = list(filter(lambda x: (x[0] != smash_targets[0]), two_distance_neighbors))
            else:
                two_distance_neighbors = list(filter(lambda x: (x[0]), two_distance_neighbors))
            smash_targets.append(two_distance_neighbors[0][0])
            if len(two_distance_neighbors) > 1:
                smash_targets.append(two_distance_neighbors[1][0])

            # go through the possibilities and try to find a free hex
            free_hex = None
            coords = field.coordinates
            for s in smash_targets:
                if s[0] >= 0 and s[0] <= 7 and s[1] >= 0 and s[1] <= 6:
                    c = coords[s[0]][s[1]]
                    if not c:
                        free_hex = s
                        break

            # move to the target if there's a fee hex available
            if free_hex:
                sett_target = [champion.target.y, champion.target.x]
                champion.target.move(free_hex[0], free_hex[1], True)
                champion.move(sett_target[0], sett_target[1], True)

        # slamming area
        damaged_hexes = field.hexes_in_distance(champion.target.y, champion.target.x, 2)
        damaged_hexes = list(filter(lambda x: x != [champion.target.y, champion.target.x], damaged_hexes))

        target_dmg = champion.target.max_health * stats.ABILITY_DMG[champion.name][champion.stars]
        secondary_target_dmg = champion.target.max_health * stats.ABILITY_SECONDARY_DMG[champion.name][champion.stars]
        champion.spell(champion.target, target_dmg)
        coords = field.coordinates
        for d in damaged_hexes:
            c = coords[d[0]][d[1]]
            if c and c.team != champion.team and c.champion:
                champion.spell(c, secondary_target_dmg)


def shen(champion):
    default_ability_calls(champion)

    # target neighbors and their distances to champion
    target_neighbors = field.find_neighbors(champion.target.y, champion.target.x)
    for i, t in enumerate(target_neighbors):
        d = field.distance({'y': champion.y, 'x': champion.x}, {'y': t[0], 'x': t[1]}, False)
        target_neighbors[i] = [t, d]

    random.shuffle(target_neighbors)
    target_neighbors = sorted(target_neighbors, key=lambda x: x[1], reverse=True)

    # dash to a neighbor of the target that's as far as possible from shen
    # if there are no free hexes, stay at the current hex
    dash_target = [champion.y, champion.x]

    coords = field.coordinates
    for t in target_neighbors:
        t = t[0]
        c = coords[t[0]][t[1]]
        if not c:
            dash_target = t
            break

    if dash_target != [champion.y, champion.x]:
        champion.move(dash_target[0], dash_target[1], True)

    shield_identifier = champion_functions.MILLIS() * stats.SHIELD_AMOUNT[champion.name][champion.stars] * champion.SP
    shield_amount = stats.SHIELD_AMOUNT[champion.name][champion.stars] * champion.SP
    champion.add_que('shield', 0, None, None,
                     {'amount': shield_amount, 'identifier': shield_identifier, 'applier': champion,
                      'original_amount': shield_amount},
                     {'increase': True, 'expires': stats.ABILITY_LENGTH[champion.name][champion.stars]})

    neighbor_enemies = field.enemies_in_distance(champion, champion.y, champion.x, 1)

    for n in neighbor_enemies:
        old_target = n.target
        n.add_que('change_target', -1, None, None, champion)
        n.add_que('change_target', stats.ABILITY_LENGTH[champion.name][champion.stars], None, None, old_target)


def sylas(champion):
    default_ability_calls(champion)

    smash_line = []

    # print("SYLAS ABILITY FIELD LINE")
    # print(field.line)
    if champion.target:
        targetLine = field.line({'y': champion.y, 'x': champion.x}, {'y': champion.target.y, 'x': champion.target.x})
        if len(targetLine) > 1:
            first_hex = targetLine[1]

            neighbors = field.find_neighbors(champion.y, champion.x, True)
            primary_neighbors = []
            for p in neighbors:
                d_champion = field.distance({'y': champion.y, 'x': champion.x}, {'y': p[0], 'x': p[1]}, False)
                d_first_hex = field.distance({'y': first_hex[0], 'x': first_hex[1]}, {'y': p[0], 'x': p[1]}, False)
                if d_champion == 1 and d_first_hex == 1:
                    primary_neighbors.append(p)

            smash_line.append(first_hex)
            smash_line.append(sylas_ability(champion, {'hexes': primary_neighbors, 'distance': 2}))
            smash_line.append(sylas_ability(champion, {'hexes': primary_neighbors, 'distance': 3}))

            # print("SYLAS ABILITY SMASH LINE")
            # print(smash_line)
            coords = field.coordinates
            for s in smash_line:
                # print("IN SYLAS ABILITY")
                # print(coords)
                if coords and s[0] < 8 and s[1] < 7:
                    c = coords[s[0]][s[1]]
                    if c and c.team != champion.team and c.champion:

                        # increase next spell mana cost of every enemy
                        # in the line by 33% = reduce mana by 33% of maxmana
                        if not c.mana_cost_increased and c.maxmana > 0:
                            mana_reduce_amount = c.maxmana * stats.ABILITY_MANA_REQUIREMENT_INCREASEMENT[champion.name]
                            start_value = c.mana
                            c.mana -= mana_reduce_amount
                            c.print(' {} {} --> {}'.format('mana', round(start_value, 1), round(c.mana, 1)))
                            c.add_que('change_stat', -1, None, 'mana_cost_increased', True)

                        champion.spell(c, stats.ABILITY_DMG[champion.name][champion.stars])


# get the hexes to the smash lane
def sylas_ability(champion, data):
    distance = data['distance']
    hexes = data['hexes']

    x_from_champion = field.hexes_distance_away(champion.y, champion.x, distance, True)
    x_from_n0 = field.hexes_distance_away(hexes[0][0], hexes[0][1], distance, True)
    x_from_n1 = field.hexes_distance_away(hexes[1][0], hexes[1][1], distance, True)
    x_away = list(set(map(tuple, x_from_n0)).intersection(set(map(tuple, x_from_n1))))
    x_away = list(set(map(tuple, x_from_champion)).intersection(set(map(tuple, x_away))))

    return [x_away[0][0], x_away[0][1]]


def talon(champion):
    #
    # 1. deal spell dmg
    # 2. attack with additional damage
    # 3. if kills the target
    #   fill mana
    #   apply 1000ms ability_active delay
    #   immune during those 1000ms
    #   teleport next to an enemy unit which has dealt the most dmg
    #   change talon's target to that unit

    default_ability_calls(champion)

    if not champion.target:
        field.find_target(champion)
    original_target = champion.target

    champion.spell(champion.target, stats.ABILITY_DMG[champion.name][champion.stars])

    # if the target dies to the spell dmg, 'champion.target' will yield 'None'
    if champion.target:
        bonus_dmg = champion.AD * (stats.ABILITY_DAMAGE_MULTIPLIER[champion.name][champion.stars] - 1)
        champion.attack(bonus_dmg)

    if not champion.target or (champion.target != original_target):
        next_target = None

        # find an enemy who has the lowest armor
        enemy_team = champion.enemy_team()
        enemies = []
        for e in enemy_team:
            enemies.append([e, e.armor])
        enemies = sorted(enemies, key=lambda x: x[1])
        if len(enemies) > 0:
            next_target = enemies[0][0]
            champion.add_que('change_stat', -1, None, 'ability_active', True)
            # champion.add_que('change_stat', -1, None, 'immune', True)
            champion.add_que('change_stat', stats.ABILITY_LENGTH[champion.name], None, 'ability_active', False)
            # champion.add_que('change_stat', stats.ABILITY_LENGTH[champion.name], None, 'immune', False)
            # champion.add_que('change_stat', stats.ABILITY_LENGTH[champion.name], None, 'mana', champion.maxmana)

            # make sure that talon doesnt try to move while we simulate the jump
            champion.idle = False
            champion.clear_que_idle()
            champion.add_que('clear_idle', stats.ABILITY_LENGTH[champion.name])

            champion.add_que('execute_function', stats.ABILITY_LENGTH[champion.name],
                             [talon_ability, {'target': next_target}])


def talon_ability(champion, data):
    if data['target'].health > 0:
        target_neighbors = field.find_neighbors(data['target'].y, data['target'].x)
        random.shuffle(target_neighbors)
        jump_target = None
        coords = field.coordinates
        for t in target_neighbors:
            c = coords[t[0]][t[1]]
            if not c:
                jump_target = t
                break

        if jump_target:
            champion.print(' leaps')
            for e in champion.enemy_team():
                if e.target == champion:
                    field.find_target(e)

            champion.move(jump_target[0], jump_target[1], True)
            champion.clear_que_idle()
            champion.add_que('clear_idle', 0)
            champion.add_que('change_target', 0, None, None, data['target'])


def teemo(champion):
    default_ability_calls(champion)

    enemy_team = champion.enemy_team()
    enemies = []
    for e in enemy_team:
        enemies.append([e, e.AS])

    enemies = sorted(enemies, key=lambda x: x[1], reverse=True)
    target = enemies[0][0]

    targets = field.enemies_in_distance(champion, target.y, target.x, 1)

    # if two teemo ults will be active at the same time, just fill the spell deals back to 6 (teemo's ability slices)
    # if teemo is a mage, just let him double cast and deal the full damage
    if not ((origin_class.get_origin_class_tier(champion.team, 'mage') > 0
             and origin_class.is_trait(champion, 'mage'))):
        for i in range(0, 5):
            que = champion.que_return()
            for q in que:
                for t in targets:
                    if q[1] is champion and q[0] == 'execute_function' \
                            and q[3][0] == teemo_ability and 'target' in q[3][1] and q[3][1]['target'] == t:
                        que.remove(q)
            champion.que_replace(que)

    slice_length = stats.ABILITY_BLIND_DURATION[champion.name][champion.stars] / stats.ABILITY_SLICES[champion.name]
    for i in range(1, stats.ABILITY_SLICES[champion.name] + 1):
        for t in targets:
            champion.add_que('execute_function', i * slice_length, [teemo_ability, {'target': t}])

    for t in targets:
        t.clear_que_blinded_removal()
        t.add_que('change_stat', -1, None, 'blinded', True)
        t.add_que('change_stat', stats.ABILITY_BLIND_DURATION[champion.name][champion.stars], None, 'blinded', False)


def teemo_ability(champion, data):
    target = data['target']
    if target and target.health > 0:
        dmg = stats.ABILITY_DMG[champion.name][champion.stars] / stats.ABILITY_SLICES[champion.name]
        champion.spell(target, dmg)


def thresh(champion):
    default_ability_calls(champion)
    own_team = champion.own_team()
    teammates = []
    for o in own_team:
        teammates.append([o, o.health / o.max_health])

    if len(teammates) == 0:
        return
    teammates = sorted(teammates, key=lambda x: x[1])
    target = teammates[0][0]

    target_neighbors = field.find_neighbors(target.y, target.x)

    shielded_allies = [target]

    coords = field.coordinates
    for t in target_neighbors:
        c = coords[t[0]][t[1]]
        if c and c.team == champion.team and c.champion:
            shielded_allies.append(c)

    for a in shielded_allies:
        for i in range(0, 3):
            for s in a.shields:
                if s['applier'] == champion:
                    shield_before = a.shield_amount()
                    a.shields.remove(s)
                    a.print(' {} {} --> {}'.format('shield', ceil(shield_before), ceil(a.shield_amount())))
                    break

        identifier = champion_functions.MILLIS() * stats.SHIELD_AMOUNT[champion.name][champion.stars] * champion.SP
        shield_amount = stats.SHIELD_AMOUNT[champion.name][champion.stars] * champion.SP
        a.add_que('shield', -1, None, None, {'amount': shield_amount, 'identifier': identifier, 'applier': champion,
                                             'original_amount': shield_amount},
                  {'increase': True, 'expires': stats.SHIELD_LENGTH[champion.name]})


# Time to add a few comments because I need to understand what is happening here.
def twistedfate(champion):
    default_ability_calls(champion)

    # middle line
    # using the rectangle function which kinda sucks for something like this.
    # sometimes it misses the target be cause of rng (check 'twistedfate_ult'),
    # so fixing it by changing the orange path to the blue one
    if champion.target is None:
        return
    line = field.rectangle_from_champion_to_wall_behind_target(champion, 1, champion.target.y, champion.target.x)[0]
    distance = field.distance(champion, champion.target, True)
    if int(distance) >= len(line):
        distance = len(line) - 1
        # print("Changing distance to {}".format(distance))
    if distance != -1:
        line[int(distance)] = [champion.target.y, champion.target.x]

        # find the first hex of the side paths
        neighbors = field.find_neighbors(champion.y, champion.x)
        for i, n in enumerate(neighbors):
            d = field.distance({'y': n[0], 'x': n[1]}, {'y': champion.target.y, 'x': champion.target.x}, False)
            neighbors[i] = [n, d]

        neighbors = sorted(neighbors, key=lambda x: x[1])
        neighbors = list(filter(lambda x: x[0] != [champion.target.y, champion.target.x], neighbors))

        # print("In TF Ability")
        # print(neighbors)
        if len(neighbors) > 1 and neighbors[0][1] != neighbors[1][1]:
            neighbors = neighbors[1:]

        # print("IN TWISTED FATE ABILITY - Neighbors")
        # print(neighbors)
        if neighbors:
            side_line0 = twistedfate_ability(champion, {'c': [neighbors[0][0][0], neighbors[0][0][1]]})
            # Experiencing a bug if tf is on the side of the board and
            # one of the lines of cards starts off of the board.
            if len(neighbors) > 1:
                side_line1 = twistedfate_ability(champion, {'c': [neighbors[1][0][0], neighbors[1][0][1]]})

        all_hit_hexes = [line]
        if neighbors:
            all_hit_hexes.append(side_line0)
            if len(neighbors) > 1:
                all_hit_hexes.append(side_line1)

        already_targeted = []
        coords = field.coordinates
        for a in all_hit_hexes:
            for h in a:
                c = coords[h[0]][h[1]]
                if c and c.team != champion.team and c.champion and c not in already_targeted:
                    already_targeted.append(c)
                    champion.spell(c, stats.ABILITY_DMG[champion.name][champion.stars])


# get a straight line for the cards to fly
def twistedfate_ability(champion, data):
    c = data['c']

    y_diff = c[0] - champion.y
    x_diff = 0
    # find the change in x (always +1 or -1)
    if champion.y % 2 == 0:
        if y_diff != 0:
            if c[1] > champion.x:
                x_diff = 1
            if c[1] == champion.x:
                x_diff = -1
        else:
            x_diff = c[1] - champion.x

    if champion.y % 2 == 1:
        if y_diff != 0:
            if c[1] == champion.x:
                x_diff = 1
            if c[1] < champion.x:
                x_diff = -1
        else:
            x_diff = c[1] - champion.x

    line = []

    # flight path
    # the x coordinate behaves differently depending on if the y coordinate is odd or even
    x = champion.x
    if y_diff > 0:
        for i in range(champion.y, 8):
            if x_diff == 1 and i % 2 == 1 and i != champion.y:
                x += x_diff
            if x_diff == -1 and i % 2 == 0 and i != champion.y:
                x += x_diff
            line.append([i, x])

    if y_diff < 0:
        for i in range(champion.y, -1, -1):
            if x_diff == -1 and i % 2 == 0 and i != champion.y:
                x += x_diff
            if x_diff == 1 and i % 2 == 1 and i != champion.y:
                x += x_diff
            line.append([i, x])

    if y_diff == 0:
        end_x = -1
        if x_diff == 1:
            end_x = 7
        for i in range(champion.x, end_x, x_diff):
            line.append([champion.y, i])

    # drop off possible coordinates outside the map
    for i in range(0, 5):
        for l in line:
            if (l[0] < 0 or l[0] > 7 or l[1] < 0 or l[1] > 6):
                line.remove(l)

    return (line)


def veigar(champion):
    default_ability_calls(champion)

    enemy_team = champion.enemy_team()
    enemies = []
    for e in enemy_team:
        enemies.append([e, e.health])

    enemies = sorted(enemies, key=lambda x: x[1])
    target = enemies[0][0]

    champion.spell(target, stats.ABILITY_DMG[champion.name][champion.stars])

    # making the change here so the change goes through before mage's second cast
    if target.health <= 0:
        start_value = champion.SP
        champion.SP += stats.ABILITY_SP_GAIN[champion.name][champion.stars]
        champion.print(' {} {} --> {}'.format('SP', round(start_value, 2), round(champion.SP, 2)))


vi_armor_list = []


def vi(champion):
    default_ability_calls(champion)
    target = champion.target
    distance = field.distance(champion, target, True)

    target_neighbors = field.find_neighbors(target.y, target.x, True)
    for i, t in enumerate(target_neighbors):
        d = field.distance({'y': champion.y, 'x': champion.x}, {'y': t[0], 'x': t[1]}, False)
        target_neighbors[i] = [t, d]

    side_primary_neighbors = list(filter(lambda x: x[1] == 1, target_neighbors))

    # print("In Vi's Ability")
    # print(side_primary_neighbors)
    if side_primary_neighbors and len(side_primary_neighbors) > 1 and len(side_primary_neighbors[0]) > 1 \
            and len(side_primary_neighbors[1]) > 1:
        x_from_champion = field.hexes_distance_away(champion.y, champion.x, distance + 1, True)
        x_from_n0 = field.hexes_distance_away(side_primary_neighbors[0][0][0], side_primary_neighbors[0][0][1], 2, True)
        x_from_n1 = field.hexes_distance_away(side_primary_neighbors[1][0][0], side_primary_neighbors[1][0][1], 2, True)
        x_away = list(set(map(tuple, x_from_n0)).intersection(set(map(tuple, x_from_n1))))
        x_away = list(set(map(tuple, x_from_champion)).intersection(set(map(tuple, x_away))))

        affected_hexes = field.hexes_in_distance(x_away[0][0], x_away[0][1], 1)

        coords = field.coordinates
        for a in affected_hexes:
            c = coords[a[0]][a[1]]
            if c and c.team != champion.team and c.champion:

                # to make sure there are no double reductions,
                # check how long ago the target's armor was reduced last time by this vi
                # the armor list elements are of syntax: [reducer, target, milliseconds_when_last_reduced]
                # search for target entries
                armor_history = (list(filter(lambda x: x[1] == c, vi_armor_list)))
                can_be_changed = False
                for ar in armor_history:
                    # there can be multiple vis, so make sure that this entry was done by the current vi
                    if ar[0] == champion:

                        # amount of ms when reduced last time
                        diff = champion_functions.MILLIS() - ar[2]
                        # if more than reduction length, allow new reduction
                        if diff > stats.ABILITY_LENGTH[champion.name]:
                            can_be_changed = True

                if len(armor_history) == 0:
                    can_be_changed = True

                if can_be_changed:
                    c.print(' {} {} --> {}'.format('armor', c.armor,
                                                   c.armor * stats.ABILITY_ARMOR_DECREASE[champion.name][
                                                       champion.stars]))
                    c.armor *= stats.ABILITY_ARMOR_DECREASE[champion.name][champion.stars]
                    c.clear_que_armor_removal()
                    c.add_que('change_stat', stats.ABILITY_LENGTH[champion.name], None, 'armor', None,
                              {'vi': stats.ABILITY_ARMOR_DECREASE[champion.name][champion.stars]})

                    vi_armor_list.append([champion, c, champion_functions.MILLIS()])

                champion.spell(c, stats.ABILITY_DMG[champion.name][champion.stars])


def warwick(champion):
    default_ability_calls(champion)
    champion.add_que('change_stat', -1, None, 'ability_active', True)
    champion.add_que('change_stat', -1, None, 'lifesteal',
                     champion.lifesteal + stats.LIFESTEAL[champion.name][champion.stars])
    champion.add_que('change_stat', -1, None, 'movement_delay', stats.MOVEMENTDELAY[champion.name])

    new_as = champion.AS + ((stats.ABILITY_AS_GAIN[champion.name][champion.stars] - 1) * champion.SP + 1)
    champion.add_que('change_stat', -1, None, 'AS', new_as)


def wukong(champion):
    default_ability_calls(champion)

    target = champion.target
    target.add_que('change_stat', -1, None, 'stunned', True)
    target.clear_que_stunned_removal()
    target.add_que('change_stat', stats.ABILITY_STUN_DURATION[champion.name][champion.stars], None, 'stunned', False)

    bonus_dmg = (champion.AD * (stats.ABILITY_DAMAGE_MULTIPLIER[champion.name][champion.stars] * champion.SP - 1))
    champion.attack(bonus_dmg)


def xinzhao(champion):
    default_ability_calls(champion)

    enemies = field.enemies_in_distance(champion, champion.y, champion.x, 1)
    bonus_dmg = champion.AD * (stats.ABILITY_DAMAGE_MULTIPLIER[champion.name][champion.stars] * champion.SP - 1)
    for e in enemies:
        champion.attack(bonus_dmg, e)

    champion.add_que('change_stat', -1, None, 'armor',
                     champion.armor + stats.ABILITY_ARMOR_MR_INCREASE[champion.name][champion.stars])
    champion.add_que('change_stat', -1, None, 'MR',
                     champion.MR + stats.ABILITY_ARMOR_MR_INCREASE[champion.name][champion.stars])


def yasuo(champion):
    default_ability_calls(champion)

    hexes = yasuo_ability(champion, {})
    bonus_dmg = champion.AD * (stats.ABILITY_DAMAGE_MULTIPLIER[champion.name][champion.stars] * champion.SP - 1)

    if (len(hexes) > 0):
        target1 = hexes[0][0][1]
        target2 = hexes[0][0][2]
        if ([champion.y, champion.x] != hexes[0][0][0]):
            champion.move(hexes[0][0][0][0], hexes[0][0][0][1], True)
            champion.idle = True
            champion.clear_que_idle()
        champion.attack(bonus_dmg, target1)
        champion.attack(bonus_dmg, target2)


    else:
        champion.attack(bonus_dmg, champion.target)
    apply_attack_cooldown(champion)


# find all the possible hexes where yasuo can strike two enemies
# this is a bit tricky
#   go through all free hexes
#   find hexes has an enemy neighbor
#       then find the commong neighbors of the hex and the enemy
#           find the hex that has a distance of two to the original hex and both neighbors
#               if this hex is an enemy, we can slash from here
#               add to list and sort the list by distances to yasuo
def yasuo_ability(champion, data):
    hexes = field.hexes_in_distance(0, 0, 20)
    coords = field.coordinates

    possible_hexes = []

    for h in hexes:
        if (h == [champion.y, champion.x] or not coords[h[0]][h[1]]):
            neighbors = field.find_neighbors(h[0], h[1])
            for n in neighbors:
                c = coords[n[0]][n[1]]
                if (c and c.team != champion.team and c.champion):
                    c_neighbors = field.find_neighbors(c.y, c.x)
                    one_away = list(set(map(tuple, c_neighbors)).intersection(set(map(tuple, neighbors))))

                    if (len(one_away) == 2):
                        x_from_champion = field.hexes_distance_away(h[0], h[1], 2, False)
                        x_from_n0 = field.hexes_distance_away(one_away[0][0], one_away[0][1], 2, False)
                        x_from_n1 = field.hexes_distance_away(one_away[1][0], one_away[1][1], 2, False)
                        x_away = list(set(map(tuple, x_from_n0)).intersection(set(map(tuple, x_from_n1))))
                        x_away = list(set(map(tuple, x_from_champion)).intersection(set(map(tuple, x_away))))
                        if (len(x_away) > 0):
                            x_away = x_away[0]
                            c2 = coords[x_away[0]][x_away[1]]
                            if (c2 and c2.team != champion.team and c2.champion):
                                possible_hexes.append([h, c, c2])

    for i, p in enumerate(possible_hexes):
        distance = field.distance({'y': champion.y, 'x': champion.x}, {'y': p[0][0], 'x': p[0][1]}, False)
        possible_hexes[i] = [p, distance]
    possible_hexes = sorted(possible_hexes, key=lambda x: x[1])
    return (possible_hexes)


yone_list = []
yone_checking = False


# welcome to the loop city
# the sir lord mayor is named 'for'
# dude's a dick tho
def yone(champion):
    global yone_list
    global yone_checking
    default_ability_calls(champion)

    if (not yone_checking):
        champion.add_que('execute_function', 0, [yone_ability, {'loop': True}])
        yone_checking = True

    coords = field.coordinates

    # Seal Fate
    if (champion.maxmana == stats.MAXMANA[champion.name]):
        if (not champion.target): field.find_target(champion)
        path = field.rectangle_from_champion_to_wall_behind_target(champion, stats.ABILITY_RADIUS[champion.name],
                                                                   champion.target.y, champion.target.x)
        for i, p in enumerate(path):
            if (len(p) > 5): path[i] = p[:5]

        middle_line = floor(stats.ABILITY_RADIUS[champion.name] / 2)
        if not path[middle_line]:
            dash_coordinate = [champion.target.y, champion.target.x]
        else:
            dash_coordinate = path[middle_line][-1]

        # since the path function kinda sucks (especially when going straight up or down), check if the target is in the path
        # if not, change the path a bit to force the target there
        distance = field.distance(champion, champion.target, True)
        if (len(path[middle_line]) > distance):
            found = False
            for pp in path:
                for p in pp:
                    if ([p[0], p[1]] == [champion.target.y, champion.target.x]):
                        found = True
            if (not found):
                path[middle_line][int(distance)] = [champion.target.y, champion.target.x]

        # if the dash coordinate is taken, find the closest free hex
        possible_targets = field.hexes_in_distance(dash_coordinate[0], dash_coordinate[1], 2)
        for i, p in enumerate(possible_targets):
            d = field.distance({'y': p[0], 'x': p[1]}, {'y': dash_coordinate[0], 'x': dash_coordinate[1]}, False)
            possible_targets[i].append(d)

        random.shuffle(possible_targets)
        possible_targets = sorted(possible_targets, key=lambda x: x[2])

        # go through the sorted list of path hexes and find the first one that's free
        dash_coordinate = None
        for p in possible_targets:
            c = coords[p[0]][p[1]]
            if (not c):
                dash_coordinate = p
                break

        # find out how many enemies are there on the path
        enemies = 0
        for pp in path:
            for p in pp:
                c = coords[p[0]][p[1]]
                if (c and c.team != champion.team and c.champion):
                    enemies += 1
        if (enemies > 0):
            damage_per_enemy = stats.ABILITY_DMG[champion.name][champion.stars] / enemies
            already_targeted = []
            for pp in path:
                for p in pp:
                    c = coords[p[0]][p[1]]
                    if (c and c.team != champion.team and c.champion and c not in already_targeted):

                        c.print(
                            ' {} {} --> {}'.format('MR', c.MR, c.MR * stats.ABILITY_ARMOR_MR_DECREASE[champion.name]))
                        c.print(' {} {} --> {}'.format('armor', c.armor,
                                                       c.armor * stats.ABILITY_ARMOR_MR_DECREASE[champion.name]))
                        c.MR *= stats.ABILITY_ARMOR_MR_DECREASE[champion.name]
                        c.armor *= stats.ABILITY_ARMOR_MR_DECREASE[champion.name]

                        # c.add_que('change_stat', -1, None, 'stunned', True)
                        # c.clear_que_stunned_removal()
                        # c.add_que('change_stat', stats.ABILITY_STUN_DURATION[champion.name][champion.stars], None, 'stunned', False)

                        already_targeted.append(c)
                        champion.spell(c, damage_per_enemy)
                        if (c.health > 0):
                            yone_list.append([champion, c])
            if (yone_helper(champion) > 0):
                champion.print(
                    ' {} {} --> {}'.format('maxmana', champion.maxmana, stats.SECONDARY_MAXMANA[champion.name]))
                champion.maxmana = stats.SECONDARY_MAXMANA[champion.name]
                champion.print(' list length {} --> {}'.format(0, yone_helper(champion)))

        if (dash_coordinate):
            champion.move(dash_coordinate[0], dash_coordinate[1], True)
        apply_attack_cooldown(champion)

    # Unforgotten
    elif (champion.maxmana == stats.SECONDARY_MAXMANA[champion.name]):
        # sort the marked enemies by hp and check their neighboring hexes
        # whichever has the first free hex, is going to be the dash target
        marked_enemies = []
        for y in yone_list:
            if (y[0] == champion):
                marked_enemies.append([y[1], y[1].health])
        marked_enemies = sorted(marked_enemies, key=lambda x: x[1])

        dash_coordinate = None
        target = None

        # as we have the list sorted by health, start rolling through it
        # go through every unit's every neighbor until we have a free one
        for m in marked_enemies:
            neighbors = field.find_neighbors(m[0].y, m[0].x)
            random.shuffle(neighbors)
            for n in neighbors:
                c = coords[n[0]][n[1]]
                if (not c):
                    target = m[0]
                    dash_coordinate = n
                    break
            if (target): break

        if (target):
            damage = (target.max_health - target.health) * stats.ABILITY_MISSING_HEALTH_DAMAGE_PERCENTAGE[champion.name]
            damage += stats.ABILITY_SECONDARY_DMG[champion.name][champion.stars]

            distance = field.distance(champion, target, True)
            if (distance > 1):
                champion.print(' dashes')
                champion.move(dash_coordinate[0], dash_coordinate[1], True)
            champion.spell(target, damage)
            champion.add_que('execute_function', 0, [yone_ability, {'loop': False}])


# check if someone on the list has died
def yone_ability(champion, data):
    global yone_list

    old_length = yone_helper(champion)
    yone_list = list(filter(lambda x: x[1].health > 0, yone_list))
    new_length = yone_helper(champion)
    if (new_length != old_length):
        champion.print(' list length {} --> {}'.format(old_length, new_length))

    if (yone_helper(champion) == 0 and champion.maxmana != stats.MAXMANA[champion.name]):
        champion.print(' {} {} --> {}'.format('maxmana', champion.maxmana, stats.MAXMANA[champion.name]))
        champion.maxmana = stats.MAXMANA[champion.name]

    if (data['loop']):
        champion.add_que('execute_function', 50, [yone_ability, {'loop': True}])


def yone_helper(champion):
    global yone_list
    counter = 0
    for y in yone_list:
        if (y[0] == champion):
            counter += 1
    return counter


def yuumi(champion):
    default_ability_calls(champion)
    champion.add_que('change_stat', -1, None, 'ability_active', True)
    champion.add_que('change_stat', stats.ABILITY_LENGTH[champion.name], None, 'ability_active', False)

    own_team = champion.own_team()
    allies = []
    for e in own_team:
        allies.append([e, e.health / e.max_health])

    allies = sorted(allies, key=lambda x: x[1])
    # If the allied already died. Should basically never need to return
    if not allies:
        print("some dead yummi allies. Should never really see this.")
        return
    first_target = allies[0][0]

    # heal and change AS for the first target
    heal_amount = (first_target.max_health - first_target.health) * \
                  stats.ABILITY_HEALTH_GAIN_PERCENTAGES[champion.name][champion.stars] * champion.SP
    first_target.add_que('heal', -1, None, None, heal_amount)
    as_gain = (stats.ABILITY_AS_GAIN[champion.name][champion.stars] - 1) * champion.SP + 1

    first_target.add_que('change_stat', -1, None, 'AS', first_target.AS * as_gain)
    first_target.add_que('change_stat', stats.ABILITY_AS_CHANGE_LENGTH[champion.name], None, 'AS', None,
                         {'ezreal': as_gain})

    second_target = None
    dash_target = None
    if len(own_team) == 2:
        second_target = list(filter(lambda x: x != first_target, own_team))[0]

    # find the ally that's furthest away from the first target
    if len(own_team) > 2:
        team_distances = []
        for o in own_team:
            if o != first_target:
                d = field.distance(first_target, o, True)
                if o == champion:
                    d = 0
                team_distances.append([o, d])

        team_distances = sorted(team_distances, key=lambda x: x[1], reverse=True)
        second_target = team_distances[0][0]

        # find the closest free coordinate next to the second target
        coords = field.coordinates
        possible_targets = field.hexes_in_distance(second_target.y, second_target.x, 2)
        for i, p in enumerate(possible_targets):
            d = field.distance({'y': p[0], 'x': p[1]}, {'y': second_target.y, 'x': second_target.x}, False)
            possible_targets[i].append(d)

        random.shuffle(possible_targets)
        possible_targets = sorted(possible_targets, key=lambda x: x[2])

        # go through the sorted list of path hexes and find the first one that's free
        for p in possible_targets:
            c = coords[p[0]][p[1]]
            if not c:
                dash_target = p
                break

    if second_target:
        pause = 0

        # fixing a bug which comes from stacking stat changes in the same milliseconds.
        # yuumi will instantly heal and buff the second ally if she is a mage

        heal_amount = (second_target.max_health - second_target.health) * \
                      stats.ABILITY_HEALTH_GAIN_PERCENTAGES[champion.name][champion.stars] * champion.SP
        second_target.add_que('heal', pause, None, None, heal_amount)

        second_target.add_que('change_stat', pause, None, 'AS', second_target.AS * as_gain)
        second_target.add_que('change_stat', stats.ABILITY_AS_CHANGE_LENGTH[champion.name] + pause, None, 'AS', None,
                              {'ezreal': as_gain})

        if dash_target:
            champion.move(dash_target[0], dash_target[1], True)


def zilean(champion):
    default_ability_calls(champion)

    own_team = champion.own_team()
    target_list = []

    for o in own_team:
        # do not add zilean himself or anyone who is targeted and not dead
        if (o != champion and not o.will_revive[0][0]):
            target_list.append([o, o.health])

    target_list = sorted(target_list, key=lambda x: x[1])
    target_amount = stats.ABILITY_TARGETS[champion.name][champion.stars]

    if (len(target_list) > target_amount):
        target_list = target_list[:target_amount]

    # fill the unit's.will_revive's first slot which is reserved for zliean
    # the revive itself is handled in "champion_functions.py"'s 'die()' -function
    for a in target_list:
        champion.print(' placed orb on {} {}'.format(a[0].team, a[0].name))
        old_value = [[a[0].will_revive[0][0]], [a[0].will_revive[1][0]]]
        a[0].will_revive[0][0] = champion
        new_value = [[a[0].will_revive[0][0].name], [a[0].will_revive[1][0]]]
        a[0].print(' {} {} --> {}'.format('will_revive', old_value, new_value))


def galio(champion):
    default_ability_calls(champion)

    if (champion.stars >= 2):
        new_damage_receiving = champion.receive_decreased_damage * stats.ABILITY_TARGET_DECREASE_DAMAGE_RECEIVING[
            champion.name]
        champion.add_que('change_stat', 0, None, 'receive_decreased_damage', new_damage_receiving)

        # taunting
        neighbor_enemies = field.enemies_in_distance(champion, champion.y, champion.x, 1)
        for n in neighbor_enemies:
            old_target = n.target
            n.add_que('change_target', -1, None, None, champion)
            n.add_que('change_target', stats.ABILITY_LENGTH[champion.name], None, None, old_target)

        # execute the second part of the ult in x seconds
        champion.add_que('execute_function', stats.ABILITY_LENGTH[champion.name], [galio_ability, {}])


def galio_ability(champion, data):
    end_damage_receiving = champion.receive_decreased_damage / stats.ABILITY_TARGET_DECREASE_DAMAGE_RECEIVING[
        champion.name]
    champion.add_que('change_stat', 0, None, 'receive_decreased_damage', end_damage_receiving)

    radius = stats.ABILITY_RADIUS[champion.name]
    targets = field.enemies_in_distance(champion, champion.y, champion.x, radius)
    for t in targets:
        champion.spell(t, stats.ABILITY_DMG[champion.name][champion.stars])

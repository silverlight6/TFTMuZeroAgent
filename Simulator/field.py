import math
import Simulator.stats as stats
import Simulator.items as items

coordinates = [[None] * 7 for _ in range(8)]


def action(champion):
    if len(champion.enemy_team()) > 0 and not champion.stunned:

        # if ability cast is 'global', cast it right away
        if champion.millis() > 0 and champion.champion and not champion.ability_requires_target \
                and 0 < champion.maxmana <= champion.mana \
                and not (champion.disarmed and not stats.ABILITY_WHILE_DISARMED[champion.name]) \
                and champion.millis() > champion.castMS + stats.MANALOCK[champion.name]:
            champion.ability()

        attackable_enemies = list(filter(lambda x: (x.champion and x.health > 0), champion.enemy_team()))

        if champion.idle and len(attackable_enemies) > 0:
            # if not target --> find one
            if champion.target is None:
                find_target(champion)
            if champion.target:
                d = distance(champion, champion.target, True)
                # chase max one tile. if target is further away than one tile, find a new target
                # the new target is automatically the closest one
                if d >= champion.range + 2:
                    find_target(champion)
                d = distance(champion, champion.target, True)

                # if target is not in range --> move (chase max 1 tile)
                if d > champion.range:
                    if champion.millis() == 0:
                        pass
                    else:
                        path = find_path(champion, champion.target.y, champion.target.x)
                        # move only if there's a way to get to the target
                        if path:
                            champion.move(path[0][0], path[0][1])

                        # if there's no way to get to the target, but this champion has a longer range
                        # --> move one step closer
                        elif champion.range > 1:
                            m = find_next_ranged_move(champion)
                            if m:
                                champion.move(m[0], m[1])

                elif champion.millis() > 0 \
                        and not (champion.ability_active and not stats.ATTACK_WHILE_ABILITY_ACTIVE[champion.name]):
                    if champion.champion and 0 < champion.maxmana <= champion.mana \
                            and not (champion.disarmed and not stats.ABILITY_WHILE_DISARMED[champion.name]) \
                            and champion.millis() > champion.castMS + stats.MANALOCK[champion.name]:
                        champion.ability()
                    elif not champion.disarmed and not champion.blinded:
                        champion.attack()

    return None


# find a tile that takes the champion one step closer to the target
# doesn't require a clear path to the target (like 'find_path' does)
def find_next_ranged_move(champion):
    neighbors_original = find_neighbors(champion.y, champion.x)
    neighbors = []

    for n in neighbors_original:
        if not (coordinates[n[0]][n[1]] and coordinates[n[0]][n[1]] is not champion.target):
            neighbors.append(n)

    for n in neighbors:
        dist = distance({'y': n[0], 'x': n[1]}, {'y': champion.target.y, 'x': champion.target.x}, False)
        n.append(dist)

    neighbors = sorted(neighbors, key=lambda x: x[2])

    if neighbors:
        return neighbors[0]
    else:
        return None


# finds the shortest path from a champion's location to some target coordinates
# the algorithm is not perfect
# find the neighbors of the current tile under review,
# sort them by distance to the target and follow the closest tile until at target.
# not allowing visits at tiles that are already visited.
# runs twice: 
# 1. execute using the rules three lines above
# 2. if there are two neighbor tiles with same distance to the target, use the second one instaed of the first one.
# this sometimes brings a different, better result. a big clumsy way of doing it, but more or less does the job.
def find_path(champion, target_y, target_x, use_second=False, secondary_result=[]):
    path = []
    visited = []
    start_y = champion.y
    stary_x = champion.x

    path.append([start_y, stary_x])
    visited.append([start_y, stary_x])

    count = 0
    while path[len(path) - 1] != [target_y, target_x]:
        count += 1
        if count > 50:
            break
        neighbors_original = find_neighbors(path[-1][0], path[-1][1])
        neighbors = []
        # add into the neighbor -list if the tile is not visited, it's free or it contains the campion's target
        for n in neighbors_original:
            if not (coordinates[n[0]][n[1]] and coordinates[n[0]][n[1]] is not champion.target) or n in visited:
                neighbors.append(n)

        for n in neighbors:
            dist = distance({'y': n[0], 'x': n[1]}, {'y': target_y, 'x': target_x}, False)
            n.append(dist)

        # sort the neighbors by 'distance to the target hex'
        neighbors = sorted(neighbors, key=lambda x: x[2])

        # if 'use_second' is True AND the first two elements in the sorted list are of same distance, flip them
        if len(neighbors) > 1 and neighbors[0][2] == neighbors[1][2] and use_second is True:
            a, b = neighbors[0], neighbors[1]
            neighbors[1] = a
            neighbors[0] = b

        if len(neighbors) == 0:
            break
            # return None

        path.append([neighbors[0][0], neighbors[0][1]])
        visited.append([neighbors[0][0], neighbors[0][1]])

    if not use_second:
        return find_path(champion, target_y, target_x, True, path)

    else:
        if secondary_result and path:
            # if there was no answer found
            target = [target_y, target_x]
            if [path[-1][0], path[-1][1]] != target and [secondary_result[-1][0], secondary_result[-1][1]] != target:
                return None

            # return the shortest path of the two test runs
            # dont include the starting point
            elif len(path) < len(secondary_result):
                return path[1:]
            else:
                return secondary_result[1:]


def find_neighbors(y, x, allow_outside_map=False):
    directions = [
        [[+1, 0], [+1, +1], [0, -1],
         [0, +1], [-1, 0], [-1, +1]],
        [[+1, -1], [+1, 0], [0, -1],
         [0, +1], [-1, -1], [-1, 0]],
    ]

    parity = y & 1
    neighbors = []
    for c in directions[parity]:
        nY = c[0] + y
        nX = c[1] + x
        if allow_outside_map or (0 <= nY <= 7 and 0 <= nX <= 6):
            neighbors.append([nY, nX])
    return neighbors


def find_target(c):
    c_coords = to_cube_coords(c)
    old_target = c.target

    current_target = {'champion': None, 'distance': None}

    # roll through everyone
    # find the closest one
    for y in coordinates:
        for x in y:
            if x is not None and x.team is not c.team and x.champion and x.health > 0:
                x_coords = to_cube_coords(x)
                dist = (abs(c_coords['x'] - x_coords['x']) + abs(c_coords['y'] - x_coords['y'])
                        + abs(c_coords['z'] - x_coords['z'])) / 2
                if not current_target['distance'] or dist < current_target['distance']:
                    current_target['champion'] = x
                    current_target['distance'] = dist

    if current_target['champion']:
        c.target = current_target['champion']
        c.target_y = current_target['champion'].y
        c.target_x = current_target['champion'].x
        if c.target != old_target and c.target.team:
            c.print(' has a new target: ' + '{:<8}'.format(c.target.team) + '{:<8}'.format(c.target.name) +
                    '  [{}, {}]'.format(c.target.y, c.target.x))


# find enemies and sort them by distance
def find_enemies(champion):
    c = coordinates
    enemies = []
    for i, a_line in enumerate(c):
        for j, col in enumerate(a_line):
            if c[i][j] and c[i][j].team != champion.team and c[i][j].champion:
                d = distance(champion, c[i][j], True)
                enemies.append([c[i][j], d])
    enemies.sort(key=lambda x: x[1])
    return enemies


# find enemies in x distance of a coordinate
def enemies_in_distance(champion, target_y, target_x, radius):
    enemies_within = []
    c = coordinates
    for i, line in enumerate(c):
        for j, col in enumerate(line):
            if (c[i][j] and
                    c[i][j].team != champion.team and
                    c[i][j].champion and
                    distance({'y': j, 'x': i}, {'y': target_y, 'x': target_x}, False) <= radius):
                enemies_within.append(c[i][j])

    return enemies_within


# find hexes that are within a certain distance
def hexes_in_distance(target_y, target_x, radius, allow_outside_map=False):
    hexes_within = []
    c = coordinates
    for i in range(-100, 100):
        for j in range(-100, 100):
            if allow_outside_map or (i >= 0 and i <= 7 and j >= 0 and j <= 6):
                if distance({'y': i, 'x': j}, {'y': target_y, 'x': target_x}, False) <= radius:
                    hexes_within.append([i, j])

    return hexes_within


# find hexes exactly x away from a coordinate
def hexes_distance_away(target_y, target_x, radius, allow_outside_map=False):
    hexes_within = []
    for i in range(-10, 20):
        for j in range(-10, 20):
            if allow_outside_map or (i >= 0 and i <= 7 and j >= 0 and j <= 6):
                if distance({'y': i, 'x': j}, {'y': target_y, 'x': target_x}, False) == radius:
                    hexes_within.append([i, j])

    return hexes_within


def distance(champion1, champion2, objects):
    if objects:
        c1_coords = to_cube_coords(champion1)
        c2_coords = to_cube_coords(champion2)
    else:
        c1_coords = to_cube_coords_nonobj(champion1)
        c2_coords = to_cube_coords_nonobj(champion2)

    dist = (abs(c1_coords['x'] - c2_coords['x']) +
            abs(c1_coords['y'] - c2_coords['y']) +
            abs(c1_coords['z'] - c2_coords['z'])) / 2
    return dist


def to_cube_coords(c):
    x = c.x - (c.y + (c.y & 1)) / 2
    z = c.y
    y = -x - z
    return {'x': x, 'y': y, 'z': z}


def to_cube_coords_nonobj(c):
    x = c['x'] - (c['y'] + (c['y'] & 1)) / 2
    z = c['y']
    y = -x - z
    return {'x': x, 'y': y, 'z': z}


def to_normal_coords(c):
    x = c[0] + (c[2] + (c[2] & 1)) / 2
    y = c[2]
    return [int(y), int(x)]


# def get_line_straight(starting_point, direction_point, length):


# def get_line_diagonal()


def lerp(a, b, t):  # for floats
    return a + (b - a) * t


def cube_lerp(a, b, t):  # for hexes
    aa = to_cube_coords_nonobj(a)
    bb = to_cube_coords_nonobj(b)

    return {'x': lerp(aa['x'], bb['x'], t),
            'y': lerp(aa['y'], bb['y'], t),
            'z': lerp(aa['z'], bb['z'], t)}


def cube_round(cube):
    rx = round(cube['x'])
    ry = round(cube['y'])
    rz = round(cube['z'])

    x_diff = abs(rx - cube['x'])
    y_diff = abs(ry - cube['y'])
    z_diff = abs(rz - cube['z'])

    if x_diff > y_diff and x_diff > z_diff:
        rx = -ry - rz
    elif y_diff > z_diff:
        ry = -rx - rz
    else:
        rz = -rx - ry

    return [rx, ry, rz]


def line(starting_point, end_point):
    N = distance({'y': starting_point['y'], 'x': starting_point['x']},
                 {'y': end_point['y'], 'x': end_point['x']}, False)
    results = []
    if N != 0:
        for i in range(0, int(N) + 1):
            results.append(to_normal_coords(cube_round(cube_lerp(starting_point, end_point, 1.0 / N * i))))
    return results


# the rectangle that's used in azir's, ezreal's and TF's ults
def rectangle_from_champion_to_wall_behind_target(champion, width, target_y, target_x, allow_outside_map=False):
    direction = 'vertical'
    # Is the champion I am looking at above me or to my side?
    if abs(target_x - champion.x) >= abs(target_y - champion.y):
        direction = 'diagonal'

    # Distance in manhattan units between y and x. I'm assuming this can be negative
    change_y = target_y - champion.y
    change_x = target_x - champion.x

    # Not sure why this is included.. This just moves the distance between the two by a multiple of 10.
    target_y = 10 * change_y + champion.y
    target_x = 10 * change_x + champion.x

    hexes = []
    affected_hexes = []

    # With twisted fate, width is 1, I imagine with azir and ezreal, this is not 1.
    # So it goes from 1 to 1 for tf. 0 ^ 0 equals 1
    loop_start = math.floor(width / 2) * -1
    loop_end = math.floor(width / 2) + 1

    # Loop over the number of hexes required.
    for i in range(loop_start, loop_end):
        if direction == 'vertical':
            hexes.append(line({'y': champion.y, 'x': champion.x + i}, {'y': target_y, 'x': target_x + i}))
        if direction == 'diagonal':
            j = 0
            # hexagonal coordinates are absolute aids
            if i == 1:
                if champion.y % 2 == 1:
                    if target_x > champion.x:
                        j = -1
                if champion.y % 2 == 0:
                    if target_x < champion.x:
                        j = 1

            if i == -1:
                if champion.y % 2 == 1:
                    if target_x < champion.x:
                        j = -1
                if champion.y % 2 == 0:
                    if target_x > champion.x:
                        j = 1

            hexes.append(line({'y': champion.y + i, 'x': champion.x + j}, {'y': target_y + i, 'x': target_x + j}))

    # drop the hexes that are outside the map and get the distance to each hex
    for i, hexline in enumerate(hexes):
        affected_hexes.append([])
        for h in hexline:
            if allow_outside_map or (h[0] >= 0 and h[0] <= 7 and h[1] >= 0 and h[1] <= 6):
                d = distance({'y': champion.y, 'x': champion.x}, {'y': h[0], 'x': h[1]}, False)
                h.append(d)
                affected_hexes[i].append(h)

    return affected_hexes


def leap_to_back_line(champion, data):
    trait = data['trait']

    # set the preferred coordinate which matches the other side of the baord (y-wise) at still somewhat same x-line
    if champion.y <= 3:
        preferred_y = 7
    else:
        preferred_y = 0

    target_hex = [[preferred_y, champion.x]]
    target_hex_index = 0
    while target_hex_index < len(target_hex):
        # do the actual leaping
        if target_hex and not coordinates[target_hex[target_hex_index][0]][target_hex[target_hex_index][1]]:
            champion.print(' leaps')
            champion.move(target_hex[target_hex_index][0], target_hex[target_hex_index][1], True)
            champion.clear_que_idle()
            items.change_stat(champion, 'idle', True, trait)
            break
        else:
            neighbors = find_neighbors(target_hex[target_hex_index][0], target_hex[target_hex_index][1], False)
            for neighbor in neighbors:
                if neighbor not in target_hex:
                    target_hex.append(neighbor)
            target_hex_index += 1

    items.change_stat(champion, 'champion', True, trait)
    items.change_stat(champion, 'stunned', False, trait)

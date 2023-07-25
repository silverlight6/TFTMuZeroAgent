import Simulator.config as config
import Simulator.origin_class as origin_class
import Simulator.origin_class_stats as origin_class_stats
import Simulator.champion_functions as champion_functions
import time
import random
import itertools

from math import ceil
from Simulator.stats import AD, HEALTH, ARMOR, MR, AS, RANGE, MANA, MAXMANA, COST, MANALOCK, ABILITY_REQUIRES_TARGET, \
    DODGE, INITIATIVE_ACTIVE, ABILITY_LENGTH, DAMAGE_PER_UNIT
from Simulator.champion_functions import attack, die, MILLIS, MILLISECONDS_INCREASE, add_damage_dealt
from Simulator import ability, active, field, item_stats, items

que = []
log = []


def printt(msg):
    if config.PRINTMESSAGES:
        log.append(msg)


test_multiple = {'blue': 0, 'red': 0, 'bugged out': 0, 'draw': 0}


# I am going to have to add cost information but that should be about it.
# When updating to the new patch,
# there are going to be different edge cases but the core of the game should remain the same.


class champion:
    def __init__(self, name, team=None, y=-1, x=-1, stars=1, itemlist=None, overlord=None,
                 sandguard_overlord_coordinates=None, chosen=False, kayn_form=None, target_dummy=False):

        if itemlist is None:
            itemlist = []
        self.champion = True

        self.name = name
        self.stars = stars

        # in case we're spawning a construct, galio or a turret, the rest are handled at the bottom of the object
        if name != 'construct' and name != 'galio' and name != 'aphelios_turret':
            self.health = round(HEALTH[name] * config.STARMULTIPLIER ** (stars - 1), 1)
            self.max_health = round(HEALTH[name] * config.STARMULTIPLIER ** (stars - 1), 1)
            self.AD = round(AD[name] * config.STARMULTIPLIER ** (stars - 1), 1)

        self.SP = 1

        self.AS = AS[name]
        self.armor = ARMOR[name]
        self.MR = MR[name]
        self.range = RANGE[name]
        self.dodge = DODGE
        self.movement_delay = config.MOVEMENTDELAY

        self.mana = MANA[name]
        self.maxmana = MAXMANA[name]
        self.cost = COST[name]

        self.manalock = MANALOCK[name]
        # not going to start changing the whole structure of the manalock code since that could create some bugs
        # shen is the only unit whose manalock scales by stars so just forcing the change here.
        if self.name == 'shen':
            self.manalock = 1000 + ABILITY_LENGTH['shen'][self.stars]

        self.mana_cost_increased = False
        self.mana_generation = 1  # enlightened - trait
        self.castMS = -50000  # the timestamp of the last cast
        self.ability_requires_target = ABILITY_REQUIRES_TARGET[name]

        self.target = None
        self.target_y = None
        self.target_x = None

        self.immune = False
        self.autoimmune = False
        self.targetable = True
        self.stunned = False
        self.burning = False
        self.disarmed = False
        self.blinded = False
        self.shields = []
        self.receive_increased_damage = 1  # received damage = 100% by default
        self.receive_decreased_damage = 1  # stupid to have two variables for this, but gets messy if using only one
        self.damage_reduction = 0
        self.deal_increased_damage = 1  # attacks and spells
        self.deal_bonus_true_damage = 0  # divine - trait
        self.spell_damage_reduction_percentage = 1
        # inverse, 60% reduction (dragons_claw) would set this to 0.4. with garen it'd be 1 * 0.4 * 0.2 = 0.08
        self.heal_per_attack = 0
        self.lifesteal = 0
        self.lifesteal_spells = 0
        self.healing_strength = 1
        self.crit_chance = 0
        self.crit_damage = 1.5

        self.team = team
        self.x = x
        self.y = y
        if y < 4:
            self.starting_x = self.x
            self.starting_y = self.y
        else:
            self.starting_x = abs(x - 6)
            self.starting_y = abs(y - 7)
        # Did this unit survive the last combat
        self.survive_combat = False
        # Did this unit participate in the last combat
        self.participated_in_combat = False
        self.bench_loc = -1
        self.underlords = []
        self.overlord = overlord
        self.overlord_coordinates = sandguard_overlord_coordinates
        self.origin = origin_class_stats.origin_class[self.name].copy()

        self.will_revive = [[None], [None]]  # consists of [[zilean_champion], [GA]]

        field.coordinates[y][x] = self

        self.idle = True
        self.ability_active = False

        self.items = itemlist
        self.num_items = 0
        self.ionic_sparked = 0  # just a helper to know who to zap
        self.spell_has_used_ludens = False  # helps with ludens_echo
        self.AD_reduction_cc = False  # ludens counts dazzler ad reduction as crowd control so adding a flag for dat
        self.pumped_up = False  # the_boss - trait
        self.done_situps = False  # the_boss - trait

        self.start_time = time.time_ns()

        self.chosen = origin_class.chosen(self, chosen)

        self.kayn_form = kayn_form

        self.target_dummy = target_dummy

        if chosen:
            self.health = round(HEALTH[name] * config.STARMULTIPLIER ** (stars - 1), 1)
            self.max_health = round(HEALTH[name] * config.STARMULTIPLIER ** (stars - 1), 1)
            self.AD = round(AD[name] * config.STARMULTIPLIER ** (stars - 1), 1)
            self.stars = 2
            # self.cost = cost_star_values[COST[name]][self.stars]
            self.health += 200
            self.max_health += 200

        if name != 'aphelios_turret':
            items.initiate(self)

        if name in INITIATIVE_ACTIVE:
            getattr(active, name + '_init')(self)

        # zzrot_portal's construct uses the same object so we have to correct the data a little
        if name == 'construct':
            self.health = HEALTH[name][stars]
            self.max_health = HEALTH[name][stars]
            self.AD = AD[name]

        if name == 'galio':
            self.health = HEALTH[name][stars] + HEALTH[name][stars] * config.GALIO_MULTIPLIER * \
                          origin_class.cultist_stars[team]
            self.max_health = HEALTH[name][stars] + HEALTH[name][stars] * config.GALIO_MULTIPLIER * \
                              origin_class.cultist_stars[team]
            self.AD = AD[name][stars] + AD[name][stars] * config.GALIO_MULTIPLIER * origin_class.cultist_stars[team]

        if name == 'aphelios_turret':
            self.health = 1
            self.max_health = 1
            self.AD = self.overlord.AD
            self.AS = self.overlord.AS

    def attack(self, bonus_dmg=0, target=None, item_attack=False, trait_attack='', set_AD=None):
        attackable_enemies = list(filter(lambda x: (x.champion and x.health > 0), self.enemy_team()))
        if target or self.target:
            if (not self.target or self.target.health <= 0) and len(attackable_enemies) > 0:
                field.find_target(self)
            if not target or target.health <= 0:
                target = self.target

            # aphelios turrets don't attack if aphelios is cc'd,
            # but still do attack if he is recovering from a zilean or GA revive
            if not (self.overlord and (self.overlord.stunned or self.overlord.disarmed or self.overlord.blinded)) \
                    or (self.overlord and self.overlord.stunned and not self.overlord.champion):

                # enforcing ashe ult here
                if self.name == 'ashe' and self.ability_active \
                        and not trait_attack and len(self.enemy_team()) > 0 and not item_attack:
                    ability.ashe_helper(self, {'target': target, 'bonus_dmg': bonus_dmg})

                else:
                    attack(self, target, bonus_dmg, item_attack, trait_attack, set_AD)

    def spell(self, target, dmg, true_dmg=0, item_damage=False, burn_damage=False, trait_damage=False):
        enemy_team = 'red' if self.team == 'blue' else 'blue'
        if self == target:
            enemy_team = self.team  # when ionic sparking themselves

        if not ('trap_claw' in target.items and not item_damage and not burn_damage):  # trap_claw

            if self.pumped_up:  # the_boss -trait
                true_dmg += dmg
                dmg = 0

            items.gargoyle_stoneplate(target)  # gargoyle_stoneplate (needs to take effect before armor or MR is used)
            if not item_damage:
                items.morellonomicon(self, target)  # morellonomicon
                if not self.spell_has_used_ludens:
                    items.ludens_echo(self, target)  # ludens_echo

            # damage = 0
            if not item_damage:
                dmg *= items.giant_slayer(self, target)  # giants_slayer
            if target.MR >= 0:
                damage = dmg * (100 / (100 + target.MR)) * self.SP
            else:
                damage = dmg * (2 - 100 / (100 - target.MR)) * self.SP

            # SP doesnt affect items' damage
            if item_damage:
                damage /= self.SP

            damage += self.deal_bonus_true_damage * damage  # divine -trait

            damage *= target.receive_increased_damage
            damage *= target.receive_decreased_damage

            damage += true_dmg
            damage -= target.damage_reduction
            damage *= self.deal_increased_damage

            if not item_damage:
                damage = damage * target.spell_damage_reduction_percentage

            if damage < 0:
                damage = 0
            if target.immune:
                damage = 0

            crit_random = random.randint(1, 100) / 100
            crit_string = ''
            # jeweled gauntlet -item     #bramble vest -item
            if 'jeweled_gauntlet' in self.items and crit_random < self.crit_chance \
                    and self != target and not 'bramble_vest' in target.items and not item_damage:
                damage *= self.crit_damage
                crit_string = 'crit'

            # assassins spell criting
            # the above one is already such a mess. just do another statement (mess) here
            # it's long as shit but gets the job done
            if crit_random < self.crit_chance and self != target and not 'bramble_vest' in target.items \
                    and not item_damage and crit_string == '' and origin_class.is_trait(self, 'assassin') \
                    and origin_class.get_origin_class_tier(self.team, 'assassin') > 0:
                damage *= self.crit_damage
                crit_string = 'crit'

            burn_string = ''
            if burn_damage:
                burn_string = ' burning'

            item_string = ''
            if item_damage:
                item_string = ' item'

            trait_string = ''
            if trait_damage:
                trait_string = ' {}'.format(trait_damage)

            # if the target has died to luden's, don't continue
            if target in eval(enemy_team):

                if self.lifesteal_spells > 0 and not item_damage:
                    self.add_que('heal', -1, None, None, damage * self.lifesteal_spells)

                add_damage_dealt(self, damage, target)

                origin_class.dazzler(self, target)  # dazzler -trait

                # shield
                shield_old = target.shield_amount()
                if len(target.shields) > 0:
                    while damage > 0 and target.shield_amount() > 0:
                        top_shield = target.shields[0]['amount']
                        target.shields[0]['amount'] -= damage
                        if target.shields[0]['amount'] < 0:
                            damage -= top_shield
                            target.shields = target.shields[1:]
                        else:
                            damage = 0

                if not item_damage:
                    items.blue_buff(self)  # blue_buff

                items.deathblade(self, target)  # deathblade
                items.hextech_gunblade(self, damage)  # hextech_gunblade

                self.print(' deals ' + '{:<8}'.format(enemy_team) + ' ' + '{:<13}'.format(target.name) +
                           '{:<5}--> {:<8}   shield {:<5}--> {:<5} {}{}{}{}'.
                           format(ceil(target.health), ceil(target.health - damage), ceil(shield_old),
                                  ceil(target.shield_amount()), crit_string, burn_string, item_string, trait_string))
                target.health -= damage
                if MILLIS() > target.castMS + target.manalock and not target.ability_active and target.maxmana > 0:
                    if not target.name == 'riven' or ability.riven_helper(target, {}):
                        old_mana = target.mana
                        target.mana += min((damage * config.MANA_DAMAGE_GAIN) *
                                           target.mana_generation, config.MAX_MANA_FROM_DAMAGE)
                        target.print(' mana {} --> {}'.format(round(old_mana, 1), round(target.mana, 1)))

                # titans_resolve -item
                # add bonus damage and armors after the values have been used.
                # this way they will be added now but used only in the next event
                items.titans_resolve(self, target, crit_string)

                # the_boss -trait
                if target.name == 'sett' and not target.done_situps \
                        and target.health < target.max_health * origin_class_stats.threshold['the_boss']:
                    if target.health <= 0:
                        target.health = 1
                    origin_class.the_boss(target)

                if target.health <= 0:
                    target.die()
                else:
                    origin_class.divine(self, target, False)  # divine -trait

                # sharpshooter -trait
                if not trait_damage and not item_damage and not burn_damage:
                    origin_class.sharpshooter(self, target, dmg, true_dmg, True)

        else:
            items.trap_claw(self, target)  # trap_claw

    def move(self, y, x, forced=False, sett=False):
        if self.idle or forced:
            if sett:
                self.print(' moves from sit-ups   to   ({} , {})        '.format(y, x))
            else:
                self.print(' moves from ({} , {})   to   ({} , {})        '.format(self.y, self.x, y, x))

            field.coordinates[self.y][self.x] = None
            self.x = x
            self.y = y
            field.coordinates[y][x] = self
            self.idle = False
            self.add_que('clear_idle', self.movement_delay)

            items.frozen_heart(self)  # frozen_heart - item
            items.ionic_spark(self)  # ionic_spark - item

    def die(self):
        die(self)

        items.redemption(self)  # redemption - item
        items.frozen_heart(self)  # frozen_heart - item
        items.ionic_spark(self)  # ionic_spark - item

    def shield_amount(self):
        shield = 0
        for s in self.shields:
            shield += s['amount']
        return shield

    def enemy_team(self):
        enemy_team = 'red' if self.team == 'blue' else 'blue'
        return eval(enemy_team)

    def own_team(self):
        try:
            return eval(self.team)
        except TypeError:
            return False

    def ability(self):
        attackable_enemies = list(filter(lambda x: (x.champion and x.health > 0), self.enemy_team()))
        if not self.target and len(attackable_enemies) > 0:
            field.find_target(self)
        if self.target:  # if still no target, the remaining enemies are under GA or zilean revive
            getattr(ability, self.name)(self)
            if origin_class.get_origin_class_tier(self.team, 'mage') > 0 and origin_class.is_trait(self, 'mage'):
                if len(self.enemy_team()) > 0:
                    getattr(ability, self.name)(self)

    def active(self):
        pass

    # Only use when creating a champion from round. Use player commands otherwise.
    def add_item(self, item):
        # Not sure why I need to initialize the array each time but it doesn't work if I don't.
        self.items = []
        self.items.append(item)

    def add_que(self, action, length, function=None, stat=None, value=None, data=None):
        if data is None:
            data = {}
        if 'underlord' in data.keys():
            que.append([action, data['underlord'], MILLIS() + length, function, stat, value, data])
        else:
            if action == 'change_stat' and length < 1:
                change_stat(self, action, length, function, stat, value, data)
            elif action == 'shield' and length < 1:
                shield(self, action, length, function, stat, value, data)
            else:
                que.append([action, self, MILLIS() + length, function, stat, value, data])

        que.sort(key=lambda x: x[2])

    def burn(self, target):
        target.clear_que_burn_removal()

        target.add_que('change_stat', -1, None, 'healing_strength', config.BURN_HEALING_REDUCE)
        target.clear_que_healing_reduction()
        target.add_que('change_stat', config.BURN_SECONDS * 1000, None, 'healing_strength', 1)

        for i in range(1, 11):
            self.add_que('burn', i * 1000, None, None, target)

        pass

    def clear_que_idle(self):
        # not very beautiful is it? not trying to impress anyone tho
        for i in range(0, 15):
            for q in que:
                if q[1] is self and q[0] == 'clear_idle':
                    que.remove(q)

    def clear_que_healing_reduction(self):
        for q in que:
            if q[1] is self and q[0] == 'change_stat' and q[4] == 'healing_strength' and q[5] == 1:
                que.remove(q)

    def clear_que_stunned_removal(self):
        for q in que:
            if q[1] is self and q[0] == 'change_stat' and q[4] == 'stunned' and not q[5]:
                que.remove(q)

    def clear_que_blinded_removal(self):
        for q in que:
            if q[1] is self and q[0] == 'change_stat' and q[4] == 'blinded' and not q[5]:
                que.remove(q)

    def clear_que_armor_removal(self):
        for q in que:
            if q[1] is self and q[0] == 'change_stat' and q[4] == 'armor':
                que.remove(q)

    def clear_que_burn_removal(self):
        for i in range(0, 15):
            for q in que:
                if q[0] == 'burn' and q[5] == self:
                    que.remove(q)

    def clear_que_dazzler(self):
        for i in range(0, 15):
            for q in que:
                if q[1] == self and q[0] == 'change_stat' and 'dazzler' in q[6]:
                    que.remove(q)

    def red_append(self, champion):
        red.append(champion)

    def blue_append(self, champion):
        blue.append(champion)

    def red_return(self):
        return red

    def blue_return(self):
        return blue

    def que_return(self):
        return que

    def spawn(self, name, stars, y, x, team=None, is_champion=True):
        if not team:
            team = self.team

        overlord = None
        items = []
        if name == 'aphelios_turret':
            overlord = self
            for i in self.items:
                if i == 'spear_of_shojin':
                    items.append(i)
        unit = champion(name, stars=stars, team=team, y=y, x=x, itemlist=items, overlord=overlord)
        unit.champion = is_champion
        eval(team).append(unit)
        return unit

    def que_replace(self, q):
        global que
        que = q

    def millis(self):
        return MILLIS()

    def print(self, msg):
        if self.team:
            printt('{:<120}'.format('{:<8}'.format(self.team) + '{:<15}'.format(self.name) + msg)
                   + '{:<12}'.format(str(MILLIS())) + str(time.time_ns() - self.start_time))
        else:
            printt('{:<120}'.format('team_unassigned' + '{:<15}'.format(self.name) + msg) + str(MILLIS()))

    def golden(self):
        self.stars += 1
        self.health = round(self.max_health * config.STARMULTIPLIER ** (self.stars - 1), 1)
        self.max_health = round(self.max_health * config.STARMULTIPLIER ** (self.stars - 1), 1)
        self.AD = round(self.AD * config.STARMULTIPLIER ** (self.stars - 1), 1)

    def new_chosen(self):
        self.health += 200
        self.max_health += 200


global blue
global red

blue = []
red = []


# I think I am going to redo parts of this function. 
# Essentially, I am just going to change the first 10 lines so it reads in the data from the two teams.
# This will be an area I will look to optimize on later if need be but for now,
# I want to keep things as simple as possible.
def run(champion_q, player_1, player_2, round_damage=0):
    reset_global_variables()

    for x in range(0, 7):
        for y in range(0, 4):
            if player_1.board[x][y]:
                blue.append(champion_q(player_1.board[x][y].name, 'blue', y, x, player_1.board[x][y].stars,
                                       player_1.board[x][y].items, False, None, player_1.board[x][y].chosen
                                       , player_1.board[x][y].kayn_form, player_1.board[x][y].target_dummy))
            if player_2.board[x][y]:
                # Inverting because the combat system uses the whole board and does not mirror at start.
                red.append(champion_q(player_2.board[x][y].name, 'red', 7 - y, 6 - x, player_2.board[x][y].stars,
                                      player_2.board[x][y].items, False, None, player_2.board[x][y].chosen,
                                      player_2.board[x][y].kayn_form, player_2.board[x][y].target_dummy))

    printt('Player 1 (Blue) Team')
    for unit in blue:
        printt(unit.name)

    printt('Player 2 (Red) Team')
    for unit in red:
        printt(unit.name)

    if len(blue) == 0 or len(red) == 0:
        if len(red) == 0 and len(blue) == 0:
            printt('DRAW')
            return 0, round_damage
        elif len(red) == 0:
            printt('BLUE TEAM WON')
            survive_combat(player_1, blue)
            return 1, round_damage + DAMAGE_PER_UNIT[len(blue)]
        elif len(blue) == 0:
            printt('RED TEAM WON')
            survive_combat(player_2, red)
            return 2, round_damage + DAMAGE_PER_UNIT[len(red)]

    # Not quite sure what is happening in these lines. 
    # They are effects that happen at the start of the fight.
    # But blue[0] feels odd
    items.chalice_of_power(blue[0])  # chalice_of_power
    items.zekes_herald(blue[0])  # zekes_herald
    items.frozen_heart(blue[0])  # frozen_heart
    items.ionic_spark(blue[0])  # ionic_spark
    items.hand_of_justice(blue[0])  # hand_of_justice
    items.locket_of_the_iron_solari(blue[0])  # locket_of_the_iron_solari
    items.shroud_of_stillness(blue[0])  # shroud_of_stillness
    items.zzrot_portal(blue[0])  # zzrot_portal
    items.zephyr(blue[0])  # zephyr

    origin_class.total_health(blue, red)
    origin_class.total_origin_class(blue[0], red[0])  # count and execute some traits
    # Not sure what changed the length of one of these arrays at this point but this seems to fix the issue
    if len(blue) == 0 or len(red) == 0:
        return 0, round_damage
    items.infinity_edge(blue[0])
    # There appears to be some issue with infinity edge
    # infinity_edge made sure that the crit damage bonus gets registered after everything else has gone through

    while True:
        if MILLIS() > 150000:
            test_multiple['bugged out'] += 1
            break
        if MILLIS() > 0 and MILLIS() % origin_class_stats.length['elderwood'] == 0:
            origin_class.elderwood(blue, red)  # elderwood -trait
        if MILLIS() > 0 and MILLIS() % \
                origin_class_stats.threshold['hunter'][origin_class.get_origin_class_tier('blue', 'hunter')] == 0:
            origin_class.hunter(blue)  # hunter -trait
        if MILLIS() > 0 and MILLIS() % \
                origin_class_stats.threshold['hunter'][origin_class.get_origin_class_tier('red', 'hunter')] == 0:
            origin_class.hunter(red)  # hunter -trait

        for b, o in itertools.zip_longest(blue, red):
            if b and not b.target_dummy:
                field.action(b)
            if o and not o.target_dummy:
                field.action(o)

        while len(que) > 0 and MILLIS() > que[0][2]:
            champion_q = que[0][1]
            data = que[0][6]
            # make sure that teemo's poison darts deal damage even after teemo himself has died
            # morgana deals if the ult is running and she dies
            # if ahri dies, she will still ult. range reduced in the executed function
            if (champion_q in blue or champion_q in red) or \
                    (champion_q.name == 'teemo' and champion_q.health <= 0 and que[0][3] and 'target' in que[0][3][1]) \
                    or (champion_q.name == 'morgana' and champion_q.health <= 0 and que[0][3] and
                        'coordinates' in que[0][3][1]) or \
                    (champion_q.name == 'ahri' and champion_q.health <= 0 and que[0][3] and 'y' in que[0][3][1]):

                if que[0][0] == 'clear_idle':
                    champion_q.idle = True
                    champion_q.print(' cleared idle     ')

                if que[0][0] == 'change_stat':
                    change_stat(champion_q, que[0][0], 0, que[0][3], que[0][4], que[0][5], data)

                if que[0][0] == 'heal':
                    start_value = round(champion_q.health, 2)
                    champion_q.health += (que[0][5] * champion_q.healing_strength)
                    if champion_q.health > champion_q.max_health:
                        champion_q.health = champion_q.max_health
                    champion_q.print(' {} {} --> {}'.format('health', start_value, round(champion_q.health, 2)))

                if que[0][0] == 'shield':
                    shield(champion_q, que[0][0], 0, que[0][3], que[0][4], que[0][5], data)

                if que[0][0] == 'change_target':
                    old_target = champion_q.target
                    new_target = que[0][5]
                    if new_target and new_target.health > 0:
                        champion_q.target = new_target
                        champion_q.target_y = new_target.y
                        champion_q.target_x = new_target.x
                        if champion_q.target != old_target:
                            champion_q.print(' has a new target: ' + '{:<8}'.format(champion_q.target.team) +
                                             '{:<8}'.format(champion_q.target.name) +
                                             '  [{}, {}]'.format(champion_q.target.y, champion_q.target.x))
                    else:
                        field.find_target(champion_q)

                if que[0][0] == 'execute_function':
                    if len(que[0][3]) > 1:
                        (que[0][3][0])(champion_q, que[0][3][1])

                if que[0][0] == 'burn':
                    champion_q.spell(que[0][5], 0, que[0][5].max_health * config.BURN_DMG_PER_SLICE, True, True)

                if que[0][0] == 'kill':
                    que[0][5].die()

            que.pop(0)

        MILLISECONDS_INCREASE()
        if len(blue) == 0 or len(red) == 0:
            if len(red) == 0:
                printt('BLUE TEAM WON')
                for unit in blue:
                    printt(unit.name)
                printt("player_1 dealt round damage = {}".format(round_damage + DAMAGE_PER_UNIT[len(blue)]))
                survive_combat(player_1, blue)
                return 1, (round_damage + DAMAGE_PER_UNIT[len(blue)])
            elif len(blue) == 0:
                printt('RED TEAM WON')
                for unit in red:
                    printt(unit.name)
                printt("player_2 dealt round damage = {}".format(round_damage + DAMAGE_PER_UNIT[len(red)]))
                survive_combat(player_2, red)
                return 2, (round_damage + DAMAGE_PER_UNIT[len(red)])
            break
        if MILLIS() > 150000:
            # print("Round has gone on too long")
            return 0, round_damage
    return 0, round_damage


def shield(champion, action, length, function, stat, value, data):
    shield_before = champion.shield_amount()
    if 'shield_before' in data and data['shield_before']:
        shield_before = data['shield_before']

    action_happened = False
    try:
        if data['increase']:
            # mainly for riven to refresh her shield (remove the old one if still applier and give a new one)
            if 'expires' not in data:
                for s in champion.shields:
                    if s['original_amount'] == value['original_amount'] and s['applier'] == value['applier']:
                        champion.shields.remove(s)

            champion.shields.insert(0, value)
            action_happened = True
            if 'expires' in data and data['expires']:
                champion.add_que('shield', data['expires'], None, None, None,
                                 {'increase': False, 'identifier': value['identifier']})
        else:
            for s in champion.shields:
                if s['identifier'] == data['identifier']:
                    action_happened = True
                    champion.shields.remove(s)
                    break
    except KeyError:
        pass
    if action_happened:
        champion.print(' {} {} --> {}'.format('shield', round(shield_before, 2), round(champion.shield_amount(), 2)))


def change_stat(a_champion, action, length, function, stat, value, data):
    # jhin AD goes up (and down) by every percentage of AS change
    if stat == 'AS' and a_champion.name == 'jhin':
        AD_change = None
        if value:
            AD_change = (value / a_champion.AS * 100 - 100) * 0.8
        if 'ezreal' in data:
            AD_change = (data['ezreal'] * 100 - 100) * -0.8

        start_value = a_champion.AD
        a_champion.AD += AD_change
        a_champion.print(' {} {} --> {}'.format('AD', round(start_value, 2), round(a_champion.AD, 2)))

    else:
        if not ('quicksilver' in a_champion.items and MILLIS() <= item_stats.item_change_length['quicksilver'] and
                value and (stat == 'stunned' or stat == 'disarmed' or stat == 'blinded')):
            if not ('rapid_firecannon' in a_champion.items and value and stat == 'blinded'):
                if not (a_champion.name == 'galio' and (MILLIS() - origin_class.galio_spawn_time[a_champion.team] <=
                                                        origin_class_stats.cc_immune['cultist'])
                        and (stat == 'stunned' or stat == 'disarmed' or stat == 'blinded')):
                    end_value = value
                    start_value = getattr(a_champion, stat)
                    if 'ezreal' in data:
                        end_value = a_champion.AS / data['ezreal']
                    if 'morgana' in data:
                        end_value = a_champion.MR / data['morgana']
                    if 'vi' in data:
                        end_value = a_champion.armor / data['vi']
                    if 'ashe' in data:
                        end_value = a_champion.AD / data['ashe']
                    if 'garen' in data:
                        end_value = a_champion.spell_damage_reduction_percentage / data['garen']

                    if stat == 'will_revive' and a_champion.will_revive[0][0]:
                        start_value = [[a_champion.will_revive[0][0].name], [a_champion.will_revive[1][0]]]

                    if end_value != start_value:
                        if isinstance(start_value, float):
                            start_value = round(start_value, 3)
                        if isinstance(end_value, float):
                            end_value = round(end_value, 3)
                        a_champion.print(' {} {} --> {}'.format(stat, start_value, end_value))
                    setattr(a_champion, stat, end_value)
            else:
                a_champion.print(' not blinded because wears rapid firecannon')
        else:
            a_champion.print(' not {} because wears quicksilver'.format(stat))


def reset_global_variables():
    global blue
    global red
    global que
    global log
    blue = []
    red = []
    que = []
    log = []

    champion_functions.MILLISECONDS = 0
    champion_functions.damage_dealt = []
    champion_functions.damage_dealt_teams = {'blue': 0, 'red': 0}
    champion_functions.galio_spawned = {'blue': False, 'red': False}

    # global kennen_hits
    # global l
    ability.kennen_hits = []
    ability.lulu_targeted = []
    ability.morgana_MR_list = []
    ability.riven_counter = []
    ability.riven_identifier_list = []
    ability.vi_armor_list = []
    ability.yone_list = []
    ability.yone_checking = False

    active.jhin_shots = []
    active.kalista_targets = []
    active.vayne_targets = []
    active.zed_counter = []

    field.coordinates = [[None] * 7 for _ in range(8)]

    items.bramble_vest_list = []
    items.deathblade_list = []
    items.frozen_heart_list = []
    items.gargoyle_stoneplate_list = []
    items.hextech_gunblade_list = []
    items.ionic_spark_list = []
    items.last_whisper_list = []
    items.statikk_shiv_list = []
    items.titans_resolve_list = []

    origin_class.cultist_stars = {'blue': 0, 'red': 0}
    origin_class.total_health_teams = {'blue': 0, 'red': 0}
    origin_class.galio_spawn_time = {'blue': 0, 'red': 0}

    for o in origin_class.amounts:
        origin_class.amounts[o] = {'blue': 0, 'red': 0}

    origin_class.divine_attack_list = []
    origin_class.divine_list = []
    origin_class.elderwood_list = {'blue': 0, 'red': 0}
    origin_class.spirit_list = []
    origin_class.duelist_helper_list = []
    origin_class.shade_helper_list = []


def survive_combat(player, champ_list):
    for champ in champ_list:
        if player.board[champ.starting_x][champ.starting_y]:
            player.board[champ.starting_x][champ.starting_y].survive_combat = True
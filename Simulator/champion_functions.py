import Simulator.config as config
import Simulator.origin_class as origin_class
import Simulator.origin_class_stats as origin_class_stats
import Simulator.stats as stats
import random
from math import ceil
from Simulator import ability, active, field, item_stats, items
from Simulator.stats import *

MILLISECONDS = 0


def MILLIS():
    return MILLISECONDS


def MILLISECONDS_INCREASE():
    global MILLISECONDS
    MILLISECONDS += 10


damage_dealt = []
damage_dealt_teams = {'blue': 0, 'red': 0}


def get_damage_dealt():
    return damage_dealt


galio_spawned = {'blue': False, 'red': False}


def add_damage_dealt(champion, damage, target):
    global damage_dealt
    global damage_dealt_teams
    global galio_spawned
    added = False

    if champion.team == 'blue':
        damage_dealt_teams['blue'] += damage
    if champion.team == 'red':
        damage_dealt_teams['red'] += damage

    # cultists galio spawning
    teams = [['blue', 'red'], ['red', 'blue']]
    for t in teams:
        if (not galio_spawned[t[0]] and origin_class.amounts['cultist'][t[0]]
                >= origin_class_stats.tiers['cultist'][0]):
            if damage_dealt_teams[t[1]] > origin_class.total_health_teams[t[0]] * config.GALIO_TEAM_HEALTH_PERCENTAGE:
                galio_spawned[t[0]] = True
                origin_class.cultist(target, t[0])

    for d in damage_dealt:
        if d['champion'] == champion:
            d['damage'] += damage
            added = True
            break
    if not added:
        damage_dealt.append({'champion': champion, 'damage': damage})


def reset_stat(champion, stat):
    if stat in ['movement_delay']:
        return config.MOVEMENTDELAY
    else:
        return eval(stat)[champion.name]


def attack(champion, target, bonus_dmg=0, item_attack=False, trait_attack='', set_ad=None):
    attackable_enemies = list(filter(lambda x: (x.champion and x.health > 0), champion.enemy_team()))
    if not target and len(attackable_enemies) > 0:
        field.find_target(champion)
        target = champion.target
        # allow forced attacks (xinzhao spin etc.)
    if (champion.idle or bonus_dmg or item_attack or trait_attack) and target:

        # enforcing the max AS rule. remembered that one too late
        # the best way for not creating a shitload of bugs
        if champion.AS > 5:
            champion.add_que('change_stat', -1, None, 'AS', 5.00)

        dodge_random = random.randint(1, 100) / 100
        crit_random = random.randint(1, 100) / 100

        items.deathblade(champion, target)  # deathblade (needs to take effect before AD is used)
        items.gargoyle_stoneplate(target)  # gargoyle_stoneplate (needs to take effect before armor or MR is used)

        if (not item_attack and not trait_attack) or trait_attack == 'hunter':
            items.runaans_hurricane(champion, target)  # runaans_hurricane
            items.guinsoos_rageblade(champion)  # guinsoos_rageblade
            items.statikk_shiv(champion, target)  # statikk_shiv

        # testing whether the hit is going to be a crit. the crit dmg add is done later.
        if crit_random < champion.crit_chance and 'bramble_vest' not in target.items:
            items.last_whisper(champion, target)  # last_whisper (needs to change the armor value before calculations)

        enemy_team = 'red' if champion.team == 'blue' else 'blue'

        true_damage = 0
        damage = 0

        # passives
        if champion.name in config.ATTACK_PASSIVES:
            if not item_attack:
                active_data = getattr(active, champion.name)(champion, target)
                if active_data['true_damage']:
                    true_damage += active_data['damage'] * items.giant_slayer(champion, target)  # gians_slayer -item
                else:
                    damage += active_data['damage'] * items.giant_slayer(champion, target)  # gians_slayer -item

                # because warwick needs to simulate whether he's going to kill the target,
                # let's keep the same parameters here as well
                if active_data['dodge_random']:
                    dodge_random = active_data['dodge_random']
                if active_data['crit_random']:
                    crit_random = active_data['crit_random']

        # calculating damage after armor reductions
        if not set_ad:
            damage += champion.AD * items.giant_slayer(champion, target)  # -item
        else:
            damage += set_ad * items.giant_slayer(champion, target)  # gians_slayer -item
        damage += bonus_dmg
        if not champion.pumped_up:  # the_boss -trait
            if target.armor >= 0:
                damage = damage * (100 / (100 + target.armor))
            else:
                damage = damage * (2 - 100 / (100 - target.armor))

        # dodging
        dodge_string = ''
        # applying rapid_firecannon's unmissableness
        if 'rapid_firecannon' not in champion.items and dodge_random < target.dodge:
            damage = 0
            dodge_string = ' dodge'

        damage += champion.deal_bonus_true_damage * damage  # divine -trait

        damage *= target.receive_increased_damage
        damage *= target.receive_decreased_damage
        damage += true_damage
        damage *= champion.deal_increased_damage

        damage -= target.damage_reduction
        if damage < 0:
            damage = 0
        if target.immune or target.autoimmune:
            damage = 0

        crit_string = ''  # bramble vest -item
        if crit_random < champion.crit_chance and 'bramble_vest' not in target.items:
            damage *= champion.crit_damage
            crit_string = ' crit'

        item_string = ''
        if item_attack:
            item_string = ' item'

        trait_string = ''
        if trait_attack:
            trait_string = ' {}'.format(trait_attack)
        if not (champion.name == 'galio' and crit_string):

            if champion.lifesteal > 0:
                champion.add_que('heal', -1, None, None, damage * champion.lifesteal)

            # if runaans_hurricane has killed the target before the actual attack finishes.
            # there's another check below but not going to touch that
            if target.health >= 0:

                add_damage_dealt(champion, damage, target)

                # bramble_vest -item
                items.bramble_vest(target)

                # shield
                shield_old = target.shield_amount()

                if len(target.shields) > 0:
                    while not (damage <= 0 or target.shield_amount() <= 0):
                        top_shield = target.shields[0]['amount']
                        target.shields[0]['amount'] -= damage
                        if target.shields[0]['amount'] < 0:
                            damage -= top_shield
                            target.shields = target.shields[1:]
                        else:
                            damage = 0

                # kalista is programmed to add a new spear -->
                # and pull the spears IF spears_in_target * spear_dmg_after_MR + auto_dmg_after_armor >
                # target.hp + target.shield_amount()
                # sometimes the spears deal enough dmg to kill the target and
                # since the spear_pull happens just the same time as the auto,
                # but is registered before, the target may already be dead (also happens with zed)
                if target.health > 0:
                    champion.print(' attacks ' + '{:<8}'.format(enemy_team) + ' ' + '{:<13}'.format(
                        target.name) + '{:<5}--> {:<8}   shield {:<5}--> {:<5} {}{}{}{}'.format(ceil(target.health),
                                                                                                ceil(
                                                                                                    target.health - damage),
                                                                                                ceil(shield_old), ceil(
                            target.shield_amount()), crit_string, dodge_string, item_string, trait_string))
                    # dealing the damage and killing the enemy if necessary
                    target.health -= damage
                    if (MILLIS() > target.castMS + target.manalock
                            and not target.ability_active and target.maxmana > 0):
                        if not target.name == 'riven' or ability.riven_helper(target, {}):
                            old_mana = target.mana
                            target.mana += min((damage * config.MANA_DAMAGE_GAIN) * target.mana_generation,
                                               config.MAX_MANA_FROM_DAMAGE)
                            target.print(' mana {} --> {}'.format(round(old_mana, 1), round(target.mana, 1)))

                    # titans_resolve -item
                    # add bonus damage and armors after the values have been used.
                    # this way they will be added now but used only in the next event
                    items.titans_resolve(champion, target, crit_string)

                    # the_boss -trait
                    if (target.name == 'sett' and not target.done_situps and target.health < target.max_health *
                            origin_class_stats.threshold['the_boss']):
                        if target.health <= 0:
                            target.health = 1
                        origin_class.the_boss(target)

                    origin_class.duelist_helper(champion)  # duelist -trait

                    if target.health <= 0:
                        target.die()
                    elif not item_attack:
                        origin_class.divine(champion, target, True)  # divine -trait

                    # sharpshooter -trait
                    if not item_attack and not trait_attack:
                        origin_class.sharpshooter(champion, target, None, bonus_dmg, False)

                # apply manalock. only give mana of the attack if it has been 1000ms since the last ability cast
                if (champion.champion and MILLIS() > champion.castMS + champion.manalock
                        and not champion.ability_active and champion.maxmana > 0
                        and not item_attack and not trait_attack):
                    if not champion.name == 'riven' or ability.riven_helper(champion, {}):
                        old_mana = champion.mana
                        champion.mana += (config.MANA_PER_ATTACK * champion.mana_generation)
                        champion.mana += (
                                items.spear_of_shojin(champion) * champion.mana_generation)  # spear of shojin -item
                        champion.print(' mana {} --> {}'.format(round(old_mana, 1), round(champion.mana, 1)))

                # aphelios turret triggering aphelios's shojins
                if champion.name == 'aphelios_turret' and \
                        MILLIS() > champion.overlord.castMS + champion.overlord.manalock and not trait_attack:
                    old_mana = champion.overlord.mana
                    champion.overlord.mana += (items.spear_of_shojin(champion) * champion.overlord.mana_generation)
                    # spear of shojin -item
                    champion.overlord.print(' mana {} --> {}'.format(round(old_mana, 1),
                                                                     round(champion.overlord.mana, 1)))

                if champion.heal_per_attack > 0:
                    health_old = champion.health
                    champion.health += (champion.heal_per_attack * champion.healing_strength)

                    if champion.health > champion.max_health:
                        champion.health = champion.max_health
                    champion.print(' heals ' + '{:<5}--> {:<8}'.format(ceil(health_old), ceil(champion.health)))

                # applying attack speed pause
                champion.idle = False
                if champion.name == 'aphelios_turret':
                    champion.overlord.add_que('clear_idle', 1 / champion.overlord.AS * 1000, None, None, None,
                                              {'underlord': champion})

        else:
            origin_class.cultist_helper(champion, damage, target)
        if (not item_attack and not trait_attack) or champion.name == 'ashe':
            champion.idle = False
            champion.clear_que_idle()
            champion.add_que('clear_idle', 1 / champion.AS * 1000)
            origin_class.shade_helper(champion)


def die(champion):
    enemy_team = champion.enemy_team()
    # mark everyone's target to be 'None' who targeted this champion
    for c in enemy_team:
        if c.target == champion: c.target = None

    if not champion.will_revive[0][0] and not champion.will_revive[1][0]:

        # free the coordinates
        field.coordinates[champion.y][champion.x] = None
        # Ran into a bug with this being removed. I'll look into where own_team is defined later
        if champion in champion.own_team():
            champion.own_team().remove(champion)
        champion.print(' dies ')

        # zzrot_portal
        if 'zzrot_portal' in champion.items:
            items.zzrot_portal_helper(champion)

        if not champion.champion and hasattr(champion, 'overlord') and not champion.name == 'sett':
            if champion.overlord and champion in champion.overlord.underlords:
                champion.overlord.underlords.remove(champion)

        # kill the dependants
        for u in champion.underlords:
            for c in enemy_team:
                if c.target == u:
                    c.target = None
            field.coordinates[u.y][u.x] = None
            if u.name == 'aphelios_turret' or (u.name == 'sandguard' and u.health >= 0):
                if u in champion.own_team():
                    champion.own_team().remove(u)
                    champion.print(' {:<15}'.format(u.name) + ' dies ')

    # if the champion has zilean orb or guardian angel equipped
    else:
        # set the unit in a state where it's not targetable or attackable but still resides in that hex
        # also stun the unit so it's not able to move etc.
        champion.add_que('change_stat', -1, None, 'stunned', True)
        champion.add_que('change_stat', -1, None, 'champion', False)
        champion.clear_que_stunned_removal()
        champion.print(' is reviving')

        # zilean revive
        if champion.will_revive[0][0]:
            zilean = champion.will_revive[0][0]
            revive_delay = ABILITY_LENGTH[zilean.name][zilean.stars]
            revive_hp = ABILITY_HEAL[zilean.name][zilean.stars] * zilean.SP

            if revive_hp > champion.max_health:
                revive_hp = champion.max_health

            as_gain = (stats.ABILITY_AS_GAIN[zilean.name][zilean.stars] - 1) * zilean.SP + 1
            champion.add_que('change_stat', revive_delay, None, 'AS', champion.AS * as_gain)
            # the first slot is for zilean. clear it and keep the second (GA) if equipped
            champion.add_que('change_stat', revive_delay, None, 'will_revive', [[None], [champion.will_revive[1][0]]])

        # GA revive
        else:
            revive_delay = item_stats.cooldown['guardian_angel']
            revive_hp = item_stats.heal['guardian_angel']
            # clear the second slot but keep zilean still in the first.
            # tho this never happens since zilean is set to activate first.
            champion.add_que('change_stat', revive_delay, None, 'will_revive', [[champion.will_revive[0][0]], [None]])

        # reviving
        if revive_hp > champion.max_health:
            revive_hp = champion.max_health

        champion.add_que('change_stat', revive_delay, None, 'health', revive_hp)
        champion.add_que('change_stat', revive_delay, None, 'burning', False)
        champion.add_que('change_stat', revive_delay, None, 'disarmed', False)
        champion.add_que('change_stat', revive_delay, None, 'blinded', False)
        champion.add_que('change_stat', revive_delay, None, 'stunned', False)
        champion.add_que('change_stat', revive_delay, None, 'damage_reduction', 0)
        champion.add_que('change_stat', revive_delay, None, 'idle', True)
        champion.add_que('change_stat', revive_delay, None, 'champion', True)

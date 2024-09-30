from Simulator import field, item_stats
import Simulator.champion_functions as champion_functions
import Simulator.stats as stats
import Simulator.origin_class as origin_class
import random

# ALL FUNCTIONS REGARDING ITEMS ARE HERE
# functions are named as just 'item_name'
# yea, kinda hard coded the item names inside the functions
# no biggie tho.


def change_stat(champion, stat, value, message=''):
    start_value = getattr(champion, stat)
    end_value = value

    if message == 'initiate_item_stat_change':
        message = ''
        if stat == 'health':
            change_stat(champion, 'max_health', value)

    if value != start_value and champion.health > 0:
        if isinstance(start_value, float):
            start_value = round(start_value, 2)
        if isinstance(end_value, float):
            end_value = round(value, 2)
        champion.print(' {} {} --> {} {}'.format(stat, start_value, end_value, message))
    setattr(champion, stat, value)


def initiate(champion):
    items = item_stats.items
    for i in champion.items:
        data = items[i]
        for stat in data:

            value = data[stat]
            original_value = getattr(champion, stat)
            if stat == 'AS':
                change_stat(champion, stat, original_value * value)
            elif stat == 'spell_damage_reduction_percentage':
                change_stat(champion, stat, original_value * value)
            elif stat == 'will_revive':
                change_stat(champion, stat, value)
            else:
                change_stat(champion, stat, original_value + value, 'initiate_item_stat_change')
            
        if i in item_stats.initiative_items:
            eval(i)(champion)

# where item functions are based at

def blue_buff(champion):
    if('blue_buff' in champion.items):
        change_stat(champion, 'mana', 20)


bramble_vest_list = []
def bramble_vest(champion):
    millis = champion_functions.MILLIS()

    if('bramble_vest' in champion.items):
        item_amount = len(list(filter(lambda x: x == 'bramble_vest', champion.items)))

        last_activation = list(filter(lambda x: x[0] == champion, bramble_vest_list))
        last_activation = sorted(last_activation, key=lambda x: x[1], reverse=True)
        if(len(last_activation) == 0 or (millis - last_activation[0][1]) > item_stats.cooldown['bramble_vest']):
        
            bramble_vest_list.append([champion, millis])
            neighboring_enemies = field.enemies_in_distance(champion, champion.y, champion.x, 1)
            for i in range(0, item_amount):
                for n in neighboring_enemies:
                    champion.spell(n, item_stats.damage['bramble_vest'][champion.stars], 0, True)


def adaptive_helm(champion):
    return


#adding stack whenever dealing damage to a target
#at the same time checking if any of the old stacked enemies are dead. if so, add x AD
deathblade_list = []
def deathblade(champion, target):
    if('deathblade' in champion.items):
        item_amount = len(list(filter(lambda x: x == 'deathblade', champion.items)))

        old_list = list(filter(lambda x: x[0] == champion, deathblade_list))
        for t in old_list:
            if(t[1].health <= 0):
                change_stat(champion, 'AD', champion.AD + item_stats.AD['deathblade'] * item_amount)
                deathblade_list.remove([champion, t[1]])

        if([champion, target] not in deathblade_list):
            deathblade_list.append([champion, target])

def edge_of_night(champion):
    return

def protectors_vow(champion):
    return


gargoyle_stoneplate_list = []


def gargoyle_stoneplate(target):
    if 'gargoyle_stoneplate' in target.items:
        item_amount = len(list(filter(lambda x: x == 'gargoyle_stoneplate', target.items)))
        
        # current enemies targeting this unit
        new_targeters = len(list(filter(lambda x: x.target == target, target.enemy_team())))

        old_targeters = 0

        # old targeters when looking at the list. if the list regarding this unit is empty, the value is zero
        holder_list = list(filter(lambda x: x[0] == target, gargoyle_stoneplate_list))
        if len(holder_list) > 0:

            # store the old value and replace it with the new one
            old_targeters = holder_list[0][1]
            for g in gargoyle_stoneplate_list:
                if g[0] == target: g[1] = new_targeters
        else:
            gargoyle_stoneplate_list.append([target, new_targeters])

        # calculate the difference and adjust the MR and armor -values
        difference = new_targeters - old_targeters
        change_stat(target, 'MR', target.MR + difference * item_stats.MR['gargoyle_stoneplate'] * item_amount)
        change_stat(target, 'armor', target.armor + difference * item_stats.armor['gargoyle_stoneplate'] * item_amount)


def giant_slayer(champion, target):
    if('giant_slayer' in champion.items):
        item_amount = len(list(filter(lambda x: x == 'giant_slayer', champion.items)))

        if(target.max_health > 1750): return 1 + (item_amount * 0.90)
        else: return 1 + (item_amount * 0.10)

    else:
        return 1.00


def guinsoos_rageblade(champion):
    if 'guinsoos_rageblade' in champion.items and champion.AS < 5.00:
        item_amount = len(list(filter(lambda x: x == 'guinsoos_rageblade', champion.items)))

        for i in range(0, item_amount):
            change_stat(champion, 'AS', champion.AS * item_stats.item_as_increase['guinsoos_rageblade'])


def hand_of_justice(champion):
    units = champion.own_team() + champion.enemy_team()

    holders = list(filter(lambda x: 'hand_of_justice' in x.items, units))
    for h in holders:
        item_amount = len(list(filter(lambda x: x == 'hand_of_justice', h.items)))

        for i in range(0, item_amount):
            r = random.randint(1, 2)
            if r == 1:
                change_stat(h, 'SP', h.SP + item_stats.SP['hand_of_justice'])
                change_stat(h, 'AD', h.AD * item_stats.AD_percentage['hand_of_justice'])
            if r == 2:
                change_stat(h, 'lifesteal', h.lifesteal + item_stats.lifesteal['hand_of_justice'])
                change_stat(h, 'lifesteal_spells', h.lifesteal_spells + item_stats.lifesteal_spells['hand_of_justice'])


hextech_gunblade_list = []


def hextech_gunblade(champion, damage):
    if 'hextech_gunblade' in champion.items:
        item_amount = len(list(filter(lambda x: x == 'hextech_gunblade', champion.items)))
        
        healable = champion.max_health - champion.health

        heal = damage * (item_stats.heal_percentage['hextech_gunblade'] * item_amount)

        # if we can just heal some amount without shields, initiate the heal
        if healable > heal:
            champion.add_que('heal', -1, None, None, heal)
        
        else:
            # if we know there's more healing to do than hp missing, heal the amount (if the champion isn't max hp)
            if healable > 0: champion.add_que('heal', -1, None, None, healable)
            heal -= healable

            # figure out the shield amount
            shield_max = item_stats.shield_max['hextech_gunblade'] * item_amount
            

            # if there is an old shield, get the identifier from 'hextech_gunblade_list' which uses the following syntax:
            # [champion, identifier]
            old_shield_identifier = list(filter(lambda x: x[0] == champion, hextech_gunblade_list))

            # now get that shield (if it still exists) from the champion's shield list using the identifier
            # those unique identifiers were a clever idea
            old_shield = []
            shield_before = champion.shield_amount()
            if len(old_shield_identifier) > 0:
                old_shield_identifier = old_shield_identifier[0]
                old_shield = list(filter(lambda x: (x['identifier'] == old_shield_identifier[1] and x['applier'] == champion), champion.shields))

                # remove the old hextech shield and replace it with a new one
                for s in champion.shields:
                    if s['applier'] == champion and s['identifier'] == old_shield_identifier[1]:
                        champion.shields.remove(s)

            if len(old_shield) > 0:
                heal += old_shield[0]['amount']

            if heal > shield_max:
                heal = shield_max
            heal = round(heal, 1)

            # PROBABLY COULD'VE JUST MODIFIED THE OLD SHIELD THO
            # when I rly think about it
            
            shield_identifier = round(champion_functions.MILLIS() * heal)
            # setting these shields to expire a long long time after the battle is over.
            # this way we can just remove the old shield here.
            champion.add_que('shield', 0, None, None, {'amount': heal, 'identifier': shield_identifier,
                                                       'applier': champion, 'original_amount': heal},
                             {'increase': True, 'expires': 9999999, 'shield_before': shield_before})

            # if there's no entries in the list, add one. otherwise just replace the list's identifier
            if len(old_shield_identifier) == 0:
                hextech_gunblade_list.append([champion, shield_identifier])
            else:
                for i, h in enumerate(hextech_gunblade_list):
                    if h[0] == champion:
                        hextech_gunblade_list[i][1] = shield_identifier


def infinity_edge(champion):
    units = champion.own_team() + champion.enemy_team()

    holders = list(filter(lambda x: 'infinity_edge' in x.items, units))
    for h in holders:
        bonus_damage = h.crit_chance + item_stats.crit_chance['infinity_edge'] - 1

        new_crit_chance = h.crit_chance + item_stats.crit_chance['infinity_edge']
        if new_crit_chance > 1: new_crit_chance = 1
        change_stat(h, 'crit_chance', new_crit_chance)

        if bonus_damage >= 0.01:
            change_stat(h, 'crit_damage', h.crit_damage + bonus_damage)


# how many ionic spark holding enemies are in the range
ionic_spark_list = []


def ionic_spark(champion):
    units = champion.own_team() + champion.enemy_team()
    for u in units:
        old_counter = u.ionic_sparked

        counter = 0
        enemies_in_distance = field.enemies_in_distance(u, u.y, u.x, item_stats.item_range['ionic_spark'])
        for e in enemies_in_distance:
            if e and 'ionic_spark' in e.items:
                counter += 1
        change_stat(u, 'ionic_sparked', counter)

        if old_counter == 0 and counter > 0:
            change_stat(u, 'MR', u.MR * item_stats.item_mr_decrease['ionic_spark'])
        elif old_counter > 0 and counter == 0:
            change_stat(u, 'MR', u.MR / item_stats.item_mr_decrease['ionic_spark'])


last_whisper_list = [] #[target, ms]
def last_whisper(champion, target):
    millis = champion_functions.MILLIS()

    if('last_whisper' in champion.items):

        last_activation = list(filter(lambda x: x[0] == target, last_whisper_list))
        last_activation = sorted(last_activation, key=lambda x: x[1], reverse=True)

        length = item_stats.item_change_length['last_whisper']
        if(len(last_activation) == 0 or (millis - last_activation[0][1]) > length):
            change_stat(target, 'armor', target.armor * item_stats.item_armor_decrease['last_whisper'])

            #using old logic which was needed last time at vi's ult
            target.add_que('change_stat', length, None, 'armor', None, {'vi': item_stats.item_armor_decrease['last_whisper']})
            last_whisper_list.append([target, millis])

def crownguard(champion):
    return


def archangels_staff(champion, target):
    return


def morellonomicon(champion, target):
    if 'morellonomicon' in champion.items:
        champion.burn(target)


def red_buff(champion):
    return

def redemption(champion):
    if('redemption' in champion.items):

        item_amount = len(list(filter(lambda x: x == 'redemption', champion.items)))

        own_team = champion.own_team()
        if(len(own_team) > 0):
            for o in own_team:
                o.add_que('heal', -1, None, None, item_stats.heal['redemption'] * item_amount)

        champion.items = list(filter(lambda x: x != 'redemption', champion.items))


def runaans_hurricane(champion, target):
    if('runaans_hurricane' in champion.items):
        item_amount = len(list(filter(lambda x: x == 'runaans_hurricane', champion.items)))
        for i in range(0, item_amount):
            
            
            enemy_team = champion.enemy_team() 
            targets = list(filter(lambda x: x != target, enemy_team))
            random.shuffle(targets)

            runaans_target = target
            if(len(targets) > 0): runaans_target = targets[0]
            champion.attack(champion.AD * item_stats.damage['runaans_hurricane'] - champion.AD, runaans_target, True)


#hexagonal coordinates are fun and all
def steadfast_heart(champion):
    return



def spear_of_shojin(champion):
    if('spear_of_shojin' in champion.items):
        item_amount = len(list(filter(lambda x: x == 'spear_of_shojin', champion.items)))
        return item_stats.mana['spear_of_shojin'] * item_amount
    else: return 0




statikk_shiv_list = []
def statikk_shiv(champion, target):
    if('statikk_shiv' in champion.items):
        item_amount = len(list(filter(lambda x: x == 'statikk_shiv', champion.items)))

        inx = -1
        for i, s in enumerate(statikk_shiv_list):
                if(s[0] == champion): inx = i



        if(inx == -1):
            statikk_shiv_list.append([champion, 1])
        else:
            statikk_shiv_list[inx][1] += 1
            current_count = statikk_shiv_list[inx][1]
            if(current_count == item_stats.item_activate_every_x_attacks['statikk_shiv']):
                statikk_shiv_list[inx][1] = 0

                #find all enemy units except current target and randomize the list
                enemy_team = champion.enemy_team() 
                targets = list(filter(lambda x: x != target, enemy_team))
                random.shuffle(targets)

                #limit the list to the max extra targets specified in item_stats.py
                #finally add the current target to the list
                max_extra_targets = item_stats.item_targets['statikk_shiv'][champion.stars]
                if(len(targets) > max_extra_targets): targets = targets[:max_extra_targets]
                targets.append(target)

                for t in targets:
                    damage = item_stats.damage['statikk_shiv'] * item_amount
                    true_damage = 0
                    if(t.shield_amount() > 0): true_damage += item_stats.true_damage['statikk_shiv'] * item_amount

                    champion.spell(t, damage, true_damage, True)



def sunfire_cape(champion, data = {'loop': False}):

    #the function gets called immediately which means that the enemy team could still be empty
    #call it again in one millisecond when the round has actually started
    if(not data['loop']): 
        champion.add_que('execute_function', 1, [sunfire_cape, {'loop': True}])
    else:

        enemies = field.enemies_in_distance(champion, champion.y, champion.x, item_stats.item_range['sunfire_cape'])
        random.shuffle(enemies)
        if(len(enemies) > 0):
            target = enemies[0]

            champion.burn(target)

            #just call this same function every x seconds
            champion.add_que('execute_function', item_stats.cooldown['sunfire_cape'], [sunfire_cape, {'loop': True}])


def thieves_gloves(champion):
    ...

titans_resolve_list = [] #[champion, stacks, maxxed]
def titans_resolve(champion, target, crit):

    # attacker in spells and physical attacks
    if 'titans_resolve' in champion.items:
        if(crit): titans_resolve_helper(champion)

    # target in both cases
    if 'titans_resolve' in target.items:
        titans_resolve_helper(target)


#increases stacks


def titans_resolve_helper(unit):
    current_stacks = list(filter(lambda x: x[0] == unit, titans_resolve_list))
    maxxed = False
    if len(current_stacks) == 0:
        current_stacks = 0
        titans_resolve_list.append([unit, 0, False])
    else: 
        maxxed = current_stacks[0][2]
        current_stacks = current_stacks[0][1]

    if current_stacks < 25:
        increase = item_stats.item_deal_increased_damage_increase['titans_resolve']
        change_stat(unit, 'deal_increased_damage', unit.deal_increased_damage + increase)

        for i, t in enumerate(titans_resolve_list):
            if t[0] == unit: titans_resolve_list[i][1] += 1
        current_stacks += 1
    
    if current_stacks == 25 and not maxxed:
        change_stat(unit, 'MR', unit.MR + item_stats.armor['titans_resolve'])
        change_stat(unit, 'armor', unit.armor + item_stats.MR['titans_resolve'])
        for i, t in enumerate(titans_resolve_list):
            if t[0] == unit:
                titans_resolve_list[i][2] = True

    
# see item notes
def guardbreaker(champion, target):
    return


def steraks_gage(champion):
    return

def evenshroud(champion):
    return

def nashors_tooth(champion):
    return

def absolution(champion):
    return

def blessed_bloodthirster(champion):
    return

def blue_blessing(champion):
    return

def brink_of_dawn(champion):
    return

def bulwarks_oath(champion):
    return

def covalent_spark(champion):
    return

def crest_of_cinders(champion):
    return

def demonslayer(champion):
    return

def dragons_will(champion):
    return

def dvarapala_stoneplate(champion):
    return

def equinox(champion):
    return

def eternal_whisper(champion):
    return

def fist_of_fairness(champion):
    return

def glamourous_gauntlet(champion):
    return

def guinsoos_reckoning(champion):
    return

def hextech_lifeblade(champion):
    return

def jaksho_the_protean(champion):
    return

def legacy_of_the_colossus(champion):
    return

def luminous_deathblade(champion):
    return

def more_more_ellonomicon(champion):
    return

def quickestsilver(champion):
    return

def rabadons_ascended_deathcap(champion):
    return

def rascals_gloves(champion):
    return

def rosethrorn_vest(champion):
    return

def royal_crownshield(champion):
    return

def runaans_tempest(champion):
    return

def spear_of_hirana(champion):
    return

def statikk_favor(champion):
    return

def steraks_megashield(champion):
    return

def sunlight_cape(champion):
    return

def the_barons_gift(champion):
    return

def titans_vow(champion):
    return

def urf_angels_staff(champion):
    return

def warmogs_pride(champion):
    return

def willbreaker(champion):
    return

def zenith_edge(champion):
    return

def accomplices_glove(champion):
    return

def aegis_of_the_legion(champion):
    return

def banshees_veil(champion):
    return

def chalice_of_power(champion):
    return

def knights_vow(champion):
    return

def locket_of_the_iron_solari(champion):
    return

def moonstone_renewer(champion):
    return

def needlessly_big_gem(champion):
    return

def obsidian_cleaver(champion):
    return

def randuins_omen(champion):
    return

def shroud_of_stillness(champion):
    return

def spite(champion):
    return

def the_eternal_flame(champion):
    return

def unstable_treasure_chest(champion):
    return

def virtue_of_the_martyr(champion):
    return

def zekes_herald(champion):
    return

def zephyr(champion):
    return

def zzrot_portal(champion):
    return

def anima_visage(champion):
    return

def blacksmiths_gloves(champion):
    return

def blighting_jewel(champion):
    return

def corrupt_vampiric_scepter(champion):
    return

def deaths_defiance(champion):
    return

def deathfire_grasp(champion):
    return

def eternal_winter(champion):
    return

def fishbones(champion):
    return

def forbidden_idol(champion):
    return

def gamblers_blade(champion):
    return

def gold_collector(champion):
    return

def horizon_focus(champion):
    return

def hullcrusher(champion):
    return

def infinity_force(champion):
    return

def innervating_locket(champion):
    return

def lich_bane(champion):
    return

def lightshield_crest(champion):
    return

def ludens_tempest(champion):
    return

def manazane(champion):
    return

def mittens(champion):
    return

def moguls_mail(champion):
    return

def prowlers_claw(champion):
    return

def rapid_firecannon(champion):
    return

def seekers_armguard(champion):
    return

def silvermere_dawn(champion):
    return

def snipers_focues(champion):
    return

def spectral_cutlass(champion):
    return

def talisman_of_ascension(champion):
    return

def tricksters_glass(champion):
    return

def unending_despair(champion):
    return

def wits_end(champion):
    return

def zhonyas_paradox(champion):
    return

def sugarcraft_emblem(champion):
    return

def frost_emblem(champion):
    return

def eldritch_emblem(champion):
    return

def portal_emblem(champion):
    return

def witchcraft_emblem(champion):
    return

def pyro_emblem(champion):
    return

def honeymancy_emblem(champion):
    return

def faerie_emblem(champion):
    return

def hunter_emblem(champion):
    return

def bastion_emblem(champion):
    return

def shapeshifter_emblem(champion):
    return

def mage_emblem(champion):
    return

def preserver_emblem(champion):
    return

def multistriker_emblem(champion):
    return

def warrior_emblem(champion):
    return

def scholar_emblem(champion):
    return

def tacticians_crown(champion):
    return

def tacticians_cape(champion):
    return

def tacticians_shield(champion):
    return
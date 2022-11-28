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
            if(stat == 'AS'):                                   change_stat(champion, stat, original_value * value)
            elif(stat == 'spell_damage_reduction_percentage'):  change_stat(champion, stat, original_value * value)
            elif(stat == 'will_revive'):                        change_stat(champion, stat, value)
            
            else: change_stat(champion, stat, original_value + value, 'initiate_item_stat_change')
            
        if(i in item_stats.initiative_items):
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


def chalice_of_power(champion):
    units = champion.own_team() + champion.enemy_team()
    holders = list(filter(lambda x: 'chalice_of_power' in x.items, units))

    coords = field.coordinates
    for holder in holders:
        item_amount = len(list(filter(lambda x: x == 'chalice_of_power', holder.items)))

        hexes = []
        if(holder.x >= 1): hexes.append(coords[holder.y][holder.x - 1])
        hexes.append(coords[holder.y][holder.x])
        if(holder.x <= 5): hexes.append(coords[holder.y][holder.x + 1])
    
        for h in hexes:
            if(h and h.team == champion.team and h.champion):
                change_stat(h, 'SP', h.SP + item_stats.SP['chalice_of_power'] * item_amount)


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


frozen_heart_list = []
def frozen_heart(champion):
    units = champion.own_team() + champion.enemy_team()

    #if a unit has died and they had some enemies affected, clear those debuffs
    for i in range(0, 5):
        for u in frozen_heart_list:
            if(u[0] not in units):
                for c in u[1]:
                    change_stat(c, 'AS', c.AS / item_stats.item_as_decrease['frozen_heart'])
                frozen_heart_list.remove(u)

    has_item = list(filter(lambda x: 'frozen_heart' in x.items, units))

    #loop through every unit with the item
    for c in has_item:
        items = 0
        for i in c.items:
            if(i == 'frozen_heart'): items += 1

        c_in_list = list(filter(lambda x: x[0] == c, frozen_heart_list))
        units_in_c_list = []
        new_c_list = []

        #current affected list by this holder
        if(len(c_in_list) > 0):
            c_in_list = c_in_list[0]
            units_in_c_list = c_in_list[1]

        #if current neighbors are not on the list, debuff
        #if they are on the list, keep them on it
        current_neighbors = field.enemies_in_distance(c, c.y, c.x, items)
        for n in current_neighbors:
            if(n not in units_in_c_list):
                change_stat(n, 'AS', n.AS * item_stats.item_as_decrease['frozen_heart'])
            if(n not in new_c_list):
                new_c_list.append(n)

        #if there are some units on the list that aren't neighbors, clear debuff
        #otherwise keep the neighbor in the list
        for u in units_in_c_list:
            if(u not in current_neighbors):
                change_stat(u, 'AS', u.AS / item_stats.item_as_decrease['frozen_heart'])
            elif(n not in new_c_list): new_c_list.append(u)

        #if there are no entries of this unit's list, append one
        if(len(c_in_list) == 0):
            frozen_heart_list.append([c, new_c_list])
        #otherwise just replace the list of affected enemy units
        else:
            for u in frozen_heart_list:
                if(u[0] == c):
                    u[1] = new_c_list
                    break


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

def locket_of_the_iron_solari(champion):
    units = champion.own_team() + champion.enemy_team()
    holders = list(filter(lambda x: 'locket_of_the_iron_solari' in x.items, units))

    coords = field.coordinates
    for holder in holders:
        item_amount = len(list(filter(lambda x: x == 'locket_of_the_iron_solari', holder.items)))


        hexes = []
        if(holder.x >= 2): hexes.append(coords[holder.y][holder.x - 2])
        if(holder.x >= 1): hexes.append(coords[holder.y][holder.x - 1])
        hexes.append(coords[holder.y][holder.x])
        if(holder.x <= 5): hexes.append(coords[holder.y][holder.x + 1])
        if(holder.x <= 4): hexes.append(coords[holder.y][holder.x + 2])

        for h in hexes:
            if(h and h.team == holder.team and h.champion):
                shield_size = item_stats.shield['locket_of_the_iron_solari'][holder.stars] * item_amount

                shield_identifier = round(champion_functions.MILLIS() * shield_size + holder.armor)
                shield_length = item_stats.item_change_length['locket_of_the_iron_solari']
                h.add_que('shield', -1, None, None, {'amount': shield_size, 'identifier': shield_identifier, 'applier': holder, 'original_amount': shield_size}, {'increase': True, 'expires': shield_length})


def ludens_echo(champion, target):

    if 'ludens_echo' in champion.items:
        item_amount = len(list(filter(lambda x: x == 'ludens_echo', champion.items)))

        # change the flag
        champion.spell_has_used_ludens = True

        # find the enemies in range and sort out the proper amount. always include the target
        enemies_in_range = field.enemies_in_distance(champion, target.y, target.x, item_stats.item_range['ludens_echo'])
        if target in enemies_in_range:
            enemies_in_range.remove(target)

        target_amount = item_stats.item_targets['ludens_echo']
        if len(enemies_in_range) > target_amount:
            enemies_in_range = enemies_in_range[:target_amount]

        targets = [target]
        targets += enemies_in_range

        champion.print(' luden\'s echo hits {} target{}'.format(len(targets), 's' if (len(targets) > 1) else ''))
        for t in targets:
            damage = item_stats.damage['ludens_echo']
            if (t.stunned or t.disarmed or t.blinded or t.AD_reduction_cc) and not t.name == 'sandguard':
                damage *= 2
            
            damage *= item_amount
            champion.spell(t, damage, 0, True)



def morellonomicon(champion, target):
    if 'morellonomicon' in champion.items:
        champion.burn(target)


def rapid_firecannon(champion):
        change_stat(champion, 'range', champion.range + stats.RANGE[champion.name] * 2)


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
def shroud_of_stillness(champion):
    units = champion.own_team() + champion.enemy_team()
    holders = list(filter(lambda x: 'locket_of_the_iron_solari' in x.items, units))

    for holder in holders:
        affect_odd_x = []
        affect_even_x = []

        #find which hexes will be affected by the shroud
        #depends on champion's position
        if(champion.y % 2 == 0):
            affect_odd_x.append(champion.x)
            if(champion.x <= 5): affect_odd_x.append(champion.x + 1)
            
            if(champion.x >= 1): affect_even_x.append(champion.x - 1)
            affect_even_x.append(champion.x)
            if(champion.x <= 5): affect_even_x.append(champion.x + 1)

        if(champion.y % 2 == 1):
            affect_even_x.append(champion.x)
            if(champion.x >= 1): affect_even_x.append(champion.x - 1)

            if(champion.x >= 1): affect_odd_x.append(champion.x - 1)
            affect_odd_x.append(champion.x)
            if(champion.x <= 5): affect_odd_x.append(champion.x + 1)


        affected_hexes = []
        for i in range(0, 8):
            if(i % 2 == 0):
                for j in affect_even_x:
                    affected_hexes.append([i, j])
            
            if(i % 2 == 1):
                for j in affect_odd_x:
                    affected_hexes.append([i, j])
                    

        coords = field.coordinates
        for h in affected_hexes:
            c = coords[h[0]][h[1]]
            if(c and c.team != champion.team and c.champion):

                #increase next spell mana cost of every enemy in the line by x% = reduce mana by x% of maxmana (can go negative)
                if(not c.mana_cost_increased and c.maxmana > 0):
                    mana_reduce_amount = c.maxmana * item_stats.item_mana_cost_increase['shroud_of_stillness']
                    start_value = c.mana
                    c.mana -= mana_reduce_amount
                    c.print(' {} {} --> {}'.format('mana', round(start_value,1), round(c.mana,1)))
                    c.add_que('change_stat', -1, None, 'mana_cost_increased', True)



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


def thiefs_gloves(champion):
    item_list = item_stats.items
    item_names = list(item_list.keys())
    item_names = item_names[9:]

    item_names.remove('thiefs_gloves')
    item_names.remove('force_of_nature')
    item_names.remove('duelists_zeal')
    item_names.remove('elderwood_heirloom')
    item_names.remove('mages_cap')
    item_names.remove('mantle_of_dusk')
    item_names.remove('sword_of_the_divine')
    item_names.remove('vanguards_cuirass')
    item_names.remove('warlords_banner')
    item_names.remove('youmuus_ghostblade')

    r1 = random.randint(0,len(item_names) - 1)
    r2 = random.randint(0,len(item_names) - 1)
    while r1 == r2:
        r2 = random.randint(0,len(item_names) - 1)

    champion.print(" thiefs_gloves: {} and {}".format(item_names[r1], item_names[r2]))

    champion.items.append(item_names[r1])
    champion.items.append(item_names[r2])


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
def trap_claw(champion, target):

    target.items.remove('trap_claw')

    change_stat(champion, 'stunned', True)
    champion.add_que('change_stat', item_stats.item_stun_duration['trap_claw'], None, 'stunned', False)

    if not target.stunned:
        target.add_que('change_stat', 1, None, 'stunned', False)
    if not target.disarmed:
        target.add_que('change_stat', 1, None, 'disarmed', False)
    if not target.blinded:
        target.add_que('change_stat', 1, None, 'blinded', False)


def zekes_herald(champion):
    units = champion.own_team() + champion.enemy_team()
    holders = list(filter(lambda x: 'zekes_herald' in x.items, units))

    coords = field.coordinates
    for holder in holders:
        item_amount = len(list(filter(lambda x: x == 'zekes_herald', holder.items)))

        hexes = []
        if holder.x >= 1:
            hexes.append(coords[holder.y][holder.x - 1])
        hexes.append(coords[holder.y][holder.x])
        if holder.x <= 5:
            hexes.append(coords[holder.y][holder.x + 1])
    
        for h in hexes:
            if h and h.team == champion.team and h.champion:

                change_stat(h, 'AS', h.AS * ((item_stats.item_as_increase['zekes_herald'] - 1) * item_amount + 1))


def zephyr(champion):
    units = champion.own_team() + champion.enemy_team()
    holders = list(filter(lambda x: 'zephyr' in x.items, units))

    coords = field.coordinates
    for holder in holders:
        item_amount = len(list(filter(lambda x: x == 'zephyr', holder.items)))

        # targeted hex
        target_coords = [7 - holder.y, 6 - holder.x]
        targets = list(filter(lambda x: x.champion, holder.enemy_team()))
        targets = sorted(targets, key=lambda x: field.distance({'y': x.y, 'x': x.x}, {'y': target_coords[0], 'x': target_coords[1]}, False))
        targets = targets[:item_amount]
        for t in targets:
            c = coords[t.y][t.x]
            if c and c.team != holder.team and c.champion:
                c.print(' targeted by zephyr')
                change_stat(c, 'stunned', True)
                change_stat(c, 'champion', False)

                length = item_stats.item_stun_duration['zephyr']
                c.add_que('change_stat', length, None, 'stunned', False)
                c.add_que('change_stat', length, None, 'champion', True)


def zzrot_portal(champion):
    units = champion.own_team() + champion.enemy_team()
    holders = list(filter(lambda x: 'zzrot_portal' in x.items, units))
    for holder in holders:

        # taunting
        neighbor_enemies = field.enemies_in_distance(champion, champion.y, champion.x, 1)
        for n in neighbor_enemies:
            old_target = n.target
            n.add_que('change_target', -1, None, None, champion)
            n.add_que('change_target', item_stats.item_change_length['zzrot_portal'], None, None, old_target)


# summon the blobs
def zzrot_portal_helper(champion):

    item_amount = len(list(filter(lambda x: x == 'zzrot_portal', champion.items)))
    spawn_hexes = field.hexes_in_distance(champion.y, champion.x, 3)

    coords = field.coordinates

    for i, s in enumerate(spawn_hexes):
        d = field.distance({'y': s[0], 'x': s[1]}, {'y': champion.y, 'x': champion.x}, False)
        spawn_hexes[i].append(d)
    random.shuffle(spawn_hexes)
    spawn_hexes = sorted(spawn_hexes, key=lambda x: x[2])

    # find us a list of free hexes
    spawn_hexes = list(filter(lambda x: not coords[x[0]][x[1]], spawn_hexes))
    spawn_hexes = spawn_hexes[:item_amount]

    for s in spawn_hexes:
        champion.print(' zzrot_portal spawns a construct at {}'.format([s[0], s[1]]))
        champion.spawn('construct', champion.stars, s[0], s[1])


def duelists_zeal(champion):
    if champion.team:
        origin_class.amounts['duelist'][champion.team] += \
            len(list(filter(lambda x: x == 'duelists_zeal', champion.items)))


def elderwood_heirloom(champion):
    if champion.team:
        origin_class.amounts['elderwood'][champion.team] += \
            len(list(filter(lambda x: x == 'elderwood_heirloom', champion.items)))


def mages_cap(champion):
    if champion.team:
        origin_class.amounts['mage'][champion.team] += \
            len(list(filter(lambda x: x == 'mages_cap', champion.items)))


def mantle_of_dusk(champion):
    if champion.team:
        origin_class.amounts['dusk'][champion.team] += \
            len(list(filter(lambda x: x == 'mantle_of_dusk', champion.items)))


def sword_of_the_divine(champion):
    if champion.team:
        origin_class.amounts['divine'][champion.team] +=\
            len(list(filter(lambda x: x == 'sword_of_the_divine', champion.items)))


def vanguards_cuirass(champion):
    if champion.team:
        origin_class.amounts['vanguard'][champion.team] += \
            len(list(filter(lambda x: x == 'vanguards_cuirass', champion.items)))


def warlords_banner(champion):
    if champion.team:
        origin_class.amounts['warlord'][champion.team] += \
            len(list(filter(lambda x: x == 'warlords_banner', champion.items)))


def youmuus_ghostblade(champion):
    if champion.team:
        origin_class.amounts['assassin'][champion.team] += \
            len(list(filter(lambda x: x == 'youmuus_ghostblade', champion.items)))

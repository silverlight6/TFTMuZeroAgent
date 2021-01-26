items = {
    'bf_sword'                  : {'AD': 15},
    'chain_vest'                : {'armor': 25},
    'giants_belt'               : {'health': 200},
    'needlessly_large_rod'      : {'SP': 0.15},
    'negatron_cloak'            : {'MR': 25},
    'recurve_bow'               : {'AS': 1.15},
    'sparring_gloves'           : {'crit_chance': 0.10, 'dodge': 0.1},
    'spatula'                   : {},
    'tear_of_the_goddess'       : {'mana': 15},

    'bloodthirster'             : {'AD': 15, 'MR': 25, 'lifesteal': 0.40},
    'blue_buff'                 : {'mana': 30}, #in champions.py: spell()
    'bramble_vest'              : {'armor': 50}, #in champions.py: spell() and champion_functions.py: attack()
    'chalice_of_power'          : {'mana': 15, 'MR': 25}, #in champion.py: main()
    'deathblade'                : {'AD': 50}, #in champions.py: spell() and champion_functions.py: attack()
    'dragons_claw'              : {'spell_damage_reduction_percentage': 0.4, 'MR': 50}, #inverse value in 'spell_damage_reduction_percentage'
    'duelists_zeal'             : {'AS': 1.15}, 
    'elderwood_heirloom'        : {'MR': 15}, 
    'force_of_nature'           : {}, 
    'frozen_heart'              : {'armor': 25, 'mana': 15}, #in champions.py: move(), champions.py: main() and champions.py: die()
    'gargoyle_stoneplate'       : {'armor': 25, 'MR': 25}, #in champions.py: spell() and champion_functions.py: attack()
    'giant_slayer'              : {'AD': 15, 'AS': 1.15}, #in champions.py: spell() and champion_functions.py: attack()
    'guardian_angel'            : {'AD': 15, 'armor': 25, 'will_revive': [[None], ['guardian_angel']]}, #in items.py: initiate(), gets triggered in champion_functions.py: die()
    'guinsoos_rageblade'        : {'AS': 1.15, 'SP': 0.15}, #in champion_functions.py: attack(), ability.py: ashe_helper()
    'hand_of_justice'           : {'crit_chance': 0.10, 'dodge': 0.10, 'mana': 15}, #in champions.py: main()
    'hextech_gunblade'          : {'AD': 15, 'SP': 0.15}, #in champions.py: spell()
    'infinity_edge'             : {'AD': 15, 'crit_chance': 0.2}, #in champions.py: main() 
    'ionic_spark'               : {'SP': 0.15, 'MR': 25}, #in champions.py: move(), champions.py: main() and champions.py: die(). zapping in ability.py: default_ability_calls()
    'jeweled_gauntlet'          : {'SP': 0.15, 'crit_chance': 0.2, 'crit_damage': 0.4}, #in champions.py: spell()
    'last_whisper'              : {'AS': 1.15, 'crit_chance': 0.2}, #in champion_functions.py: attack()
    'locket_of_the_iron_solari' : {'SP': 0.15, 'armor': 25}, #in champion.py: main()
    'ludens_echo'               : {'SP': 0.15, 'mana': 15}, #in champions.py: spell(), ability.py: default_ability_calls()
    'mages_cap'                 : {'mana': 15}, #in items.py: initiate()
    'mantle_of_dusk'            : {'SP': 0.15},
    'morellonomicon'            : {'SP': 0.15, 'health': 200}, #in champions.py: spell()
    'quicksilver'               : {'MR': 25, 'dodge': 0.20}, #in champions.py: change_stat(), ability.py: cassiopeia(), 
    'rabadons_deathcap'         : {'SP': 0.80}, 
    'rapid_firecannon'          : {'AS': 1.30}, #in champions.py: change_stat(), champion_functions: attack(), items.py: initiate()
    'redemption'                : {'mana': 15, 'health': 200}, #in champions.py: die()
    'runaans_hurricane'         : {'AS': 1.15, 'MR': 25}, #in champion_functions.py: attack()
    'shroud_of_stillness'       : {'armor': 25, 'dodge': 0.20}, #in champions.py: main()
    'spear_of_shojin'           : {'AD': 15, 'mana': 15}, #in champion_functions.py: attack()
    'statikk_shiv'              : {'AS': 1.15, 'mana': 15}, #in champion_functions.py: attack()
    'sunfire_cape'              : {'armor': 25, 'health': 200}, #in items.py: initiate()
    'sword_of_the_divine'       : {'AD': 15}, 
    'thiefs_gloves'             : {'crit_chance': 0.2, 'dodge': 0.2}, #in items.py: initiate()
    'titans_resolve'            : {'AS': 1.15, 'armor': 20}, #in champions.py: spell() and champion_functions.py: attack()
    'trap_claw'                 : {'health': 200, 'dodge': 0.20}, #in champions.py: spell()
    'vanguards_cuirass'         : {'armor': 25}, 
    'warlords_banner'           : {'health': 200}, 
    'warmogs_armor'             : {'health': 1000},
    'youmuus_ghostblade'        : {'crit_chance': 0.10, 'dodge': 0.10}, 
    'zekes_herald'              : {'AD': 15, 'health': 200}, #in champion.py: main()
    'zephyr'                    : {'health': 200, 'MR': 25}, #in champion.py: main()
    'zzrot_portal'              : {'AS': 1.15, 'health': 200}, #in champion.py: main(), champion object, champion_functions.py: die(), 
}

trait_items = {
    'duelist'       : 'duelists_zeal',
    'elderwood'     : 'elderwood_heirloom',
    'mage'          : 'mages_cap',
    'dusk'          : 'mantle_of_dusk',
    'divine'        : 'sword_of_the_divine',
    'vanguard'      : 'vanguards_cuirass',
    'warlord'       : 'warlords_banner',
    'assassin'      : 'youmuus_ghostblade',

}

SP = {
    'chalice_of_power': 0.35,
    'hand_of_justice': 0.45,
}

AD ={
    'deathblade': 20,
}

AD_percentage = {
    'hand_of_justice': 1.45,
}

crit_chance = {
    'infinity_edge': 0.55
}

damage = {
    'bramble_vest': [0, 80, 100, 150],
    'ionic_spark': 2.25, #% of max mana
    'ludens_echo': 180,
    'statikk_shiv': 80,
    'runaans_hurricane': 0.90
}

true_damage = {'statikk_shiv': 240}

armor = {
    'gargoyle_stoneplate': 15,
    'titans_resolve': 25,
}

MR = {
    'gargoyle_stoneplate': 15,
    'titans_resolve': 25,
}

item_mr_decrease    = {'ionic_spark':  0.60} #inverted
item_armor_decrease = {'last_whisper': 0.25} #inverted
item_deal_increased_damage_increase = {'titans_resolve': 0.02}

heal = {
    'guardian_angel': 400,
    'redemption': 800,
}

heal_percentage = {
    'hextech_gunblade': 0.33
}

shield = {
    'locket_of_the_iron_solari': [0, 300, 375, 500, 800],
}

shield_max = {
    'hextech_gunblade': 400
}

item_as_decrease = {
    'frozen_heart': 0.5,
}

item_as_increase = {
    'guinsoos_rageblade': 1.06,
    'zekes_herald': 1.35
}

lifesteal = {
    'hand_of_justice': 0.45,
}

lifesteal_spells = {
    'hand_of_justice': 0.45,
}

item_mana_cost_increase = {'shroud_of_stillness': 0.33}
mana = {'spear_of_shojin': 5}

item_stun_duration = {
    'trap_claw': 4000,
    'zephyr': 5000,
}

item_range = {
    'ionic_spark': 2,
    'ludens_echo': 2,
    'sunfire_cape': 2
}

item_targets = {
    'ludens_echo': 3, #target + 3
    'statikk_shiv': [0, 2, 3, 4, 8] #target + x
}

item_max_stacks = {'titans_resolve': 25}

item_change_length = {
    'last_whisper': 3000,
    'locket_of_the_iron_solari': 8000,
    'quicksilver': 10000,
    'zzrot_portal': 1000
}

cooldown = {
    'bramble_vest': 2500,
    'guardian_angel': 2000,
    'sunfire_cape': 2000,
}

item_activate_every_x_attacks = {'statikk_shiv': 3}

initiative_items = [
    'rapid_firecannon', 
    'sunfire_cape', 
    'thiefs_gloves', 
    'duelists_zeal',
    'elderwood_heirloom',
    'mages_cap',
    'mantle_of_dusk',
    'sword_of_the_divine',
    'vanguards_cuirass',
    'warlords_banner',
    'youmuus_ghostblade'
] #these items will call their items.py -function from items.initiate() 
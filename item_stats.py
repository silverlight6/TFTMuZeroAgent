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
    'blue_buff'                 : {'mana': 30},
    'bramble_vest'              : {'armor': 50},
    'chalice_of_power'          : {'mana': 15, 'MR': 25},
    'deathblade'                : {'AD': 50},
    'dragons_claw'              : {'spell_damage_reduction_percentage': 0.4, 'MR': 50},
    'duelists_zeal'             : {'AS': 1.15}, 
    'elderwood_heirloom'        : {'MR': 15}, 
    'force_of_nature'           : {}, 
    'frozen_heart'              : {'armor': 25, 'mana': 15},
    'gargoyle_stoneplate'       : {'armor': 25, 'MR': 25},
    'giant_slayer'              : {'AD': 15, 'AS': 1.15},
    'guardian_angel'            : {'AD': 15, 'armor': 25, 'will_revive': [[None], ['guardian_angel']]},
    'guinsoos_rageblade'        : {'AS': 1.15, 'SP': 0.15},
    'hand_of_justice'           : {'crit_chance': 0.10, 'dodge': 0.10, 'mana': 15},
    'hextech_gunblade'          : {'AD': 15, 'SP': 0.15},
    'infinity_edge'             : {'AD': 15, 'crit_chance': 0.2},
    'ionic_spark'               : {'SP': 0.15, 'MR': 25},
    'jeweled_gauntlet'          : {'SP': 0.15, 'crit_chance': 0.2, 'crit_damage': 0.4},
    'last_whisper'              : {'AS': 1.15, 'crit_chance': 0.2},
    'locket_of_the_iron_solari' : {'SP': 0.15, 'armor': 25},
    'ludens_echo'               : {'SP': 0.15, 'mana': 15},
    'mages_cap'                 : {'mana': 15},
    'mantle_of_dusk'            : {'SP': 0.15},
    'morellonomicon'            : {'SP': 0.15, 'health': 200},
    'quicksilver'               : {'MR': 25, 'dodge': 0.20},
    'rabadons_deathcap'         : {'SP': 0.80}, 
    'rapid_firecannon'          : {'AS': 1.30},
    'redemption'                : {'mana': 15, 'health': 200},
    'runaans_hurricane'         : {'AS': 1.15, 'MR': 25},
    'shroud_of_stillness'       : {'armor': 25, 'dodge': 0.20},
    'spear_of_shojin'           : {'AD': 15, 'mana': 15},
    'statikk_shiv'              : {'AS': 1.15, 'mana': 15},
    'sunfire_cape'              : {'armor': 25, 'health': 200},
    'sword_of_the_divine'       : {'AD': 15}, 
    'thiefs_gloves'             : {'crit_chance': 0.2, 'dodge': 0.2},
    'titans_resolve'            : {'AS': 1.15, 'armor': 20},
    'trap_claw'                 : {'health': 200, 'dodge': 0.20},
    'vanguards_cuirass'         : {'armor': 25}, 
    'warlords_banner'           : {'health': 200}, 
    'warmogs_armor'             : {'health': 1000},
    'youmuus_ghostblade'        : {'crit_chance': 0.10, 'dodge': 0.10}, 
    'zekes_herald'              : {'AD': 15, 'health': 200},
    'zephyr'                    : {'health': 200, 'MR': 25},
    'zzrot_portal'              : {'AS': 1.15, 'health': 200},
}

item_builds = {
    'bloodthirster'             : ['bf_sword', 'negatron_cloak'],
    'blue_buff'                 : ['tear_of_the_goddess', 'tear_of_the_goddess'],
    'bramble_vest'              : ['chain_vest', 'chain_vest'],
    'chalice_of_power'          : ['tear_of_the_goddess', 'negatron_cloak'],
    'deathblade'                : ['bf_sword', 'bf_sword'],
    'dragons_claw'              : ['negatron_cloak', 'negatron_cloak'],
    'duelists_zeal'             : ['recurve_bow', 'spatula'],
    'elderwood_heirloom'        : ['negatron_cloak', 'spatula'],
    'force_of_nature'           : ['spatula', 'spatula'],
    'frozen_heart'              : ['tear_of_the_goddess', 'chain_vest'],
    'gargoyle_stoneplate'       : ['chain_vest', 'negatron_cloak'],
    'giant_slayer'              : ['bf_sword', 'recurve_bow'],
    'guardian_angel'            : ['bf_sword', 'chain_vest'],
    'guinsoos_rageblade'        : ['recurve_bow', 'needlessly_large_rod'],
    'hand_of_justice'           : ['tear_of_the_goddess', 'sparring_gloves'],
    'hextech_gunblade'          : ['bf_sword', 'needlessly_large_rod'],
    'infinity_edge'             : ['bf_sword', 'sparring_gloves'],
    'ionic_spark'               : ['needlessly_large_rod', 'negatron_cloak'],
    'jeweled_gauntlet'          : ['needlessly_large_rod', 'sparring_gloves'],
    'last_whisper'              : ['recurve_bow', 'sparring_gloves'],
    'locket_of_the_iron_solari' : ['needlessly_large_rod', 'chain_vest'],
    'ludens_echo'               : ['needlessly_large_rod', 'tear_of_the_goddess'],
    'mages_cap'                 : ['tear_of_the_goddess', 'spatula'],
    'mantle_of_dusk'            : ['needlessly_large_rod', 'spatula'],
    'morellonomicon'            : ['needlessly_large_rod', 'giants_belt'],
    'quicksilver'               : ['negatron_cloak', 'sparring_gloves'],
    'rabadons_deathcap'         : ['needlessly_large_rod', 'needlessly_large_rod'],
    'rapid_firecannon'          : ['recurve_bow', 'recurve_bow'],
    'redemption'                : ['tear_of_the_goddess', 'giants_belt'],
    'runaans_hurricane'         : ['recurve_bow', 'negatron_cloak'],
    'shroud_of_stillness'       : ['chain_vest', 'sparring_gloves'],
    'spear_of_shojin'           : ['bf_sword', 'tear_of_the_goddess'],
    'statikk_shiv'              : ['recurve_bow', 'tear_of_the_goddess'],
    'sunfire_cape'              : ['chain_vest', 'giants_belt'],
    'sword_of_the_divine'       : ['bf_sword', 'spatula'],
    'thiefs_gloves'             : ['sparring_gloves', 'sparring_gloves'],
    'titans_resolve'            : ['chain_vest', 'recurve_bow'],
    'trap_claw'                 : ['giants_belt', 'sparring_gloves'],
    'vanguards_cuirass'         : ['chain_vest', 'spatula'],
    'warlords_banner'           : ['giants_belt', 'spatula'],
    'warmogs_armor'             : ['giants_belt', 'giants_belt'],
    'youmuus_ghostblade'        : ['sparring_gloves', 'spatula'],
    'zekes_herald'              : ['bf_sword', 'giants_belt'],
    'zephyr'                    : ['giants_belt', 'negatron_cloak'],
    'zzrot_portal'              : ['recurve_bow', 'giants_belt'],
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
    'ionic_spark': 2.25,  # % of max mana
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

item_mr_decrease = {'ionic_spark':  0.60}  # inverted
item_armor_decrease = {'last_whisper': 0.25}  # inverted
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
    'ludens_echo': 3, # target + 3
    'statikk_shiv': [0, 2, 3, 4, 8] # target + x
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
] # these items will call their items.py -function from items.initiate() 

basic_items = [
    'bf_sword',
    'chain_vest',
    'giants_belt',
    'needlessly_large_rod',
    'negatron_cloak',
    'recurve_bow',
    'sparring_gloves',
    'spatula',
    'tear_of_the_goddess',
]

starting_items = [
    'bf_sword',
    'chain_vest',
    'giants_belt',
    'needlessly_large_rod',
    'negatron_cloak',
    'recurve_bow',
    'sparring_gloves',
    'tear_of_the_goddess',
]
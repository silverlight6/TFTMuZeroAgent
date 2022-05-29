# level_percentage = [
#     [1, 0, 0, 0, 0],
#     [1, 0, 0, 0, 0],
#     [.75, .25, 0, 0, 0],
#     [.55, .30, .15, 0, 0],
#     [.45, .33, .20, .2, 0],
#     [.25, .40, .30, .5, 0],
#     [.19, .30, .35, .15, .1],
#     [.15, .20, .35, .25, .5],
#     [.1, .15, .3, .3, .15],
#     [.5, .10, .20, .40, .25],
#     [.1, .2, .12, .5, .35],
# ]

# The code should never get past the 1 so the 10s are irrelevant. Only there to catch possible bugs
level_percentage = [
    [1, 10, 10, 10, 10],
    [1, 10, 10, 10, 10],
    [1, 10, 10, 10, 10],
    [.75, 1, 10, 10, 10],
    [.55, .85, 1, 10, 10],
    [.45, .775, .975, 1, 10],
    [.25, .65, .95, 1, 10],
    [.19, .49, .84, .99, 1],
    [.15, .35, .7, .95, 1],
    [.1, .25, .55, .85, 1],
    [.5, .15, .35, .75, 1],
    [.01, .03, .15, .65, 1],
]

# 29
COST_1 = {
    'diana': 29,
    'elise': 29,
    'fiora': 29,
    'garen': 29,
    'lissandra': 29,
    'maokai': 29,
    'nami': 29,
    'nidalee': 29,
    'tahmkench': 29,
    'twistedfate': 29,
    'vayne': 29,
    'wukong': 29,
    'yasuo': 29,
}

# 22
COST_2 = {
    'annie': 22,
    'aphelios': 22,
    'hecarim': 22,
    'janna': 22,
    'jarvaniv': 22,
    'jax': 22,
    'lulu': 22,
    'pyke': 22,
    'sylas': 22,
    'teemo': 22,
    'thresh': 22,
    'vi': 22,
    'zed': 22,
}

# 18
COST_3 = {
    'akali': 18,
    'evelynn': 18,
    'irelia': 18,
    'jinx': 18,
    'kalista': 18,
    'katarina': 18,
    'kennen': 18,
    'kindred': 18,
    'lux': 18,
    'nunu': 18,
    'veigar': 18,
    'xinzhao': 18,
    'yuumi': 18,
}

# 12 
COST_4 = {
    'aatrox': 12,
    'ahri': 12,
    'ashe': 12,
    'cassiopeia': 12,
    'jhin': 12,
    'morgana': 12,
    'riven': 12,
    'sejuani': 12,
    'shen': 12,
    'talon': 12,
    'warwick': 12,
}

# 10
COST_5 = {
    'azir': 10,
    'ezreal': 10,
    'kayn': 10,
    'leesin': 10,
    'lillia': 10,
    'sett': 10,
    'yone': 10,
    'zilean': 10,
}

chosen_stats = [
    [1.0, 10, 10, 10, 10],
    [1.0, 10, 10, 10, 10],
    [1.0, 10, 10, 10, 10],
    [1.0, 10, 10, 10, 10],
    [.8, 1.0, 10, 10, 10],
    [.4, .95, 1.0, 10, 10],
    [0, .6, 1.0, 10, 10],
    [0, .4, .98, 1.0, 10],
    [0, 0, .6, 1.0, 10],
    [0, 0, 0, .6, 1.0]
]

cost_star_values = [
    [1, 3, 9],
    [2, 5, 17],
    [3, 8, 26],
    [4, 11, 35],
    [5, 14, 44]
]

base_pool_values = [29, 22, 18, 12, 10]

import config
from Set12Simulator.item_stats import uncraftable_items, items

CHAMPION_INFORMATION = 12
BOARD_X = 7
BOARD_Y = 4
BOARD_SIZE = BOARD_X * BOARD_Y
BENCH_SIZE = 9
SHOP_SIZE = 5
ITEM_BENCH_SIZE = 10
MAX_CHAMPION_IN_SET = 62
UNCRAFTABLE_ITEM = len(uncraftable_items)
MAX_BENCH_SPACE = 10
MAX_ITEMS_IN_SET = len(list(items.keys()))


NUM_PLAYERS = config.NUM_PLAYERS
LOG_COMBAT = False

PRINTMESSAGES = True
LOGMESSAGES = True
MANA_DAMAGE_GAIN = 0.06
MAX_MANA_FROM_DAMAGE = 42.5

MOVEMENTDELAY = 550
STARMULTIPLIER = 1.8

ATTACK_PASSIVES = ["vayne", "jhin", "kalista", "warwick", "zed"]

MANA_PER_ATTACK = 10

BURN_SECONDS = 10
BURN_DMG_PER_SLICE = 0.025
BURN_HEALING_REDUCE = 0.5

# unit name
CHOSEN = None

GALIO_MULTIPLIER = 0.14
GALIO_TEAM_HEALTH_PERCENTAGE = 0.50

WARLORD_WINS = {"blue": 0, "red": 0}

LEAP_DELAY = 395  # assassins and shades

MATCHMAKING_WEIGHTS = 10
WEIGHTS_INCREMENT = 3

CRIT_CHANCE = 0
CRIT_DAMAGE = 1.5
SP = 1

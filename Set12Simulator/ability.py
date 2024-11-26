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



def ahri(champion):
    return

def akali(champion):
    return

def ashe(champion):
    return

def bard(champion):
    return

def blitzcrank(champion):
    return

def briar(champion):
    return

def camille(champion):
    return

def cassiopeia(champion):
    return

def diana(champion):
    return

def elise(champion):
    return

def ezreal(champion):
    return

def fiora(champion):
    return

def galio(champion):
    return

def gwen(champion):
    return

def hecarim(champion):
    return

def hwei(champion):
    return

def jax(champion):
    return

def jayce(champion):
    return

def jinx(champion):
    return

def kalista(champion):
    return

def karma(champion):
    return

def kassadin(champion):
    return

def katarina(champion):
    return

def kogmaw(champion):
    return

def lillia(champion):
    return

def milio(champion):
    return

def mordekaiser(champion):
    return

def morgana(champion):
    return

def nami(champion):
    return

def nasus(champion):
    return

def neeko(champion):
    return

def nilah(champion):
    return

def nomsy(champion):
    return

def norra_yuumi(champion):
    return

def nunu(champion):
    return

def olaf(champion):
    return

def poppy(champion):
    return

def rakan(champion):
    return

def rumble(champion):
    return

def ryze(champion):
    return

def seraphine(champion):
    return

def shen(champion):
    return

def shyvana(champion):
    return

def smolder(champion):
    return

def soraka(champion):
    return

def swain(champion):
    return

def syndra(champion):
    return

def tahm_kench(champion):
    return

def taric(champion):
    return

def tristana(champion):
    return

def twitch(champion):
    return

def varus(champion):
    return

def veigar(champion):
    return

def vex(champion):
    return

def warwick(champion):
    return

def wukong(champion):
    return

def xerath(champion):
    return

def ziggs(champion):
    return

def zilean(champion):
    return

def zoe(champion):
    return


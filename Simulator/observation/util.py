from Simulator.item_stats import items, trait_items, basic_items, item_builds
from Simulator.pool_stats import cost_star_values
import numpy as np

CHAMPIONS = [
    'aatrox', 'ahri', 'akali', 'annie', 'aphelios', 'ashe', 'azir',
    'cassiopeia',
    'diana',
    'elise', 'evelynn', 'ezreal',
    'fiora',
    'garen',
    'hecarim',
    'irelia',
    'janna', 'jarvaniv', 'jax', 'jhin', 'jinx',
    'kalista', 'katarina', 'kayn', 'kennen', 'kindred',
    'leesin', 'lillia', 'lissandra', 'lulu', 'lux',
    'maokai', 'morgana',
    'nami', 'nidalee', 'nunu', 
    'pyke',
    'riven',
    'sejuani', 'sett', 'shen', 'sylas',
    'tahmkench', 'talon', 'teemo', 'thresh', 'twistedfate',
    'vayne', 'veigar', 'vi',
    'warwick', 'wukong',
    'xinzhao',
    'yasuo', 'yone', 'yuumi',
    'zed', 'zilean',
    'target_dummy'
]

ITEMS = [
    'bf_sword', 'chain_vest', 'giants_belt', 
    'needlessly_large_rod', 'negatron_cloak', 'recurve_bow', 
    'sparring_gloves', 'spatula', 'tear_of_the_goddess',

    'bloodthirster', 'blue_buff', 'bramble_vest',
    'chalice_of_power',
    'deathblade', 'dragons_claw',
    'duelists_zeal',
    'elderwood_heirloom',
    'force_of_nature', 'frozen_heart',
    'gargoyle_stoneplate', 'giant_slayer', 'guardian_angel', 'guinsoos_rageblade',
    'hand_of_justice', 'hextech_gunblade',
    'infinity_edge', 'ionic_spark',
    'jeweled_gauntlet', 
    'last_whisper', 'locket_of_the_iron_solari', 'ludens_echo',
    'mages_cap', 'mantle_of_dusk', 'morellonomicon',
    'quicksilver',
    'rabadons_deathcap', 'rapid_firecannon', 'redemption', 'runaans_hurricane',
    'shroud_of_stillness', 'spear_of_shojin',
    'statikk_shiv', 'sunfire_cape', 'sword_of_the_divine',
    'thieves_gloves', 'titans_resolve', 'trap_claw',
    'vanguards_cuirass',
    'warlords_banner', 'warmogs_armor',
    'youmuus_ghostblade',
    'zekes_herald', 'zephyr', 'zzrot_portal',

    'kayn_rhast', 'kayn_shadowassassin',

    'champion_duplicator', 'magnetic_remover', 'reforger',
]

TRAITS = [
    'cultist',
    'divine', 'dusk',
    'elderwood', 'enlightened', 'exile',
    'ninja',
    'spirit',
    'the_boss',
    'warlord',

    'adept', 'assassin',
    'brawler',
    'dazzler', 'duelist',
    'emperor',
    'fortune',
    'hunter',
    'keeper',
    'mage', 'moonlight', 'mystic',
    'shade', 'sharpshooter',
    'tormented',
    'vanguard'
]

class Util:
    def __init__(self):
        # -- Champions -- #
        self.champion_ids = {
            champion: i + 1 for i, champion in enumerate(CHAMPIONS)
        }   

        # -- Items -- #
        self.item_ids = {
            item: i + 1 for i, item in enumerate(ITEMS)
        }
        self.item_components = set(basic_items)
        self.full_items = set(item_builds.keys())
        
        # -- Traits -- #
        self.trait_ids = {
            trait: i + 1 for i, trait in enumerate(TRAITS)
        }
        # Reverse trait_items dictionary to get the trait from the item
        self.item_traits = {v: k for k, v in trait_items.items()}
        
        # -- Other -- #
        self.chosen_table = np.eye(2)
        self.stars_table = np.eye(4)

    # -- Champions -- #
    def get_champion_id(self, champion) -> int:
        if champion.target_dummy:
            return self.champion_ids['target_dummy']
        return self.champion_ids[champion.name]
    
    def get_champion_cost(self, champion) -> int:
        return cost_star_values[champion.cost - 1][champion.stars - 1]
    
    # -- Items -- #
    def get_item_id(self, item) -> int:
        if item is None:
            return 0
        return self.item_ids[item]
    
    def get_item_stats(self, item) -> dict:
        return items[item]

    def is_item_component(self, item) -> bool:
        return item in self.item_components

    def is_full_item(self, item) -> bool:
        return item in self.full_items
    
    # -- Traits -- #
    def get_trait_id(self, trait) -> int:
        return self.trait_ids[trait]

    def get_item_trait(self, item) -> str:
        if item in self.item_traits:
            return self.item_traits[item]
        return None

    # -- Other -- #
    def chosen_one_hot(self, chosen) -> np.ndarray:
        chosenID = 1 if chosen else 0
        return self.chosen_table[chosenID]
    
    def stars_one_hot(self, stars) -> np.ndarray:
        return self.stars_table[stars - 1]

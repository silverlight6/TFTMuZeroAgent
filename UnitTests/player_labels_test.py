import numpy as np

from Simulator.battle_generator import BattleGenerator, base_level_config
from Simulator.stats import COST

def tier_label_set_comp_test(battle_generator):
    [player, _, _] = battle_generator.generate_set_battle(8)
    initial_tier_labels = player.get_tier_labels()
    player.update_team_tiers()
    tier_labels = list(player.team_tiers.values())
    for i, tier in enumerate(initial_tier_labels):
        assert np.argmax(tier) == tier_labels[i]

def comp_label_set_comp_test(battle_generator):
    [player, _, _] = battle_generator.generate_set_battle(8)
    initial_comp_labels = player.get_champion_labels()
    keys = list(COST.keys())
    for x in range(len(player.board)):
        for y in range(len(player.board[0])):
            if player.board[x][y]:
                index = keys.index(player.board[x][y].name)
                assert np.argmax(initial_comp_labels[index - 1]) == 1


# Create a test using the battle generator and a set comp to compare the labels against.
def test_list():
    generator_config = {**base_level_config, "stationary": True}
    battle_generator = BattleGenerator(generator_config)
    tier_label_set_comp_test(battle_generator)
    comp_label_set_comp_test(battle_generator)
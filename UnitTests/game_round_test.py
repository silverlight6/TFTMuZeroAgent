import config
from copy import deepcopy
from Simulator import pool
from Simulator.battle_generator import BattleGenerator
from Simulator.player_manager import PlayerManager
from Simulator.observation.vector.observation import ObservationVector
from Simulator.game_round import Game_Round
from Simulator.tft_simulator import TFTConfig


def test_single_player_combat():
    base_level_config = {
        "num_unique_champions": 3,
        "max_cost": 1,
        "num_items": 0,
        "current_level": 3,
        "chosen": False,
        "sample_from_pool": False,
        "two_star_unit_percentage": 0,
        "three_star_unit_percentage": 0,
        "scenario_info": True,
        "extra_randomness": False
    }
    battle_generator = BattleGenerator(base_level_config)
    [player, opponent, other_players] = battle_generator.generate_battle()

    PLAYER = player
    PLAYER.opponent = opponent
    opponent.opponent = PLAYER

    pool_obj = pool.pool()
    player_manager = PlayerManager(config.NUM_PLAYERS, pool_obj, TFTConfig(observation_class=ObservationVector))
    player_manager.reinit_player_set([PLAYER] + list(other_players.values()))

    player_copy = deepcopy(PLAYER)
    assert player_copy == PLAYER, "before single combat"

    game_round = Game_Round(PLAYER, pool_obj, player_manager)
    result, damage = game_round.single_combat_phase([PLAYER, PLAYER.opponent])

    if result == 0 or result == 2:
        player_copy.loss_round(damage)

    if result == 1:
        player_copy.won_round(damage)

    assert player_copy == PLAYER, "after single combat"

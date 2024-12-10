import config
from copy import deepcopy
from Simulator import pool
from Simulator.battle_generator import BattleGenerator, base_level_config
from Simulator.player_manager import PlayerManager
from Simulator.observation.vector.observation import ObservationVector
from Simulator.game_round import Game_Round
from Simulator.tft_config import TFTConfig


def test_single_player_combat():
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

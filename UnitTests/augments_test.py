from Set12Simulator.augment import AugmentType, Augment
from Set12Simulator.player import Player
from Set12Simulator import pool


def test_afk_augment():
    pool_pointer = pool.pool()
    player_id = 0
    player = Player(pool_pointer=pool_pointer, player_num=player_id)
    player.augments = [Augment(AugmentType.AFK, 0)]
    player.start_round(1)
    assert player.is_afk == True, "afk augment is_afk"
    assert player.gold == 2, "afk augment is_afk"
    player.start_round(2)
    assert player.is_afk == True, "afk augment is_afk"
    assert player.gold == 4, "afk augment is_afk"
    player.start_round(3)
    assert player.gold == 27, "after afk augment"
    assert player.is_afk == False, "after afk augment"
from enum import Enum, auto


class Augment(Enum):
    AFK = auto()

    @staticmethod
    def is_augment_round(round_:int) -> bool:
        return round_ in [3,10,17]


class SelectedAugmentConfig:
    round: int
    augment: Augment

    def __init__(self, augment: Augment, round_: int):
        self.round = round_
        self.augment = augment


def handle_start_round_augment(player, augment_config: SelectedAugmentConfig):
    """
    Handle modifications to the player at the start of the round
    """
    if augment_config.augment == Augment.AFK:
        afk_augment_get_golds(player,augment_config)



def handle_can_perform_action_augment(round_:int, augment_config:SelectedAugmentConfig)-> bool:
    """
    Check if the player can perform an action based on the augment
    """
    if augment_config.augment == Augment.AFK:
        return afk_augment_can_perform_action(round_, augment_config)
    return True



def afk_augment_get_golds(player, augment_config: SelectedAugmentConfig):
    """
    Update the player gold after the afk augment become inactive
    """
    if player.round == augment_config.round + 3:
        player.gold += 20


def afk_augment_can_perform_action(round_:int, augment_config: SelectedAugmentConfig) -> bool:
    """
    Check if the afk augment is active
    """
    return round_ >= augment_config.round + 3

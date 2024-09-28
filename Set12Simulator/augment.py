from enum import Enum, auto


class AugmentType(Enum):
    AFK = auto()

class Augment:
    type_: AugmentType
    round_: int

    def __init__(self, type_: AugmentType, round_: int):
        self.type_ = type_
        self.round_ = round_


    def start_round(self, player):
        if self.type_ == AugmentType.AFK:
            remaining_round = player.round - self.round_ - 3
            if remaining_round < 0:
                player.is_afk = True
            elif remaining_round == 0:
                player.gold += 20


from dataclasses import dataclass

from Simulator.observation.interface import ObservationBase, ActionBase
from Simulator.observation.token.action import ActionToken
from Simulator.observation.token.observation import ObservationToken


@dataclass
class TFTConfig:
    num_players: int = 8
    max_actions_per_round: int = 15
    reward_type: str = "winloss"
    render_mode: str = None  # "json" or None
    render_path: str = "Games"
    observation_class: ObservationBase = ObservationToken
    action_class: ActionBase = ActionToken
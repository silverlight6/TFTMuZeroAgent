from gymnasium.envs.registration import register
from Simulator.tft_simulator import TFT_Simulator

register(
    id="TFT-Set4",
    entry_point="Simulator.tft_simulator:TFT_Simulator"
)

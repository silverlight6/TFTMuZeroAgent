from gymnasium.envs.registration import register
from Simulator.tft_simulator import TFT_Simulator

register(
    id="Simulator/TFT-Set4",
    entry_point="Simulator:TFT_Simulator"
)

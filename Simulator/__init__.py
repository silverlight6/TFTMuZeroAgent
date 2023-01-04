from gymnasium.envs.registration import register

register(
    id='TFTSet4-v0',
    entry_point='Simulator.tft_simulator:TFT_Simulator'
)

register(
    id='tftSingle-v0',
    entry_point='Simulator.single_player_env:Single_Player_Game'
)


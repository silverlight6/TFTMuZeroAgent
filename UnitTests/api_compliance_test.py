"""Official PettingZoo and Gymnasium API checkers for every public env."""

from gymnasium.utils.env_checker import check_env
from pettingzoo.test import api_test, parallel_api_test

from Simulator.encoding.token.basic_observation import ObservationToken
from Simulator.encoding.vector.observation import ObservationVector
from Simulator.simulators.tft_item_simulator import TFT_Item_Simulator
from Simulator.simulators.tft_position_simulator import TFT_Position_Simulator
from Simulator.simulators.tft_simulator import TFTConfig, env as tft_env, parallel_env
from Simulator.simulators.tft_single_player_simulator import TFT_Single_Player_Simulator


def test_pettingzoo_parallel_api_token():
    env = parallel_env(TFTConfig(observation_class=ObservationToken, num_players=8))
    parallel_api_test(env, num_cycles=50)
    env.close()


def test_pettingzoo_parallel_api_vector():
    env = parallel_env(TFTConfig(observation_class=ObservationVector, num_players=8))
    parallel_api_test(env, num_cycles=50)
    env.close()


def test_pettingzoo_aec_api():
    env = tft_env(TFTConfig(observation_class=ObservationToken, num_players=8))
    api_test(env, num_cycles=50)
    env.close()


def test_gymnasium_position_env():
    check_env(TFT_Position_Simulator(), skip_render_check=True)


def test_gymnasium_item_env():
    check_env(TFT_Item_Simulator(), skip_render_check=True)


def test_gymnasium_single_player_env():
    check_env(TFT_Single_Player_Simulator(TFTConfig(num_players=1)), skip_render_check=True)

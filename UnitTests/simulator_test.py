import pytest
from pettingzoo.test import parallel_api_test, api_test

import config
from Simulator.tft_config import TFTConfig
from Simulator.tft_simulator import parallel_env, env as tft_env
from Simulator.observation.vector.observation import ObservationVector

def aec(env):
    return tft_env(env)

def parallel(env):
    return parallel_env(env)


def test_Env():
    """
    PettingZoo's api tests for the simulator.
    """
    tftConfig = TFTConfig(observation_class=ObservationVector, num_players=config.NUM_PLAYERS)
    # raw_env = aec(tftConfig)
    # api_test(raw_env, num_cycles=100000)
    local_env = parallel(tftConfig)
    parallel_api_test(local_env, num_cycles=100000)
